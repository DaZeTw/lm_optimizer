"""End-to-end pipeline orchestration: parse -> optimize -> plan -> execute."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import dataclass
from parser.semantic_parser import SemanticParser
from pathlib import Path
from typing import Any

import executor.ops  # noqa: F401  # Ensure physical variants are registered.
from catalog.catalog import SystemCatalog
from catalog.indexer import load_catalog
from cost_model.cost_aware_planner import CostAwarePlanner
from cost_model.scorer import PlanScorer
from cost_model.telemetry import default_telemetry
from executor.runner import ExecutionResult, PlanRunner
from ir.nodes import LogicalNode, PhysicalNode
from ir.ops import Op
from optimizer.engine import OptimizerEngine, RewriteEntry
from planner.llm_feedback_planner import LlmFeedbackPlanner
from planner.physical import build_physical_plan


@dataclass(frozen=True)
class PipelineResult:
    """All artifacts produced by one query run."""

    query: str
    logical_plan: LogicalNode
    optimized_plan: LogicalNode
    physical_plan: PhysicalNode
    rewrite_log: list[RewriteEntry]
    execution: ExecutionResult


class LmOptimizerPipeline:
    """
    High-level runner that wires parser, optimizer, planner, and executor.

    Typical usage:
            pipeline = LmOptimizerPipeline(corpus=corpus, llm=llm)
            result = pipeline.run_sync("What is the main contribution?")
            print(result.execution.answer)
    """

    def __init__(
        self,
        corpus,
        llm,
        parser: SemanticParser | Any | None = None,
        optimizer: OptimizerEngine | None = None,
        catalog: SystemCatalog | None = None,
        catalog_path: str | Path | None = None,
        use_cost_aware_planner: bool = True,
        planner_preset: str = "balanced",
        model_id: str | None = None,
        physical_planner_mode: str = "cost",
        planning_client: Any | None = None,
        planning_rounds: int = 1,
    ):
        self.corpus = corpus
        self.llm = llm
        self.catalog = catalog or self._load_catalog_if_present(catalog_path)
        self.model_id = model_id

        self.parser = parser or SemanticParser(catalog=self.catalog)
        if hasattr(self.parser, "catalog") and getattr(self.parser, "catalog") is None:
            setattr(self.parser, "catalog", self.catalog)

        self.optimizer = optimizer or OptimizerEngine()
        self.use_cost_aware_planner = use_cost_aware_planner
        self.planner_preset = planner_preset
        # Keep for compatibility, but planning now uses a single planner path.
        self.physical_planner_mode = physical_planner_mode
        self.planning_client = planning_client
        self.planning_rounds = planning_rounds
        self._latest_planning_feedback: dict[str, Any] | None = None

        self.runner = PlanRunner(corpus=self.corpus, llm=self.llm, catalog=self.catalog)

    async def run(
        self, query: str, log_path: str | Path | None = None
    ) -> PipelineResult:
        """Execute the full pipeline for one query and return all artifacts."""
        self._latest_planning_feedback = None
        logical = self.parse(query)
        optimized, rewrites = self.optimize(logical)
        physical = self.build_physical(optimized)
        execution = await self.runner.run(physical)
        self._record_execution_feedback(physical, execution)
        self._latest_planning_feedback = self._build_planning_feedback(
            physical,
            execution,
        )
        result = PipelineResult(
            query=query,
            logical_plan=logical,
            optimized_plan=optimized,
            physical_plan=physical,
            rewrite_log=rewrites,
            execution=execution,
        )

        if log_path:
            self._save_logs(result, Path(log_path))

        return result

    async def run_with_physical_feedback(
        self,
        query: str,
        corpus,
        iterations: int,
        log_path: str | Path | None = None,
    ) -> PipelineResult:
        """Run iterative physical planning where each run feeds telemetry for the next."""
        if iterations < 1:
            raise ValueError("iterations must be >= 1")

        runner = PlanRunner(corpus=corpus, llm=self.llm, catalog=self.catalog)

        logical = self.parse(query)
        optimized, rewrites = self.optimize(logical)
        self._latest_planning_feedback = None

        best: PipelineResult | None = None
        for _ in range(iterations):
            physical = self.build_physical(optimized)
            execution = await runner.run(physical)
            self._record_execution_feedback(physical, execution)
            self._latest_planning_feedback = self._build_planning_feedback(
                physical,
                execution,
            )
            candidate = PipelineResult(
                query=query,
                logical_plan=logical,
                optimized_plan=optimized,
                physical_plan=physical,
                rewrite_log=rewrites,
                execution=execution,
            )
            if best is None:
                best = candidate
                continue

            # Keep the run with fewer errors and then fewer tokens.
            best_err = len(best.execution.errors)
            cand_err = len(candidate.execution.errors)
            if cand_err < best_err:
                best = candidate
                continue
            if cand_err == best_err:
                best_tokens = sum(best.execution.token_counts.values())
                cand_tokens = sum(candidate.execution.token_counts.values())
                if cand_tokens <= best_tokens:
                    best = candidate

        if best is None:
            raise RuntimeError("No pipeline result produced")

        if log_path:
            self._save_logs(best, Path(log_path))
        return best

    def _save_logs(self, result: PipelineResult, path: Path):
        """Serializes each phase into a JSON file."""
        log_data = {
            "query": result.query,
            "phases": {
                "parse": result.logical_plan.to_dict(),
                "optimize": {
                    "plan": result.optimized_plan.to_dict(),
                    "rewrite_count": len(result.rewrite_log),
                    "rules_fired": [entry.rule for entry in result.rewrite_log],
                },
                "physical_plan": {
                    "root_variant": result.physical_plan.variant,
                    "params": result.physical_plan.params,
                },
                "execute": {
                    "answer": result.execution.answer,
                    "trace": result.execution.trace,
                    "token_counts": result.execution.token_counts,
                    "errors": result.execution.errors,
                },
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(log_data, indent=2), encoding="utf-8")

    def run_sync(self, query: str) -> PipelineResult:
        """Synchronous wrapper for CLI/scripts."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(query))
        raise RuntimeError("run_sync() cannot be called from an active event loop")

    def run_sync_with_physical_feedback(
        self,
        query: str,
        corpus,
        iterations: int,
    ) -> PipelineResult:
        """Synchronous wrapper for iterative physical-feedback execution."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.run_with_physical_feedback(query, corpus, iterations)
            )
        raise RuntimeError(
            "run_sync_with_physical_feedback() cannot be called from an active event loop"
        )

    def parse(self, query: str) -> LogicalNode:
        return self.parser.parse(query)

    def optimize(self, logical: LogicalNode) -> tuple[LogicalNode, list[RewriteEntry]]:
        return self.optimizer.run(logical)

    def build_physical(self, logical: LogicalNode) -> PhysicalNode:
        if not self.use_cost_aware_planner:
            return build_physical_plan(logical)

        cost_planner = CostAwarePlanner(
            preset=self.planner_preset,
            corpus=self.corpus,
            catalog=self.catalog,
            model_id=self.model_id,
        )

        if (
            self.physical_planner_mode == "llm_feedback"
            and self.planning_client is not None
        ):
            planner = LlmFeedbackPlanner(
                base_planner=cost_planner,
                planning_client=self.planning_client,
                catalog=self.catalog,
                rounds=self.planning_rounds,
            )
            return planner.build(logical, feedback=self._latest_planning_feedback)

        return cost_planner.build(logical)

    def _build_planning_feedback(
        self,
        physical: PhysicalNode,
        execution: ExecutionResult,
    ) -> dict[str, Any]:
        """Build planner feedback payload with overall and per-operator signals."""
        report = PlanScorer(catalog=self.catalog).score(physical)
        per_operator = {}
        for op_id, vector in report.per_node.items():
            per_operator[op_id] = {
                "variant": vector.variant,
                "scalar": round(vector.scalar(), 4),
                "token_cost": round(vector.token_cost, 4),
                "call_cost": round(vector.call_cost, 4),
                "latency_cost": round(vector.latency_cost, 4),
                "quality_risk": round(vector.quality_risk, 4),
                "sample_count": vector.sample_count,
                "accuracy_score": vector.accuracy_score,
            }

        total_tokens = sum(execution.token_counts.values())
        error_count = len(execution.errors)
        quality_score = max(0.0, 1.0 - (error_count / max(1, len(execution.trace))))

        return {
            "overall": {
                "scalar": report.scalar,
                "token_cost": report.total_token_cost,
                "call_cost": report.total_call_cost,
                "latency_cost": report.total_latency_cost,
                "quality_risk": report.total_quality_risk,
                "quality_score": round(quality_score, 4),
                "bottleneck": report.bottleneck,
                "warnings": list(report.warnings),
            },
            "per_operator": per_operator,
            "execution": {
                "errors": list(execution.errors),
                "error_count": error_count,
                "trace": list(execution.trace),
                "trace_count": len(execution.trace),
                "token_counts": dict(execution.token_counts),
                "total_tokens": total_tokens,
            },
        }

    def _record_execution_feedback(
        self,
        physical: PhysicalNode,
        execution: ExecutionResult,
    ) -> None:
        """Record coarse operator-variant telemetry from one execution.

        This is intentionally lightweight for the first iteration: it records
        per-op per-variant token/call/latency/risk signals so later planning
        rounds can exploit observed outcomes.
        """

        nodes: list[PhysicalNode] = []

        def walk(node: PhysicalNode) -> None:
            for child in node.inputs:
                walk(child)
            nodes.append(node)

        walk(physical)
        if not nodes:
            return

        per_variant_nodes = Counter(n.variant for n in nodes)
        trace_counts = Counter(execution.trace)
        total_errors = len(execution.errors)
        accuracy_proxy = max(0.0, 1.0 - (total_errors / max(1, len(execution.trace))))

        for node in nodes:
            variant = node.variant
            op_name = node.logical_ref.op.value
            token_total = float(execution.token_counts.get(variant, 0))
            token_per_node = token_total / max(1, per_variant_nodes[variant])
            call_cost = float(trace_counts.get(variant, 0)) / max(
                1,
                per_variant_nodes[variant],
            )

            # Keep latency as a simple call-proportional proxy until node-level
            # timing is available from the executor.
            latency_cost = max(0.1, call_cost)
            quality_risk = 1.0 - accuracy_proxy
            if node.logical_ref.op == Op.AGGREGATE:
                quality_risk += 0.1

            default_telemetry.record(
                op=op_name,
                variant=variant,
                token_cost=token_per_node,
                call_cost=call_cost,
                latency_cost=latency_cost,
                quality_risk=quality_risk,
                accuracy_score=accuracy_proxy,
            )

    @staticmethod
    def _load_catalog_if_present(
        catalog_path: str | Path | None,
    ) -> SystemCatalog | None:
        if catalog_path is None:
            return None
        path = Path(catalog_path)
        if not path.exists():
            raise FileNotFoundError(f"Catalog file not found: {path}")
        return load_catalog(path)


def run_pipeline_sync(
    query: str,
    corpus,
    llm,
    parser: SemanticParser | Any | None = None,
    optimizer: OptimizerEngine | None = None,
    catalog: SystemCatalog | None = None,
    catalog_path: str | Path | None = None,
    use_cost_aware_planner: bool = True,
    planner_preset: str = "balanced",
    model_id: str | None = None,
    physical_planner_mode: str = "cost",
    planning_client: Any | None = None,
    planning_rounds: int = 1,
) -> PipelineResult:
    """Convenience function for one-off synchronous runs."""
    pipeline = LmOptimizerPipeline(
        corpus=corpus,
        llm=llm,
        parser=parser,
        optimizer=optimizer,
        catalog=catalog,
        catalog_path=catalog_path,
        use_cost_aware_planner=use_cost_aware_planner,
        planner_preset=planner_preset,
        model_id=model_id,
        physical_planner_mode=physical_planner_mode,
        planning_client=planning_client,
        planning_rounds=planning_rounds,
    )
    return pipeline.run_sync(query)
