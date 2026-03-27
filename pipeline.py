"""End-to-end pipeline orchestration: parse -> optimize -> plan -> execute."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from parser.semantic_parser import SemanticParser
from pathlib import Path
from typing import Any

import executor.ops  # noqa: F401  # Ensure physical variants are registered.
from catalog.catalog import SystemCatalog
from catalog.indexer import load_catalog
from cost_model.cost_aware_planner import CostAwarePlanner
from executor.runner import ExecutionResult, PlanRunner
from ir.nodes import LogicalNode, PhysicalNode
from optimizer.engine import OptimizerEngine, RewriteEntry
from planner.llm_feedback_planner import LLMFeedbackPhysicalPlanner
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
        self.physical_planner_mode = physical_planner_mode
        self.planning_client = planning_client
        self.planning_rounds = planning_rounds

        self.runner = PlanRunner(corpus=self.corpus, llm=self.llm, catalog=self.catalog)

    async def run(
        self, query: str, log_path: str | Path | None = None
    ) -> PipelineResult:
        """Execute the full pipeline for one query and return all artifacts."""
        logical = self.parse(query)
        optimized, rewrites = self.optimize(logical)
        physical = self.build_physical(optimized)
        execution = await self.runner.run(physical)
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

        if self.physical_planner_mode == "llm_feedback":
            client = self.planning_client
            if client is None and hasattr(self.parser, "client"):
                client = getattr(self.parser, "client")
            feedback_planner = LLMFeedbackPhysicalPlanner(
                cost_planner=cost_planner,
                client=client,
                model=self.model_id or "gpt-4o-mini",
                rounds=self.planning_rounds,
            )
            return feedback_planner.build(logical)

        return cost_planner.build(logical)

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
    return pipeline.run_sync(query)
