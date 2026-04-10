"""End-to-end pipeline orchestration: parse -> optimize -> plan -> execute."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from parser.semantic_parser import SemanticParser
from pathlib import Path
from typing import Any

import executor.ops  # noqa: F401  # Ensure physical variants are registered.
from catalog.catalog import SystemCatalog
from catalog.indexer import load_catalog
from cost_model.judge import AccuracyJudge
from executor.runner import ExecutionResult, PlanRunner
from ir.feedback import Feedback
from ir.nodes import LogicalNode, PhysicalNode
from optimizer.engine import OptimizerEngine, RewriteEntry
from planner.llm_planner import LLMPhysicalPlanner


@dataclass(frozen=True)
class PipelineResult:
    """All artifacts produced by one pipeline run over a task."""

    task_description: str
    sample_queries: list[str]
    logical_plan: LogicalNode
    optimized_plan: LogicalNode
    physical_plan: PhysicalNode
    rewrite_log: list[RewriteEntry]
    executions: list[ExecutionResult]   # one per sample (last iteration)
    feedbacks: list[Feedback]           # one per sample (last iteration)

    # Convenience accessors for single-sample callers.
    @property
    def execution(self) -> ExecutionResult | None:
        return self.executions[-1] if self.executions else None

    @property
    def feedback(self) -> Feedback | None:
        return self.feedbacks[-1] if self.feedbacks else None


class LmOptimizerPipeline:
    """
    Wires parser, optimizer, LLM physical planner, executor, and judge.

    Usage::

        pipeline = LmOptimizerPipeline(corpus=corpus, llm=llm,
                                        planning_client=client)
        result = pipeline.run_sync_with_samples(
            task_description="QA over scientific papers",
            samples=[("What is the main contribution?", "gold answer")],
            iterations=2,
        )
        print(result.execution.answer)
    """

    def __init__(
        self,
        corpus,
        llm,
        planning_client: Any | None = None,
        parser: SemanticParser | Any | None = None,
        optimizer: OptimizerEngine | None = None,
        catalog: SystemCatalog | None = None,
        catalog_path: str | Path | None = None,
        model_id: str | None = None,
        planning_model: str = "gpt-4o-mini",
        judge_model: str = "gpt-4o-mini",
    ):
        self.corpus = corpus
        self.llm = llm
        self.catalog = catalog or self._load_catalog_if_present(catalog_path)
        self.model_id = model_id

        self.parser = parser or SemanticParser(catalog=self.catalog)
        if hasattr(self.parser, "catalog") and getattr(self.parser, "catalog") is None:
            setattr(self.parser, "catalog", self.catalog)

        self.optimizer = optimizer or OptimizerEngine()

        self.planner = (
            LLMPhysicalPlanner(
                client=planning_client,
                model=planning_model,
                catalog=self.catalog,
            )
            if planning_client is not None
            else None
        )

        self.judge = (
            AccuracyJudge(client=planning_client, model=judge_model)
            if planning_client is not None
            else None
        )

        self.runner = PlanRunner(corpus=self.corpus, llm=self.llm, catalog=self.catalog)

    # ── public entry points ────────────────────────────────────────

    async def run_with_samples(
        self,
        task_description: str,
        samples: list[tuple[str, str]],
        iterations: int = 2,
        log_path: str | Path | None = None,
    ) -> PipelineResult:
        """
        Multi-sample iterative pipeline.

        Args:
            task_description: High-level task label passed to the semantic parser.
            samples:          List of (query, gold_answer) pairs sharing this pipeline's
                              corpus. Pass gold_answer="" to skip judging.
            iterations:       Number of execute→judge→revise cycles. Must be >= 1.
        """
        if iterations < 1:
            raise ValueError("iterations must be >= 1")

        sample_queries = [q for q, _ in samples]
        gold_answers = [g for _, g in samples]

        logical = self.parse(task_description, sample_queries)
        optimized, rewrites = self.optimize(logical)
        physical = self._init_physical(task_description, optimized)

        last_executions: list[ExecutionResult] = []
        last_feedbacks: list[Feedback] = []

        for i in range(iterations):
            run_results = await asyncio.gather(
                *[self.runner.run(physical) for _ in samples]
            )

            last_executions = []
            last_feedbacks = []
            for (execution, node_feedbacks), gold_ans in zip(run_results, gold_answers):
                accuracy = await self._judge(execution.answer, gold_ans)
                feedback = Feedback(
                    items=node_feedbacks,
                    accuracy=accuracy,
                    result=execution.answer,
                    gold_ans=gold_ans,
                )
                last_executions.append(execution)
                last_feedbacks.append(feedback)

            if i < iterations - 1 and self.planner is not None:
                physical = self.planner.revise(
                    task_description, optimized, physical, last_feedbacks
                )

        result = PipelineResult(
            task_description=task_description,
            sample_queries=sample_queries,
            logical_plan=logical,
            optimized_plan=optimized,
            physical_plan=physical,
            rewrite_log=rewrites,
            executions=last_executions,
            feedbacks=last_feedbacks,
        )
        if log_path:
            self._save_logs(result, Path(log_path))
        return result

    def run_sync_with_samples(
        self,
        task_description: str,
        samples: list[tuple[str, str]],
        iterations: int = 2,
        log_path: str | Path | None = None,
    ) -> PipelineResult:
        return self._run_async(
            self.run_with_samples(task_description, samples, iterations, log_path)
        )

    # ── pipeline stages ────────────────────────────────────────────

    def parse(self, task_description: str, sample_queries: list[str]) -> LogicalNode:
        return self.parser.parse(task_description, sample_queries)

    def optimize(self, logical: LogicalNode) -> tuple[LogicalNode, list[RewriteEntry]]:
        return self.optimizer.run(logical)

    # ── internals ─────────────────────────────────────────────────

    def _init_physical(self, task_description: str, logical: LogicalNode) -> PhysicalNode:
        if self.planner is not None:
            return self.planner.init(task_description, logical)
        from planner.plan_parser import parse_physical_plan
        return parse_physical_plan(logical.to_dict())

    async def _judge(self, result: str, gold_ans: str) -> float:
        if self.judge is None or not gold_ans.strip():
            return 0.0
        return await self.judge.score(result, gold_ans)

    def _save_logs(self, result: PipelineResult, path: Path) -> None:
        import json

        sample_logs = []
        for query, exec_, fb in zip(
            result.sample_queries, result.executions, result.feedbacks
        ):
            sample_logs.append({
                "query": query,
                "execute": {
                    "answer": exec_.answer,
                    "trace": exec_.trace,
                    "token_counts": exec_.token_counts,
                    "errors": exec_.errors,
                },
                "feedback": {
                    "accuracy": fb.accuracy,
                    "result": fb.result,
                    "gold_ans": fb.gold_ans,
                    "items": [
                        {
                            "op_id": item.op_id,
                            "variant": item.variant,
                            "token_cost": item.token_cost,
                            "latency_ms": round(item.latency_ms, 2),
                            "output_summary": item.output_summary,
                        }
                        for item in fb.items
                    ],
                },
            })

        log_data = {
            "task_description": result.task_description,
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
                "samples": sample_logs,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(log_data, indent=2), encoding="utf-8")

    @staticmethod
    def _run_async(coro):
        import sys
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            def _suppress_loop_closed(loop, context):
                exc = context.get("exception")
                if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
                    return
                loop.default_exception_handler(context)

            loop.set_exception_handler(_suppress_loop_closed)
            try:
                result = loop.run_until_complete(coro)
            finally:
                try:
                    # Drain any lingering cleanup tasks (e.g. httpx aclose)
                    pending = asyncio.all_tasks(loop)
                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                    loop.run_until_complete(loop.shutdown_asyncgens())
                finally:
                    asyncio.set_event_loop(None)
                    loop.close()
            return result
        raise RuntimeError("run_sync*() cannot be called from an active event loop")

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
