"""End-to-end pipeline orchestration: task-plan → parse → optimize → plan → execute → revise.

Execution model
---------------
Task-level   (once per run, revised between iterations)
    TaskPlanner.generate()   → TST dict  (logical skeleton + physical policy + adaptation policy)

Query-level  (once per sample per iteration)
    SemanticParser.parse(query, tst)       → LogicalNode  (fills TST skeleton slots for this query)
    OptimizerEngine.run(logical)           → optimized LogicalNode
    LLMPhysicalPlanner.plan(query, ...)    → PhysicalNode (honours TST physical policy)
    PlanRunner.run(physical)               → ExecutionResult

Revision     (once per iteration boundary, when iterations > 1)
    SampleAnalyzer.analyze()  × N         → SampleFeedback dicts
    aggregate_feedback()                   → PatternSummary
    TSTRevisor.revise()                    → revised TST  (or same TST if no patterns)
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from parser.semantic_parser import SemanticParser, TaskPlanner
from pathlib import Path
from typing import Any

import executor.ops  # noqa: F401  # registers physical variants
from catalog.catalog import SystemCatalog
from catalog.indexer import load_catalog
from cost_model.judge import AccuracyJudge
from executor.runner import ExecutionResult, PlanRunner
from ir.feedback import Feedback, NodeFeedback
from ir.nodes import LogicalNode, PhysicalNode
from optimizer.engine import OptimizerEngine, RewriteEntry
from planner.llm_planner import LLMPhysicalPlanner
from reviser.aggregator import aggregate_feedback
from reviser.sample_analyzer import SampleAnalyzer
from reviser.store import FeedbackStore
from reviser.tst_revisor import TSTRevisor


@dataclass(frozen=True)
class SamplePlan:
    """Query-level plans produced for one sample in one iteration."""

    query: str
    logical: LogicalNode
    optimized: LogicalNode
    physical: PhysicalNode
    rewrites: list[RewriteEntry]


@dataclass(frozen=True)
class PipelineResult:
    """All artifacts produced by one pipeline run over a task."""

    task_description: str
    sample_queries: list[str]
    sample_plans: list[SamplePlan]  # one per sample, last iteration
    executions: list[ExecutionResult]  # one per sample, last iteration
    feedbacks: list[Feedback]  # one per sample, last iteration
    tst_versions: list[dict]  # [initial_tst, revised_tst, ...]

    @property
    def execution(self) -> ExecutionResult | None:
        return self.executions[-1] if self.executions else None

    @property
    def feedback(self) -> Feedback | None:
        return self.feedbacks[-1] if self.feedbacks else None

    @property
    def final_tst(self) -> dict | None:
        return self.tst_versions[-1] if self.tst_versions else None


class LmOptimizerPipeline:
    """
    Wires TaskPlanner, SemanticParser, optimizer, LLMPhysicalPlanner,
    PlanRunner, AccuracyJudge, and the Reviser loop.

    Each sample gets its own query-level logical and physical plan derived
    from the current TST skeleton and policies.  The TST is revised between
    iterations based on aggregated feedback across all samples.

    Usage::

        pipeline = LmOptimizerPipeline(corpus=corpus, llm=llm,
                                        planning_client=client)
        result = pipeline.run_sync_with_samples(
            task_description="QA over scientific papers",
            evaluation_criteria="F1 against gold evidence passages",
            samples=[("What is X?", "X is Y", ["evidence passage..."])],
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

        self.task_planner = (
            TaskPlanner(
                client=planning_client, model=planning_model, catalog=self.catalog
            )
            if planning_client is not None
            else None
        )

        self.planner = (
            LLMPhysicalPlanner(
                client=planning_client, model=planning_model, catalog=self.catalog
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

        self.sample_analyzer = (
            SampleAnalyzer(client=planning_client, model=planning_model)
            if planning_client is not None
            else None
        )

        self.tst_revisor = (
            TSTRevisor(client=planning_client, model=planning_model)
            if planning_client is not None
            else None
        )

    # ── public entry point ─────────────────────────────────────────

    def run_sync_with_samples(
        self,
        task_description: str,
        evaluation_criteria: str,
        samples: list[
            tuple[str, str, list[str]]
        ],  # (query, candidate_answer, gold_evidence)
        iterations: int = 1,
        log_path: Path | None = None,
    ) -> PipelineResult:
        return self._run_async(
            self._run_with_samples_async(
                task_description, evaluation_criteria, samples, iterations, log_path
            )
        )

    # ── async core ────────────────────────────────────────────────

    async def _run_with_samples_async(
        self,
        task_description: str,
        evaluation_criteria: str,
        samples: list[tuple[str, str, list[str]]],
        iterations: int = 1,
        log_path: Path | None = None,
    ) -> PipelineResult:

        sample_queries = [q for q, _, _ in samples]

        # ── Task-level planning (once per run) ────────────────────
        # Produces the TST shared across all queries.
        # Revised between iterations but never per-query.
        tst = self._generate_tst(task_description, evaluation_criteria, sample_queries)
        tst_versions: list[dict] = [tst]
        tst_version_idx = 0

        store = FeedbackStore()
        log_entries: list[dict] = []
        feedbacks: list[Feedback] = []
        executions: list[ExecutionResult] = []
        sample_plans: list[SamplePlan] = []

        for iteration in range(iterations):
            feedbacks = []
            executions = []
            sample_plans = []

            # ── Query-level planning + execution (per sample) ─────
            # Each query independently fills the TST skeleton slots,
            # then gets its own optimized logical and physical plan.
            for query, candidate_answer, gold_evidence in samples:

                # Step A: query-level logical plan
                logical, optimized, rewrites = self._build_logical(query, tst)

                # Step B: query-level physical plan
                physical = self._build_physical(query, optimized, tst)

                # Step C: execute
                execution, node_feedbacks = await self.runner.run(physical)
                executions.append(execution)

                # Step D: judge
                feedback = await self._judge_evidence(
                    execution.answer, gold_evidence, node_feedbacks
                )
                feedbacks.append(feedback)

                # Step E: per-sample analysis (Reviser Step 1)
                sample_fb = self._analyze_sample(
                    query=query,
                    logical=optimized,
                    physical=physical,
                    execution=execution,
                    feedback=feedback,
                    tst=tst,
                )

                # Step F: store (Reviser Step 2)
                store.add(
                    iteration=iteration, tst_version=tst_version_idx, sample=sample_fb
                )

                # Collect per-sample plans for PipelineResult and log
                sample_plans.append(
                    SamplePlan(
                        query=query,
                        logical=logical,
                        optimized=optimized,
                        physical=physical,
                        rewrites=rewrites,
                    )
                )

                log_entries.append(
                    {
                        "iteration": iteration,
                        "tst_version": tst_version_idx,
                        "query": query,
                        "candidate_answer": candidate_answer,
                        "answer": execution.answer,
                        "gold_evidence": gold_evidence,
                        "accuracy": feedback.accuracy,
                        "errors": execution.errors,
                        "logical_plan": logical.to_dict(),
                        "optimized_plan": optimized.to_dict(),
                        "rewrite_log": rewrites,
                        "sample_analysis": sample_fb,
                    }
                )

            # ── TST revision between iterations (Reviser Steps 3-6) ──
            if iteration < iterations - 1:
                pattern_summary = aggregate_feedback(
                    [r["sample"] for r in store.by_iteration(iteration)]
                )
                new_tst = self._revise_tst(tst, pattern_summary)
                if new_tst is not tst:
                    tst = new_tst
                    tst_version_idx += 1
                    tst_versions.append(tst)

        if log_path is not None:
            self._write_log(
                log_path=log_path,
                task_description=task_description,
                iterations=iterations,
                tst_versions=tst_versions,
                log_entries=log_entries,
            )

        return PipelineResult(
            task_description=task_description,
            sample_queries=sample_queries,
            sample_plans=sample_plans,
            executions=executions,
            feedbacks=feedbacks,
            tst_versions=tst_versions,
        )

    # ── pipeline stages ────────────────────────────────────────────

    def _generate_tst(
        self,
        task_description: str,
        evaluation_criteria: str,
        sample_queries: list[str],
    ) -> dict:
        if self.task_planner is not None:
            return self.task_planner.generate(
                task_description=task_description,
                evaluation_criteria=evaluation_criteria,
                sample_queries=sample_queries,
            )
        return {}

    def _build_logical(
        self,
        query: str,
        tst: dict,
    ) -> tuple[LogicalNode, LogicalNode, list[RewriteEntry]]:
        """Query-level: fill TST skeleton slots for one query, then optimise."""
        logical = self.parser.parse(
            task_description=query,
            sample_queries=[query],
            task_strategy=tst,
        )
        optimized, rewrites = self.optimizer.run(logical)
        return logical, optimized, rewrites

    def _build_physical(
        self,
        query: str,
        optimized: LogicalNode,
        tst: dict,
    ) -> PhysicalNode:
        """Query-level: assign physical variants under TST policy for one query."""
        if self.planner is not None:
            return self.planner.plan(
                query=query,
                logical=optimized,
                physical_policy=tst.get("physical_policy", {}),
                corpus_stats=self._corpus_stats(),
                adaptation_policy=tst.get("adaptation_policy", {}),
            )
        from planner.plan_parser import parse_physical_plan

        return parse_physical_plan(optimized.to_dict())

    def _analyze_sample(
        self,
        query: str,
        logical: LogicalNode,
        physical: PhysicalNode,
        execution: ExecutionResult,
        feedback: Feedback,
        tst: dict,
    ) -> dict:
        if self.sample_analyzer is not None:
            return self.sample_analyzer.analyze(
                query=query,
                logical=logical,
                physical=physical,
                execution=execution,
                feedback=feedback,
                tst=tst,
            )
        return {
            "query": query,
            "accuracy": feedback.accuracy,
            "total_tokens": sum(i.token_cost for i in feedback.items),
            "total_latency_ms": sum(i.latency_ms for i in feedback.items),
            "query_features": {},
            "failure_points": [],
            "successful_adaptations": [],
            "suggested_fixes": [],
        }

    def _revise_tst(self, prev_tst: dict, pattern_summary: dict) -> dict:
        if self.tst_revisor is not None:
            return self.tst_revisor.revise(prev_tst, pattern_summary)
        return prev_tst

    # ── internals ─────────────────────────────────────────────────

    def _corpus_stats(self) -> dict:
        if self.catalog is not None:
            return {
                "context_window": self.catalog.context_window(),
                "avg_chunk_tokens": self.catalog.avg_chunk_tokens(),
            }
        return {"context_window": 128_000, "avg_chunk_tokens": 180.0}

    async def _judge_evidence(
        self,
        result: str,
        gold_evidence: list[str],
        node_feedbacks: list[NodeFeedback] | None = None,
    ) -> Feedback:
        if self.judge is None or not gold_evidence:
            accuracy = 0.0
        else:
            retrieval_texts = [
                item.output_summary
                for item in (node_feedbacks or [])
                if item.op_id.startswith(("I_", "RANK_", "COMPOSE_"))
                and item.output_summary.strip()
            ]
            _MAX_PASSAGE_CHARS = 2_000
            _MAX_TOTAL_CHARS = 80_000
            if retrieval_texts:
                truncated = [
                    self._truncate_text(t, _MAX_PASSAGE_CHARS) for t in retrieval_texts
                ]
                judge_text = self._truncate_text(
                    "\n\n".join(truncated), _MAX_TOTAL_CHARS
                )
            else:
                judge_text = self._truncate_text(result, _MAX_TOTAL_CHARS)
            accuracy = await self.judge.score_evidence(judge_text, gold_evidence)

        return Feedback(
            accuracy=accuracy,
            result=result,
            gold_ans="; ".join(gold_evidence),
            items=node_feedbacks or [],
        )

    def _write_log(
        self,
        log_path: Path,
        task_description: str,
        iterations: int,
        tst_versions: list[dict],
        log_entries: list[dict],
    ) -> None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps(
                {
                    "task": task_description,
                    "iterations": iterations,
                    # ── TST evolution ──────────────────────────────
                    # One entry per TST version (initial + each revision).
                    # Each entry contains the full TST detail so the log
                    # is self-contained for offline analysis.
                    "tst_versions": [
                        {
                            "version": idx,
                            "logical_skeleton": {
                                "template": v.get("logical_skeleton", {}).get(
                                    "template", ""
                                ),
                                "slots": v.get("logical_skeleton", {}).get("slots", []),
                                "core_operators": v.get("logical_skeleton", {}).get(
                                    "core_operators", []
                                ),
                                "optional_operators": v.get("logical_skeleton", {}).get(
                                    "optional_operators", []
                                ),
                            },
                            "physical_policy": {
                                op_id: {
                                    "op_name": node.get("op_name", ""),
                                    "variant": node.get("variant", ""),
                                    "params": node.get("params", {}),
                                    "param_ranges": node.get("param_ranges", {}),
                                }
                                for op_id, node in v.get("physical_policy", {}).items()
                            },
                            "adaptation_policy": {
                                "mutable_slots": v.get("adaptation_policy", {}).get(
                                    "mutable_slots", []
                                ),
                                "immutable_slots": v.get("adaptation_policy", {}).get(
                                    "immutable_slots", []
                                ),
                                "mutable_ops": v.get("adaptation_policy", {}).get(
                                    "mutable_ops", []
                                ),
                                "immutable_ops": v.get("adaptation_policy", {}).get(
                                    "immutable_ops", []
                                ),
                                "allowed_rewrites": v.get("adaptation_policy", {}).get(
                                    "allowed_rewrites", []
                                ),
                                "forbidden_rewrites": v.get(
                                    "adaptation_policy", {}
                                ).get("forbidden_rewrites", []),
                            },
                        }
                        for idx, v in enumerate(tst_versions)
                    ],
                    # ── Per-sample, per-iteration runs ─────────────
                    # Each entry already contains logical_plan, optimized_plan,
                    # rewrite_log, and sample_analysis from the loop above.
                    "runs": log_entries,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

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

    @staticmethod
    def _truncate_text(text: str, max_chars: int = 200) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "...[truncated]"
