"""Simplified cost model tests for runtime-only physical feedback flow."""

from __future__ import annotations

import tempfile
from pathlib import Path

import executor.ops  # noqa: F401
from catalog import ModelStats, SystemCatalog
from cost_model.cost_aware_planner import CostAwarePlanner
from cost_model.scorer import PlanScorer
from cost_model.telemetry import TelemetryStore
from cost_model.vectors import PlanCostReport
from executor.corpus import InMemoryCorpus
from executor.registry import REGISTRY
from ir.evidence import Chunk
from ir.nodes import LogicalNode, PhysicalNode
from optimizer.engine import OptimizerEngine
from planner.physical import build_physical_plan


def _corpus() -> InMemoryCorpus:
    return InMemoryCorpus(
        chunks=[
            Chunk(text="Transformer method uses attention.", doc_id="p1", score=0.9),
            Chunk(
                text="BLEU and ROUGE were used for evaluation.", doc_id="p1", score=0.8
            ),
            Chunk(
                text="High compute cost remains a limitation.", doc_id="p2", score=0.7
            ),
        ]
    )


def _tmp_telemetry() -> TelemetryStore:
    f = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    f.close()
    return TelemetryStore(path=f.name)


def _seed_minimal_observations(store: TelemetryStore) -> None:
    # I
    store.record(
        op="I",
        variant="HybridRetrieve",
        token_cost=1200,
        call_cost=1,
        latency_cost=1,
        quality_risk=0.2,
        accuracy_score=0.8,
    )
    store.record(
        op="I",
        variant="BM25Retrieve",
        token_cost=1400,
        call_cost=1,
        latency_cost=1,
        quality_risk=0.22,
        accuracy_score=0.75,
    )

    # TRANSFORM
    store.record(
        op="TRANSFORM",
        variant="IdentityTransform",
        token_cost=0,
        call_cost=0,
        latency_cost=0,
        quality_risk=0.02,
        accuracy_score=0.8,
    )
    store.record(
        op="TRANSFORM",
        variant="ExtractiveCompress",
        token_cost=400,
        call_cost=0,
        latency_cost=0.2,
        quality_risk=0.06,
        accuracy_score=0.78,
    )

    # COMPOSE
    store.record(
        op="COMPOSE",
        variant="ConcatCompose",
        token_cost=700,
        call_cost=0,
        latency_cost=0.1,
        quality_risk=0.18,
        accuracy_score=0.7,
    )
    store.record(
        op="COMPOSE",
        variant="LLMCompose",
        token_cost=550,
        call_cost=1,
        latency_cost=1.2,
        quality_risk=0.09,
        accuracy_score=0.82,
    )

    # RANK
    store.record(
        op="RANK",
        variant="SimilarityRank",
        token_cost=500,
        call_cost=0,
        latency_cost=0.3,
        quality_risk=0.07,
        accuracy_score=0.76,
    )

    # UNION / DIFF
    store.record(
        op="UNION",
        variant="SimpleUnion",
        token_cost=800,
        call_cost=0,
        latency_cost=0,
        quality_risk=0.01,
        accuracy_score=0.8,
    )
    store.record(
        op="DIFF",
        variant="SemanticDiff",
        token_cost=500,
        call_cost=1,
        latency_cost=0.6,
        quality_risk=0.05,
        accuracy_score=0.8,
    )

    # AGGREGATE / VERIFY
    store.record(
        op="AGGREGATE",
        variant="DirectGenerate",
        token_cost=650,
        call_cost=1,
        latency_cost=1.1,
        quality_risk=0.2,
        accuracy_score=0.75,
    )
    store.record(
        op="AGGREGATE",
        variant="HierarchicalGenerate",
        token_cost=520,
        call_cost=2,
        latency_cost=1.4,
        quality_risk=0.14,
        accuracy_score=0.8,
    )
    store.record(
        op="VERIFY",
        variant="CitationVerify",
        token_cost=300,
        call_cost=1,
        latency_cost=0.8,
        quality_risk=-0.1,
        accuracy_score=0.84,
    )


def _simple_plan() -> PhysicalNode:
    logical = LogicalNode.aggregate(
        LogicalNode.isolate("main contribution"), goal="summary"
    )
    return build_physical_plan(logical)


def _holistic_plan() -> PhysicalNode:
    naive = LogicalNode.aggregate(
        LogicalNode.union(
            LogicalNode.isolate("method"),
            LogicalNode.isolate("evaluation"),
            LogicalNode.isolate("limitations"),
        ),
        goal="holistic answer",
    )
    optimized, _ = OptimizerEngine().run(naive)
    return build_physical_plan(optimized)


class TestPlanScorer:
    def test_simple_plan_scores_from_observed_data(self):
        telemetry = _tmp_telemetry()
        _seed_minimal_observations(telemetry)
        scorer = PlanScorer(telemetry=telemetry)

        report = scorer.score(_simple_plan())

        assert isinstance(report, PlanCostReport)
        assert report.scalar > 0.0
        assert report.total_token_cost > 0.0

    def test_per_node_includes_variant_options(self):
        telemetry = _tmp_telemetry()
        _seed_minimal_observations(telemetry)
        scorer = PlanScorer(telemetry=telemetry)

        report = scorer.score(_simple_plan())

        assert report.per_node
        first = next(iter(report.per_node.values()))
        assert isinstance(first.variant_costs, dict)
        assert len(first.variant_costs) > 0

    def test_annotate_plan_writes_selected_and_variant_costs(self):
        telemetry = _tmp_telemetry()
        _seed_minimal_observations(telemetry)
        scorer = PlanScorer(telemetry=telemetry)

        annotated, _ = scorer.annotate_plan(_holistic_plan())

        assert (annotated.cost.token_cost + annotated.cost.call_cost) >= 0.0
        assert isinstance(annotated.variant_costs, dict)


class TestCostAwarePlanner:
    def test_simple_plan_builds(self):
        telemetry = _tmp_telemetry()
        _seed_minimal_observations(telemetry)
        planner = CostAwarePlanner(corpus=_corpus())

        physical = planner.build(
            LogicalNode.aggregate(LogicalNode.isolate("q"), goal="ans")
        )

        assert physical.variant in REGISTRY or physical.variant == "IdentityTransform"

    def test_forced_rules_still_apply(self):
        planner = CostAwarePlanner(corpus=_corpus())

        t = planner.build(LogicalNode.transform(LogicalNode.isolate("q"), schema=""))
        c = planner.build(
            LogicalNode.compose(
                LogicalNode.isolate("a"),
                LogicalNode.isolate("b"),
                condition="",
            )
        )
        d = planner.build(
            LogicalNode.diff(
                LogicalNode.union(LogicalNode.isolate("q1"), LogicalNode.isolate("q2")),
                LogicalNode.isolate("__overlap__"),
            )
        )

        assert t.variant == "IdentityTransform"
        assert c.variant == "ConcatCompose"
        assert d.variant == "SemanticDiff"

    def test_catalog_context_window_injected_into_aggregate_params(self):
        catalog = SystemCatalog(
            model_stats={
                "tiny": ModelStats(
                    model_id="tiny",
                    context_window=4096,
                    input_cost_per_1k=0.0,
                    output_cost_per_1k=0.0,
                    avg_latency_ms=0,
                    supports_tools=False,
                )
            },
            default_model_id="tiny",
        )
        planner = CostAwarePlanner(corpus=_corpus(), catalog=catalog, model_id="tiny")

        physical = planner.build(
            LogicalNode.aggregate(LogicalNode.isolate("q"), goal="ans")
        )

        assert physical.params.get("context_window") == 4096
        assert physical.params.get("model_id") == "tiny"
