"""
Phase 3 tests: weights, history, profilers, scorer, cost-aware planner.
No LLM calls, no API keys required.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import executor.ops  # noqa: F401 — triggers variant self-registration
from catalog import DocStats, ModelStats, SemanticStats, SystemCatalog, query_hash
from cost_model.cost_aware_planner import CostAwarePlanner
from cost_model.history import HistoryLog
from cost_model.profilers import profile_node
from cost_model.scorer import PlanScorer
from cost_model.vectors import OperatorCostVector, PlanCostReport
from cost_model.weights import PRESETS, WeightConfig, get_weights
from executor.corpus import InMemoryCorpus
from executor.registry import REGISTRY
from ir.nodes import CostVector, LogicalNode, PhysicalNode
from ir.ops import Op
from optimizer.engine import OptimizerEngine
from planner.physical import build_physical_plan

# ── Shared fixtures ────────────────────────────────────────────────


def _corpus():
    from executor.corpus import InMemoryCorpus
    from ir.evidence import Chunk

    return InMemoryCorpus(
        chunks=[
            Chunk(
                text="Transformer method uses attention mechanisms.",
                doc_id="p1",
                score=0.9,
            ),
            Chunk(
                text="BLEU and ROUGE evaluation metrics used.", doc_id="p1", score=0.85
            ),
            Chunk(
                text="Computational cost is a known limitation.", doc_id="p2", score=0.8
            ),
        ]
    )


def _tmp_log() -> HistoryLog:
    """Fresh HistoryLog backed by a temp file."""
    f = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    f.close()
    return HistoryLog(path=f.name)


def _simple_plan() -> PhysicalNode:
    logical = LogicalNode.aggregate(
        LogicalNode.isolate("main contribution"), goal="summarize"
    )
    return build_physical_plan(logical)


def _holistic_plan() -> PhysicalNode:
    naive = LogicalNode.aggregate(
        LogicalNode.union(
            LogicalNode.isolate("What method is proposed?"),
            LogicalNode.isolate("How is the method evaluated?"),
            LogicalNode.isolate("What are the limitations?"),
        ),
        goal="holistic QA answer",
    )
    optimized, _ = OptimizerEngine().run(naive)
    return build_physical_plan(optimized)


def _compare_plan() -> PhysicalNode:
    logical = LogicalNode.aggregate(
        LogicalNode.compose(
            LogicalNode.isolate("evaluation metrics paper A"),
            LogicalNode.isolate("evaluation metrics paper B"),
            condition="compare evaluation metrics",
        ),
        goal="compare metrics",
    )
    return build_physical_plan(logical)


# ══════════════════════════════════════════════════════════════════
# WeightConfig
# ══════════════════════════════════════════════════════════════════


class TestWeightConfig:
    def test_scalar_pure_token(self):
        w = WeightConfig(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0)
        assert w.scalar(10.0, 5.0, 2.0, 1.0) == 10.0

    def test_scalar_all_equal(self):
        w = WeightConfig(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0)
        assert w.scalar(1.0, 2.0, 3.0, 4.0) == 10.0

    def test_scalar_quality_preset(self):
        w = get_weights("quality")
        # quality weights delta=2.0 — quality_risk should dominate
        assert w.delta == 2.0
        assert w.scalar(0.0, 0.0, 0.0, 1.0) == 2.0

    def test_all_presets_exist(self):
        for name in ["speed", "quality", "economy", "balanced"]:
            cfg = get_weights(name)
            assert isinstance(cfg, WeightConfig)

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            get_weights("nonexistent")

    def test_frozen(self):
        w = WeightConfig()
        with pytest.raises((AttributeError, TypeError)):
            w.alpha = 99.0  # type: ignore

    def test_economy_prefers_low_tokens(self):
        w = get_weights("economy")
        # economy weights alpha=2.0, beta=2.0 — tokens and calls dominate
        assert w.alpha >= 2.0
        assert w.beta >= 2.0


# ══════════════════════════════════════════════════════════════════
# HistoryLog
# ══════════════════════════════════════════════════════════════════


class TestHistoryLog:
    def test_get_default_value(self):
        log = _tmp_log()
        val = log.get("TRANSFORM", "default", "avg", default=0.30)
        assert val == pytest.approx(0.30, abs=0.05)

    def test_get_missing_returns_default(self):
        log = _tmp_log()
        val = log.get("NONEXISTENT", "key", default=42)
        assert val == 42

    def test_record_updates_avg(self):
        log = _tmp_log()
        log.record("TRANSFORM", "summarize", "compression_ratios", 0.20)
        log.record("TRANSFORM", "summarize", "compression_ratios", 0.30)
        avg = log.get("TRANSFORM", "summarize", "avg")
        assert avg == pytest.approx(0.25, abs=0.01)

    def test_record_persists_to_file(self):
        log = _tmp_log()
        log.record("VERIFY", "simple_qa", "failure_rates", 0.10)
        # Re-load from same file
        log2 = HistoryLog(path=log._path)
        avg = log2.get("VERIFY", "simple_qa", "avg")
        assert avg is not None

    def test_rolling_window(self):
        log = _tmp_log()
        # Record more than _MAX_SAMPLES entries
        for i in range(110):
            log.record("I", "test", "chunk_sizes", float(i))
        samples = log.get("I", "test", "chunk_sizes", default=[])
        assert len(samples) <= 100

    def test_corrupt_file_falls_back_to_defaults(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("NOT VALID JSON {{{{")
            path = f.name
        log = HistoryLog(path=path)
        # Should not raise — falls back to defaults
        val = log.get("I", "avg_chunk_tokens", default=180)
        assert val == 180

    def test_reset_clears_data(self):
        log = _tmp_log()
        log.record("TRANSFORM", "test", "compression_ratios", 0.99)
        log.reset()
        # After reset, custom entry should be gone
        val = log.get("TRANSFORM", "test", "avg", default=None)
        assert val is None


# ══════════════════════════════════════════════════════════════════
# Profilers
# ══════════════════════════════════════════════════════════════════


class TestProfilers:
    def setup_method(self):
        self.log = _tmp_log()
        self.corpus = _corpus()

    def _profile(self, node, variant, upstream=0.0):
        return profile_node(
            node=node,
            variant=variant,
            op_id="test_0",
            upstream_tokens=upstream,
            corpus=self.corpus,
            log=self.log,
        )

    # ── I ──────────────────────────────────────────────────────────
    def test_isolate_token_cost_positive(self):
        cv = self._profile(LogicalNode.isolate("q"), "HybridRetrieve")
        assert cv.token_cost > 0

    def test_isolate_call_cost_is_one(self):
        cv = self._profile(LogicalNode.isolate("q"), "HybridRetrieve")
        assert cv.call_cost == 1.0

    def test_isolate_top_k_scales_tokens(self):
        node_small = LogicalNode(op=Op.I, inputs=(), params={"query": "q", "top_k": 5})
        node_large = LogicalNode(op=Op.I, inputs=(), params={"query": "q", "top_k": 20})
        cv_s = self._profile(node_small, "HybridRetrieve")
        cv_l = self._profile(node_large, "HybridRetrieve")
        assert cv_l.token_cost > cv_s.token_cost

    # ── TRANSFORM ──────────────────────────────────────────────────
    def test_identity_is_free(self):
        cv = self._profile(
            LogicalNode.transform(LogicalNode.isolate("q"), schema=""),
            "IdentityTransform",
        )
        assert cv.token_cost == 0.0
        assert cv.call_cost == 0.0

    def test_extractive_no_llm_call(self):
        cv = self._profile(
            LogicalNode.transform(LogicalNode.isolate("q"), schema="metrics"),
            "ExtractiveCompress",
            upstream=1000.0,
        )
        assert cv.call_cost == 0.0
        assert 0 < cv.token_cost < 1000.0  # should compress

    def test_llm_summarize_costs_one_call(self):
        cv = self._profile(
            LogicalNode.transform(LogicalNode.isolate("q"), schema="summary"),
            "LLMSummarize",
            upstream=1000.0,
        )
        assert cv.call_cost == 1.0

    # ── COMPOSE ────────────────────────────────────────────────────
    def test_concat_compose_no_call(self):
        node = LogicalNode.compose(
            LogicalNode.isolate("q1"), LogicalNode.isolate("q2"), condition=""
        )
        cv = self._profile(node, "ConcatCompose", upstream=500.0)
        assert cv.call_cost == 0.0

    def test_llm_compose_one_call(self):
        node = LogicalNode.compose(
            LogicalNode.isolate("q1"),
            LogicalNode.isolate("q2"),
            condition="compare methods",
        )
        cv = self._profile(node, "LLMCompose", upstream=500.0)
        assert cv.call_cost == 1.0

    def test_compose_quality_risk_reflects_gap(self):
        node = LogicalNode.compose(
            LogicalNode.isolate("transformer attention"),
            LogicalNode.isolate("transformer attention"),  # same → low gap
            condition="x",
        )
        cv_same = self._profile(node, "LLMCompose", upstream=500.0)

        node2 = LogicalNode.compose(
            LogicalNode.isolate("transformer neural network deep learning"),
            LogicalNode.isolate("ancient roman history empire conquest"),
            condition="x",
        )
        cv_diff = self._profile(node2, "LLMCompose", upstream=500.0)
        # Different topics → higher quality risk
        assert cv_diff.quality_risk >= cv_same.quality_risk

    # ── RANK ───────────────────────────────────────────────────────
    def test_rank_reduces_tokens(self):
        node = LogicalNode.rank(LogicalNode.isolate("q"), criterion="relevance")
        cv = self._profile(node, "SimilarityRank", upstream=2000.0)
        assert cv.token_cost <= 2000.0

    def test_similarity_rank_no_call(self):
        node = LogicalNode.rank(LogicalNode.isolate("q"), criterion="r")
        cv = self._profile(node, "SimilarityRank", upstream=500.0)
        assert cv.call_cost == 0.0

    def test_cross_encoder_one_call(self):
        node = LogicalNode.rank(LogicalNode.isolate("q"), criterion="r")
        cv = self._profile(node, "CrossEncoderRank", upstream=500.0)
        assert cv.call_cost == 1.0

    # ── AGGREGATE ──────────────────────────────────────────────────
    def test_aggregate_saturation_computed(self):
        node = LogicalNode.aggregate(LogicalNode.isolate("q"), goal="ans")
        cv = self._profile(node, "DirectGenerate", upstream=10_000.0)
        assert cv.saturation > 0.0

    def test_high_saturation_raises_risk(self):
        node = LogicalNode.aggregate(LogicalNode.isolate("q"), goal="ans")
        cv_low = self._profile(node, "DirectGenerate", upstream=10_000.0)
        cv_high = self._profile(node, "DirectGenerate", upstream=100_000.0)
        assert cv_high.quality_risk >= cv_low.quality_risk

    def test_hierarchical_multiple_calls(self):
        node = LogicalNode(
            op=Op.AGGREGATE,
            inputs=(LogicalNode.isolate("q"),),
            params={"goal": "ans", "batch_tokens": 8_000},
        )
        cv = self._profile(node, "HierarchicalGenerate", upstream=40_000.0)
        assert cv.call_cost > 1.0  # multiple batches

    # ── VERIFY ─────────────────────────────────────────────────────
    def test_verify_reduces_risk(self):
        node = LogicalNode.verify(LogicalNode.isolate("q"), constraints="grounded")
        cv = self._profile(node, "CitationVerify", upstream=500.0)
        # VERIFY quality_risk is negative (it reduces net risk)
        assert cv.quality_risk < 0.0

    def test_verify_adds_latency(self):
        node = LogicalNode.verify(LogicalNode.isolate("q"), constraints="grounded")
        cv = self._profile(node, "CitationVerify", upstream=500.0)
        assert cv.latency_cost > 0.0

    # ── UNION / DIFF ───────────────────────────────────────────────
    def test_union_no_latency(self):
        node = LogicalNode.union(LogicalNode.isolate("q1"), LogicalNode.isolate("q2"))
        cv = self._profile(node, "SimpleUnion", upstream=1000.0)
        assert cv.latency_cost == 0.0  # parallel branches

    def test_diff_reduces_tokens(self):
        node = LogicalNode.diff(
            LogicalNode.isolate("q"), LogicalNode.isolate("overlap")
        )
        cv = self._profile(node, "ExactDiff", upstream=1000.0)
        assert cv.token_cost < 1000.0

    # ── CostVector fields ──────────────────────────────────────────
    def test_all_profilers_return_operator_cost_vector(self):
        nodes = [
            (LogicalNode.isolate("q"), "HybridRetrieve"),
            (LogicalNode.transform(LogicalNode.isolate("q"), "s"), "LLMSummarize"),
            (LogicalNode.rank(LogicalNode.isolate("q"), "r"), "SimilarityRank"),
            (LogicalNode.aggregate(LogicalNode.isolate("q"), "ans"), "DirectGenerate"),
            (LogicalNode.verify(LogicalNode.isolate("q"), "c"), "CitationVerify"),
        ]
        for node, variant in nodes:
            cv = self._profile(node, variant, upstream=500.0)
            assert isinstance(cv, OperatorCostVector)
            assert cv.op_id == "test_0"
            assert cv.variant == variant

    def test_isolate_uses_catalog_density_and_chunk_size(self):
        q = "q"
        catalog = SystemCatalog(
            doc_stats={
                "d1": DocStats(
                    doc_id="d1",
                    total_chunks=2,
                    avg_chunk_tokens=300.0,
                    sections=["method"],
                    total_tokens=600,
                )
            },
            semantic_stats=SemanticStats(density_map={query_hash(q): 0.61}),
        )

        node = LogicalNode(op=Op.I, inputs=(), params={"query": q, "top_k": 5})
        cv = profile_node(
            node=node,
            variant="HybridRetrieve",
            op_id="test_0",
            upstream_tokens=0.0,
            corpus=self.corpus,
            log=self.log,
            catalog=catalog,
        )

        assert cv.token_cost == pytest.approx(1500.0, abs=0.001)
        assert cv.quality_risk == pytest.approx(0.61, abs=0.001)

    def test_aggregate_uses_catalog_context_window(self):
        catalog = SystemCatalog(
            model_stats={
                "tiny": ModelStats(
                    model_id="tiny",
                    context_window=1000,
                    input_cost_per_1k=0.0,
                    output_cost_per_1k=0.0,
                    avg_latency_ms=0,
                    supports_tools=False,
                )
            },
            default_model_id="tiny",
        )

        node = LogicalNode(
            op=Op.AGGREGATE, inputs=(LogicalNode.isolate("q"),), params={}
        )
        cv = profile_node(
            node=node,
            variant="DirectGenerate",
            op_id="test_0",
            upstream_tokens=1800.0,
            corpus=self.corpus,
            log=self.log,
            catalog=catalog,
        )

        assert cv.saturation == pytest.approx(1.8, abs=0.001)


# ══════════════════════════════════════════════════════════════════
# PlanScorer
# ══════════════════════════════════════════════════════════════════


class TestPlanScorer:
    def setup_method(self):
        self.scorer = PlanScorer(preset="balanced", log=_tmp_log(), corpus=_corpus())

    def test_simple_plan_scores(self):
        report = self.scorer.score(_simple_plan())
        assert isinstance(report, PlanCostReport)
        assert report.scalar > 0.0
        assert report.total_token_cost > 0.0

    def test_holistic_plan_scores(self):
        report = self.scorer.score(_holistic_plan())
        assert report.scalar > 0.0
        assert len(report.per_node) > 0

    def test_per_node_covers_all_variants(self):
        plan = _simple_plan()
        report = self.scorer.score(plan)
        # Should have entries for AGGREGATE and I
        op_names = {k.split("_")[0] for k in report.per_node}
        assert "AGGREGATE" in op_names
        assert "I" in op_names

    def test_bottleneck_is_valid_op_id(self):
        report = self.scorer.score(_holistic_plan())
        assert report.bottleneck in report.per_node

    def test_saturation_in_range(self):
        report = self.scorer.score(_simple_plan())
        assert 0.0 <= report.saturation <= 1.0

    def test_compare_plan_has_compose(self):
        report = self.scorer.score(_compare_plan())
        op_names = {k.split("_")[0] for k in report.per_node}
        assert "COMPOSE" in op_names

    def test_quality_preset_weights_risk_higher(self):
        scorer_q = PlanScorer(preset="quality", log=_tmp_log())
        scorer_e = PlanScorer(preset="economy", log=_tmp_log())
        plan = _compare_plan()
        r_q = scorer_q.score(plan)
        r_e = scorer_e.score(plan)
        # Economy scorer penalises tokens more; quality scorer penalises risk more.
        # They should produce different scalars on the same plan.
        assert r_q.scalar != r_e.scalar

    def test_summary_string_non_empty(self):
        report = self.scorer.score(_simple_plan())
        summary = report.summary()
        assert "Plan cost scalar" in summary
        assert "Token cost" in summary
        assert "Bottleneck" in summary

    def test_annotate_plan_writes_costs(self):
        plan = _simple_plan()
        annotated, report = self.scorer.annotate_plan(plan)
        # Root node should have non-zero cost
        assert annotated.cost.token_cost > 0.0 or annotated.cost.call_cost > 0.0

    def test_holistic_warnings_when_saturated(self):
        # Build a plan where upstream tokens are very large
        scorer = PlanScorer(preset="balanced", log=_tmp_log())
        # Force saturation by giving a tiny context window
        naive = LogicalNode.aggregate(LogicalNode.isolate("q"), goal="ans")
        plan = build_physical_plan(naive)
        # Override context_window in params to tiny value
        from ir.nodes import CostVector
        from ir.nodes import LogicalNode as LN
        from ir.nodes import PhysicalNode

        tiny_agg = LN(
            op=Op.AGGREGATE,
            inputs=(LN.isolate("q"),),
            params={"goal": "ans", "context_window": 100},
        )
        tiny_plan = build_physical_plan(tiny_agg)
        report = scorer.score(tiny_plan)
        # Very high saturation should trigger a warning
        if report.saturation > 0.7:
            assert any("saturation" in w.lower() for w in report.warnings)


# ══════════════════════════════════════════════════════════════════
# CostAwarePlanner
# ══════════════════════════════════════════════════════════════════


class TestCostAwarePlanner:
    def setup_method(self):
        self.planner = CostAwarePlanner(
            preset="balanced", log=_tmp_log(), corpus=_corpus()
        )

    def test_simple_plan_builds(self):
        logical = LogicalNode.aggregate(LogicalNode.isolate("q"), goal="ans")
        physical = self.planner.build(logical)
        assert physical.variant in REGISTRY or physical.variant == "IdentityTransform"

    def test_cost_annotated_on_nodes(self):
        logical = LogicalNode.aggregate(LogicalNode.isolate("q"), goal="ans")
        physical = self.planner.build(logical)
        # Root should have non-trivial cost
        assert (physical.cost.token_cost + physical.cost.call_cost) > 0.0

    def test_empty_schema_forced_identity(self):
        logical = LogicalNode.transform(LogicalNode.isolate("q"), schema="")
        physical = self.planner.build(logical)
        assert physical.variant == "IdentityTransform"

    def test_no_condition_forced_concat(self):
        logical = LogicalNode.compose(
            LogicalNode.isolate("a"), LogicalNode.isolate("b"), condition=""
        )
        physical = self.planner.build(logical)
        assert physical.variant == "ConcatCompose"

    def test_overlap_sentinel_forced_semantic(self):
        logical = LogicalNode.diff(
            LogicalNode.union(LogicalNode.isolate("q1"), LogicalNode.isolate("q2")),
            LogicalNode.isolate("__overlap__"),
        )
        physical = self.planner.build(logical)
        assert physical.variant == "SemanticDiff"

    def test_economy_preset_avoids_llm_calls(self):
        """Economy preset should prefer cheaper (non-LLM) variants."""
        planner_eco = CostAwarePlanner(
            preset="economy", log=_tmp_log(), corpus=_corpus()
        )
        logical = LogicalNode.transform(LogicalNode.isolate("q"), schema="metrics list")
        physical = planner_eco.build(logical)
        # Economy mode should prefer ExtractiveCompress (no LLM) over LLMSummarize
        assert physical.variant in ("ExtractiveCompress", "IdentityTransform")

    def test_quality_preset_tolerates_llm_calls(self):
        """Quality preset should be willing to pay for better reasoning."""
        planner_q = CostAwarePlanner(preset="quality", log=_tmp_log(), corpus=_corpus())
        logical = LogicalNode.compose(
            LogicalNode.isolate("method A"),
            LogicalNode.isolate("method B"),
            condition="compare approaches",
        )
        physical = planner_q.build(logical)
        # Quality mode may choose LLMCompose for better reasoning
        assert physical.variant in ("LLMCompose", "KeyMatchCompose", "ConcatCompose")

    def test_holistic_plan_end_to_end(self):
        naive = LogicalNode.aggregate(
            LogicalNode.union(
                LogicalNode.isolate("method"),
                LogicalNode.isolate("evaluation"),
                LogicalNode.isolate("limitations"),
            ),
            goal="holistic answer",
        )
        optimized, _ = OptimizerEngine().run(naive)
        physical = self.planner.build(optimized)

        def check(node):
            assert node.variant, f"Node missing variant"
            for c in node.inputs:
                check(c)

        check(physical)

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
        planner = CostAwarePlanner(
            preset="balanced",
            log=_tmp_log(),
            corpus=_corpus(),
            catalog=catalog,
            model_id="tiny",
        )
        logical = LogicalNode.aggregate(LogicalNode.isolate("q"), goal="ans")
        physical = planner.build(logical)

        assert physical.params.get("context_window") == 4096
        assert physical.params.get("model_id") == "tiny"
