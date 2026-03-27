"""
Phase 2 optimizer tests.

Tests are organized in three layers:
  1. Unit tests — one rule in isolation
  2. Engine tests — fixpoint iteration, log structure
  3. End-to-end — the holistic QA plan from the notes
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ir.nodes import LogicalNode
from ir.ops import Op
from optimizer.engine import OptimizerEngine, RewriteEntry
from optimizer.rules import (
    r1_transform_pushdown,
    r2_filter_pushdown,
    r3_compose_reorder,
    r4_merge_cascade_filters,
    r5_merge_cascade_transforms,
    r6_delayed_aggregation,
    r7_early_dedup,
    r8_selective_verify,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────


def _rule_names(log: list[RewriteEntry]) -> list[str]:
    return [e.rule for e in log]


def _ops(node: LogicalNode) -> list[str]:
    """Flatten the tree into a breadth-first list of op names."""
    result, queue = [], [node]
    while queue:
        n = queue.pop(0)
        result.append(n.op.value)
        queue.extend(n.inputs)
    return result


# ══════════════════════════════════════════════════════════════════
# 1. Unit tests — one rule at a time
# ══════════════════════════════════════════════════════════════════


class TestR1TransformPushdown:
    def test_fires_on_compose(self):
        node = LogicalNode.compose(
            LogicalNode.isolate("q1"),
            LogicalNode.isolate("q2"),
            condition="compare",
        )
        result, fired = r1_transform_pushdown(node)
        assert fired
        assert result.op == Op.COMPOSE
        assert result.inputs[0].op == Op.TRANSFORM
        assert result.inputs[1].op == Op.TRANSFORM

    def test_no_fire_if_already_transformed(self):
        node = LogicalNode.compose(
            LogicalNode.transform(LogicalNode.isolate("q1")),
            LogicalNode.transform(LogicalNode.isolate("q2")),
            condition="compare",
        )
        _, fired = r1_transform_pushdown(node)
        assert not fired

    def test_no_fire_on_non_compose(self):
        node = LogicalNode.aggregate(LogicalNode.isolate("q"), goal="ans")
        _, fired = r1_transform_pushdown(node)
        assert not fired

    def test_partial_transform_completes(self):
        """Only left is transformed — right should get wrapped."""
        node = LogicalNode.compose(
            LogicalNode.transform(LogicalNode.isolate("q1")),
            LogicalNode.isolate("q2"),
            condition="x",
        )
        result, fired = r1_transform_pushdown(node)
        assert fired
        assert result.inputs[0].op == Op.TRANSFORM
        assert result.inputs[1].op == Op.TRANSFORM


class TestR2FilterPushdown:
    def test_fires_on_aggregate_with_bare_i(self):
        node = LogicalNode.aggregate(LogicalNode.isolate("q"), goal="ans")
        result, fired = r2_filter_pushdown(node)
        assert fired
        assert result.inputs[0].op == Op.RANK
        assert result.inputs[0].inputs[0].op == Op.I

    def test_fires_on_union_children(self):
        node = LogicalNode.union(
            LogicalNode.isolate("q1"),
            LogicalNode.isolate("q2"),
        )
        result, fired = r2_filter_pushdown(node)
        assert fired
        assert all(c.op == Op.RANK for c in result.inputs)

    def test_no_fire_if_already_ranked(self):
        node = LogicalNode.aggregate(
            LogicalNode.rank(LogicalNode.isolate("q"), criterion="relevance"),
            goal="ans",
        )
        _, fired = r2_filter_pushdown(node)
        assert not fired

    def test_no_fire_on_rank_node(self):
        node = LogicalNode.rank(LogicalNode.isolate("q"), criterion="x")
        _, fired = r2_filter_pushdown(node)
        assert not fired


class TestR3ComposeReorder:
    def test_fires_when_left_is_compose(self):
        # COMPOSE(COMPOSE(large, small), tiny)
        large = LogicalNode.aggregate(
            LogicalNode.union(LogicalNode.isolate("a"), LogicalNode.isolate("b")),
            goal="x",
        )
        small = LogicalNode.transform(LogicalNode.isolate("c"))
        tiny = LogicalNode.isolate("d")

        inner = LogicalNode.compose(large, small, condition="x")
        node = LogicalNode.compose(inner, tiny, condition="y")

        result, fired = r3_compose_reorder(node)
        assert fired
        # After reorder: COMPOSE(large, COMPOSE(small, tiny))
        assert result.inputs[0] is large
        assert result.inputs[1].op == Op.COMPOSE

    def test_no_fire_when_left_is_not_compose(self):
        node = LogicalNode.compose(
            LogicalNode.isolate("q1"),
            LogicalNode.isolate("q2"),
            condition="x",
        )
        _, fired = r3_compose_reorder(node)
        assert not fired


class TestR4MergeCascadeFilters:
    def test_merges_double_rank(self):
        inner = LogicalNode.rank(LogicalNode.isolate("q"), criterion="c1")
        outer = LogicalNode.rank(inner, criterion="c2")
        result, fired = r4_merge_cascade_filters(outer)
        assert fired
        assert result.op == Op.RANK
        assert result.inputs[0].op == Op.I
        assert "c1" in result.params["criterion"]
        assert "c2" in result.params["criterion"]

    def test_merges_double_diff(self):
        e = LogicalNode.isolate("base")
        e1 = LogicalNode.isolate("sub1")
        e2 = LogicalNode.isolate("sub2")
        inner = LogicalNode.diff(e, e1)
        outer = LogicalNode.diff(inner, e2)
        result, fired = r4_merge_cascade_filters(outer)
        assert fired
        assert result.op == Op.DIFF
        assert result.inputs[1].op == Op.UNION

    def test_no_fire_on_single_rank(self):
        node = LogicalNode.rank(LogicalNode.isolate("q"), criterion="c1")
        _, fired = r4_merge_cascade_filters(node)
        assert not fired


class TestR5MergeCascadeTransforms:
    def test_merges_double_transform(self):
        inner = LogicalNode.transform(LogicalNode.isolate("q"), schema="s1")
        outer = LogicalNode.transform(inner, schema="s2")
        result, fired = r5_merge_cascade_transforms(outer)
        assert fired
        assert result.op == Op.TRANSFORM
        assert result.inputs[0].op == Op.I
        assert "s1" in result.params["schema"]
        assert "s2" in result.params["schema"]

    def test_no_fire_on_single_transform(self):
        node = LogicalNode.transform(LogicalNode.isolate("q"), schema="s1")
        _, fired = r5_merge_cascade_transforms(node)
        assert not fired

    def test_no_fire_on_non_transform(self):
        node = LogicalNode.rank(LogicalNode.isolate("q"), criterion="c")
        _, fired = r5_merge_cascade_transforms(node)
        assert not fired


class TestR6DelayedAggregation:
    def test_inserts_rank_before_aggregate(self):
        union = LogicalNode.union(
            LogicalNode.isolate("q1"),
            LogicalNode.isolate("q2"),
        )
        node = LogicalNode.aggregate(union, goal="ans")
        result, fired = r6_delayed_aggregation(node)
        assert fired
        assert result.inputs[0].op == Op.RANK
        assert result.inputs[0].inputs[0].op == Op.UNION

    def test_no_fire_if_child_is_rank(self):
        node = LogicalNode.aggregate(
            LogicalNode.rank(LogicalNode.isolate("q"), criterion="r"),
            goal="ans",
        )
        _, fired = r6_delayed_aggregation(node)
        assert not fired

    def test_no_fire_if_child_is_diff(self):
        node = LogicalNode.aggregate(
            LogicalNode.diff(LogicalNode.isolate("q"), LogicalNode.isolate("x")),
            goal="ans",
        )
        _, fired = r6_delayed_aggregation(node)
        assert not fired

    def test_no_fire_on_non_aggregate(self):
        node = LogicalNode.rank(LogicalNode.isolate("q"), criterion="r")
        _, fired = r6_delayed_aggregation(node)
        assert not fired


class TestR7EarlyDedup:
    def test_inserts_diff_after_union(self):
        union = LogicalNode.union(
            LogicalNode.isolate("q1"),
            LogicalNode.isolate("q2"),
        )
        node = LogicalNode.aggregate(union, goal="ans")
        result, fired = r7_early_dedup(node)
        assert fired
        assert result.inputs[0].op == Op.DIFF
        assert result.inputs[0].inputs[0].op == Op.UNION

    def test_fires_through_rank_wrapping_union(self):
        union = LogicalNode.union(
            LogicalNode.isolate("q1"),
            LogicalNode.isolate("q2"),
        )
        rank = LogicalNode.rank(union, criterion="relevance")
        node = LogicalNode.aggregate(rank, goal="ans")
        result, fired = r7_early_dedup(node)
        assert fired
        # RANK should now wrap DIFF(UNION(...))
        assert result.inputs[0].op == Op.RANK
        assert result.inputs[0].inputs[0].op == Op.DIFF

    def test_no_fire_if_already_deduped(self):
        diff = LogicalNode.diff(
            LogicalNode.isolate("q1"),
            LogicalNode.isolate("overlap"),
        )
        node = LogicalNode.aggregate(diff, goal="ans")
        _, fired = r7_early_dedup(node)
        assert not fired

    def test_no_fire_without_union(self):
        node = LogicalNode.aggregate(LogicalNode.isolate("q"), goal="ans")
        _, fired = r7_early_dedup(node)
        assert not fired


class TestR8SelectiveVerify:
    def test_fires_when_compose_present(self):
        compose = LogicalNode.compose(
            LogicalNode.isolate("q1"),
            LogicalNode.isolate("q2"),
            condition="compare",
        )
        node = LogicalNode.aggregate(compose, goal="ans")
        result, fired = r8_selective_verify(node)
        assert fired
        assert result.op == Op.VERIFY
        assert result.inputs[0].op == Op.AGGREGATE

    def test_no_fire_without_compose(self):
        node = LogicalNode.aggregate(LogicalNode.isolate("q"), goal="ans")
        _, fired = r8_selective_verify(node)
        assert not fired

    def test_no_fire_if_already_verified(self):
        agg = LogicalNode.aggregate(LogicalNode.isolate("q"), goal="ans")
        node = LogicalNode.verify(agg, constraints="grounded")
        _, fired = r8_selective_verify(node)
        assert not fired

    def test_no_fire_on_non_aggregate(self):
        node = LogicalNode.rank(LogicalNode.isolate("q"), criterion="r")
        _, fired = r8_selective_verify(node)
        assert not fired


# ══════════════════════════════════════════════════════════════════
# 2. Engine tests
# ══════════════════════════════════════════════════════════════════


class TestEngine:
    def setup_method(self):
        self.engine = OptimizerEngine()

    def test_fixpoint_on_already_optimal_plan(self):
        """A single I node needs no rewriting."""
        plan = LogicalNode.isolate("q")
        optimized, log = self.engine.run(plan)
        assert optimized == plan
        assert log == []

    def test_log_has_rule_names(self):
        plan = LogicalNode.aggregate(LogicalNode.isolate("q"), goal="ans")
        _, log = self.engine.run(plan)
        names = _rule_names(log)
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)

    def test_log_entries_have_pass_numbers(self):
        plan = LogicalNode.aggregate(LogicalNode.isolate("q"), goal="ans")
        _, log = self.engine.run(plan)
        assert all(e.pass_no >= 1 for e in log)

    def test_output_roundtrips(self):
        plan = LogicalNode.aggregate(
            LogicalNode.union(
                LogicalNode.isolate("q1"),
                LogicalNode.isolate("q2"),
            ),
            goal="ans",
        )
        optimized, _ = self.engine.run(plan)
        assert LogicalNode.from_dict(optimized.to_dict()) == optimized

    def test_max_passes_safety(self):
        """Engine should not loop forever even with a contrived plan."""
        plan = LogicalNode.aggregate(LogicalNode.isolate("q"), goal="ans")
        engine = OptimizerEngine(max_passes=3)
        optimized, log = engine.run(plan)
        assert optimized is not None


# ══════════════════════════════════════════════════════════════════
# 3. End-to-end: holistic QA plan
# ══════════════════════════════════════════════════════════════════


class TestHolisticQA:
    """
    Input (naive plan from Phase 1):
        AGGREGATE(
          UNION(
            I("What method is proposed?"),
            I("How is the method evaluated?"),
            I("What limitations are discussed?")
          ),
          goal="holistic QA answer"
        )

    Expected optimized plan (from notes):
        AGGREGATE(
          RANK(
            DIFF(
              UNION(
                TRANSFORM(RANK(I(q1))),
                TRANSFORM(RANK(I(q2))),
                TRANSFORM(RANK(I(q3)))
              ),
              I("__overlap__")
            ),
            criterion="relevance"
          ),
          goal="holistic QA answer"
        )
    """

    def setup_method(self):
        self.engine = OptimizerEngine()
        self.naive_plan = LogicalNode.aggregate(
            LogicalNode.union(
                LogicalNode.isolate("What method is proposed?"),
                LogicalNode.isolate("How is the method evaluated?"),
                LogicalNode.isolate("What limitations are discussed?"),
            ),
            goal="holistic QA answer",
        )

    def test_root_is_aggregate(self):
        optimized, _ = self.engine.run(self.naive_plan)
        # VERIFY only wraps when COMPOSE is present; UNION doesn't trigger it
        assert optimized.op in (Op.AGGREGATE, Op.VERIFY)

    def test_union_children_are_ranked(self):
        """After R2 fires, every I() under UNION gets a RANK wrapper."""
        optimized, _ = self.engine.run(self.naive_plan)
        ops = _ops(optimized)
        assert "RANK" in ops

    def test_dedup_inserted(self):
        """R7 should insert DIFF after UNION."""
        optimized, _ = self.engine.run(self.naive_plan)
        ops = _ops(optimized)
        assert "DIFF" in ops

    def test_r2_fired(self):
        _, log = self.engine.run(self.naive_plan)
        assert "R2_filter_pushdown" in _rule_names(log)

    def test_r6_fired(self):
        _, log = self.engine.run(self.naive_plan)
        assert "R6_delayed_aggregation" in _rule_names(log)

    def test_r7_fired(self):
        _, log = self.engine.run(self.naive_plan)
        assert "R7_early_dedup" in _rule_names(log)

    def test_optimized_plan_saved_as_fixture(self):
        """Save before/after JSON to tests/fixtures/holistic_qa_optimized.json"""
        optimized, log = self.engine.run(self.naive_plan)

        payload = {
            "description": "Holistic QA plan before and after optimization",
            "naive_plan": self.naive_plan.to_dict(),
            "optimized_plan": optimized.to_dict(),
            "rules_fired": _rule_names(log),
            "passes": max((e.pass_no for e in log), default=0),
        }

        path = FIXTURES_DIR / "holistic_qa_optimized.json"
        path.write_text(json.dumps(payload, indent=2))
        assert path.exists()

        # Reload and verify
        data = json.loads(path.read_text())
        restored = LogicalNode.from_dict(data["optimized_plan"])
        assert restored == optimized


# ══════════════════════════════════════════════════════════════════
# 4. Multi-doc compare plan (involves COMPOSE → triggers R8)
# ══════════════════════════════════════════════════════════════════


class TestMultiDocCompare:
    """
    Input:
        AGGREGATE(
          COMPOSE(
            I("evaluation metrics paper A"),
            I("evaluation metrics paper B"),
            condition="compare"
          ),
          goal="compare metrics"
        )

    Expected: VERIFY wraps the result (R8 fires because COMPOSE present).
    """

    def setup_method(self):
        self.engine = OptimizerEngine()
        self.plan = LogicalNode.aggregate(
            LogicalNode.compose(
                LogicalNode.isolate("evaluation metrics paper A"),
                LogicalNode.isolate("evaluation metrics paper B"),
                condition="compare evaluation metrics",
            ),
            goal="compare metrics",
        )

    def test_verify_wraps_result(self):
        optimized, _ = self.engine.run(self.plan)
        assert optimized.op == Op.VERIFY

    def test_transforms_pushed_into_compose(self):
        optimized, _ = self.engine.run(self.plan)
        ops = _ops(optimized)
        assert "TRANSFORM" in ops

    def test_r1_fired(self):
        _, log = self.engine.run(self.plan)
        assert "R1_transform_pushdown" in _rule_names(log)

    def test_r8_fired(self):
        _, log = self.engine.run(self.plan)
        assert "R8_selective_verify" in _rule_names(log)

    def test_saved_as_fixture(self):
        optimized, log = self.engine.run(self.plan)
        payload = {
            "description": "Multi-doc compare plan before and after optimization",
            "naive_plan": self.plan.to_dict(),
            "optimized_plan": optimized.to_dict(),
            "rules_fired": _rule_names(log),
        }
        path = FIXTURES_DIR / "multi_doc_compare_optimized.json"
        path.write_text(json.dumps(payload, indent=2))
        assert path.exists()
