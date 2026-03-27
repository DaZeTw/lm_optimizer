"""Phase 1 tests: IR, expression parser, and semantic parser."""

from parser.expr_parser import ParseError, parse_expression

import pytest

from ir.evidence import Chunk, EvidenceSet
from ir.nodes import CostVector, LogicalNode, PhysicalNode
from ir.ops import Op

# ── IR tests ──────────────────────────────────────────────────────


class TestLogicalNode:
    def test_roundtrip_simple(self):
        plan = LogicalNode.aggregate(
            LogicalNode.isolate("main contribution"),
            goal="summarize",
        )
        assert LogicalNode.from_dict(plan.to_dict()) == plan

    def test_roundtrip_deep(self):
        """The holistic QA naive plan from the notes."""
        plan = LogicalNode.aggregate(
            LogicalNode.union(
                LogicalNode.isolate("What method is proposed?"),
                LogicalNode.isolate("How is the method evaluated?"),
                LogicalNode.isolate("What limitations are discussed?"),
            ),
            goal="holistic QA answer",
        )
        assert LogicalNode.from_dict(plan.to_dict()) == plan

    def test_pretty_prints_without_error(self):
        plan = LogicalNode.verify(
            LogicalNode.aggregate(
                LogicalNode.rank(
                    LogicalNode.isolate("q"),
                    criterion="relevance",
                ),
                goal="answer",
            ),
            constraints="grounded",
        )
        output = plan.pretty()
        assert "VERIFY" in output
        assert "AGGREGATE" in output
        assert "RANK" in output
        assert "I(" in output

    def test_frozen(self):
        node = LogicalNode.isolate("q")
        with pytest.raises((AttributeError, TypeError)):
            node.op = Op.AGGREGATE  # type: ignore


class TestCostVector:
    def test_scalar_token_only(self):
        cv = CostVector(token_cost=10.0)
        assert cv.scalar(alpha=1, beta=0, gamma=0, delta=0) == 10.0

    def test_scalar_weighted(self):
        cv = CostVector(
            token_cost=2.0, call_cost=3.0, latency_cost=1.0, quality_risk=4.0
        )
        assert cv.scalar(alpha=1, beta=2, gamma=1, delta=0) == 2 + 6 + 1 + 0


class TestEvidenceSet:
    def test_token_estimate(self):
        es = EvidenceSet(
            chunks=[
                Chunk(text="a" * 400, doc_id="d1"),
                Chunk(text="b" * 800, doc_id="d2"),
            ]
        )
        assert es.token_estimate() == 300  # (400 + 800) / 4

    def test_append_trace(self):
        es = EvidenceSet(chunks=[], query_ref="q")
        es2 = es.append_trace("DenseRetrieve")
        es3 = es2.append_trace("LLMSummarize")
        assert es3.op_trace == ["DenseRetrieve", "LLMSummarize"]
        assert es.op_trace == []  # original unchanged

    def test_as_text(self):
        es = EvidenceSet(
            chunks=[
                Chunk(text="hello", doc_id="d1"),
                Chunk(text="world", doc_id="d2"),
            ]
        )
        assert es.as_text(separator=" | ") == "hello | world"


# ── Expression parser tests ────────────────────────────────────────


class TestExprParser:
    def test_simple_isolate(self):
        node = parse_expression('I("what is the main contribution?")')
        assert node.op == Op.I
        assert node.params["query"] == "what is the main contribution?"

    def test_aggregate_with_goal(self):
        node = parse_expression('AGGREGATE(I("main contribution"), goal="summarize")')
        assert node.op == Op.AGGREGATE
        assert node.params["goal"] == "summarize"
        assert len(node.inputs) == 1
        assert node.inputs[0].op == Op.I

    def test_verify_wrapping_aggregate(self):
        expr = (
            "VERIFY(\n"
            '  AGGREGATE(I("q"), goal="answer"),\n'
            '  constraints="grounded"\n'
            ")"
        )
        node = parse_expression(expr)
        assert node.op == Op.VERIFY
        assert node.params["constraints"] == "grounded"
        assert node.inputs[0].op == Op.AGGREGATE

    def test_union_multiple_children(self):
        expr = 'UNION(I("q1"), I("q2"), I("q3"))'
        node = parse_expression(expr)
        assert node.op == Op.UNION
        assert len(node.inputs) == 3

    def test_compose(self):
        expr = (
            'COMPOSE(I("paper A metrics"), I("paper B metrics"), condition="compare")'
        )
        node = parse_expression(expr)
        assert node.op == Op.COMPOSE
        assert len(node.inputs) == 2
        assert node.params["condition"] == "compare"

    def test_transform_with_schema(self):
        expr = 'TRANSFORM(I("methodology"), schema="metrics list")'
        node = parse_expression(expr)
        assert node.op == Op.TRANSFORM
        assert node.params["schema"] == "metrics list"

    def test_rank(self):
        expr = 'RANK(I("q"), criterion="relevance")'
        node = parse_expression(expr)
        assert node.op == Op.RANK
        assert node.params["criterion"] == "relevance"

    def test_full_holistic_plan(self):
        """The optimized holistic QA plan from the notes."""
        expr = (
            "VERIFY(\n"
            "  AGGREGATE(\n"
            "    RANK(\n"
            "      DIFF(\n"
            "        UNION(\n"
            '          TRANSFORM(RANK(I("method proposed?"), criterion="relevance"), schema="summary"),\n'
            '          TRANSFORM(RANK(I("evaluation approach?"), criterion="relevance"), schema="summary"),\n'
            '          TRANSFORM(RANK(I("limitations discussed?"), criterion="relevance"), schema="summary")\n'
            "        ),\n"
            '        I("overlap")\n'
            "      ),\n"
            '      criterion="final relevance"\n'
            "    ),\n"
            '    goal="holistic answer"\n'
            "  ),\n"
            '  constraints="grounded in evidence"\n'
            ")"
        )
        node = parse_expression(expr)
        assert node.op == Op.VERIFY
        assert node.inputs[0].op == Op.AGGREGATE

    def test_strips_markdown_fences(self):
        expr = '```\nAGGREGATE(I("q"), goal="answer")\n```'
        node = parse_expression(expr)
        assert node.op == Op.AGGREGATE

    def test_unknown_operator_raises(self):
        with pytest.raises(ParseError, match="Unknown operator"):
            parse_expression('SUMMARIZE(I("q"))')

    def test_missing_rparen_raises(self):
        with pytest.raises(ParseError):
            parse_expression('AGGREGATE(I("q"), goal="answer"')

    def test_aggregate_wrong_arity_raises(self):
        with pytest.raises(ParseError):
            parse_expression('AGGREGATE(I("q1"), I("q2"), goal="answer")')

    def test_compose_wrong_arity_raises(self):
        with pytest.raises(ParseError):
            parse_expression('COMPOSE(I("q1"), condition="x")')


# ── Roundtrip: parse → to_dict → from_dict ─────────────────────────


class TestParseRoundtrip:
    def test_expr_roundtrip(self):
        expr = 'AGGREGATE(UNION(I("q1"), I("q2")), goal="answer")'
        node = parse_expression(expr)
        restored = LogicalNode.from_dict(node.to_dict())
        assert restored == node
