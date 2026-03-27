"""
Tests for physical variants, planner, and runner.
Uses InMemoryCorpus + MockLLM — no API keys required.
"""

from __future__ import annotations

import asyncio

import pytest

import executor.ops  # noqa: F401 — triggers self-registration
from catalog import SemanticStats, SystemCatalog
from executor.corpus import InMemoryCorpus, MockLLM
from executor.registry import REGISTRY
from executor.runner import PlanRunner
from ir.evidence import Chunk, EvidenceSet
from ir.nodes import CostVector, LogicalNode, PhysicalNode
from ir.ops import Op
from planner.physical import build_physical_plan

# ── Helpers ────────────────────────────────────────────────────────


def _corpus() -> InMemoryCorpus:
    return InMemoryCorpus(
        chunks=[
            Chunk(
                text="The proposed method uses transformer architecture with attention mechanisms.",
                doc_id="p1",
                section="method",
                score=0.9,
            ),
            Chunk(
                text="Evaluation was performed on BLEU and ROUGE metrics across 5 datasets.",
                doc_id="p1",
                section="eval",
                score=0.85,
            ),
            Chunk(
                text="Limitations include high computational cost and limited context window.",
                doc_id="p1",
                section="limits",
                score=0.8,
            ),
            Chunk(
                text="The baseline model uses LSTM with standard sequence-to-sequence approach.",
                doc_id="p2",
                section="method",
                score=0.75,
            ),
            Chunk(
                text="Results show 15% improvement over baseline on standard benchmarks.",
                doc_id="p2",
                section="results",
                score=0.88,
            ),
        ]
    )


def _llm(response: str = "Based on the evidence, the answer is clear.") -> MockLLM:
    return MockLLM(response=response)


class CountingLLM(MockLLM):
    def __init__(self, response: str):
        super().__init__(response=response)
        self.calls = 0

    async def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        self.calls += 1
        return await super().complete(system=system, user=user, max_tokens=max_tokens)


def _ev(*texts: str, query: str = "test") -> EvidenceSet:
    return EvidenceSet(
        chunks=[Chunk(text=t, doc_id=f"d{i}") for i, t in enumerate(texts)],
        query_ref=query,
    )


def run(coro):
    return asyncio.run(coro)


def call(variant: str, inputs, params, corpus=None, llm=None):
    return run(REGISTRY[variant](inputs, params, corpus or _corpus(), llm or _llm()))


# ══════════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════════

EXPECTED = [
    "BM25Retrieve",
    "DenseRetrieve",
    "HybridRetrieve",
    "IdentityTransform",
    "ExtractiveCompress",
    "LLMSummarize",
    "SimilarityRank",
    "CrossEncoderRank",
    "SimpleUnion",
    "ExactDiff",
    "SemanticDiff",
    "ConcatCompose",
    "LLMCompose",
    "KeyMatchCompose",
    "DirectGenerate",
    "HierarchicalGenerate",
    "CitationVerify",
    "NliVerify",
]


@pytest.mark.parametrize("name", EXPECTED)
def test_variant_registered(name):
    assert name in REGISTRY


def test_all_variants_callable():
    for name, fn in REGISTRY.items():
        assert callable(fn)


# ══════════════════════════════════════════════════════════════════
# Retrieve
# ══════════════════════════════════════════════════════════════════


class TestRetrieve:
    @pytest.mark.parametrize(
        "variant", ["BM25Retrieve", "DenseRetrieve", "HybridRetrieve"]
    )
    def test_returns_chunks(self, variant):
        result = call(variant, [], {"query": "transformer method", "top_k": 3})
        assert len(result.chunks) > 0

    @pytest.mark.parametrize(
        "variant", ["BM25Retrieve", "DenseRetrieve", "HybridRetrieve"]
    )
    def test_trace_recorded(self, variant):
        result = call(variant, [], {"query": "transformer", "top_k": 3})
        assert variant in result.op_trace

    def test_top_k_respected(self):
        result = call("BM25Retrieve", [], {"query": "transformer", "top_k": 2})
        assert len(result.chunks) <= 2

    def test_query_ref_set(self):
        result = call("HybridRetrieve", [], {"query": "my query", "top_k": 5})
        assert result.query_ref == "my query"


# ══════════════════════════════════════════════════════════════════
# Transform
# ══════════════════════════════════════════════════════════════════


class TestTransform:
    def setup_method(self):
        self.ev = _ev(
            "Transformer uses self-attention for encoding text sequences.",
            query="method",
        )
        self.llm = _llm("Extracted: transformer, attention, BLEU.")

    def test_identity_passthrough(self):
        result = call("IdentityTransform", [self.ev], {})
        assert result.chunks == self.ev.chunks
        assert "IdentityTransform" in result.op_trace

    def test_extractive_compress_returns_chunks(self):
        result = call(
            "ExtractiveCompress", [self.ev], {"query": "attention", "top_k_sents": 2}
        )
        assert len(result.chunks) == len(self.ev.chunks)
        assert "ExtractiveCompress" in result.op_trace

    def test_llm_summarize_single_output(self):
        result = call(
            "LLMSummarize", [self.ev], {"schema": "key methods"}, llm=self.llm
        )
        assert len(result.chunks) == 1
        assert result.chunks[0].text == self.llm._response
        assert "LLMSummarize" in result.op_trace

    def test_llm_summarize_empty_input(self):
        result = call("LLMSummarize", [EvidenceSet(chunks=[])], {"schema": "test"})
        assert "LLMSummarize" in result.op_trace

    def test_extractive_no_query_passthrough(self):
        ev = EvidenceSet(chunks=[Chunk(text="hello world", doc_id="d1")], query_ref="")
        result = call("ExtractiveCompress", [ev], {})
        assert "ExtractiveCompress" in result.op_trace

    def test_llm_summarize_uses_catalog_cache(self):
        catalog = SystemCatalog(
            semantic_stats=SemanticStats(
                summarization_map={"d1/method": "Cached section summary."}
            )
        )
        llm = CountingLLM("LLM fallback summary")
        ev = EvidenceSet(
            chunks=[Chunk(text="source text", doc_id="d1", section="method")],
            query_ref="method",
        )

        result = call(
            "LLMSummarize",
            [ev],
            {"schema": "key methods", "catalog": catalog},
            llm=llm,
        )

        assert result.chunks[0].text == "Cached section summary."
        assert result.chunks[0].metadata["cached"] is True
        assert llm.calls == 0


# ══════════════════════════════════════════════════════════════════
# Rank
# ══════════════════════════════════════════════════════════════════


class TestRank:
    def setup_method(self):
        self.ev = _ev(
            "transformer attention mechanism paper",
            "LSTM baseline model comparison",
            "future work multilingual support",
            query="transformer",
        )

    def test_similarity_rank_top_k(self):
        result = call("SimilarityRank", [self.ev], {"query": "transformer", "top_k": 2})
        assert len(result.chunks) <= 2

    def test_similarity_rank_scores_valid(self):
        result = call("SimilarityRank", [self.ev], {"query": "transformer", "top_k": 3})
        assert all(0.0 <= c.score <= 1.0 for c in result.chunks)

    def test_similarity_rank_trace(self):
        result = call("SimilarityRank", [self.ev], {"query": "test"})
        assert "SimilarityRank" in result.op_trace

    def test_rank_empty_input(self):
        result = call(
            "SimilarityRank", [EvidenceSet(chunks=[], query_ref="q")], {"query": "test"}
        )
        assert result.chunks == []

    def test_cross_encoder_falls_back(self):
        # sentence-transformers likely not installed — should fall back gracefully
        result = call(
            "CrossEncoderRank", [self.ev], {"query": "transformer", "top_k": 2}
        )
        assert len(result.chunks) <= 3  # either CE or fallback to SimilarityRank


# ══════════════════════════════════════════════════════════════════
# Union / Diff
# ══════════════════════════════════════════════════════════════════


class TestUnionDiff:
    def setup_method(self):
        self.ev1 = _ev("chunk A about transformers", "chunk B about attention")
        self.ev2 = _ev("chunk C about LSTM", "chunk A about transformers")  # dup

    def test_simple_union_total(self):
        result = call("SimpleUnion", [self.ev1, self.ev2], {})
        assert len(result.chunks) == 4
        assert "SimpleUnion" in result.op_trace

    def test_simple_union_empty(self):
        result = call("SimpleUnion", [], {})
        assert result.chunks == []

    def test_exact_diff_overlap_sentinel_deduplicates(self):
        merged = EvidenceSet(chunks=self.ev1.chunks + self.ev2.chunks)
        sentinel = EvidenceSet(chunks=[Chunk(text="__overlap__", doc_id="")])
        result = call("ExactDiff", [merged, sentinel], {})
        texts = [c.text for c in result.chunks]
        assert len(texts) == len(set(texts))
        assert "ExactDiff" in result.op_trace

    def test_exact_diff_removes_subtract(self):
        merged = EvidenceSet(chunks=self.ev1.chunks + self.ev2.chunks)
        result = call("ExactDiff", [merged, self.ev2], {})
        kept = {c.text for c in result.chunks}
        for c in self.ev2.chunks:
            assert c.text not in kept

    def test_semantic_diff_self_dedup(self):
        ev = _ev(
            "The transformer uses self-attention mechanisms",
            "The transformer uses self-attention mechanisms",  # exact dup
        )
        sentinel = EvidenceSet(chunks=[Chunk(text="__overlap__", doc_id="")])
        result = call("SemanticDiff", [ev, sentinel], {"threshold": 0.99})
        assert len(result.chunks) <= 2
        assert "SemanticDiff" in result.op_trace


# ══════════════════════════════════════════════════════════════════
# Compose
# ══════════════════════════════════════════════════════════════════


class TestCompose:
    def setup_method(self):
        self.evA = _ev("Paper A uses transformer with BLEU evaluation.")
        self.evB = _ev("Paper B uses LSTM with accuracy evaluation.")
        self.llm = _llm("Both papers compare evaluation methods: BLEU and accuracy.")

    def test_concat_compose_merges(self):
        result = call("ConcatCompose", [self.evA, self.evB], {})
        assert len(result.chunks) == 2
        assert "ConcatCompose" in result.op_trace

    def test_concat_compose_single_input(self):
        result = call("ConcatCompose", [self.evA], {})
        assert len(result.chunks) == 1

    def test_llm_compose_single_output(self):
        result = call(
            "LLMCompose",
            [self.evA, self.evB],
            {"condition": "compare methods"},
            llm=self.llm,
        )
        assert len(result.chunks) == 1
        assert result.chunks[0].text == self.llm._response
        assert "LLMCompose" in result.op_trace

    def test_llm_compose_metadata(self):
        result = call(
            "LLMCompose", [self.evA, self.evB], {"condition": "compare"}, llm=self.llm
        )
        assert result.chunks[0].metadata["condition"] == "compare"
        assert result.chunks[0].metadata["left_chunks"] == 1

    def test_key_match_finds_overlap(self):
        evA = _ev("transformer evaluation method accuracy results improvement")
        evB = _ev("LSTM evaluation method baseline comparison results")
        result = call("KeyMatchCompose", [evA, evB], {"min_overlap": 1})
        assert len(result.chunks) >= 1
        assert "KeyMatchCompose" in result.op_trace

    def test_key_match_falls_back_on_no_overlap(self):
        evA = _ev("aaaa bbbb cccc dddd eeee")
        evB = _ev("xxxx yyyy zzzz wwww vvvv")
        result = call("KeyMatchCompose", [evA, evB], {"min_overlap": 1})
        assert len(result.chunks) >= 1  # concat fallback


# ══════════════════════════════════════════════════════════════════
# Aggregate
# ══════════════════════════════════════════════════════════════════


class TestAggregate:
    def setup_method(self):
        self.ev = _ev(
            "Transformer architecture with attention.",
            "Evaluated on BLEU metrics.",
            "Compute limitations discussed.",
            query="holistic summary",
        )
        self.llm = _llm(
            "The transformer method achieves strong BLEU results despite high compute."
        )

    def test_direct_generate_answer(self):
        result = call(
            "DirectGenerate", [self.ev], {"goal": "summarize paper"}, llm=self.llm
        )
        assert result.chunks[0].text == self.llm._response
        assert "DirectGenerate" in result.op_trace

    def test_direct_generate_metadata(self):
        result = call("DirectGenerate", [self.ev], {"goal": "summarize"}, llm=self.llm)
        md = result.chunks[0].metadata
        assert "goal" in md
        assert "source_chunks" in md
        assert "saturation" in md

    def test_direct_generate_empty(self):
        result = call(
            "DirectGenerate", [EvidenceSet(chunks=[])], {"goal": "test"}, llm=self.llm
        )
        assert "DirectGenerate" in result.op_trace

    def test_hierarchical_single_output(self):
        result = call(
            "HierarchicalGenerate",
            [self.ev],
            {"goal": "summarize", "batch_tokens": 10},
            llm=self.llm,
        )
        assert len(result.chunks) == 1
        assert result.chunks[0].metadata["batches"] >= 1
        assert "HierarchicalGenerate" in result.op_trace

    def test_hierarchical_llm_calls_count(self):
        result = call(
            "HierarchicalGenerate",
            [self.ev],
            {"goal": "test", "batch_tokens": 10},
            llm=self.llm,
        )
        assert (
            result.chunks[0].metadata["llm_calls"] >= 2
        )  # at least 1 summary + 1 synthesis


# ══════════════════════════════════════════════════════════════════
# Verify
# ══════════════════════════════════════════════════════════════════


class TestVerify:
    def setup_method(self):
        self.answer = _ev(
            "The paper proposes a transformer evaluated with BLEU metrics."
        )
        self.evid = _ev("transformer architecture evaluation BLEU metrics improvement")

    def test_citation_verify_score_range(self):
        result = call("CitationVerify", [self.answer, self.evid], {})
        score = result.chunks[0].metadata["grounding_score"]
        assert 0.0 <= score <= 1.0
        assert "CitationVerify" in result.op_trace

    def test_citation_verify_well_grounded(self):
        answer = _ev("transformer attention evaluation BLEU metrics results")
        evid = _ev("transformer attention evaluation BLEU metrics results")
        result = call("CitationVerify", [answer, evid], {})
        assert result.chunks[0].metadata["grounding_score"] > 0.5

    def test_citation_verify_metadata_keys(self):
        result = call("CitationVerify", [self.answer, self.evid], {})
        md = result.chunks[0].metadata
        assert "grounding_score" in md
        assert "sentences_checked" in md
        assert "unsupported_count" in md

    def test_citation_verify_no_evidence(self):
        result = call("CitationVerify", [self.answer], {})
        assert len(result.chunks) == 1

    def test_nli_falls_back_gracefully(self):
        # NLI model not installed — should not raise
        result = call("NliVerify", [self.answer, self.evid], {})
        assert len(result.chunks) == 1


# ══════════════════════════════════════════════════════════════════
# Physical planner
# ══════════════════════════════════════════════════════════════════


class TestPlanner:
    def test_simple_plan(self):
        logical = LogicalNode.aggregate(LogicalNode.isolate("q"), goal="ans")
        physical = build_physical_plan(logical)
        assert physical.variant == "DirectGenerate"
        assert physical.inputs[0].variant == "HybridRetrieve"

    def test_compose_with_condition(self):
        logical = LogicalNode.aggregate(
            LogicalNode.compose(
                LogicalNode.isolate("a"),
                LogicalNode.isolate("b"),
                condition="compare",
            ),
            goal="compare",
        )
        physical = build_physical_plan(logical)
        assert physical.inputs[0].variant == "LLMCompose"

    def test_compose_no_condition(self):
        logical = LogicalNode.compose(
            LogicalNode.isolate("a"), LogicalNode.isolate("b"), condition=""
        )
        assert build_physical_plan(logical).variant == "ConcatCompose"

    def test_empty_schema_identity(self):
        logical = LogicalNode.transform(LogicalNode.isolate("q"), schema="")
        assert build_physical_plan(logical).variant == "IdentityTransform"

    def test_nonempty_schema_extractive(self):
        logical = LogicalNode.transform(LogicalNode.isolate("q"), schema="metrics")
        assert build_physical_plan(logical).variant == "ExtractiveCompress"

    def test_overlap_sentinel_semantic_diff(self):
        logical = LogicalNode.diff(
            LogicalNode.union(LogicalNode.isolate("q1"), LogicalNode.isolate("q2")),
            LogicalNode.isolate("__overlap__"),
        )
        assert build_physical_plan(logical).variant == "SemanticDiff"

    def test_all_nodes_have_known_variants(self):
        logical = LogicalNode.verify(
            LogicalNode.aggregate(
                LogicalNode.rank(
                    LogicalNode.union(
                        LogicalNode.isolate("q1"), LogicalNode.isolate("q2")
                    ),
                    criterion="relevance",
                ),
                goal="answer",
            ),
            constraints="grounded",
        )
        physical = build_physical_plan(logical)

        def check(node):
            assert node.variant in REGISTRY or node.variant in (
                "NoDecompose",
                "IdentityTransform",
            ), f"Unknown variant: {node.variant}"
            for c in node.inputs:
                check(c)

        check(physical)


# ══════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════


class TestRunner:
    def setup_method(self):
        self.corpus = _corpus()
        self.llm = _llm(
            "The transformer method achieves strong results on BLEU metrics."
        )
        self.runner = PlanRunner(corpus=self.corpus, llm=self.llm)

    def test_simple_plan_runs(self):
        logical = LogicalNode.aggregate(
            LogicalNode.isolate("main contribution"), goal="summarize"
        )
        physical = build_physical_plan(logical)
        result = run(self.runner.run(physical))
        assert result.answer != ""
        assert not result.errors

    def test_trace_recorded(self):
        logical = LogicalNode.aggregate(LogicalNode.isolate("q"), goal="ans")
        physical = build_physical_plan(logical)
        result = run(self.runner.run(physical))
        assert len(result.trace) > 0

    def test_token_counts_recorded(self):
        logical = LogicalNode.aggregate(LogicalNode.isolate("q"), goal="ans")
        physical = build_physical_plan(logical)
        result = run(self.runner.run(physical))
        assert len(result.token_counts) > 0

    def test_unknown_variant_captured_not_raised(self):
        bad = PhysicalNode(
            variant="DoesNotExist",
            logical_ref=LogicalNode.isolate("q"),
            inputs=(),
            params={},
            cost=CostVector(),
        )
        result = run(self.runner.run(bad))
        assert len(result.errors) == 1
        assert "DoesNotExist" in result.errors[0]

    def test_holistic_qa_end_to_end(self):
        """Full optimized holistic QA plan runs without errors."""
        from optimizer.engine import OptimizerEngine

        naive = LogicalNode.aggregate(
            LogicalNode.union(
                LogicalNode.isolate("What method is proposed?"),
                LogicalNode.isolate("How is the method evaluated?"),
                LogicalNode.isolate("What are the limitations?"),
            ),
            goal="holistic QA answer",
        )
        optimized, _ = OptimizerEngine().run(naive)
        physical = build_physical_plan(optimized)
        result = run(self.runner.run(physical))

        assert result.answer != ""
        assert len(result.trace) > 0

    def test_runner_injects_catalog_into_llm_summarize(self):
        catalog = SystemCatalog(
            semantic_stats=SemanticStats(
                summarization_map={"p1/method": "Cached method summary."}
            )
        )
        runner = PlanRunner(corpus=self.corpus, llm=self.llm, catalog=catalog)

        physical = PhysicalNode(
            variant="LLMSummarize",
            logical_ref=LogicalNode.transform(
                LogicalNode.isolate("method"), schema="key"
            ),
            inputs=(
                PhysicalNode(
                    variant="HybridRetrieve",
                    logical_ref=LogicalNode.isolate("method"),
                    inputs=(),
                    params={"query": "method", "top_k": 1},
                    cost=CostVector(),
                ),
            ),
            params={"schema": "key"},
            cost=CostVector(),
        )

        result = run(runner.run(physical))
        assert result.answer == "Cached method summary."
        assert result.errors == []

    def test_parallel_branches_both_execute(self):
        """UNION children should both produce evidence."""
        logical = LogicalNode.aggregate(
            LogicalNode.union(
                LogicalNode.isolate("transformer"),
                LogicalNode.isolate("evaluation metrics"),
            ),
            goal="answer",
        )
        physical = build_physical_plan(logical)
        result = run(self.runner.run(physical))
        assert "SimpleUnion" in result.trace
        assert "HybridRetrieve" in result.trace
