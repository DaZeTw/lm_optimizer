import json
import os
from parser.semantic_parser import SemanticParser
from pathlib import Path

import pytest

from catalog import DocStats, ModelStats, SemanticStats, SystemCatalog, query_hash
from executor.corpus import InMemoryCorpus, MockLLM, OpenAILLM
from ir.evidence import Chunk
from ir.nodes import LogicalNode, PhysicalNode
from pipeline import LmOptimizerPipeline


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or os.getenv("RUN_LIVE_OPENAI_TESTS") != "1",
    reason="Requires OPENAI_API_KEY and RUN_LIVE_OPENAI_TESTS=1",
)
def test_pipeline_whole_process_live_gpt4o_mini_with_simulated_facts():
    corpus = InMemoryCorpus(
        chunks=[
            Chunk(
                text="Paper A uses a transformer encoder with BLEU and ROUGE evaluation.",
                doc_id="paper_a",
                section="method",
            ),
            Chunk(
                text="Paper B uses retrieval-augmented generation and reports ROUGE-L gains.",
                doc_id="paper_b",
                section="method",
            ),
            Chunk(
                text="Both papers discuss formula accuracy and error propagation.",
                doc_id="paper_mix",
                section="analysis",
            ),
        ]
    )

    query = "Compare evaluation metrics and formula-accuracy handling across paper A and paper B."

    catalog = SystemCatalog(
        doc_stats={
            "paper_a": DocStats(
                doc_id="paper_a",
                total_chunks=1,
                avg_chunk_tokens=120.0,
                sections=["method"],
                total_tokens=120,
            ),
            "paper_b": DocStats(
                doc_id="paper_b",
                total_chunks=1,
                avg_chunk_tokens=130.0,
                sections=["method"],
                total_tokens=130,
            ),
            "paper_mix": DocStats(
                doc_id="paper_mix",
                total_chunks=1,
                avg_chunk_tokens=110.0,
                sections=["analysis"],
                total_tokens=110,
            ),
        },
        model_stats={
            "gpt-4o-mini": ModelStats(
                model_id="gpt-4o-mini",
                context_window=128_000,
                input_cost_per_1k=0.00015,
                output_cost_per_1k=0.0006,
                avg_latency_ms=700,
                supports_tools=True,
            )
        },
        semantic_stats=SemanticStats(
            theme_clusters={
                "method": ["paper_a:method:0:0", "paper_b:method:0:0"],
                "analysis": ["paper_mix:analysis:0:0"],
            },
            summarization_map={
                "paper_a/method": "Paper A: transformer, BLEU/ROUGE.",
                "paper_b/method": "Paper B: RAG, ROUGE-L gains.",
                "paper_mix/analysis": "Shared formula-accuracy discussion.",
            },
            density_map={query_hash("evaluation metrics"): 0.55},
            overlap_graph={
                "paper_a:method:0:0": ["paper_b:method:0:0"],
                "paper_b:method:0:0": ["paper_a:method:0:0"],
            },
        ),
        default_model_id="gpt-4o-mini",
    )

    parser = SemanticParser(model="gpt-4o-mini", temperature=0.0, catalog=catalog)
    llm = OpenAILLM(model="gpt-4o-mini")

    pipeline = LmOptimizerPipeline(
        corpus=corpus,
        llm=llm,
        parser=parser,
        catalog=catalog,
        use_cost_aware_planner=True,
        planner_preset="balanced",
        model_id="gpt-4o-mini",
    )

    result = pipeline.run_sync(
        "Compare evaluation metrics across papers.", log_path="logs/latest_run.json"
    )

    assert result.logical_plan.op.value in ("AGGREGATE", "VERIFY")
    assert result.optimized_plan.op.value in ("AGGREGATE", "VERIFY")
    assert result.physical_plan.variant in ("DirectGenerate", "HierarchicalGenerate")
    assert result.execution.answer.strip() != ""
    assert result.execution.errors == []
    assert len(result.execution.trace) > 0


class LongContextStubParser:
    def parse(self, query: str) -> LogicalNode:
        return LogicalNode.aggregate(
            LogicalNode.union(
                LogicalNode.isolate(f"{query} methodology details"),
                LogicalNode.isolate(f"{query} evaluation metrics"),
                LogicalNode.isolate(f"{query} formula accuracy limitations"),
            ),
            goal="compare methodology, metrics, and formula-accuracy tradeoffs",
        )


class PlanningStubClient:
    def __init__(self):
        self.calls = 0
        self.last_messages = None

    def complete(self, messages, model, temperature):
        self.calls += 1
        self.last_messages = messages
        return json.dumps(
            {
                "variant_overrides": {
                    "AGGREGATE": "DirectGenerate",
                }
            }
        )


def _long_context_corpus() -> InMemoryCorpus:
    chunks: list[Chunk] = []
    for i in range(24):
        chunks.append(
            Chunk(
                text=(
                    "Paper A methodology section details transformer encoder, attention maps, "
                    "ablation settings, and robustness analysis. "
                )
                * 8,
                doc_id="paper_a",
                section="method",
                score=0.85,
                metadata={"chunk_index": i},
            )
        )
    for i in range(24):
        chunks.append(
            Chunk(
                text=(
                    "Paper B evaluation section reports BLEU, ROUGE-L, latency, and token budget, "
                    "including formula-level error analysis under long-context retrieval settings. "
                )
                * 8,
                doc_id="paper_b",
                section="evaluation",
                score=0.87,
                metadata={"chunk_index": i},
            )
        )
    for i in range(24):
        chunks.append(
            Chunk(
                text=(
                    "Cross-paper analysis discusses formula-accuracy failure modes, grounding checks, "
                    "and mitigation strategies for hallucination under dense retrieval overlap. "
                )
                * 8,
                doc_id="paper_mix",
                section="analysis",
                score=0.83,
                metadata={"chunk_index": i},
            )
        )
    return InMemoryCorpus(chunks=chunks)


def _long_context_catalog() -> SystemCatalog:
    return SystemCatalog(
        doc_stats={
            "paper_a": DocStats(
                doc_id="paper_a",
                total_chunks=24,
                avg_chunk_tokens=300.0,
                sections=["method"],
                total_tokens=7200,
            ),
            "paper_b": DocStats(
                doc_id="paper_b",
                total_chunks=24,
                avg_chunk_tokens=320.0,
                sections=["evaluation"],
                total_tokens=7680,
            ),
            "paper_mix": DocStats(
                doc_id="paper_mix",
                total_chunks=24,
                avg_chunk_tokens=310.0,
                sections=["analysis"],
                total_tokens=7440,
            ),
        },
        model_stats={
            "gpt-4o-mini": ModelStats(
                model_id="gpt-4o-mini",
                context_window=128_000,
                input_cost_per_1k=0.00015,
                output_cost_per_1k=0.0006,
                avg_latency_ms=700,
                supports_tools=True,
            )
        },
        semantic_stats=SemanticStats(
            theme_clusters={
                "method": ["paper_a:method:0:0"],
                "evaluation": ["paper_b:evaluation:0:0"],
                "analysis": ["paper_mix:analysis:0:0"],
            },
            summarization_map={
                "paper_a/method": "Paper A: transformer architecture and ablation results.",
                "paper_b/evaluation": "Paper B: BLEU, ROUGE-L, and latency tradeoffs.",
                "paper_mix/analysis": "Cross-paper formula-accuracy limitations and mitigations.",
            },
            density_map={
                query_hash("methodology details"): 0.58,
                query_hash("evaluation metrics"): 0.61,
                query_hash("formula accuracy limitations"): 0.64,
            },
            overlap_graph={
                "paper_a:method:0:0": ["paper_mix:analysis:0:0"],
                "paper_b:evaluation:0:0": ["paper_mix:analysis:0:0"],
            },
        ),
        default_model_id="gpt-4o-mini",
    )


def _collect_variants(root: PhysicalNode) -> list[str]:
    variants = [root.variant]
    for child in root.inputs:
        variants.extend(_collect_variants(child))
    return variants


def test_pipeline_long_context_with_catalog_and_phase_logs():
    query = (
        "Compare paper A and paper B on methodology, metrics, and formula-accuracy handling "
        "for long-context QA."
    )
    catalog = _long_context_catalog()
    pipeline = LmOptimizerPipeline(
        corpus=_long_context_corpus(),
        llm=MockLLM("Catalog-aware long-context synthesis complete."),
        parser=LongContextStubParser(),
        catalog=catalog,
        use_cost_aware_planner=True,
        planner_preset="balanced",
        model_id="gpt-4o-mini",
    )

    result = pipeline.run_sync(query)

    phase_log = {
        "parse": {
            "query": query,
            "root_op": result.logical_plan.op.value,
            "node": result.logical_plan.to_dict(),
        },
        "optimize": {
            "root_op": result.optimized_plan.op.value,
            "rewrite_count": len(result.rewrite_log),
            "rewrite_rules": [entry.rule for entry in result.rewrite_log],
            "node": result.optimized_plan.to_dict(),
        },
        "physical_plan": {
            "root_variant": result.physical_plan.variant,
            "variants": _collect_variants(result.physical_plan),
            "node": {
                "variant": result.physical_plan.variant,
                "params": result.physical_plan.params,
                "logical_root_op": result.physical_plan.logical_ref.op.value,
            },
        },
        "execute": {
            "answer_preview": result.execution.answer[:180],
            "trace": result.execution.trace,
            "token_counts": result.execution.token_counts,
            "errors": result.execution.errors,
        },
        "catalog": {
            "default_model_id": catalog.default_model_id,
            "doc_ids": sorted(catalog.doc_stats.keys()),
            "theme_labels": sorted(catalog.semantic_stats.theme_clusters.keys()),
            "avg_chunk_tokens": catalog.avg_chunk_tokens(),
            "context_window": catalog.context_window(model_id="gpt-4o-mini"),
        },
    }

    log_dir = Path(__file__).parent / "fixtures" / "pipeline_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pipeline_long_context_phase_log.json"
    log_path.write_text(json.dumps(phase_log, indent=2), encoding="utf-8")

    assert result.logical_plan.op.value == "AGGREGATE"
    assert result.optimized_plan.op.value in ("AGGREGATE", "VERIFY")
    assert result.physical_plan.variant in ("DirectGenerate", "HierarchicalGenerate")
    assert result.execution.answer == "Catalog-aware long-context synthesis complete."
    assert result.execution.errors == []
    assert len(result.execution.trace) > 0
    assert log_path.exists()

    loaded = json.loads(log_path.read_text(encoding="utf-8"))
    assert set(loaded.keys()) == {
        "parse",
        "optimize",
        "physical_plan",
        "execute",
        "catalog",
    }
    assert loaded["catalog"]["default_model_id"] == "gpt-4o-mini"


def test_pipeline_physical_plan_llm_feedback_mode_applies_override():
    planning_client = PlanningStubClient()
    pipeline = LmOptimizerPipeline(
        corpus=_long_context_corpus(),
        llm=MockLLM("Planner feedback answer."),
        parser=LongContextStubParser(),
        catalog=_long_context_catalog(),
        use_cost_aware_planner=True,
        planner_preset="balanced",
        model_id="gpt-4o-mini",
        physical_planner_mode="llm_feedback",
        planning_client=planning_client,
        planning_rounds=1,
    )

    result = pipeline.run_sync("long context compare")

    assert planning_client.calls == 1
    assert result.physical_plan.variant == "DirectGenerate"
    assert result.execution.errors == []


def test_pipeline_feedback_iterations_call_llm_planner_each_round():
    planning_client = PlanningStubClient()
    corpus = _long_context_corpus()
    pipeline = LmOptimizerPipeline(
        corpus=corpus,
        llm=MockLLM("Planner feedback answer."),
        parser=LongContextStubParser(),
        catalog=_long_context_catalog(),
        use_cost_aware_planner=True,
        planner_preset="balanced",
        model_id="gpt-4o-mini",
        physical_planner_mode="llm_feedback",
        planning_client=planning_client,
        planning_rounds=1,
    )

    result = pipeline.run_sync_with_physical_feedback(
        "long context compare",
        corpus,
        iterations=2,
    )

    assert planning_client.calls == 2
    assert result.physical_plan.variant == "DirectGenerate"
    assert result.execution.errors == []
