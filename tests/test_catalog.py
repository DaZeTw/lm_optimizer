from __future__ import annotations

import asyncio

from catalog import (
    DocStats,
    ModelStats,
    SemanticStats,
    SystemCatalog,
    build_doc_stats,
    build_summarization_map,
    build_system_catalog,
    load_catalog,
    query_hash,
    save_catalog,
)
from executor.corpus import InMemoryCorpus
from ir.evidence import Chunk


def _corpus() -> InMemoryCorpus:
    return InMemoryCorpus(
        chunks=[
            Chunk(
                text="Method details and architecture.", doc_id="d1", section="method"
            ),
            Chunk(text="Evaluation uses BLEU and ROUGE.", doc_id="d1", section="eval"),
            Chunk(text="Limitations include latency.", doc_id="d2", section="limits"),
        ]
    )


def test_build_doc_stats_from_corpus():
    stats = build_doc_stats(_corpus())

    assert set(stats.keys()) == {"d1", "d2"}
    assert stats["d1"].total_chunks == 2
    assert stats["d1"].total_tokens > 0
    assert "method" in stats["d1"].sections


def test_system_catalog_context_and_density_lookup():
    q = "method"
    density = {query_hash(q): 0.63}
    catalog = SystemCatalog(
        doc_stats={
            "d1": DocStats(
                doc_id="d1",
                total_chunks=2,
                avg_chunk_tokens=240.0,
                sections=["method", "eval"],
                total_tokens=480,
            )
        },
        model_stats={
            "tiny": ModelStats(
                model_id="tiny",
                context_window=8192,
                input_cost_per_1k=0.0,
                output_cost_per_1k=0.0,
                avg_latency_ms=100,
                supports_tools=False,
            )
        },
        semantic_stats=SemanticStats(density_map=density),
        default_model_id="tiny",
    )

    assert catalog.avg_chunk_tokens() == 240.0
    assert catalog.context_window() == 8192
    assert catalog.density_for_query(q) == 0.63


def test_build_summarization_map_without_llm_uses_fallback():
    summaries = asyncio.run(build_summarization_map(_corpus(), llm=None))

    assert "d1/method" in summaries
    assert summaries["d1/method"]


def test_indexer_save_load_roundtrip(tmp_path):
    catalog = asyncio.run(build_system_catalog(_corpus(), llm=None))
    path = tmp_path / "catalog.json"

    save_catalog(catalog, path)
    loaded = load_catalog(path)

    assert loaded.default_model_id == catalog.default_model_id
    assert set(loaded.doc_stats.keys()) == set(catalog.doc_stats.keys())
    assert loaded.semantic_stats.summarization_map
