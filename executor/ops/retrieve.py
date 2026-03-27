"""Physical variants for the I(q, C) logical operator."""

from __future__ import annotations

from ir.evidence import EvidenceSet
from executor.registry import register


@register("BM25Retrieve")
async def bm25_retrieve(
    inputs: list[EvidenceSet], params: dict, corpus, llm
) -> EvidenceSet:
    """Sparse keyword retrieval. Fast, no embeddings."""
    query = params.get("query", "")
    top_k = params.get("top_k", 10)
    chunks = corpus.bm25_search(query, top_k=top_k)
    for c in chunks:
        c.metadata["retrieval"] = "bm25"
    return EvidenceSet(chunks=chunks, query_ref=query).append_trace("BM25Retrieve")


@register("DenseRetrieve")
async def dense_retrieve(
    inputs: list[EvidenceSet], params: dict, corpus, llm
) -> EvidenceSet:
    """Embedding-based dense retrieval."""
    query = params.get("query", "")
    top_k = params.get("top_k", 10)
    chunks = corpus.dense_search(query, top_k=top_k)
    for c in chunks:
        c.metadata["retrieval"] = "dense"
    return EvidenceSet(chunks=chunks, query_ref=query).append_trace("DenseRetrieve")


@register("HybridRetrieve")
async def hybrid_retrieve(
    inputs: list[EvidenceSet], params: dict, corpus, llm
) -> EvidenceSet:
    """
    Weighted BM25 + dense retrieval (default variant).
    Best recall across most query types.

    params:
        query  — retrieval query string
        top_k  — number of results (default 10)
        alpha  — dense weight 0.0–1.0 (default 0.5)
    """
    query = params.get("query", "")
    top_k = params.get("top_k", 10)
    alpha = params.get("alpha", 0.5)
    chunks = corpus.hybrid_search(query, top_k=top_k, alpha=alpha)
    for c in chunks:
        c.metadata["retrieval"] = "hybrid"
        c.metadata["alpha"] = alpha
    return EvidenceSet(chunks=chunks, query_ref=query).append_trace("HybridRetrieve")
