"""Physical variants for the I(q, C) logical operator."""

from __future__ import annotations

from executor.registry import register
from ir.evidence import EvidenceSet


@register("BM25Retrieve")
async def bm25_retrieve(
    inputs: list[EvidenceSet], params: dict, corpus, llm, context
) -> EvidenceSet:
    """Sparse keyword retrieval. Fast, no embeddings needed."""
    query = params.get("query", "")
    top_k = params.get("top_k", 10)
    chunks = corpus.bm25_search(query, top_k=top_k)
    for c in chunks:
        c.metadata["retrieval"] = "bm25"
    return EvidenceSet(chunks=chunks, query_ref=query).append_trace("BM25Retrieve")


@register("DenseRetrieve")
async def dense_retrieve(
    inputs: list[EvidenceSet], params: dict, corpus, llm, context
) -> EvidenceSet:
    """
    Embedding-based dense retrieval.

    Requires corpus to be initialised with an OpenAIEmbedder (or any
    Embedder implementation). Falls back to BM25 with a metadata flag
    if no embedder is present — never silently degrades without notice.
    """
    query = params.get("query", "")
    top_k = params.get("top_k", 10)
    chunks = corpus.dense_search(query, top_k=top_k)
    for c in chunks:
        c.metadata.setdefault("retrieval", "dense")  # don't overwrite fallback flag
    return EvidenceSet(chunks=chunks, query_ref=query).append_trace("DenseRetrieve")


@register("HybridRetrieve")
async def hybrid_retrieve(
    inputs: list[EvidenceSet], params: dict, corpus, llm, context
) -> EvidenceSet:
    """
    RRF fusion of BM25 + dense retrieval (recommended default).

    params:
        query  — retrieval query string
        top_k  — number of results (default 10)
        alpha  — dense weight 0–1 (default 0.5); 1.0 = pure dense
    """
    query = params.get("query", "")
    top_k = params.get("top_k", 10)
    alpha = params.get("alpha", 0.5)
    chunks = corpus.hybrid_search(query, top_k=top_k, alpha=alpha)
    for c in chunks:
        c.metadata.setdefault("retrieval", "hybrid")
        c.metadata["alpha"] = alpha
    return EvidenceSet(chunks=chunks, query_ref=query).append_trace("HybridRetrieve")
