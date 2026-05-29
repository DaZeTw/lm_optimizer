"""Physical variants for the RANK logical operator."""

from __future__ import annotations

from executor.registry import register
from ir.evidence import Chunk, EvidenceSet


@register("SimilarityRank")
async def similarity_rank(
    inputs: list[EvidenceSet], params: dict, corpus, llm, context
) -> EvidenceSet:
    """
    Re-score chunks by cosine similarity to the query, keep top-k.

    Uses sentence-transformers util.cos_sim instead of manual cosine.

    params:
        query      — ranking query, falls back to evidence.query_ref
        top_k      — chunks to keep, default 5
        criterion  — stored in metadata for traceability
    """
    ev = inputs[0] if inputs else EvidenceSet(chunks=[])
    query = params.get("query", ev.query_ref)
    top_k = int(params.get("top_k", 5))
    criterion = params.get("criterion", "relevance")

    if not ev.chunks or not query:
        return ev.append_trace("SimilarityRank")

    try:
        import torch
        from sentence_transformers import util

        q_embed = torch.tensor(corpus.embed(query), dtype=torch.float32).unsqueeze(0)

        chunk_embeds = torch.tensor(
            [corpus.embed(c.text) for c in ev.chunks],
            dtype=torch.float32,
        )

        scores = util.cos_sim(q_embed, chunk_embeds)[0].tolist()

    except ImportError:
        scores = [_fallback_cosine(corpus.embed(query), corpus.embed(c.text)) for c in ev.chunks]

    ranked = sorted(zip(scores, ev.chunks), key=lambda x: -float(x[0]))

    top_chunks = [
        Chunk(
            text=c.text,
            doc_id=c.doc_id,
            section=c.section,
            span=c.span,
            score=float(score),
            metadata={
                **c.metadata,
                "rank_method": "cosine_similarity",
                "rank_criterion": criterion,
            },
        )
        for score, c in ranked[:top_k]
    ]

    return EvidenceSet(
        chunks=top_chunks,
        query_ref=ev.query_ref,
        op_trace=ev.op_trace,
    ).append_trace("SimilarityRank")


@register("CrossEncoderRank")
async def cross_encoder_rank(
    inputs: list[EvidenceSet], params: dict, corpus, llm, context
) -> EvidenceSet:
    """
    Rank using a cross-encoder for precise query-chunk relevance.
    Falls back to SimilarityRank if sentence-transformers unavailable.

    params:
        query   — ranking query
        top_k   — chunks to keep, default 5
        model   — cross-encoder model name
    """
    ev = inputs[0] if inputs else EvidenceSet(chunks=[])
    query = params.get("query", ev.query_ref)
    top_k = int(params.get("top_k", 5))
    model = params.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    if not ev.chunks or not query:
        return ev.append_trace("CrossEncoderRank")

    try:
        from sentence_transformers import CrossEncoder

        ce = CrossEncoder(model)
        scores = ce.predict([(query, c.text) for c in ev.chunks])

        ranked = sorted(zip(scores, ev.chunks), key=lambda x: -float(x[0]))

        top_chunks = [
            Chunk(
                text=c.text,
                doc_id=c.doc_id,
                section=c.section,
                span=c.span,
                score=float(score),
                metadata={
                    **c.metadata,
                    "rank_method": "cross_encoder",
                    "cross_encoder_model": model,
                },
            )
            for score, c in ranked[:top_k]
        ]

        return EvidenceSet(
            chunks=top_chunks,
            query_ref=ev.query_ref,
            op_trace=ev.op_trace,
        ).append_trace("CrossEncoderRank")

    except ImportError:
        return await similarity_rank(inputs, params, corpus, llm, context)


def _fallback_cosine(a: list[float], b: list[float]) -> float:
    """
    Lightweight fallback if sentence-transformers / torch is unavailable.
    """
    import numpy as np

    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)

    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0

    return float(np.dot(a_arr, b_arr) / denom)
