"""Physical variants for the RANK logical operator."""

from __future__ import annotations

from executor.registry import register
from ir.evidence import Chunk, EvidenceSet


@register("SimilarityRank")
async def similarity_rank(
    inputs: list[EvidenceSet], params: dict, corpus, llm
) -> EvidenceSet:
    """
    Re-score chunks by cosine similarity to the query, keep top-k.
    Uses corpus embeddings — no extra model needed.

    params:
        query      — ranking query (falls back to evidence.query_ref)
        top_k      — chunks to keep (default 5)
        criterion  — stored in metadata for traceability
    """
    ev = inputs[0] if inputs else EvidenceSet(chunks=[])
    query = params.get("query", ev.query_ref)
    top_k = params.get("top_k", 5)
    criterion = params.get("criterion", "relevance")

    if not ev.chunks or not query:
        return ev.append_trace("SimilarityRank")

    q_embed = corpus.embed(query)
    scored = sorted(
        [(_cosine(q_embed, corpus.embed(c.text[:512])), c) for c in ev.chunks],
        key=lambda x: -x[0],
    )
    top_chunks = [
        Chunk(
            text=c.text,
            doc_id=c.doc_id,
            section=c.section,
            span=c.span,
            score=score,
            metadata={**c.metadata, "rank_criterion": criterion},
        )
        for score, c in scored[:top_k]
    ]
    return EvidenceSet(
        chunks=top_chunks,
        query_ref=ev.query_ref,
        op_trace=ev.op_trace,
    ).append_trace("SimilarityRank")


@register("CrossEncoderRank")
async def cross_encoder_rank(
    inputs: list[EvidenceSet], params: dict, corpus, llm
) -> EvidenceSet:
    """
    Rank using a cross-encoder for precise query-chunk relevance.
    Falls back to SimilarityRank if sentence-transformers unavailable.

    params:
        query   — ranking query
        top_k   — chunks to keep (default 5)
        model   — cross-encoder model name
    """
    ev = inputs[0] if inputs else EvidenceSet(chunks=[])
    query = params.get("query", ev.query_ref)
    top_k = params.get("top_k", 5)
    model = params.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    if not ev.chunks or not query:
        return ev.append_trace("CrossEncoderRank")

    try:
        from sentence_transformers import CrossEncoder

        ce = CrossEncoder(model)
        scores = ce.predict([(query, c.text[:512]) for c in ev.chunks])
        ranked = sorted(zip(scores, ev.chunks), key=lambda x: -float(x[0]))
        top_chunks = [
            Chunk(
                text=c.text,
                doc_id=c.doc_id,
                section=c.section,
                span=c.span,
                score=float(s),
                metadata={**c.metadata, "rank_method": "cross_encoder"},
            )
            for s, c in ranked[:top_k]
        ]
        return EvidenceSet(
            chunks=top_chunks,
            query_ref=ev.query_ref,
            op_trace=ev.op_trace,
        ).append_trace("CrossEncoderRank")

    except ImportError:
        return await similarity_rank(inputs, params, corpus, llm)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    raw = dot / (na * nb) if na and nb else 0.0
    return max(0.0, min(1.0, raw))  # clamp: float arithmetic can exceed ±1
