"""Physical variants for the COMPOSE logical operator."""

from __future__ import annotations

import re

from ir.evidence import Chunk, EvidenceSet
from executor.registry import register


@register("ConcatCompose")
async def concat_compose(
    inputs: list[EvidenceSet], params: dict, corpus, llm
) -> EvidenceSet:
    """Concatenate both evidence sets. No reasoning — downstream AGGREGATE handles synthesis."""
    if len(inputs) < 2:
        return (inputs[0] if inputs else EvidenceSet(chunks=[])).append_trace(
            "ConcatCompose"
        )
    left, right = inputs[0], inputs[1]
    return EvidenceSet(
        chunks=left.chunks + right.chunks,
        query_ref=left.query_ref,
        op_trace=left.op_trace + right.op_trace,
    ).append_trace("ConcatCompose")


@register("LLMCompose")
async def llm_compose(
    inputs: list[EvidenceSet], params: dict, corpus, llm
) -> EvidenceSet:
    """
    Synthesize connections between two evidence sets via one LLM call.
    Best for multi-hop reasoning where explicit linking is needed.

    params:
        condition   — relationship to identify (e.g. "compare methods")
        query       — original query for context
    """
    if len(inputs) < 2:
        return (inputs[0] if inputs else EvidenceSet(chunks=[])).append_trace(
            "LLMCompose"
        )

    left, right = inputs[0], inputs[1]
    condition = params.get("condition", "relevant connections")
    query = params.get("query", left.query_ref)

    system = (
        "You are a precise reasoning assistant. "
        "Given two evidence sets, identify and synthesize the connections "
        "relevant to the specified condition. "
        "Output a single coherent passage that links the evidence."
    )
    user = (
        f"Condition: {condition}\n\n"
        f"Evidence set A:\n{left.as_text()}\n\n"
        f"Evidence set B:\n{right.as_text()}"
        + (f"\n\nQuery context: {query}" if query else "")
    )

    composed = await llm.complete(system=system, user=user, max_tokens=512)

    return EvidenceSet(
        chunks=[
            Chunk(
                text=composed,
                doc_id="llm_composed",
                section=condition,
                score=1.0,
                metadata={
                    "condition": condition,
                    "left_chunks": len(left.chunks),
                    "right_chunks": len(right.chunks),
                },
            )
        ],
        query_ref=query,
        op_trace=left.op_trace + right.op_trace,
    ).append_trace("LLMCompose")


@register("KeyMatchCompose")
async def key_match_compose(
    inputs: list[EvidenceSet], params: dict, corpus, llm
) -> EvidenceSet:
    """
    Align chunks that share significant keywords. Deterministic — no LLM.
    Falls back to ConcatCompose when no keyword overlap is found.

    params:
        min_overlap   — minimum shared keywords to form a pair (default 1)
        top_k_pairs   — maximum pairs to keep (default 10)
    """
    if len(inputs) < 2:
        return (inputs[0] if inputs else EvidenceSet(chunks=[])).append_trace(
            "KeyMatchCompose"
        )

    left = inputs[0]
    right = inputs[1]
    min_overlap = params.get("min_overlap", 1)
    top_k_pairs = params.get("top_k_pairs", 10)

    pairs: list[tuple[int, Chunk]] = []
    for lc in left.chunks:
        lk = _keywords(lc.text)
        for rc in right.chunks:
            rk = _keywords(rc.text)
            shared = lk & rk
            if len(shared) >= min_overlap:
                pairs.append(
                    (
                        len(shared),
                        Chunk(
                            text=(
                                f"[Left] {lc.text}\n\n"
                                f"[Right] {rc.text}\n\n"
                                f"[Shared: {', '.join(sorted(shared)[:5])}]"
                            ),
                            doc_id=f"{lc.doc_id}+{rc.doc_id}",
                            section="key_match",
                            score=float(len(shared)),
                            metadata={"shared_keys": list(shared)[:10]},
                        ),
                    )
                )

    pairs.sort(key=lambda x: -x[0])
    kept = [c for _, c in pairs[:top_k_pairs]] or (left.chunks + right.chunks)

    return EvidenceSet(
        chunks=kept,
        query_ref=left.query_ref,
        op_trace=left.op_trace + right.op_trace,
    ).append_trace("KeyMatchCompose")


_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "this",
    "that",
    "it",
    "we",
    "they",
    "he",
    "she",
    "have",
    "has",
    "had",
    "do",
    "does",
}


def _keywords(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"\b[a-z][a-z0-9_-]{2,}\b", text.lower())
        if t not in _STOPWORDS
    }
