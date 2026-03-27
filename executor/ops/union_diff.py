"""Physical variants for UNION and DIFF logical operators."""

from __future__ import annotations

import hashlib

from ir.evidence import Chunk, EvidenceSet
from executor.registry import register


@register("SimpleUnion")
async def simple_union(
    inputs: list[EvidenceSet], params: dict, corpus, llm
) -> EvidenceSet:
    """Concatenate all inputs. No dedup — use when DIFF follows."""
    if not inputs:
        return EvidenceSet(chunks=[])
    all_chunks = [c for ev in inputs for c in ev.chunks]
    merged_trace = [t for ev in inputs for t in ev.op_trace]
    return EvidenceSet(
        chunks=all_chunks,
        query_ref=inputs[0].query_ref,
        op_trace=merged_trace,
    ).append_trace("SimpleUnion")


@register("ExactDiff")
async def exact_diff(
    inputs: list[EvidenceSet], params: dict, corpus, llm
) -> EvidenceSet:
    """
    Remove chunks from inputs[0] whose text hash matches any chunk
    in inputs[1].

    When inputs[1] is the __overlap__ sentinel inserted by R7,
    performs hash-based self-deduplication on inputs[0].
    """
    if not inputs:
        return EvidenceSet(chunks=[])

    base = inputs[0]
    subtract = inputs[1] if len(inputs) > 1 else EvidenceSet(chunks=[])

    is_sentinel = len(subtract.chunks) == 1 and "__overlap__" in subtract.chunks[0].text
    subtract_hashes = {_hash(c.text) for c in subtract.chunks}

    seen: set[str] = set()
    kept: list[Chunk] = []

    for chunk in base.chunks:
        h = _hash(chunk.text)
        if is_sentinel:
            if h not in seen:
                seen.add(h)
                kept.append(chunk)
        else:
            if h not in subtract_hashes:
                kept.append(chunk)

    return EvidenceSet(
        chunks=kept,
        query_ref=base.query_ref,
        op_trace=base.op_trace,
    ).append_trace("ExactDiff")


@register("SemanticDiff")
async def semantic_diff(
    inputs: list[EvidenceSet], params: dict, corpus, llm
) -> EvidenceSet:
    """
    Remove near-duplicate chunks using cosine similarity.
    Catches paraphrases that ExactDiff misses.

    When inputs[1] is __overlap__ sentinel, self-deduplicates inputs[0].

    params:
        threshold   — similarity threshold for removal (default 0.92)
    """
    if not inputs:
        return EvidenceSet(chunks=[])

    base = inputs[0]
    subtract = inputs[1] if len(inputs) > 1 else EvidenceSet(chunks=[])
    threshold = params.get("threshold", 0.92)

    is_sentinel = not subtract.chunks or (
        len(subtract.chunks) == 1 and "__overlap__" in subtract.chunks[0].text
    )

    if is_sentinel:
        kept: list[Chunk] = []
        kept_embeds: list[list[float]] = []
        for chunk in base.chunks:
            emb = corpus.embed(chunk.text[:512])
            if not any(_cosine(emb, e) >= threshold for e in kept_embeds):
                kept.append(chunk)
                kept_embeds.append(emb)
    else:
        sub_embeds = [corpus.embed(c.text[:512]) for c in subtract.chunks]
        kept = [
            c
            for c in base.chunks
            if not any(
                _cosine(corpus.embed(c.text[:512]), se) >= threshold
                for se in sub_embeds
            )
        ]

    return EvidenceSet(
        chunks=kept,
        query_ref=base.query_ref,
        op_trace=base.op_trace,
    ).append_trace("SemanticDiff")


def _hash(text: str) -> str:
    return hashlib.md5(text.strip().encode()).hexdigest()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0
