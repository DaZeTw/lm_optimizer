"""Physical variants for the AGGREGATE logical operator."""

from __future__ import annotations

from ir.evidence import Chunk, EvidenceSet
from executor.registry import register

_DEFAULT_CONTEXT_WINDOW = 128_000


@register("DirectGenerate")
async def direct_generate(
    inputs: list[EvidenceSet], params: dict, corpus, llm
) -> EvidenceSet:
    """
    Single LLM call with all ranked evidence in context.
    Default variant. Switch to HierarchicalGenerate when saturation > 0.7.

    params:
        goal             — synthesis goal
        max_tokens       — output tokens (default 512)
        context_window   — model context window size (default 128k)
    """
    ev = inputs[0] if inputs else EvidenceSet(chunks=[])
    goal = params.get("goal", "answer the question")
    max_tokens = params.get("max_tokens", 512)
    context_window = params.get("context_window", _DEFAULT_CONTEXT_WINDOW)

    saturation = ev.token_estimate() / context_window
    if saturation > 1.0:
        ev = _truncate(ev, int(context_window * 0.8))

    system = (
        "You are a precise research assistant. "
        "Answer based only on the provided evidence. "
        "Be thorough but concise. Cite specific evidence when possible."
    )
    user = f"Task: {goal}\n\nEvidence:\n{ev.as_text()}\n\nAnswer:"

    answer = await llm.complete(system=system, user=user, max_tokens=max_tokens)

    return EvidenceSet(
        chunks=[
            Chunk(
                text=answer,
                doc_id="generated_answer",
                section=goal,
                score=1.0,
                metadata={
                    "goal": goal,
                    "source_chunks": len(ev.chunks),
                    "source_tokens": ev.token_estimate(),
                    "saturation": round(saturation, 3),
                },
            )
        ],
        query_ref=ev.query_ref,
        op_trace=ev.op_trace,
    ).append_trace("DirectGenerate")


@register("HierarchicalGenerate")
async def hierarchical_generate(
    inputs: list[EvidenceSet], params: dict, corpus, llm
) -> EvidenceSet:
    """
    Map-reduce generation for large evidence sets.

    1. Split evidence into batches that fit the context window.
    2. Summarize each batch (sequential — can be parallelised later).
    3. Synthesize a final answer from the summaries.

    params:
        goal          — synthesis goal
        batch_tokens  — tokens per batch (default 8000)
        max_tokens    — tokens per LLM call (default 512)
    """
    ev = inputs[0] if inputs else EvidenceSet(chunks=[])
    goal = params.get("goal", "answer the question")
    batch_tokens = params.get("batch_tokens", 8_000)
    max_tokens = params.get("max_tokens", 512)

    if not ev.chunks:
        return EvidenceSet(chunks=[], query_ref=ev.query_ref)

    batches = _batch(ev, batch_tokens)
    sys_sum = f"Summarize evidence relevant to: {goal}\nBe concise — preserve key facts and numbers."
    summaries: list[str] = []

    for i, batch in enumerate(batches):
        s = await llm.complete(
            system=sys_sum,
            user=f"Evidence batch {i+1}/{len(batches)}:\n{batch.as_text()}",
            max_tokens=max_tokens,
        )
        summaries.append(s)

    combined = "\n\n".join(f"[Summary {i+1}]\n{s}" for i, s in enumerate(summaries))
    final = await llm.complete(
        system="Synthesize the provided summaries into a final coherent answer.",
        user=f"Task: {goal}\n\nSummaries:\n{combined}\n\nFinal answer:",
        max_tokens=max_tokens,
    )

    return EvidenceSet(
        chunks=[
            Chunk(
                text=final,
                doc_id="hierarchical_answer",
                section=goal,
                score=1.0,
                metadata={
                    "goal": goal,
                    "batches": len(batches),
                    "total_source_chunks": len(ev.chunks),
                    "llm_calls": len(batches) + 1,
                },
            )
        ],
        query_ref=ev.query_ref,
        op_trace=ev.op_trace,
    ).append_trace("HierarchicalGenerate")


def _truncate(ev: EvidenceSet, max_tokens: int) -> EvidenceSet:
    kept, total = [], 0
    for c in sorted(ev.chunks, key=lambda c: -c.score):
        est = c.token_estimate()
        if total + est > max_tokens:
            break
        kept.append(c)
        total += est
    return EvidenceSet(chunks=kept, query_ref=ev.query_ref, op_trace=ev.op_trace)


def _batch(ev: EvidenceSet, batch_tokens: int) -> list[EvidenceSet]:
    batches, current, total = [], [], 0
    for c in ev.chunks:
        est = c.token_estimate()
        if current and total + est > batch_tokens:
            batches.append(EvidenceSet(chunks=current, query_ref=ev.query_ref))
            current, total = [], 0
        current.append(c)
        total += est
    if current:
        batches.append(EvidenceSet(chunks=current, query_ref=ev.query_ref))
    return batches or [ev]
