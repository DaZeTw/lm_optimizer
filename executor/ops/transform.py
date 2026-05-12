"""Physical variants for the TRANSFORM logical operator."""

from __future__ import annotations

from executor.registry import register
from ir.evidence import Chunk, EvidenceSet


@register("LLMSummarize")
async def llm_summarize(
    inputs: list[EvidenceSet], params: dict, corpus, llm
) -> EvidenceSet:
    """
    Summarize/extract evidence into a schema via one LLM call.

    params:
        schema   — extraction target (e.g. "evaluation metrics list")
        query    — optional context
    """
    ev = inputs[0] if inputs else EvidenceSet(chunks=[])
    schema = params.get("schema", "key information")
    query = params.get("query", ev.query_ref)
    catalog = params.get("catalog")

    if not ev.chunks:
        return ev.append_trace("LLMSummarize")

    cache_key = params.get("summary_key")
    if not cache_key and ev.chunks:
        first = ev.chunks[0]
        same_source = all(
            c.doc_id == first.doc_id and c.section == first.section for c in ev.chunks
        )
        if same_source:
            cache_key = (
                f"{first.doc_id}/{first.section}" if first.section else first.doc_id
            )

    if catalog is not None and cache_key:
        cached = catalog.cached_summary_by_key(cache_key)
        if cached:
            return EvidenceSet(
                chunks=[
                    Chunk(
                        text=cached,
                        doc_id="summary_cache",
                        section=schema,
                        score=1.0,
                        metadata={
                            "schema": schema,
                            "cache_key": cache_key,
                            "cached": True,
                            "source_chunks": len(ev.chunks),
                            "source_tokens": ev.token_estimate(),
                        },
                    )
                ],
                query_ref=ev.query_ref,
                op_trace=ev.op_trace,
            ).append_trace("LLMSummarize")

    system = (
        "You are a precise information extractor. "
        "Extract exactly what is asked for. Be concise. "
        "Output only the extracted content."
    )
    user = (
        f"Extract the following from the source text: {schema}\n\n"
        f"Source text:\n{ev.as_text()}"
        + (f"\n\nContext query: {query}" if query else "")
    )

    summary = await llm.complete(system=system, user=user, max_tokens=512)

    return EvidenceSet(
        chunks=[
            Chunk(
                text=summary,
                doc_id="llm_summary",
                section=schema,
                score=1.0,
                metadata={
                    "schema": schema,
                    "source_chunks": len(ev.chunks),
                    "source_tokens": ev.token_estimate(),
                },
            )
        ],
        query_ref=ev.query_ref,
        op_trace=ev.op_trace,
    ).append_trace("LLMSummarize")
