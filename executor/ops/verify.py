"""Physical variants for the VERIFY logical operator."""

from __future__ import annotations

import re

from ir.evidence import Chunk, EvidenceSet
from executor.registry import register


@register("CitationVerify")
async def citation_verify(
    inputs: list[EvidenceSet], params: dict, corpus, llm
) -> EvidenceSet:
    """
    Verify answer claims against evidence using keyword overlap.
    No LLM call — fast and deterministic.

    inputs[0]  — generated answer (from AGGREGATE)
    inputs[1]  — evidence used to generate it (optional)

    Attaches grounding_score (0–1) and unsupported claim list to metadata.
    """
    if not inputs:
        return EvidenceSet(chunks=[])

    answer_es = inputs[0]
    evidence_es = inputs[1] if len(inputs) > 1 else EvidenceSet(chunks=[])

    ev_words = set(
        re.findall(r"\b\w{4,}\b", " ".join(c.text for c in evidence_es.chunks).lower())
    )

    sentences = [
        s.strip() for s in re.split(r"[.!?]", answer_es.as_text()) if s.strip()
    ]
    supported, unsupported = 0, []

    for sent in sentences:
        sig_words = set(re.findall(r"\b\w{4,}\b", sent.lower())) - _COMMON
        if not sig_words:
            supported += 1
            continue
        if len(sig_words & ev_words) / len(sig_words) >= 0.3:
            supported += 1
        else:
            unsupported.append(sent[:100])

    score = supported / max(len(sentences), 1)

    return EvidenceSet(
        chunks=[
            Chunk(
                text=answer_es.as_text(),
                doc_id="verified_answer",
                section="citation_verify",
                score=score,
                metadata={
                    "grounding_score": round(score, 3),
                    "sentences_checked": len(sentences),
                    "unsupported_count": len(unsupported),
                    "unsupported": unsupported[:5],
                    "verify_method": "citation",
                },
            )
        ],
        query_ref=answer_es.query_ref,
        op_trace=answer_es.op_trace,
    ).append_trace("CitationVerify")


@register("NliVerify")
async def nli_verify(
    inputs: list[EvidenceSet], params: dict, corpus, llm
) -> EvidenceSet:
    """
    Verify using a local NLI model (entailment check).
    Falls back to CitationVerify if sentence-transformers unavailable.

    inputs[0]  — generated answer
    inputs[1]  — evidence (optional)

    params:
        model      — NLI model name
        threshold  — entailment probability threshold (default 0.5)
    """
    if not inputs:
        return EvidenceSet(chunks=[])

    threshold = params.get("threshold", 0.5)
    model_name = params.get("model", "cross-encoder/nli-deberta-v3-small")

    try:
        from sentence_transformers import CrossEncoder

        nli = CrossEncoder(model_name)

        answer_es = inputs[0]
        evidence_es = inputs[1] if len(inputs) > 1 else EvidenceSet(chunks=[])
        sentences = [
            s.strip() for s in re.split(r"[.!?]", answer_es.as_text()) if s.strip()
        ]
        ev_texts = [c.text for c in evidence_es.chunks]

        if not ev_texts or not sentences:
            return await citation_verify(inputs, params, corpus, llm)

        entailed, flags = 0, []
        for sent in sentences:
            scores = nli.predict(
                [(ev, sent) for ev in ev_texts[:10]], apply_softmax=True
            )
            max_ent = max(float(s[2]) for s in scores)
            if max_ent >= threshold:
                entailed += 1
            else:
                flags.append(
                    {"sentence": sent[:80], "max_entailment": round(max_ent, 3)}
                )

        score = entailed / max(len(sentences), 1)

        return EvidenceSet(
            chunks=[
                Chunk(
                    text=answer_es.as_text(),
                    doc_id="nli_verified_answer",
                    section="nli_verify",
                    score=score,
                    metadata={
                        "grounding_score": round(score, 3),
                        "sentences_checked": len(sentences),
                        "unentailed_count": len(flags),
                        "flags": flags[:5],
                        "verify_method": "nli",
                    },
                )
            ],
            query_ref=answer_es.query_ref,
            op_trace=answer_es.op_trace,
        ).append_trace("NliVerify")

    except (ImportError, Exception):
        return await citation_verify(inputs, params, corpus, llm)


_COMMON = {
    "this",
    "that",
    "these",
    "those",
    "with",
    "have",
    "from",
    "they",
    "will",
    "been",
    "were",
    "their",
    "there",
    "about",
    "which",
    "when",
    "also",
    "more",
    "than",
    "then",
    "some",
    "what",
    "into",
    "over",
}
