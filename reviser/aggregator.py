"""Step 3: aggregate SampleFeedback dicts into a PatternSummary.

Pure Python — no LLM call.  Groups failure points, successful adaptations,
and cost patterns across all samples in a batch and counts occurrences.
Only patterns that appear in at least `min_frequency` fraction of samples
are promoted to the summary (conservatism rule from Step 5).

PatternSummary schema
---------------------
{
    "num_samples": int,
    "avg_accuracy": float,
    "avg_tokens": float,
    "avg_latency_ms": float,

    "failure_patterns": [
        {
            "op_id":       str,
            "issue_type":  str,
            "description": str,   # most common description for this (op_id, issue_type)
            "count":       int,
            "frequency":   float, # count / num_samples
        },
        ...  # sorted descending by frequency
    ],

    "success_patterns": [
        {
            "op_id":      str,
            "what_worked": str,
            "count":       int,
            "frequency":   float,
        },
        ...
    ],

    "cost_patterns": [
        {
            "op_id":          str,   # or "TOTAL"
            "issue":          str,   # e.g. "high_token_cost", "high_latency"
            "avg_tokens":     float | None,
            "avg_latency_ms": float | None,
            "frequency":      float,
        },
        ...
    ],

    "recommendation": str,  # short human-readable summary for the revision prompt
}
"""

from __future__ import annotations

from collections import Counter, defaultdict


def aggregate_feedback(
    samples: list[dict],
    min_frequency: float = 0.5,
    high_token_threshold: int = 2000,
    high_latency_threshold_ms: float = 5000.0,
) -> dict:
    """Aggregate a batch of SampleFeedback dicts into a PatternSummary.

    Args:
        samples:                 List of SampleFeedback dicts from FeedbackStore.samples().
        min_frequency:           Minimum fraction of samples a pattern must appear
                                 in to be included (default 0.5 = majority).
        high_token_threshold:    Per-node token count considered "high cost".
        high_latency_threshold_ms: Per-node latency considered "slow".

    Returns:
        PatternSummary plain dict.
    """
    n = len(samples)
    if n == 0:
        return _empty_summary()

    avg_accuracy = sum(s["accuracy"] for s in samples) / n
    avg_tokens = sum(s["total_tokens"] for s in samples) / n
    avg_latency_ms = sum(s["total_latency_ms"] for s in samples) / n

    failure_patterns = _aggregate_failures(samples, n, min_frequency)
    success_patterns = _aggregate_successes(samples, n, min_frequency)
    cost_patterns = _aggregate_costs(
        samples, n, min_frequency, high_token_threshold, high_latency_threshold_ms
    )
    recommendation = _build_recommendation(
        avg_accuracy, failure_patterns, success_patterns, cost_patterns
    )

    return {
        "num_samples": n,
        "avg_accuracy": round(avg_accuracy, 3),
        "avg_tokens": round(avg_tokens, 1),
        "avg_latency_ms": round(avg_latency_ms, 1),
        "failure_patterns": failure_patterns,
        "success_patterns": success_patterns,
        "cost_patterns": cost_patterns,
        "recommendation": recommendation,
    }


# ── Renderers (PatternSummary → prompt-ready strings) ─────────────


def render_failure_patterns(summary: dict) -> str:
    lines = []
    for p in summary.get("failure_patterns", []):
        lines.append(
            f"  {p['op_id']} | {p['issue_type']} | {p['description']} "
            f"(in {p['frequency']:.0%} of samples)"
        )
    return "\n".join(lines) if lines else "(none)"


def render_success_patterns(summary: dict) -> str:
    lines = []
    for p in summary.get("success_patterns", []):
        lines.append(
            f"  {p['op_id']} | {p['what_worked']} "
            f"(in {p['frequency']:.0%} of samples)"
        )
    return "\n".join(lines) if lines else "(none)"


def render_cost_patterns(summary: dict) -> str:
    lines = [
        f"  avg_accuracy={summary.get('avg_accuracy', 0):.2f}  "
        f"avg_tokens={summary.get('avg_tokens', 0):.0f}  "
        f"avg_latency={summary.get('avg_latency_ms', 0):.0f}ms"
    ]
    for p in summary.get("cost_patterns", []):
        detail = ""
        if p.get("avg_tokens") is not None:
            detail += f"  avg_tokens={p['avg_tokens']:.0f}"
        if p.get("avg_latency_ms") is not None:
            detail += f"  avg_latency={p['avg_latency_ms']:.0f}ms"
        lines.append(
            f"  {p['op_id']} | {p['issue']}{detail} "
            f"(in {p['frequency']:.0%} of samples)"
        )
    return "\n".join(lines)


# ── Internal helpers ──────────────────────────────────────────────


def _aggregate_failures(samples: list[dict], n: int, min_freq: float) -> list[dict]:
    # Count (op_id, issue_type) pairs; keep most common description per pair
    counts: Counter = Counter()
    descriptions: defaultdict[tuple, Counter] = defaultdict(Counter)

    for s in samples:
        seen_this_sample: set[tuple] = set()
        for fp in s.get("failure_points", []):
            key = (fp["op_id"], fp["issue_type"])
            if key not in seen_this_sample:
                counts[key] += 1
                seen_this_sample.add(key)
            descriptions[key][fp["description"]] += 1

    results = []
    for (op_id, issue_type), count in counts.most_common():
        freq = count / n
        if freq < min_freq:
            continue
        top_desc = descriptions[(op_id, issue_type)].most_common(1)[0][0]
        results.append(
            {
                "op_id": op_id,
                "issue_type": issue_type,
                "description": top_desc,
                "count": count,
                "frequency": round(freq, 3),
            }
        )
    return results


def _aggregate_successes(samples: list[dict], n: int, min_freq: float) -> list[dict]:
    counts: Counter = Counter()
    what_worked_bag: defaultdict[tuple, Counter] = defaultdict(Counter)

    for s in samples:
        seen: set[tuple] = set()
        for sa in s.get("successful_adaptations", []):
            key = (sa["op_id"], sa["what_worked"])
            if key not in seen:
                counts[(sa["op_id"], sa["what_worked"])] += 1
                seen.add(key)
            what_worked_bag[(sa["op_id"], sa["what_worked"])][sa["what_worked"]] += 1

    results = []
    for (op_id, what_worked), count in counts.most_common():
        freq = count / n
        if freq < min_freq:
            continue
        results.append(
            {
                "op_id": op_id,
                "what_worked": what_worked,
                "count": count,
                "frequency": round(freq, 3),
            }
        )
    return results


def _aggregate_costs(
    samples: list[dict],
    n: int,
    min_freq: float,
    high_token_threshold: int,
    high_latency_threshold_ms: float,
) -> list[dict]:
    """Flag per-node cost issues that appear consistently across samples."""
    # Collect per-node metrics from node-level feedback items embedded in samples.
    # SampleFeedback doesn't carry raw node items — we work from suggested_fixes
    # with fix_type=="tune_param" as a proxy for cost issues flagged by the LLM.
    op_token_counts: defaultdict[str, list[float]] = defaultdict(list)
    op_latency_counts: defaultdict[str, list[float]] = defaultdict(list)
    high_token_hits: Counter = Counter()
    high_latency_hits: Counter = Counter()

    for s in samples:
        # Use suggested_fixes as signal: "tune_param" on a node that had a cost issue
        for fix in s.get("suggested_fixes", []):
            if fix["fix_type"] == "tune_param" and "token" in fix["detail"].lower():
                high_token_hits[fix["op_id"]] += 1
            if fix["fix_type"] == "tune_param" and "latency" in fix["detail"].lower():
                high_latency_hits[fix["op_id"]] += 1

    results = []
    all_op_ids = set(high_token_hits) | set(high_latency_hits)
    for op_id in all_op_ids:
        token_freq = high_token_hits.get(op_id, 0) / n
        latency_freq = high_latency_hits.get(op_id, 0) / n

        if token_freq >= min_freq:
            results.append(
                {
                    "op_id": op_id,
                    "issue": "high_token_cost",
                    "avg_tokens": None,
                    "avg_latency_ms": None,
                    "frequency": round(token_freq, 3),
                }
            )
        if latency_freq >= min_freq:
            results.append(
                {
                    "op_id": op_id,
                    "issue": "high_latency",
                    "avg_tokens": None,
                    "avg_latency_ms": None,
                    "frequency": round(latency_freq, 3),
                }
            )

    return sorted(results, key=lambda x: x["frequency"], reverse=True)


def _build_recommendation(
    avg_accuracy: float,
    failure_patterns: list[dict],
    success_patterns: list[dict],
    cost_patterns: list[dict],
) -> str:
    parts: list[str] = []

    if avg_accuracy < 0.5:
        parts.append(
            f"Low average accuracy ({avg_accuracy:.2f}) — prioritise fixing retrieval failures."
        )
    elif avg_accuracy >= 0.8:
        parts.append(
            f"Good accuracy ({avg_accuracy:.2f}) — focus on cost/latency reduction if needed."
        )

    if failure_patterns:
        top = failure_patterns[0]
        parts.append(
            f"Most common failure: {top['op_id']} / {top['issue_type']} "
            f"({top['frequency']:.0%} of samples) — consider changing variant or params."
        )

    if success_patterns:
        top = success_patterns[0]
        parts.append(
            f"Preserve working adaptation: {top['op_id']} / {top['what_worked']} "
            f"({top['frequency']:.0%} of samples)."
        )

    if cost_patterns:
        top = cost_patterns[0]
        parts.append(
            f"Cost issue: {top['op_id']} shows {top['issue']} "
            f"in {top['frequency']:.0%} of samples — tune params or downgrade variant."
        )

    return (
        " ".join(parts)
        if parts
        else "No strong patterns detected — keep TST unchanged."
    )


def _empty_summary() -> dict:
    return {
        "num_samples": 0,
        "avg_accuracy": 0.0,
        "avg_tokens": 0.0,
        "avg_latency_ms": 0.0,
        "failure_patterns": [],
        "success_patterns": [],
        "cost_patterns": [],
        "recommendation": "No samples to aggregate.",
    }
