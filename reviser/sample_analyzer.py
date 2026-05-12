"""Step 1: per-sample execution analysis.

SampleAnalyzer calls the LLM once per execution record and parses the
four-section report into a plain SampleFeedback dict.  No dataclasses.

SampleFeedback schema
---------------------
{
    "query":          str,
    "accuracy":       float,
    "total_tokens":   int,
    "total_latency_ms": float,
    "query_features": {           # from QUERY FEATURES section
        "query_type":         str,
        "complexity":         str,
        "evidence_scope":     str,
        "comparison_required": str,
    },
    "failure_points": [           # from FAILURE POINTS section
        {"op_id": str, "issue_type": str, "description": str},
        ...
    ],
    "successful_adaptations": [   # from SUCCESSFUL ADAPTATIONS section
        {"op_id": str, "what_worked": str},
        ...
    ],
    "suggested_fixes": [          # from SUGGESTED FIXES section
        {"op_id": str, "fix_type": str, "detail": str},
        ...
    ],
}
"""

from __future__ import annotations

import re

from executor.runner import ExecutionResult
from ir.feedback import Feedback
from ir.nodes import PhysicalNode, LogicalNode
from parser.semantic_parser import _tst_dict_to_text

from .prompts import (
    SAMPLE_ANALYSIS_SYSTEM_PROMPT,
    build_sample_analysis_user_message,
)

# ── Text renderers (execution record → prompt-ready strings) ──────


def _render_logical_plan(logical: LogicalNode) -> str:
    """Render a LogicalNode as a one-line algebraic summary per node."""

    def _walk(node: LogicalNode, depth: int = 0) -> list[str]:
        indent = "  " * depth
        params_str = ", ".join(f"{k}={v!r}" for k, v in node.params.items())
        line = f"{indent}{node.op.value}({params_str})"
        lines = [line]
        for child in node.inputs:
            lines.extend(_walk(child, depth + 1))
        return lines

    return "\n".join(_walk(logical))


def _render_physical_plan(physical: PhysicalNode) -> str:
    """Render a PhysicalNode tree as op_id | variant | params lines."""
    lines: list[str] = []
    counter: dict[str, int] = {}

    def _walk(node: PhysicalNode) -> None:
        op_name = node.logical_ref.op.value
        idx = counter.get(op_name, 0)
        counter[op_name] = idx + 1
        op_id = f"{op_name}_{idx}"
        params_str = (
            ", ".join(f"{k}={v}" for k, v in node.params.items())
            if node.params
            else "—"
        )
        lines.append(f"  {op_id:<12} variant={node.variant!r}  params={{{params_str}}}")
        for child in node.inputs:
            _walk(child)

    _walk(physical)
    return "\n".join(lines)


def _render_node_trace(feedback: Feedback) -> str:
    """Render per-node execution trace from a Feedback object."""
    lines: list[str] = []
    for item in feedback.items:
        summary = item.output_summary[:300]
        if len(item.output_summary) > 300:
            summary += "...[truncated]"
        lines.append(
            f"  {item.op_id:<12} ({item.variant}): "
            f"tokens={item.token_cost}, latency={item.latency_ms:.1f}ms\n"
            f"    output: {summary!r}"
        )
    return "\n".join(lines) if lines else "  (none)"


# ── Parser for the four-section LLM output ────────────────────────

_SECTION_HEADERS = (
    "QUERY FEATURES",
    "FAILURE POINTS",
    "SUCCESSFUL ADAPTATIONS",
    "SUGGESTED FIXES",
)


def _split_sections(text: str) -> dict[str, str]:
    """Split the LLM output into named section bodies."""
    positions: dict[str, int] = {}
    for header in _SECTION_HEADERS:
        idx = text.find(header)
        if idx == -1:
            raise ValueError(f"Missing section: {header!r}")
        positions[header] = idx

    ordered = sorted(positions.items(), key=lambda kv: kv[1])
    bodies: dict[str, str] = {}
    for i, (header, start) in enumerate(ordered):
        body_start = start + len(header)
        body_end = ordered[i + 1][1] if i + 1 < len(ordered) else len(text)
        bodies[header] = text[body_start:body_end].strip()
    return bodies


def _parse_sample_feedback(
    raw: str,
    query: str,
    accuracy: float,
    total_tokens: int,
    total_latency_ms: float,
) -> dict:
    """Parse the LLM's four-section report into a SampleFeedback dict."""
    # Strip markdown fences
    raw = re.sub(r"```[a-z]*", "", raw, flags=re.IGNORECASE).strip().strip("`").strip()

    bodies = _split_sections(raw)

    # QUERY FEATURES — key: value lines
    query_features: dict[str, str] = {}
    for line in bodies["QUERY FEATURES"].splitlines():
        line = line.strip()
        if ":" in line:
            k, _, v = line.partition(":")
            query_features[k.strip().lower().replace(" ", "_")] = v.strip()

    # FAILURE POINTS — op_id | issue_type | description
    failure_points: list[dict] = []
    for line in bodies["FAILURE POINTS"].splitlines():
        line = line.strip()
        if not line or line == "(none)":
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3:
            failure_points.append(
                {
                    "op_id": parts[0],
                    "issue_type": parts[1],
                    "description": parts[2],
                }
            )

    # SUCCESSFUL ADAPTATIONS — op_id | what_worked
    successful_adaptations: list[dict] = []
    for line in bodies["SUCCESSFUL ADAPTATIONS"].splitlines():
        line = line.strip()
        if not line or line == "(none)":
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 2:
            successful_adaptations.append(
                {
                    "op_id": parts[0],
                    "what_worked": parts[1],
                }
            )

    # SUGGESTED FIXES — op_id | fix_type | detail
    suggested_fixes: list[dict] = []
    for line in bodies["SUGGESTED FIXES"].splitlines():
        line = line.strip()
        if not line or line == "(none)":
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3:
            suggested_fixes.append(
                {
                    "op_id": parts[0],
                    "fix_type": parts[1],
                    "detail": parts[2],
                }
            )

    return {
        "query": query,
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "total_latency_ms": total_latency_ms,
        "query_features": query_features,
        "failure_points": failure_points,
        "successful_adaptations": successful_adaptations,
        "suggested_fixes": suggested_fixes,
    }


# ── SampleAnalyzer ────────────────────────────────────────────────


class SampleAnalyzer:
    """Step 1: analyse one execution record via LLM, return a SampleFeedback dict.

    Args:
        client      LLMClient (or any object with .complete(messages, model, temp))
        model       LLM model identifier.
        temperature Sampling temperature.
        max_retries Self-correction retries on parse failure.
    """

    def __init__(
        self,
        client,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_retries: int = 2,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

    def analyze(
        self,
        query: str,
        logical: LogicalNode,
        physical: PhysicalNode,
        execution: ExecutionResult,
        feedback: Feedback,
        tst: dict,
    ) -> dict:
        """Analyse one execution record and return a SampleFeedback dict.

        Args:
            query:     The natural-language query for this sample.
            logical:   Logical plan used (post-optimization).
            physical:  Physical plan used for this sample.
            execution: ExecutionResult from PlanRunner.run().
            feedback:  Feedback (accuracy + per-node items) from the judge.
            tst:       Current Task Strategy Template dict.

        Returns:
            SampleFeedback plain dict.
        """
        total_tokens = sum(
            feedback.items and [item.token_cost for item in feedback.items] or [0]
        )
        total_latency_ms = (
            sum(item.latency_ms for item in feedback.items) if feedback.items else 0.0
        )

        user_msg = build_sample_analysis_user_message(
            query=query,
            logical_plan_text=_render_logical_plan(logical),
            physical_plan_text=_render_physical_plan(physical),
            node_trace=_render_node_trace(feedback),
            accuracy=feedback.accuracy,
            total_tokens=total_tokens,
            total_latency_ms=total_latency_ms,
            errors=execution.errors,
            tst_text=_tst_dict_to_text(tst),
        )

        messages: list[dict] = [
            {"role": "system", "content": SAMPLE_ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        last_err: Exception | None = None

        for _ in range(self.max_retries):
            raw = self.client.complete(messages, self.model, self.temperature)
            try:
                return _parse_sample_feedback(
                    raw,
                    query=query,
                    accuracy=feedback.accuracy,
                    total_tokens=total_tokens,
                    total_latency_ms=total_latency_ms,
                )
            except ValueError as exc:
                last_err = exc
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"That output has an error: {exc}\n\n"
                            "Output ONLY the corrected four-section report. "
                            "No markdown, no explanation."
                        ),
                    }
                )

        # Fallback: return minimal feedback without LLM analysis
        return {
            "query": query,
            "accuracy": feedback.accuracy,
            "total_tokens": total_tokens,
            "total_latency_ms": total_latency_ms,
            "query_features": {},
            "failure_points": [],
            "successful_adaptations": [],
            "suggested_fixes": [],
            "_parse_error": str(last_err),
        }
