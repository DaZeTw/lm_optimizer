"""Step 1: per-sample execution analysis.

SampleAnalyzer calls the LLM once per execution record and parses the
JSON report into a plain SampleFeedback dict.  No dataclasses.

SampleFeedback schema
---------------------
{
    "query":          str,
    "accuracy":       float,
    "total_tokens":   int,
    "total_latency_ms": float,
    "plan_feedback": {
        "supports_task":       bool,
        "main_structural_gap": str,
        "reason":              str,
    },
    "physical_feedback": [
        {
            "op_id": str,
            "variant": str,
            "issue_type": str,
            "description": str,
            "suggested_change": str,
        },
        ...
    ],
    "successful_adaptations": [
        {"op_id": str, "what_worked": str},
        ...
    ],
}
"""

from __future__ import annotations

import json
import re

from executor.runner import ExecutionResult
from ir.feedback import Feedback
from ir.nodes import PhysicalNode, LogicalNode
from parser.semantic_parser import _tst_dict_to_text

from .prompts import (
    SAMPLE_ANALYSIS_SYSTEM_PROMPT,
    build_sample_analysis_user_message,
)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)

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


# ── Parser for the JSON LLM output ────────────────────────────────


def _parse_sample_feedback(
    raw: str,
    query: str,
    accuracy: float,
    total_tokens: int,
    total_latency_ms: float,
) -> dict:
    """Parse the LLM's JSON report into a SampleFeedback dict."""
    # Strip markdown fences
    raw = re.sub(r"```[a-z]*", "", raw, flags=re.IGNORECASE).strip().strip("`").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Sample analysis must be a JSON object")

    sample = data.get("sample_analysis", data)
    if not isinstance(sample, dict):
        raise ValueError("'sample_analysis' must be a JSON object")

    plan_feedback = sample.get("plan_feedback")
    if not isinstance(plan_feedback, dict):
        raise ValueError("Missing object field: plan_feedback")

    normalized_plan_feedback = {
        "supports_task": _as_bool(plan_feedback.get("supports_task", False)),
        "main_structural_gap": str(
            plan_feedback.get("main_structural_gap", "unspecified")
        ),
        "reason": str(plan_feedback.get("reason", "")),
    }

    physical_feedback = sample.get("physical_feedback", [])
    if not isinstance(physical_feedback, list):
        raise ValueError("'physical_feedback' must be a list")

    normalized_physical_feedback: list[dict] = []
    for item in physical_feedback:
        if not isinstance(item, dict):
            continue
        normalized_physical_feedback.append(
            {
                "op_id": str(item.get("op_id", "")),
                "variant": str(item.get("variant", "")),
                "issue_type": str(item.get("issue_type", "")),
                "description": str(item.get("description", "")),
                "suggested_change": str(item.get("suggested_change", "")),
            }
        )

    successful_adaptations = sample.get("successful_adaptations", [])
    if not isinstance(successful_adaptations, list):
        raise ValueError("'successful_adaptations' must be a list")

    return {
        "query": str(sample.get("query", query)),
        "accuracy": float(sample.get("accuracy", accuracy)),
        "total_tokens": int(sample.get("total_tokens", total_tokens)),
        "total_latency_ms": float(sample.get("total_latency_ms", total_latency_ms)),
        "plan_feedback": normalized_plan_feedback,
        "physical_feedback": normalized_physical_feedback,
        "successful_adaptations": [
            item for item in successful_adaptations if isinstance(item, dict)
        ],
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
                            "Output ONLY the corrected JSON report. "
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
            "plan_feedback": {
                "supports_task": False,
                "main_structural_gap": "analysis_parse_error",
                "reason": str(last_err),
            },
            "physical_feedback": [],
            "successful_adaptations": [],
            "_parse_error": str(last_err),
        }
