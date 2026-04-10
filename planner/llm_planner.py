"""LLM-driven physical planner with init and revise phases."""

from __future__ import annotations

import json
import re

from catalog.catalog import SystemCatalog
from ir.feedback import Feedback
from ir.nodes import LogicalNode, PhysicalNode
from planner.plan_parser import PlanParseError, parse_physical_plan
from planner.variant_candidates import CANDIDATE_VARIANTS

_SYSTEM_PROMPT = """\
You are a physical query planner for a long-context reasoning system.

Your job: given a logical query plan and corpus/model statistics, choose the best
physical variant and parameters for each operator node.

## Operator → available variants
{variant_catalog}

## Output format
Output ONLY a JSON object (no markdown, no explanation) representing the physical
plan tree. Each node must have:
  "op"      — the logical operator name (same as input)
  "variant" — one of the listed variants for that op
  "params"  — dict of operator parameters (preserve from logical plan; add extras if helpful)
  "inputs"  — list of child nodes (same structure, recursively)

Constraints:
- Every node in the logical plan must appear exactly once.
- Use only the listed variants — no others.
- Prefer cheaper variants unless the task clearly warrants a stronger one.
"""

_INIT_USER_TEMPLATE = """\
## Task
{query}

## Corpus / model context
- Model context window : {context_window:,} tokens
- Avg chunk size       : {avg_chunk_tokens:.0f} tokens

## Logical plan (JSON)
{logical_json}

Produce the physical plan JSON now.
"""

_REVISE_USER_TEMPLATE = """\
## Task
{query}

## Corpus / model context
- Model context window : {context_window:,} tokens
- Avg chunk size       : {avg_chunk_tokens:.0f} tokens

## Logical plan (JSON)
{logical_json}

## Previous physical plan
{prev_physical_json}

## Execution feedback across {num_samples} sample(s)

{feedback_blocks}

Revise the physical plan to improve accuracy and/or reduce cost across all samples. \
Output only the revised physical plan JSON.
"""


def _variant_catalog_str() -> str:
    lines = []
    for op, variants in CANDIDATE_VARIANTS.items():
        lines.append(f"  {op.value}: {', '.join(variants)}")
    return "\n".join(lines)


def _format_feedback_block(idx: int, fb: "Feedback") -> str:
    node_lines = "\n".join(
        f"    {item.op_id} ({item.variant}): "
        f"tokens={item.token_cost}, latency={item.latency_ms:.1f}ms, "
        f"output={item.output_summary!r}"
        for item in fb.items
    )
    return (
        f"### Sample {idx + 1}\n"
        f"Accuracy: {fb.accuracy:.2f}\n"
        f"Result  : {fb.result}\n"
        f"Gold ans: {fb.gold_ans}\n"
        f"Per-node:\n"
        f"{node_lines or '    (none)'}"
    )


def _physical_to_dict(node: PhysicalNode) -> dict:
    return {
        "op": node.logical_ref.op.value,
        "variant": node.variant,
        "params": dict(node.params),
        "inputs": [_physical_to_dict(c) for c in node.inputs],
    }


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


class LLMPhysicalPlanner:
    """
    Physical planner backed by an LLM.

    Two phases
    ----------
    init   — cold start: maps the optimised logical plan to physical variants.
    revise — warm start: refines a previous plan using execution feedback.
    """

    def __init__(
        self,
        client,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_retries: int = 2,
        catalog: SystemCatalog | None = None,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.catalog = catalog

    # ── public API ────────────────────────────────────────────────

    def init(self, query: str, logical: LogicalNode) -> PhysicalNode:
        """Produce an initial physical plan from the logical plan."""
        context_window, avg_chunk_tokens = self._catalog_stats()
        user_msg = _INIT_USER_TEMPLATE.format(
            query=query,
            context_window=context_window,
            avg_chunk_tokens=avg_chunk_tokens,
            logical_json=json.dumps(logical.to_dict(), indent=2),
        )
        return self._call_and_parse(user_msg)

    def revise(
        self,
        query: str,
        logical: LogicalNode,
        prev_physical: PhysicalNode,
        feedbacks: list[Feedback],
    ) -> PhysicalNode:
        """Refine a physical plan using execution feedback from one or more samples."""
        context_window, avg_chunk_tokens = self._catalog_stats()
        feedback_blocks = "\n\n".join(
            _format_feedback_block(i, fb) for i, fb in enumerate(feedbacks)
        )
        user_msg = _REVISE_USER_TEMPLATE.format(
            query=query,
            context_window=context_window,
            avg_chunk_tokens=avg_chunk_tokens,
            logical_json=json.dumps(logical.to_dict(), indent=2),
            prev_physical_json=json.dumps(_physical_to_dict(prev_physical), indent=2),
            num_samples=len(feedbacks),
            feedback_blocks=feedback_blocks,
        )
        return self._call_and_parse(user_msg)

    # ── internals ─────────────────────────────────────────────────

    def _catalog_stats(self) -> tuple[int, float]:
        if self.catalog is None:
            return 128_000, 180.0
        return self.catalog.context_window(), self.catalog.avg_chunk_tokens()

    def _system_prompt(self) -> str:
        return _SYSTEM_PROMPT.format(variant_catalog=_variant_catalog_str())

    def _call_and_parse(self, user_msg: str) -> PhysicalNode:
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": user_msg},
        ]
        last_err: Exception | None = None
        for _ in range(self.max_retries):
            raw = self.client.complete(messages, self.model, self.temperature)
            cleaned = _strip_code_fence(raw)
            try:
                data = json.loads(cleaned)
                return parse_physical_plan(data)
            except (json.JSONDecodeError, PlanParseError) as exc:
                last_err = exc
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        f"That output has an error: {exc}\n"
                        "Output ONLY valid JSON for the physical plan, nothing else."
                    ),
                })
        raise PlanParseError(
            f"LLMPhysicalPlanner failed after {self.max_retries} attempts. "
            f"Last error: {last_err}"
        )
