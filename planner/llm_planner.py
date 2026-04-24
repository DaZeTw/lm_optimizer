"""LLM-driven physical planner.

Single entry point: ``LLMPhysicalPlanner.plan()``.

Takes the logical plan plus the three TST components that are relevant to
physical planning — corpus stats, physical policy, and adaptation policy —
and returns a PhysicalNode DAG.  No revise, no bind.
"""

from __future__ import annotations

import json
import re

from catalog.catalog import SystemCatalog
from ir.nodes import LogicalNode, PhysicalNode
from planner.plan_parser import PlanParseError, parse_physical_plan
from planner.variant_candidates import CANDIDATE_VARIANTS

# ── Prompt ────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a physical query planner for a long-context reasoning system.

Your job: given a logical query plan, corpus/model statistics, a physical
policy, and adaptation rules, assign the best physical variant and parameters
to every operator node.

## Operator → available variants
{variant_catalog}

## Output format
Output ONLY a JSON object (no markdown, no explanation) representing the
physical plan tree. Each node must have:
  "op"      — the logical operator name (same as input)
  "variant" — one of the listed variants for that op
  "params"  — dict of operator parameters (preserve from logical plan; add extras if needed)
  "inputs"  — list of child nodes (same structure, recursively)

Constraints:
- Every node in the logical plan must appear exactly once.
- Use only the listed variants — no others.
- Locked nodes must be copied verbatim (variant + params).
- Tunable nodes may deviate only within the stated param ranges.
- Apply only the listed allowed rewrites; never apply a forbidden one.
"""

_USER_TEMPLATE = """\
## Task
{query}

## Corpus / model context
- Model context window : {context_window:,} tokens
- Avg chunk size       : {avg_chunk_tokens:.0f} tokens

## Logical plan (JSON)
{logical_json}

## Physical policy
{physical_block}

## Adaptation rules
{adaptation_block}

Produce the physical plan JSON now.
"""

# ── Helpers ───────────────────────────────────────────────────────


def _variant_catalog_str() -> str:
    lines = []
    for op, variants in CANDIDATE_VARIANTS.items():
        lines.append(f"  {op.value}: {', '.join(variants)}")
    return "\n".join(lines)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _render_physical_block(
    physical_policy: dict,
    adaptation_policy: dict,
) -> str:
    """Render the physical policy section of the user message.

    Splits nodes into LOCKED (immutable_ops + unlisted) and TUNABLE
    (mutable_ops), showing variant, params, and tunable ranges per node.
    """
    immutable_ids: set[str] = set(adaptation_policy.get("immutable_ops", []))
    mutable_ids: set[str] = set(adaptation_policy.get("mutable_ops", []))

    locked_lines: list[str] = []
    tunable_lines: list[str] = []

    for op_id, node in physical_policy.items():
        variant = node.get("variant", "")
        params = node.get("params", {})
        ranges = node.get("param_ranges", {})
        op_name = node.get("op_name", "")
        params_str = ", ".join(f"{k}={v}" for k, v in params.items()) if params else "—"
        base = (
            f"  {op_id:<12} {op_name:<12} variant={variant!r}  params={{{params_str}}}"
        )

        if op_id in mutable_ids:
            ranges_str = (
                "  tunable: " + ", ".join(f"{k}={v}" for k, v in ranges.items())
                if ranges
                else ""
            )
            tunable_lines.append(base + ranges_str)
        else:
            # immutable_ids or not listed — lock by default
            locked_lines.append(base)

    lines: list[str] = []
    if locked_lines:
        lines.append("Locked nodes — copy variant and params verbatim:")
        lines.extend(locked_lines)
    if tunable_lines:
        lines.append("Tunable nodes — may adjust variant/params within stated ranges:")
        lines.extend(tunable_lines)

    return "\n".join(lines) if lines else "(no physical policy provided)"


def _render_adaptation_block(adaptation_policy: dict) -> str:
    """Render the adaptation rules section of the user message."""
    lines: list[str] = []
    for rule in adaptation_policy.get("allowed_rewrites", []):
        lines.append(f"  allowed  : {rule}")
    for rule in adaptation_policy.get("forbidden_rewrites", []):
        lines.append(f"  forbidden: {rule}")
    return "\n".join(lines) if lines else "(no adaptation rules)"


# ── Planner ───────────────────────────────────────────────────────


class LLMPhysicalPlanner:
    """Physical planner backed by an LLM.

    Single method: ``plan()``.  Takes the logical plan and the three TST
    components relevant to physical planning and returns a PhysicalNode DAG.
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

    def plan(
        self,
        query: str,
        logical: LogicalNode,
        physical_policy: dict,
        corpus_stats: dict,
        adaptation_policy: dict,
    ) -> PhysicalNode:
        """Produce a physical plan from the logical plan and TST components.

        Args:
            query:             The natural-language query being planned.
            logical:           Root of the logical plan DAG.
            physical_policy:   ``tst["physical_policy"]`` — per-node variant
                               and param decisions from task-level planning.
            corpus_stats:      Dict with keys ``context_window`` (int) and
                               ``avg_chunk_tokens`` (float).  Falls back to
                               catalog or hardcoded defaults for missing keys.
            adaptation_policy: ``tst["adaptation_policy"]`` — mutable_ops,
                               immutable_ops, allowed/forbidden rewrites.
        """
        context_window, avg_chunk_tokens = self._resolve_stats(corpus_stats)
        user_msg = _USER_TEMPLATE.format(
            query=query,
            context_window=context_window,
            avg_chunk_tokens=avg_chunk_tokens,
            logical_json=json.dumps(logical.to_dict(), indent=2),
            physical_block=_render_physical_block(physical_policy, adaptation_policy),
            adaptation_block=_render_adaptation_block(adaptation_policy),
        )
        return self._call_and_parse(user_msg)

    # ── internals ─────────────────────────────────────────────────

    def _resolve_stats(self, corpus_stats: dict) -> tuple[int, float]:
        """Resolve corpus stats, falling back to catalog then hardcoded defaults."""
        if "context_window" in corpus_stats and "avg_chunk_tokens" in corpus_stats:
            return corpus_stats["context_window"], corpus_stats["avg_chunk_tokens"]
        if self.catalog is not None:
            return self.catalog.context_window(), self.catalog.avg_chunk_tokens()
        return (
            corpus_stats.get("context_window", 128_000),
            corpus_stats.get("avg_chunk_tokens", 180.0),
        )

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
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"That output has an error: {exc}\n"
                            "Output ONLY valid JSON for the physical plan, nothing else."
                        ),
                    }
                )
        raise PlanParseError(
            f"LLMPhysicalPlanner failed after {self.max_retries} attempts. "
            f"Last error: {last_err}"
        )
