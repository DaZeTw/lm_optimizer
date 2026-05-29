"""LLM-driven physical planner.

Single entry point: ``LLMPhysicalPlanner.plan()``.

Takes the logical plan plus corpus stats and returns a PhysicalNode DAG.
No revise, no bind, and no TST policy input.
"""

from __future__ import annotations

import json
import re

from catalog.catalog import SystemCatalog
from ir.nodes import LogicalNode, PhysicalNode
from planner.plan_parser import PlanParseError, parse_physical_plan
from planner.variant_candidates import CANDIDATE_VARIANTS
from planner.variant_schemas import (
    render_variant_param_schemas,
    validate_physical_plan_params,
)

# ── Prompt ────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a physical query planner for a long-context reasoning system.

Your job: given a logical query plan and corpus/model statistics, assign the
best physical variant and parameters to every operator node.

## Operator → available variants and parameter schemas
{variant_catalog}

## Output format
Output ONLY a JSON object (no markdown, no explanation) representing the
physical plan tree. Each node must have:
  "op"      — the logical operator name (same as input)
  "variant" — one of the listed variants for that op
  "params"  — dict of physical operator-control parameters
  "inputs"  — list of child nodes (same structure, recursively)

Physical params must be self-contained for execution:
- Copy any logical intent needed by the executor into params explicitly.
- Do not rely on logical_ref params being merged during execution.
- Do not include runtime resources in params.
- Never include corpus, llm, catalog, index, client, model_client, or embedder.

## Retrieval query rule

For each I node, rewrite the query as a precise evidence query
that matches how the retrieval model scores text.

Focus on:
- WHAT exact evidence to retrieve (entities, lists, metrics, methods)
- how it appears in documents (natural phrasing, not instructions)

Guidelines:
- use natural text (like a sentence or phrase from a paper)
- include key entities, terms, or values from the candidate answer
- keep it short and specific
- add 1–2 synonyms only if needed for recall

Model-aware hints:
- BM25 → include exact keywords and surface forms
- Dense → use natural semantic phrasing
- CrossEncoder → ensure query clearly matches the target evidence

Do NOT:
- copy the full question
- use instruction-style text (e.g., "find", "retrieve", "answer")
- add unrelated terms
- over-expand into long keyword lists

Constraints:
- Every node in the logical plan must appear exactly once.
- Use only the listed variants — no others.
- Preserve parameters from the logical plan when they express task intent.
- Add physical parameters when they improve evidence quality, latency, or cost.
- Do not change the logical operator structure.
"""

_USER_TEMPLATE = """\
## Task
{task_description}

## Evaluation criteria
{evaluation_criteria}

## Query
{query}

## Corpus / model context
- Model context window : {context_window:,} tokens
- Avg chunk size       : {avg_chunk_tokens:.0f} tokens

## Logical plan (JSON)
{logical_json}

Produce the physical plan JSON now.
"""

# ── Helpers ───────────────────────────────────────────────────────


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


# ── Planner ───────────────────────────────────────────────────────


class LLMPhysicalPlanner:
    """Physical planner backed by an LLM.

    Single method: ``plan()``.  Takes the logical plan and corpus stats, then
    returns a PhysicalNode DAG.
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
        task_description: str,
        evaluation_criteria: str,
        query: str,
        logical: LogicalNode,
        corpus_stats: dict,
    ) -> PhysicalNode:
        """Produce a physical plan from the logical plan and corpus stats.

        Args:
            task_description:    Task family description.
            evaluation_criteria: How final outputs are judged.
            query:               The natural-language query being planned.
            logical:             Root of the logical plan DAG.
            corpus_stats:        Dict with keys ``context_window`` (int) and
                                 ``avg_chunk_tokens`` (float).  Falls back to
                                 catalog or hardcoded defaults for missing keys.
        """
        context_window, avg_chunk_tokens = self._resolve_stats(corpus_stats)
        user_msg = _USER_TEMPLATE.format(
            task_description=task_description,
            evaluation_criteria=evaluation_criteria,
            query=query,
            context_window=context_window,
            avg_chunk_tokens=avg_chunk_tokens,
            logical_json=json.dumps(logical.to_dict(), indent=2),
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
        return _SYSTEM_PROMPT.format(
            variant_catalog=render_variant_param_schemas(CANDIDATE_VARIANTS)
        )

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
                plan = parse_physical_plan(data)
                validate_physical_plan_params(plan)
                return plan
            except (json.JSONDecodeError, PlanParseError, ValueError) as exc:
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
