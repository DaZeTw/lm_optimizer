"""Natural language → LogicalNode DAG via LLM + algebraic expression parser.

Two planners live here:

SemanticParser   (query-level)
    Translates a single natural-language query into a LogicalNode DAG.
    Optionally constrained by a Task Strategy Template (TST) produced by
    TaskPlanner — when a TST is provided, the query planner fills its slots
    instead of designing a plan from scratch.

TaskPlanner      (task-level)
    Runs once per task family.  Calls the LLM with task_prompts, parses the
    three-section TST text via expr_parser.parse_task_strategy(), and returns
    a plain dict.  Supports cold-start generation and feedback-driven revision.
"""

from __future__ import annotations

import os

from openai import OpenAI

from catalog.catalog import SystemCatalog
from ir.feedback import Feedback
from ir.nodes import LogicalNode

from .expr_parser import ParseError, parse_expression, parse_task_strategy
from .prompts import SYSTEM_PROMPT, build_user_message
from .task_prompts import (
    TASK_SYSTEM_PROMPT,
    build_task_revise_message,
    build_task_user_message,
)

# ── OpenAI client wrapper ──────────────────────────────────────────


class LLMClient:
    """
    Thin wrapper around the OpenAI chat completions API.
    Reads OPENAI_API_KEY from the environment automatically.
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self._client = OpenAI(
            api_key=api_key or os.environ["OPENAI_API_KEY"],
            **({"base_url": base_url} if base_url else {}),
        )

    def complete(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
    ) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


# ── Semantic parser ────────────────────────────────────────────────


class SemanticParser:
    """
    Translates a natural-language query into a LogicalNode DAG.

    Workflow:
        1. Send the query to the LLM with a prompt asking for an
           algebraic plan expression (not JSON).
        2. Parse the expression with the deterministic expr_parser.
        3. If parsing fails, feed the error back to the LLM and retry.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_retries: int = 3,
        catalog: SystemCatalog | None = None,
    ):
        self.client = client or LLMClient()
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.catalog = catalog

    def parse(
        self,
        task_description: str,
        sample_queries: list[str],
        task_strategy: dict | None = None,
    ) -> LogicalNode:
        """
        Translate a task description and sample queries into a LogicalNode DAG.

        Args:
            task_description: High-level description of the overall task (e.g.
                              "QA over scientific papers").
            sample_queries:   Representative example questions the plan must handle.
            task_strategy:    Optional TST dict returned by TaskPlanner.generate()
                              or TaskPlanner.revise().  When provided, it is
                              rendered into the user message so the query planner
                              fills the skeleton's slots rather than designing
                              from scratch.

        Returns:
            The root LogicalNode of the unoptimized logical plan.

        Raises:
            ParseError: If the LLM fails to produce a valid plan after
                        max_retries attempts.
        """
        context_window: int | None = None
        avg_chunk_tokens: float | None = None

        if self.catalog is not None:
            model_stats = self.catalog.get_model(self.model)
            context_window = model_stats.context_window or None
            avg_chunk_tokens = self.catalog.avg_chunk_tokens()

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_message(
                    task_description,
                    sample_queries,
                    context_window=context_window,
                    avg_chunk_tokens=avg_chunk_tokens,
                    task_strategy=task_strategy,
                ),
            },
        ]

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            raw = self.client.complete(messages, self.model, self.temperature)

            try:
                node = parse_expression(raw)
                return node

            except ParseError as e:
                last_error = e
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"That plan has a syntax error: {e}\n\n"
                            "Please output ONLY the corrected algebraic expression. "
                            "No explanation, no markdown, just the expression."
                        ),
                    }
                )

        raise ParseError(
            f"SemanticParser failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )


# ── Task-level planner ────────────────────────────────────────────


class TaskParseError(Exception):
    pass


def _format_feedback_block(idx: int, fb: Feedback) -> str:
    """Format one Feedback object as a readable block for the revise prompt."""
    _MAX_CHARS = 400
    node_lines = "\n".join(
        f"    {item.op_id} ({item.variant}): "
        f"tokens={item.token_cost}, latency={item.latency_ms:.1f}ms, "
        f"output={item.output_summary[:_MAX_CHARS]!r}"
        + ("...[truncated]" if len(item.output_summary) > _MAX_CHARS else "")
        for item in fb.items
    )
    return (
        f"### Sample {idx + 1}\n"
        f"Accuracy : {fb.accuracy:.2f}\n"
        f"Result   : {fb.result[:500]}\n"
        f"Gold ans : {fb.gold_ans[:300]}\n"
        f"Per-node :\n{node_lines or '    (none)'}"
    )


class TaskPlanner:
    """
    Task-level planner: generates and revises Task Strategy Templates (TSTs).

    A TST is a plain dict with three keys — logical_skeleton, physical_policy,
    adaptation_policy — parsed from the LLM's three-section text output by
    expr_parser.parse_task_strategy().  No dataclasses are used.

    Workflow:
        1. generate() — cold start: send task description + criteria to the LLM,
           parse the three-section TST text, return the dict.
        2. revise()   — warm start: send the previous TST text + execution
           feedback to the LLM, parse the revised TST, return the new dict.

    The returned dict is then passed to SemanticParser.parse(task_strategy=...)
    so query-level planning fills slots instead of designing from scratch.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_retries: int = 3,
        catalog: SystemCatalog | None = None,
    ):
        self.client = client or LLMClient()
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.catalog = catalog

    def generate(
        self,
        task_description: str,
        evaluation_criteria: str,
        sample_queries: list[str] | None = None,
        prior_heuristics: list[str] | None = None,
    ) -> dict:
        """
        Cold-start: generate a TST from a task description.

        Args:
            task_description:    Free-text description of the task family.
            evaluation_criteria: How answers will be judged.
            sample_queries:      Representative example questions.
            prior_heuristics:    Known-good planning rules to keep.

        Returns:
            TST as a plain dict with keys logical_skeleton, physical_policy,
            adaptation_policy.

        Raises:
            TaskParseError: If the LLM fails to produce a valid TST after
                            max_retries attempts.
        """
        context_window, avg_chunk_tokens = self._catalog_stats()
        user_msg = build_task_user_message(
            task_description=task_description,
            evaluation_criteria=evaluation_criteria,
            sample_queries=sample_queries,
            prior_heuristics=prior_heuristics,
            context_window=context_window,
            avg_chunk_tokens=avg_chunk_tokens,
        )
        return self._call_and_parse(user_msg)

    def revise(
        self,
        task_description: str,
        evaluation_criteria: str,
        prev_tst: dict,
        feedbacks: list[Feedback],
        sample_queries: list[str] | None = None,
    ) -> dict:
        """
        Feedback-driven revision: refine an existing TST.

        Args:
            task_description:    Same description used in generate().
            evaluation_criteria: Same criteria used in generate().
            prev_tst:            The TST dict to revise (from generate() or a
                                 previous revise() call).
            feedbacks:           Feedback objects from query-level execution.
            sample_queries:      Queries associated with the feedback runs.

        Returns:
            Revised TST as a plain dict.

        Raises:
            TaskParseError: If the LLM fails to produce a valid TST after
                            max_retries attempts.
        """
        context_window, avg_chunk_tokens = self._catalog_stats()
        feedback_blocks = "\n\n".join(
            _format_feedback_block(i, fb) for i, fb in enumerate(feedbacks)
        )
        # Reconstruct the raw three-section text from the previous TST dict
        # so the revise prompt shows exactly what the model produced last time.
        prev_tst_text = _tst_dict_to_text(prev_tst)
        user_msg = build_task_revise_message(
            task_description=task_description,
            evaluation_criteria=evaluation_criteria,
            prev_tst_text=prev_tst_text,
            feedback_blocks=feedback_blocks,
            num_samples=len(feedbacks),
            sample_queries=sample_queries,
            context_window=context_window,
            avg_chunk_tokens=avg_chunk_tokens,
        )
        return self._call_and_parse(user_msg)

    # ── Internals ────────────────────────────────────────────────

    def _catalog_stats(self) -> tuple[int, float]:
        if self.catalog is None:
            return 128_000, 180.0
        return self.catalog.context_window(), self.catalog.avg_chunk_tokens()

    def _call_and_parse(self, user_msg: str) -> dict:
        messages: list[dict] = [
            {"role": "system", "content": TASK_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            raw = self.client.complete(messages, self.model, self.temperature)
            try:
                return parse_task_strategy(raw)
            except ParseError as e:
                last_error = e
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"That output has an error: {e}\n\n"
                            "Output ONLY the corrected Task Strategy Template "
                            "in the exact three-section format. "
                            "No markdown fences, no explanation."
                        ),
                    }
                )

        raise TaskParseError(
            f"TaskPlanner failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )


# ── TST round-trip helper ─────────────────────────────────────────


def _tst_dict_to_text(tst: dict) -> str:
    """Re-render a parsed TST dict back into the three-section text format.

    Used by TaskPlanner.revise() so the revision prompt shows the model's
    own prior output rather than a JSON blob it didn't produce.
    """
    skel = tst.get("logical_skeleton", {})
    phys = tst.get("physical_policy", {})
    adap = tst.get("adaptation_policy", {})

    lines: list[str] = ["LOGICAL SKELETON", skel.get("template", ""), ""]

    lines.append("PHYSICAL POLICY")
    for op_id, node in phys.items():
        raw_params = node.get("params", {})
        params_str = (
            ", ".join(f"{k}={v}" for k, v in raw_params.items()) if raw_params else "{}"
        )
        lines.append(f"{op_id} | {node['op_name']} | {node['variant']} | {params_str}")
    lines.append("")

    lines.append("ADAPTATION POLICY")
    for key in ("mutable_slots", "immutable_slots", "mutable_ops", "immutable_ops"):
        values = adap.get(key, [])
        lines.append(f"{key}: {', '.join(values)}")
    for rule in adap.get("allowed_rewrites", []):
        lines.append(f"allowed_rewrites: {rule}")
    for rule in adap.get("forbidden_rewrites", []):
        lines.append(f"forbidden_rewrites: {rule}")

    return "\n".join(lines)
