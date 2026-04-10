"""Natural language → LogicalNode DAG via LLM + algebraic expression parser."""

from __future__ import annotations

import os

from openai import OpenAI

from catalog.catalog import SystemCatalog
from ir.nodes import LogicalNode

from .expr_parser import ParseError, parse_expression
from .prompts import SYSTEM_PROMPT, build_user_message

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

    def parse(self, task_description: str, sample_queries: list[str]) -> LogicalNode:
        """
        Translate a task description and sample queries into a LogicalNode DAG.

        Args:
            task_description: High-level description of the overall task (e.g.
                              "QA over scientific papers").
            sample_queries:   Representative example questions the plan must handle.

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
                # Return the bad output + error to the model so it can self-correct
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

