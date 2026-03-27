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

    def parse(self, query: str) -> LogicalNode:
        """
        Translate a natural-language query into a LogicalNode DAG.

        Args:
            query: The user's natural-language question or task.

        Returns:
            The root LogicalNode of the unoptimized logical plan.

        Raises:
            ParseError: If the LLM fails to produce a valid plan after
                        max_retries attempts.
        """
        theme_labels = self._theme_labels()
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_message(query, theme_labels=theme_labels),
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

    def _theme_labels(self) -> list[str] | None:
        if self.catalog is None:
            return None
        labels = sorted(self.catalog.semantic_stats.theme_clusters.keys())
        return labels[:12] if labels else None
