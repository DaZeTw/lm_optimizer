"""Steps 4-6: revise the Task Strategy Template from an aggregated PatternSummary.

TSTRevisor takes:
  - the previous TST dict
  - a PatternSummary dict from aggregator.aggregate_feedback()

and calls the LLM once to produce a revised TST, parsed by
expr_parser.parse_task_strategy() into the same plain dict format.

Conservatism is enforced at the prompt level (TST_REVISION_SYSTEM_PROMPT)
and structurally: we only call revise() when there is at least one repeated
failure pattern (frequency >= min_frequency), otherwise we return the
previous TST unchanged.
"""

from __future__ import annotations

import re
from parser.expr_parser import ParseError, parse_task_strategy
from parser.semantic_parser import _tst_dict_to_text

from .aggregator import (
    render_cost_patterns,
    render_failure_patterns,
    render_success_patterns,
)
from .prompts import TST_REVISION_SYSTEM_PROMPT, build_tst_revision_user_message


class TSTRevisor:
    """Steps 4-6: produce a revised TST from a PatternSummary.

    Args:
        client      LLMClient (any object with .complete(messages, model, temp)).
        model       LLM model identifier.
        temperature Sampling temperature.
        max_retries Self-correction retries on parse failure.
    """

    def __init__(
        self,
        client,
        model: str = "gpt-5.4",
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

    def revise(
        self,
        prev_tst: dict,
        pattern_summary: dict,
        task_description: str,
        evaluation_metrics: str,
    ) -> dict:
        """Produce a revised TST, or return the previous one unchanged if no
        repeated patterns warrant a revision.

        Args:
            prev_tst: Current TST dict from TaskPlanner.generate() or previous revision.
            pattern_summary: PatternSummary dict from aggregator.aggregate_feedback().
            task_description: Original task description used to create the TST.
            evaluation_metrics: Evaluation criteria or metrics for judging task success.

        Returns:
            Revised TST dict, same plain-dict schema as the input.
        """
        # Step 5 conservatism gate: skip revision if nothing repeated enough
        if not pattern_summary.get("failure_patterns") and not pattern_summary.get(
            "cost_patterns"
        ):
            return prev_tst

        user_msg = build_tst_revision_user_message(
            task_description=task_description,
            evaluation_metrics=evaluation_metrics,
            tst_text=_tst_dict_to_text(prev_tst),
            num_samples=pattern_summary.get("num_samples", 0),
            failure_patterns=render_failure_patterns(pattern_summary),
            success_patterns=render_success_patterns(pattern_summary),
            cost_patterns=render_cost_patterns(pattern_summary),
            recommendation=pattern_summary.get("recommendation", ""),
        )

        messages: list[dict] = [
            {"role": "system", "content": TST_REVISION_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        for _ in range(self.max_retries):
            raw = self.client.complete(messages, self.model, self.temperature)

            cleaned = (
                re.sub(r"```[a-z]*", "", raw, flags=re.IGNORECASE)
                .strip()
                .strip("`")
                .strip()
            )

            try:
                return parse_task_strategy(cleaned)
            except ParseError as exc:
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"That output has an error: {exc}\n\n"
                            "Output ONLY the corrected TST in the exact three-section format. "
                            "No markdown, no explanation."
                        ),
                    }
                )

        return prev_tst
