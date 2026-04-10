"""LLM-as-judge accuracy scorer."""

from __future__ import annotations

_SYSTEM_PROMPT = """\
You are an impartial accuracy evaluator for a question-answering system.

Given a predicted answer and a gold (reference) answer, rate how accurately the
predicted answer addresses the question compared to the gold answer.

Respond with ONLY a single float between 0.0 and 1.0:
  1.0 — the predicted answer is fully correct and complete
  0.5 — partially correct or incomplete
  0.0 — wrong, irrelevant, or empty

Output only the number. No explanation.
"""

_USER_TEMPLATE = """\
Gold answer : {gold_ans}
Predicted   : {result}

Accuracy score (0.0–1.0):"""


class AccuracyJudge:
    """
    Calls an LLM to score how well ``result`` matches ``gold_ans``.

    Parameters
    ----------
    client:
        Any object with a ``complete(messages, model, temperature) -> str``
        interface (e.g. ``LLMClient`` from ``parser.semantic_parser``).
    model:
        Model name to use for judging.
    temperature:
        Keep at 0.0 for deterministic scoring.
    """

    def __init__(self, client, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.client = client
        self.model = model
        self.temperature = temperature

    async def score(self, result: str, gold_ans: str) -> float:
        """Return a 0.0–1.0 accuracy score comparing *result* to *gold_ans*."""
        if not gold_ans.strip():
            return 0.0

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(gold_ans=gold_ans, result=result),
            },
        ]
        raw = self.client.complete(messages, self.model, self.temperature)
        return _parse_score(raw)


def _parse_score(text: str) -> float:
    try:
        value = float(text.strip().split()[0])
        return max(0.0, min(1.0, value))
    except (ValueError, IndexError):
        return 0.0
