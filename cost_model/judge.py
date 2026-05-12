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

_EVIDENCE_SYSTEM_PROMPT = """\
You are an impartial evidence coverage evaluator.

Given a list of gold grounding evidence passages and a retrieved text,
rate how well the retrieved text covers the gold evidence.

Respond with ONLY a single integer between 1 and 10:

10 — all gold evidence passages are fully covered  
7–9 — most evidence is covered with minor gaps  
4–6 — partial coverage  
2–3 — very little coverage  
1 — none of the gold evidence is present  

Output only the number. No explanation.
"""

_EVIDENCE_USER_TEMPLATE = """\
Gold evidence passages:
{gold_evidence}

Retrieved text:
{result}

Coverage score (1–10):"""


class AccuracyJudge:
    def __init__(self, client, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.client = client
        self.model = model
        self.temperature = temperature

    async def score(self, result: str, gold_ans: str) -> float:
        """Return a 0.0–1.0 accuracy score comparing result to gold_ans."""
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

    async def score_evidence(self, result: str, gold_evidence: list[str]) -> float:
        """Return a 0.0–1.0 coverage score comparing result to grounding evidence."""
        if not gold_evidence:
            return 0.0
        formatted = "\n".join(f"  - {e}" for e in gold_evidence)
        messages = [
            {"role": "system", "content": _EVIDENCE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _EVIDENCE_USER_TEMPLATE.format(
                    gold_evidence=formatted, result=result
                ),
            },
        ]
        raw = self.client.complete(messages, self.model, self.temperature)
        return _parse_score(raw)


def _parse_score(text: str) -> float:
    try:
        value = float(text.strip().split()[0])
        value = max(1.0, min(10.0, value))
        return (value - 1) / 9  # normalize to 0–1
    except (ValueError, IndexError):
        return 0.0
