"""LLM-backed TST compiler."""

from __future__ import annotations

import json
import re

from .models import CompilerInput, CompilerParseError, CompilerResult
from .prompts import COMPILER_SYSTEM_PROMPT, build_compiler_user_message


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def parse_compiler_result(raw: str) -> CompilerResult:
    cleaned = _strip_code_fence(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise CompilerParseError(f"Invalid compiler JSON: {exc}") from exc
    return CompilerResult.from_dict(data)


class TSTCompiler:
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

    def compile(self, compiler_input: CompilerInput) -> CompilerResult:
        messages: list[dict] = [
            {"role": "system", "content": COMPILER_SYSTEM_PROMPT},
            {"role": "user", "content": build_compiler_user_message(compiler_input)},
        ]
        last_error: Exception | None = None

        for _ in range(self.max_retries):
            raw = self.client.complete(messages, self.model, self.temperature)
            try:
                return parse_compiler_result(raw)
            except CompilerParseError as exc:
                last_error = exc
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"That compiler output is invalid: {exc}\n\n"
                            "Output ONLY strict JSON with keys executable and errors."
                        ),
                    }
                )

        raise CompilerParseError(
            f"TSTCompiler failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )
