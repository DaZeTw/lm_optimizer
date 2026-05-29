"""Dataclasses for TST compiler input and output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class CompilerParseError(Exception):
    pass


@dataclass
class CompilerInput:
    task_description: str
    evaluation_criteria: str
    tst: dict
    operator_descriptions: dict


@dataclass
class CompilerIssue:
    section: str
    component: str | None
    message: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompilerIssue:
        if not isinstance(data, dict):
            raise CompilerParseError("Compiler issue must be an object")
        section = data.get("section")
        message = data.get("message")
        if not isinstance(section, str) or not section:
            raise CompilerParseError("Compiler issue missing non-empty 'section'")
        if not isinstance(message, str) or not message:
            raise CompilerParseError("Compiler issue missing non-empty 'message'")
        component = data.get("component")
        if component is not None and not isinstance(component, str):
            raise CompilerParseError("Compiler issue 'component' must be string or null")
        return cls(section=section, component=component, message=message)


@dataclass
class CompilerResult:
    executable: bool
    errors: list[CompilerIssue]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompilerResult:
        if not isinstance(data, dict):
            raise CompilerParseError("Compiler result must be an object")
        executable = data.get("executable")
        if not isinstance(executable, bool):
            raise CompilerParseError("Compiler result missing boolean 'executable'")
        raw_errors = data.get("errors", [])
        if not isinstance(raw_errors, list):
            raise CompilerParseError("Compiler result 'errors' must be a list")
        errors = [CompilerIssue.from_dict(item) for item in raw_errors]
        if not executable and not errors:
            raise CompilerParseError("Non-executable compiler result must include errors")
        return cls(executable=executable, errors=errors)
