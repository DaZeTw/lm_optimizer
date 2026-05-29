"""Prompt rendering helpers for the TST compiler."""

from __future__ import annotations

from typing import Any


def normalize_operator_descriptions(operator_descriptions: dict) -> dict[str, dict]:
    normalized: dict[str, dict] = {}
    for op, meta in operator_descriptions.items():
        op_name = getattr(op, "value", str(op))
        normalized[op_name] = dict(meta)
    return normalized


def render_operator_descriptions(operator_descriptions: dict) -> str:
    normalized = normalize_operator_descriptions(operator_descriptions)
    lines: list[str] = []
    for op_name, meta in normalized.items():
        lines.append(f"{op_name}")
        lines.append(f"  description: {meta.get('description', '')}")
        lines.append(f"  inputs: {meta.get('inputs', '')}")
        lines.append(f"  outputs: {meta.get('outputs', '')}")
        lines.append(f"  use_when: {meta.get('use_when', '')}")
    return "\n".join(lines)


def render_tst(tst: dict[str, Any]) -> str:
    skel = tst.get("logical_skeleton", {})
    adap = tst.get("adaptation_policy", {})
    lines: list[str] = [
        "LOGICAL SKELETON",
        str(skel.get("template", "")),
        "",
        "ADAPTATION POLICY",
    ]
    for key in ("mutable_slots", "immutable_slots", "mutable_ops", "immutable_ops"):
        values = adap.get(key, [])
        lines.append(f"{key}: {', '.join(str(v) for v in values)}")
    for rule in adap.get("allowed_rewrites", []):
        lines.append(f"allowed_rewrites: {rule}")
    for rule in adap.get("forbidden_rewrites", []):
        lines.append(f"forbidden_rewrites: {rule}")
    return "\n".join(lines)
