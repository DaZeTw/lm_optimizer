"""Parse an LLM-produced JSON dict into a PhysicalNode DAG."""

from __future__ import annotations

import json

from ir.nodes import CostVector, LogicalNode, PhysicalNode
from ir.ops import Op
from planner.variant_candidates import CANDIDATE_VARIANTS


class PlanParseError(Exception):
    pass


def parse_physical_plan(data: dict | str) -> PhysicalNode:
    """
    Convert a JSON dict (or JSON string) describing a physical plan into a
    PhysicalNode DAG.

    Expected format::

        {
            "op": "AGGREGATE",
            "variant": "DirectGenerate",
            "params": {"goal": "summarize findings"},
            "inputs": [
                {
                    "op": "RANK",
                    "variant": "SimilarityRank",
                    "params": {"criterion": "relevance", "top_k": 5},
                    "inputs": [...]
                }
            ]
        }

    Raises:
        PlanParseError: if ``op`` is missing or not a recognised Op value.
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as exc:
            raise PlanParseError(f"Invalid JSON: {exc}") from exc

    return _parse_node(data)


def _parse_node(data: dict) -> PhysicalNode:
    if not isinstance(data, dict):
        raise PlanParseError(f"Expected a dict node, got {type(data).__name__}")

    op_str = data.get("op")
    if op_str is None:
        raise PlanParseError("Node is missing required 'op' field")

    try:
        op = Op(op_str)
    except ValueError:
        raise PlanParseError(f"Unknown op '{op_str}'. Valid ops: {[o.value for o in Op]}")

    candidates = CANDIDATE_VARIANTS.get(op, [])
    variant = data.get("variant")
    if not isinstance(variant, str) or not variant:
        raise PlanParseError(f"{op.value} node is missing required 'variant' field")
    if variant not in candidates:
        raise PlanParseError(
            f"Unknown variant '{variant}' for op '{op.value}'. "
            f"Valid variants: {candidates}"
        )

    params = dict(data.get("params") or {})
    inputs = tuple(_parse_node(child) for child in (data.get("inputs") or []))

    logical_ref = LogicalNode(op=op, inputs=tuple(n.logical_ref for n in inputs), params=params)

    return PhysicalNode(
        variant=variant,
        logical_ref=logical_ref,
        inputs=inputs,
        params=params,
        cost=CostVector(),
    )
