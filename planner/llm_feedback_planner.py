"""LLM-driven physical planner with iterative refinement support.

This planner starts from a deterministic base physical plan and then asks an
LLM planning client for per-operator variant overrides. Overrides are validated
against allowed candidates before being applied.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from catalog.catalog import SystemCatalog
from cost_model.cost_aware_planner import CostAwarePlanner
from cost_model.scorer import PlanScorer
from ir.nodes import LogicalNode, PhysicalNode
from ir.ops import Op
from planner.variant_candidates import CANDIDATE_VARIANTS


class LlmFeedbackPlanner:
    """Applies LLM-proposed physical variant overrides over a base plan."""

    def __init__(
        self,
        *,
        base_planner: CostAwarePlanner,
        planning_client: Any,
        catalog: SystemCatalog | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        rounds: int = 1,
    ):
        self.base_planner = base_planner
        self.planning_client = planning_client
        self.catalog = catalog
        self.model = model
        self.temperature = temperature
        self.rounds = max(1, rounds)

    def build(
        self,
        root: LogicalNode,
        feedback: dict[str, Any] | None = None,
    ) -> PhysicalNode:
        """Build physical plan and refine with one or more LLM planning rounds."""
        scorer = PlanScorer(catalog=self.catalog)
        current, current_report = scorer.annotate_plan(self.base_planner.build(root))

        for round_index in range(self.rounds):
            response = self._request_overrides(
                logical=root,
                current_plan=current,
                current_feedback=feedback,
                current_score=current_report,
                round_index=round_index,
            )
            overrides = self._parse_overrides(response)
            if not overrides:
                continue

            refined = self._apply_overrides(current, overrides)
            current, current_report = scorer.annotate_plan(refined)

        return current

    def _request_overrides(
        self,
        *,
        logical: LogicalNode,
        current_plan: PhysicalNode,
        current_feedback: dict[str, Any] | None,
        current_score,
        round_index: int,
    ) -> Any:
        allowed = {
            op.value: list(variants) for op, variants in CANDIDATE_VARIANTS.items()
        }
        payload = {
            "task": "Choose physical variant overrides that reduce scalar cost while preserving quality.",
            "round": round_index + 1,
            "logical_plan": logical.to_dict(),
            "current_physical_plan": self._physical_to_dict(current_plan),
            "allowed_variants": allowed,
            "current_score": {
                "scalar": current_score.scalar,
                "total_token_cost": current_score.total_token_cost,
                "total_call_cost": current_score.total_call_cost,
                "total_latency_cost": current_score.total_latency_cost,
                "total_quality_risk": current_score.total_quality_risk,
                "bottleneck": current_score.bottleneck,
            },
            "execution_feedback": current_feedback or {},
            "output_schema": {"variant_overrides": {"<OP or OP_index>": "<variant>"}},
            "rules": [
                "Only return JSON.",
                "Only choose variants in allowed_variants for that operator.",
                "Prefer lower scalar cost and lower quality risk.",
                "Do not change operators, only variants.",
            ],
        }

        messages = [
            {
                "role": "system",
                "content": "You are a query-optimizer planner that returns strict JSON.",
            },
            {
                "role": "user",
                "content": json.dumps(payload, indent=2),
            },
        ]
        return self.planning_client.complete(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
        )

    def _parse_overrides(self, response: Any) -> dict[str, str]:
        if isinstance(response, dict):
            body = response
        else:
            text = str(response or "").strip()
            if text.startswith("```"):
                text = self._strip_code_fence(text)
            try:
                body = json.loads(text)
            except json.JSONDecodeError:
                return {}

        overrides = body.get("variant_overrides") if isinstance(body, dict) else None
        if not isinstance(overrides, dict):
            return {}

        out: dict[str, str] = {}
        for k, v in overrides.items():
            if isinstance(k, str) and isinstance(v, str):
                out[k.strip()] = v.strip()
        return out

    def _strip_code_fence(self, text: str) -> str:
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _apply_overrides(
        self,
        root: PhysicalNode,
        overrides: dict[str, str],
    ) -> PhysicalNode:
        counters: dict[str, int] = {}

        def walk(node: PhysicalNode) -> PhysicalNode:
            op_name = node.logical_ref.op.value
            op_index = counters.get(op_name, 0)
            counters[op_name] = op_index + 1
            op_id = f"{op_name}_{op_index}"

            updated_inputs = tuple(walk(c) for c in node.inputs)

            desired = overrides.get(op_id)
            if desired is None:
                desired = overrides.get(op_name)

            candidates = self._filter_candidates(
                node.logical_ref,
                list(CANDIDATE_VARIANTS.get(node.logical_ref.op, [node.variant])),
            )
            if not candidates:
                candidates = [node.variant]

            variant = node.variant
            if desired in candidates:
                variant = desired

            return replace(
                node,
                variant=variant,
                inputs=updated_inputs,
            )

        return walk(root)

    def _filter_candidates(
        self,
        logical_node: LogicalNode,
        candidates: list[str],
    ) -> list[str]:
        op = logical_node.op

        if op == Op.TRANSFORM:
            schema = str(logical_node.params.get("schema", ""))
            if not schema.strip():
                return ["IdentityTransform"]

        if op == Op.COMPOSE:
            condition = str(logical_node.params.get("condition", ""))
            if not condition.strip():
                return ["ConcatCompose"]

        if op == Op.DIFF and len(logical_node.inputs) > 1:
            subtract = logical_node.inputs[1]
            if subtract.op == Op.I and "__overlap__" in str(
                subtract.params.get("query", "")
            ):
                return ["SemanticDiff"]

        seen: set[str] = set()
        filtered: list[str] = []
        for name in candidates:
            if name not in seen:
                seen.add(name)
                filtered.append(name)
        return filtered

    def _physical_to_dict(self, node: PhysicalNode) -> dict[str, Any]:
        return {
            "op": node.logical_ref.op.value,
            "variant": node.variant,
            "params": dict(node.params),
            "inputs": [self._physical_to_dict(c) for c in node.inputs],
        }
