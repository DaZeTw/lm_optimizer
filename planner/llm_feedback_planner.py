"""LLM-guided physical planner with cost-model feedback.

This planner starts from a cost-aware baseline, asks an LLM for variant overrides,
then scores each proposed plan and keeps the best one.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from cost_model.cost_aware_planner import _CANDIDATES, CostAwarePlanner
from cost_model.scorer import PlanScorer
from ir.nodes import LogicalNode, PhysicalNode


class LLMFeedbackPhysicalPlanner:
    """Refine physical variants using LLM proposals and cost-model scoring."""

    def __init__(
        self,
        *,
        cost_planner: CostAwarePlanner,
        client: Any | None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        rounds: int = 1,
    ):
        self._cost_planner = cost_planner
        self._client = client
        self._model = model
        self._temperature = temperature
        self._rounds = max(1, rounds)

    def build(self, root: LogicalNode) -> PhysicalNode:
        """Build and refine a physical plan. Falls back gracefully on any LLM issues."""
        baseline = self._cost_planner.build(root)
        if self._client is None:
            return baseline

        best_plan = baseline
        best_scalar = self._score(best_plan)

        for _ in range(self._rounds):
            overrides = self._propose_overrides(best_plan, best_scalar)
            if not overrides:
                continue
            candidate = self._apply_overrides(best_plan, overrides)
            candidate_scalar = self._score(candidate)
            if candidate_scalar < best_scalar:
                best_plan = candidate
                best_scalar = candidate_scalar

        return best_plan

    def _score(self, plan: PhysicalNode) -> float:
        scorer = PlanScorer(
            weights=self._cost_planner.weights,
            log=self._cost_planner.log,
            corpus=self._cost_planner.corpus,
            catalog=self._cost_planner.catalog,
        )
        return scorer.score(plan).scalar

    def _propose_overrides(
        self,
        plan: PhysicalNode,
        current_scalar: float,
    ) -> dict[str, str]:
        prompt = self._build_prompt(plan, current_scalar)
        messages = [
            {
                "role": "system",
                "content": (
                    "You optimize physical operator variants for a retrieval+LLM pipeline. "
                    "Return only strict JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            raw = self._client.complete(
                messages=messages,
                model=self._model,
                temperature=self._temperature,
            )
            payload = json.loads(raw)
        except Exception:
            return {}

        overrides = payload.get("variant_overrides", {})
        if not isinstance(overrides, dict):
            return {}

        normalized: dict[str, str] = {}
        for op_name, variant in overrides.items():
            if isinstance(op_name, str) and isinstance(variant, str):
                normalized[op_name.strip().upper()] = variant.strip()
        return normalized

    def _build_prompt(self, plan: PhysicalNode, current_scalar: float) -> str:
        lines: list[str] = []
        self._render_plan(plan, lines, 0)
        return (
            "Current physical plan variants by node:\n"
            + "\n".join(lines)
            + "\n\n"
            + f"Current total scalar cost: {current_scalar:.3f}\n"
            + "You may override variants by logical op name. "
            + "Only use known variants from this set:\n"
            + json.dumps({k.value: v for k, v in _CANDIDATES.items()}, indent=2)
            + "\n\nReturn JSON only in this form:\n"
            + '{"variant_overrides": {"AGGREGATE": "HierarchicalGenerate"}}'
        )

    def _render_plan(self, node: PhysicalNode, lines: list[str], depth: int) -> None:
        indent = "  " * depth
        lines.append(
            f"{indent}{node.logical_ref.op.value}: variant={node.variant} params={node.params}"
        )
        for child in node.inputs:
            self._render_plan(child, lines, depth + 1)

    def _apply_overrides(
        self,
        plan: PhysicalNode,
        overrides: dict[str, str],
    ) -> PhysicalNode:
        logical_op = plan.logical_ref.op
        op_name = logical_op.value
        desired = overrides.get(op_name)

        allowed = self._cost_planner._filter_candidates(  # noqa: SLF001
            plan.logical_ref,
            list(_CANDIDATES.get(logical_op, [plan.variant])),
        )
        next_variant = plan.variant
        if desired and desired in allowed:
            next_variant = desired

        new_inputs = tuple(
            self._apply_overrides(child, overrides) for child in plan.inputs
        )
        rebuilt = replace(plan, variant=next_variant, inputs=new_inputs)

        scorer = PlanScorer(
            weights=self._cost_planner.weights,
            log=self._cost_planner.log,
            corpus=self._cost_planner.corpus,
            catalog=self._cost_planner.catalog,
        )
        annotated, _ = scorer.annotate_plan(rebuilt)
        return annotated
