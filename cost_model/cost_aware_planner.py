"""Single physical feedback planner.

This planner selects and refines variants using only observed scorer outputs.
"""

from __future__ import annotations

from dataclasses import replace

from catalog.catalog import SystemCatalog
from cost_model.scorer import PlanScorer
from ir.nodes import LogicalNode, PhysicalNode
from ir.ops import Op
from planner.physical import build_physical_plan
from planner.variant_candidates import CANDIDATE_VARIANTS


class CostAwarePlanner:
    """Single planner for initial build and feedback-driven refinement."""

    def __init__(
        self,
        preset: str = "balanced",
        weights=None,
        log=None,
        corpus=None,
        catalog: SystemCatalog | None = None,
        model_id: str | None = None,
    ):
        del preset
        del weights
        del log
        self.corpus = corpus
        self.catalog = catalog
        self.model_id = model_id

    def build(self, root: LogicalNode) -> PhysicalNode:
        """Build a physical plan and annotate observed variant score options."""
        base = build_physical_plan(root)
        selected = self._select_tree(base)
        annotated, _ = PlanScorer(catalog=self.catalog).annotate_plan(selected)
        return annotated

    def _select_tree(self, node: PhysicalNode) -> PhysicalNode:
        selected_inputs = tuple(self._select_tree(child) for child in node.inputs)
        params = self._inject_runtime_params(node.logical_ref, node.params)

        candidates = self._filter_candidates(
            node.logical_ref,
            list(CANDIDATE_VARIANTS.get(node.logical_ref.op, [node.variant])),
        )
        if not candidates:
            candidates = [node.variant]

        scorer = PlanScorer(catalog=self.catalog)
        best_variant = node.variant
        best_scalar: float | None = None

        for variant in candidates:
            candidate = replace(
                node,
                variant=variant,
                inputs=selected_inputs,
                params=params,
            )
            score = scorer.score(candidate).scalar
            if best_scalar is None or score < best_scalar:
                best_scalar = score
                best_variant = variant

        return replace(
            node,
            variant=best_variant,
            inputs=selected_inputs,
            params=params,
        )

    def _inject_runtime_params(self, logical: LogicalNode, params: dict) -> dict:
        out = dict(params)
        if logical.op == Op.AGGREGATE and self.catalog is not None:
            model_id = self.model_id or self.catalog.default_model_id
            out.setdefault("model_id", model_id)
            out.setdefault("context_window", self.catalog.context_window(model_id))
        return out

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
