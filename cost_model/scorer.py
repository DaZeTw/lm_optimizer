"""Plan scorer based on observed runtime telemetry.

This scorer uses one fixed scalar function and does not rely on preset weights,
history priors, or heuristic fallback estimates.
"""

from __future__ import annotations

from catalog.catalog import SystemCatalog
from cost_model.telemetry import TelemetryStore, default_telemetry
from cost_model.vectors import OperatorCostVector, PlanCostReport
from ir.nodes import CostVector, PhysicalNode
from ir.ops import Op
from planner.variant_candidates import CANDIDATE_VARIANTS

_SATURATION_WARN = 0.70
_QUALITY_RISK_WARN = 0.50
_COMPOSE_GAP_WARN = 0.70
_UNOBSERVED_PENALTY = 1_000_000.0


class PlanScorer:
    """Scores physical plans using observed per-operator per-variant telemetry."""

    def __init__(
        self,
        preset: str = "balanced",
        weights=None,
        log=None,
        corpus=None,
        catalog: SystemCatalog | None = None,
        telemetry: TelemetryStore | None = None,
    ):
        del preset
        del weights
        del log
        del corpus
        self.catalog = catalog
        self.telemetry = telemetry or default_telemetry
        self._counter: dict[str, int] = {}

    def score(self, root: PhysicalNode) -> PlanCostReport:
        self._counter = {}
        per_node: dict[str, OperatorCostVector] = {}
        warnings: list[str] = []

        self._walk(root, per_node, warnings)

        total_token = sum(v.token_cost for v in per_node.values())
        total_call = sum(v.call_cost for v in per_node.values())
        self._counter = {}
        total_latency = self._critical_path_latency(root, per_node)
        total_risk = max(0.0, sum(v.quality_risk for v in per_node.values()))

        scalar = self._scalar(total_token, total_call, total_latency, total_risk)

        bottleneck = max(
            per_node, key=lambda k: self._vector_scalar(per_node[k]), default=""
        )

        saturation = max((v.saturation for v in per_node.values()), default=0.0)
        if saturation > _SATURATION_WARN:
            warnings.append(
                f"Context saturation {saturation:.1%} - consider HierarchicalGenerate"
            )
        if total_risk > _QUALITY_RISK_WARN:
            warnings.append(
                f"High quality risk {total_risk:.2f} - consider adding VERIFY"
            )

        return PlanCostReport(
            total_token_cost=round(total_token, 1),
            total_call_cost=round(total_call, 2),
            total_latency_cost=round(total_latency, 1),
            total_quality_risk=round(total_risk, 3),
            scalar=round(scalar, 2),
            per_node=per_node,
            bottleneck=bottleneck,
            saturation=round(saturation, 3),
            warnings=warnings,
        )

    def annotate_plan(self, root: PhysicalNode) -> tuple[PhysicalNode, PlanCostReport]:
        report = self.score(root)
        self._counter = {}
        annotated = self._annotate(root, report.per_node)
        return annotated, report

    def _scalar(
        self,
        token_cost: float,
        call_cost: float,
        latency_cost: float,
        quality_risk: float,
    ) -> float:
        # Single simple objective for now.
        return token_cost + call_cost + latency_cost + quality_risk

    def _vector_scalar(self, vector: OperatorCostVector) -> float:
        return self._scalar(
            vector.token_cost,
            vector.call_cost,
            vector.latency_cost,
            vector.quality_risk,
        )

    def _op_id(self, node: PhysicalNode) -> str:
        name = node.logical_ref.op.value
        idx = self._counter.get(name, 0)
        self._counter[name] = idx + 1
        return f"{name}_{idx}"

    def _walk(
        self,
        node: PhysicalNode,
        per_node: dict[str, OperatorCostVector],
        warnings: list[str],
    ) -> float:
        upstream_tokens = sum(
            self._walk(child, per_node, warnings) for child in node.inputs
        )

        op_id = self._op_id(node)
        variant_costs = self._score_all_variants(
            node=node, upstream_tokens=upstream_tokens
        )

        selected = variant_costs.get(node.variant)
        if selected is None:
            selected = {
                "token_cost": _UNOBSERVED_PENALTY,
                "call_cost": 0.0,
                "latency_cost": 0.0,
                "quality_risk": 1.0,
                "saturation": 0.0,
                "key_metric": "unobserved_variant",
                "sample_count": 0,
                "accuracy_score": None,
            }

        cv = OperatorCostVector(
            op_id=op_id,
            variant=node.variant,
            token_cost=float(selected["token_cost"]),
            call_cost=float(selected["call_cost"]),
            latency_cost=float(selected["latency_cost"]),
            quality_risk=float(selected["quality_risk"]),
            key_metric=str(selected["key_metric"]),
            saturation=float(selected["saturation"]),
            sample_count=int(selected["sample_count"]),
            accuracy_score=(
                float(selected["accuracy_score"])
                if selected["accuracy_score"] is not None
                else None
            ),
            variant_costs={k: dict(v) for k, v in variant_costs.items()},
        )
        per_node[op_id] = cv

        if "semantic_gap" in cv.key_metric:
            try:
                gap = float(cv.key_metric.split("semantic_gap=")[1].split()[0])
                if gap > _COMPOSE_GAP_WARN:
                    warnings.append(
                        f"{op_id}: COMPOSE semantic gap {gap:.2f} - high join risk"
                    )
            except (IndexError, ValueError):
                pass

        return cv.token_cost

    def _critical_path_latency(
        self,
        node: PhysicalNode,
        per_node: dict[str, OperatorCostVector],
    ) -> float:
        if node.inputs:
            if node.logical_ref.op == Op.UNION:
                child_latency = max(
                    self._critical_path_latency(c, per_node) for c in node.inputs
                )
            else:
                child_latency = sum(
                    self._critical_path_latency(c, per_node) for c in node.inputs
                )
        else:
            child_latency = 0.0

        op_id = self._op_id(node)
        own_latency = per_node.get(
            op_id, OperatorCostVector(op_id=op_id, variant=node.variant)
        ).latency_cost
        return child_latency + own_latency

    def _annotate(
        self,
        node: PhysicalNode,
        per_node: dict[str, OperatorCostVector],
    ) -> PhysicalNode:
        annotated_inputs = tuple(self._annotate(c, per_node) for c in node.inputs)
        op_id = self._op_id(node)
        cv = per_node.get(op_id)

        selected_cost = CostVector()
        per_variant_costs: dict[str, CostVector] = {}
        if cv is not None:
            selected_cost = CostVector(
                token_cost=cv.token_cost,
                call_cost=cv.call_cost,
                latency_cost=cv.latency_cost,
                quality_risk=cv.quality_risk,
            )
            for variant, metrics in cv.variant_costs.items():
                per_variant_costs[variant] = CostVector(
                    token_cost=float(metrics.get("token_cost", 0.0)),
                    call_cost=float(metrics.get("call_cost", 0.0)),
                    latency_cost=float(metrics.get("latency_cost", 0.0)),
                    quality_risk=float(metrics.get("quality_risk", 0.0)),
                )

        return PhysicalNode(
            variant=node.variant,
            logical_ref=node.logical_ref,
            inputs=annotated_inputs,
            params=node.params,
            cost=selected_cost,
            variant_costs=per_variant_costs,
        )

    def _score_all_variants(
        self,
        *,
        node: PhysicalNode,
        upstream_tokens: float,
    ) -> dict[str, dict[str, object]]:
        del upstream_tokens
        candidates = list(CANDIDATE_VARIANTS.get(node.logical_ref.op, [node.variant]))
        if node.variant not in candidates:
            candidates.append(node.variant)

        op_name = node.logical_ref.op.value
        out: dict[str, dict[str, object]] = {}
        for variant in candidates:
            observed = self.telemetry.estimate(op=op_name, variant=variant)
            if observed is None:
                out[variant] = {
                    "token_cost": _UNOBSERVED_PENALTY,
                    "call_cost": 0.0,
                    "latency_cost": 0.0,
                    "quality_risk": 1.0,
                    "saturation": 0.0,
                    "key_metric": "unobserved_variant",
                    "sample_count": 0,
                    "accuracy_score": None,
                }
                continue

            out[variant] = {
                "token_cost": round(observed.token_cost, 4),
                "call_cost": round(observed.call_cost, 4),
                "latency_cost": round(observed.latency_cost, 4),
                "quality_risk": round(observed.quality_risk, 4),
                "saturation": round(observed.saturation, 6),
                "key_metric": observed.key_metric,
                "sample_count": observed.sample_count,
                "accuracy_score": (
                    round(float(observed.accuracy_score), 6)
                    if observed.accuracy_score is not None
                    else None
                ),
            }
        return out
