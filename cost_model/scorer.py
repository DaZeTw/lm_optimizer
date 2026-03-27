"""
Plan scorer — walks a PhysicalNode DAG bottom-up, calls the
appropriate profiler for each node, accumulates totals, and
produces a PlanCostReport.

Also provides annotate_plan() which writes cost estimates back
onto each PhysicalNode.cost field so the physical planner can
use them for cost-aware variant selection.

Usage:
    from cost_model.scorer import PlanScorer

    scorer = PlanScorer(preset="balanced")
    report = scorer.score(physical_root)
    print(report.summary())
"""

from __future__ import annotations

from catalog.catalog import SystemCatalog
from cost_model.history import HistoryLog, default_log
from cost_model.profilers import profile_node
from cost_model.vectors import OperatorCostVector, PlanCostReport
from cost_model.weights import WeightConfig, get_weights
from ir.nodes import CostVector, PhysicalNode

# Thresholds for automatic warnings
_SATURATION_WARN = 0.70
_QUALITY_RISK_WARN = 0.50
_COMPOSE_GAP_WARN = 0.70


class PlanScorer:
    """
    Scores a PhysicalNode DAG and returns a PlanCostReport.

    Args:
        preset   — weight preset name: "balanced", "speed", "quality", "economy"
        weights  — explicit WeightConfig (overrides preset)
        log      — history log (defaults to module singleton)
        corpus   — optional corpus for embedding-based estimates in COMPOSE
        catalog  — optional system catalog for offline statistics
    """

    def __init__(
        self,
        preset: str = "balanced",
        weights: WeightConfig | None = None,
        log: HistoryLog | None = None,
        corpus=None,
        catalog: SystemCatalog | None = None,
    ):
        self.weights = weights or get_weights(preset)
        self.log = log or default_log
        self.corpus = corpus
        self.catalog = catalog
        self._counter: dict[str, int] = {}  # op_name → count for unique op_ids

    def score(self, root: PhysicalNode) -> PlanCostReport:
        """Score the full plan and return a PlanCostReport."""
        self._counter = {}
        per_node: dict[str, OperatorCostVector] = {}
        warnings: list[str] = []

        # Post-order walk: score children first, propagate token estimates up
        self._walk(root, per_node, warnings)

        # Aggregate totals
        total_token = sum(v.token_cost for v in per_node.values())
        total_call = sum(v.call_cost for v in per_node.values())
        total_latency = self._critical_path_latency(root, per_node)
        total_risk = max(0.0, sum(v.quality_risk for v in per_node.values()))

        scalar = self.weights.scalar(total_token, total_call, total_latency, total_risk)

        # Find bottleneck (node with highest individual scalar)
        bottleneck = max(
            per_node,
            key=lambda k: per_node[k].scalar(
                self.weights.alpha,
                self.weights.beta,
                self.weights.gamma,
                self.weights.delta,
            ),
            default="",
        )

        # Max saturation across all AGGREGATE nodes
        saturation = max(
            (v.saturation for v in per_node.values()),
            default=0.0,
        )

        # Automatic warnings
        if saturation > _SATURATION_WARN:
            warnings.append(
                f"Context saturation {saturation:.1%} — consider HierarchicalGenerate"
            )
        if total_risk > _QUALITY_RISK_WARN:
            warnings.append(
                f"High quality risk {total_risk:.2f} — consider adding VERIFY"
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
        """
        Score the plan and write CostVector estimates back onto each
        PhysicalNode.cost field. Returns the annotated root and the report.

        Since PhysicalNode is frozen, this rebuilds the tree with updated
        cost fields rather than mutating in place.
        """
        report = self.score(root)
        annotated = self._annotate(root, report.per_node)
        return annotated, report

    # ── Internal ───────────────────────────────────────────────────

    def _op_id(self, node: PhysicalNode) -> str:
        """Generate a unique op_id for this node, e.g. AGGREGATE_0."""
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
        """
        Post-order walk. Returns the estimated output token count for
        this node, which is passed as upstream_tokens to the parent.
        """
        # Recurse into children and sum their output tokens
        upstream_tokens = sum(
            self._walk(child, per_node, warnings) for child in node.inputs
        )

        op_id = self._op_id(node)
        cv = profile_node(
            node=node.logical_ref,
            variant=node.variant,
            op_id=op_id,
            upstream_tokens=upstream_tokens,
            corpus=self.corpus,
            log=self.log,
            catalog=self.catalog,
        )
        per_node[op_id] = cv

        # Warn on high-risk COMPOSE
        if "semantic_gap" in cv.key_metric:
            try:
                gap = float(cv.key_metric.split("semantic_gap=")[1].split()[0])
                if gap > _COMPOSE_GAP_WARN:
                    warnings.append(
                        f"{op_id}: COMPOSE semantic gap {gap:.2f} — high join risk"
                    )
            except (IndexError, ValueError):
                pass

        return cv.token_cost

    def _critical_path_latency(
        self,
        node: PhysicalNode,
        per_node: dict[str, OperatorCostVector],
    ) -> float:
        """
        Compute the longest sequential chain of latency steps.
        Parallel branches (UNION children) contribute only their max,
        not their sum — reflecting asyncio.gather parallelism.
        """
        from ir.ops import Op

        if not node.inputs:
            # Find this node's op_id in per_node
            op_name = node.logical_ref.op.value
            for op_id, cv in per_node.items():
                if op_id.startswith(op_name) and cv.variant == node.variant:
                    return cv.latency_cost
            return 0.0

        # For UNION: children run in parallel — take max child latency
        if node.logical_ref.op == Op.UNION:
            child_latency = max(
                self._critical_path_latency(c, per_node) for c in node.inputs
            )
        else:
            # Sequential: sum child latencies
            child_latency = sum(
                self._critical_path_latency(c, per_node) for c in node.inputs
            )

        op_name = node.logical_ref.op.value
        own_latency = 0.0
        for op_id, cv in per_node.items():
            if op_id.startswith(op_name) and cv.variant == node.variant:
                own_latency = cv.latency_cost
                break

        return child_latency + own_latency

    def _annotate(
        self,
        node: PhysicalNode,
        per_node: dict[str, OperatorCostVector],
    ) -> PhysicalNode:
        """Rebuild tree with CostVector written to each node.cost."""
        annotated_inputs = tuple(self._annotate(c, per_node) for c in node.inputs)

        # Match this node to its per_node entry
        op_name = node.logical_ref.op.value
        cost_cv = CostVector()
        for op_id, cv in per_node.items():
            if op_id.startswith(op_name) and cv.variant == node.variant:
                cost_cv = CostVector(
                    token_cost=cv.token_cost,
                    call_cost=cv.call_cost,
                    latency_cost=cv.latency_cost,
                    quality_risk=cv.quality_risk,
                )
                break

        return PhysicalNode(
            variant=node.variant,
            logical_ref=node.logical_ref,
            inputs=annotated_inputs,
            params=node.params,
            cost=cost_cv,
        )
