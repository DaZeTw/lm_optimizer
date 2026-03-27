"""
Physical planner — maps an optimized LogicalNode DAG to a
PhysicalNode DAG by selecting a concrete variant for each node.

Selection is rule-based. Phase 3 (cost model) will upgrade this
to cost-aware selection once estimates are available.
"""

from __future__ import annotations

from ir.nodes import CostVector, LogicalNode, PhysicalNode
from ir.ops import Op


def build_physical_plan(root: LogicalNode) -> PhysicalNode:
    """
    Recursively convert a LogicalNode DAG into a PhysicalNode DAG.
    Children are resolved before parents (post-order).
    """
    physical_inputs = tuple(build_physical_plan(c) for c in root.inputs)
    variant = _select(root)
    return PhysicalNode(
        variant=variant,
        logical_ref=root,
        inputs=physical_inputs,
        params=dict(root.params),
        cost=CostVector(),  # populated by Phase 3 scorer
    )


def _select(node: LogicalNode) -> str:
    op = node.op

    if op == Op.I:
        return "HybridRetrieve"

    if op == Op.TRANSFORM:
        schema = node.params.get("schema", "")
        return "IdentityTransform" if not schema.strip() else "ExtractiveCompress"

    if op == Op.COMPOSE:
        condition = node.params.get("condition", "")
        return "LLMCompose" if condition.strip() else "ConcatCompose"

    if op == Op.RANK:
        return "SimilarityRank"

    if op == Op.UNION:
        return "SimpleUnion"

    if op == Op.DIFF:
        # Use SemanticDiff when subtracting the __overlap__ sentinel from R7
        if len(node.inputs) > 1:
            sub = node.inputs[1]
            if sub.op == Op.I and "__overlap__" in sub.params.get("query", ""):
                return "SemanticDiff"
        return "ExactDiff"

    if op == Op.AGGREGATE:
        # Heuristic: deep subtrees likely produce large evidence sets
        return "HierarchicalGenerate" if _depth(node) > 3 else "DirectGenerate"

    if op == Op.VERIFY:
        return "CitationVerify"

    return "IdentityTransform"  # safe no-op fallback


def _depth(node: LogicalNode) -> int:
    if not node.inputs:
        return 0
    return 1 + max(_depth(c) for c in node.inputs)
