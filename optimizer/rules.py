"""
8 algebraic rewrite rules for the logical plan optimizer.

Each rule is a pure function:
    rule(node: LogicalNode) -> tuple[LogicalNode, bool]

Returns (new_node, True)  if the rule fired (tree changed).
Returns (node,     False) if the rule did not match.

Rules only inspect the current node and its direct children.
The engine handles the tree walk.
"""

from __future__ import annotations

from ir.nodes import LogicalNode
from ir.ops import Op

# ── Helpers ────────────────────────────────────────────────────────


def _subtree_depth(node: LogicalNode) -> int:
    """Proxy for evidence size: deeper subtree = more evidence."""
    if not node.inputs:
        return 0
    return 1 + max(_subtree_depth(c) for c in node.inputs)


def _contains_op(node: LogicalNode, op: Op) -> bool:
    """Return True if op appears anywhere in the subtree."""
    if node.op == op:
        return True
    return any(_contains_op(c, op) for c in node.inputs)


def _merge_str(a: str, b: str, sep: str = " + ") -> str:
    """Combine two param strings, skipping empty ones."""
    parts = [x for x in (a, b) if x]
    return sep.join(parts) if parts else ""


# ── R1: Transform Pushdown ─────────────────────────────────────────


def r1_transform_pushdown(node: LogicalNode) -> tuple[LogicalNode, bool]:
    """
    Push TRANSFORM inside COMPOSE so evidence is compressed
    before the expensive semantic join.

    COMPOSE(E1, E2)
    → COMPOSE(TRANSFORM(E1, schema=""), TRANSFORM(E2, schema=""))

    Only fires when neither child is already a TRANSFORM.
    """
    if node.op != Op.COMPOSE:
        return node, False

    left, right = node.inputs
    if left.op == Op.TRANSFORM and right.op == Op.TRANSFORM:
        return node, False  # already pushed

    new_left = left if left.op == Op.TRANSFORM else LogicalNode.transform(left)
    new_right = right if right.op == Op.TRANSFORM else LogicalNode.transform(right)

    return (
        LogicalNode(op=Op.COMPOSE, inputs=(new_left, new_right), params=node.params),
        True,
    )


# ── R2: Filter Pushdown ────────────────────────────────────────────


def r2_filter_pushdown(node: LogicalNode) -> tuple[LogicalNode, bool]:
    """
    Insert RANK immediately after every bare I() that feeds into
    AGGREGATE, COMPOSE, or UNION, so noisy evidence is pruned early.

    AGGREGATE(I(q))  →  AGGREGATE(RANK(I(q), criterion="relevance"))

    Does not fire if the child is already a RANK.
    """
    if node.op not in (Op.AGGREGATE, Op.COMPOSE, Op.UNION):
        return node, False

    new_inputs = []
    changed = False
    for child in node.inputs:
        if child.op == Op.I:
            new_inputs.append(LogicalNode.rank(child, criterion="relevance"))
            changed = True
        else:
            new_inputs.append(child)

    if not changed:
        return node, False

    return (
        LogicalNode(op=node.op, inputs=tuple(new_inputs), params=node.params),
        True,
    )


# ── R3: Compose Reorder ────────────────────────────────────────────


def r3_compose_reorder(node: LogicalNode) -> tuple[LogicalNode, bool]:
    """
    When composing nested COMPOSEs, ensure smaller subtrees are
    composed first to minimise intermediate evidence size.

    COMPOSE(COMPOSE(Elarge, Esmall), Etiny)
    → COMPOSE(Elarge, COMPOSE(Esmall, Etiny))

    Uses subtree depth as a proxy for evidence size.
    Only fires when the left child is a COMPOSE and swapping
    would reduce the depth of the intermediate result.
    """
    if node.op != Op.COMPOSE:
        return node, False

    left, right = node.inputs
    if left.op != Op.COMPOSE:
        return node, False

    # left = COMPOSE(A, B),  right = C
    a, b = left.inputs
    c = right

    depth_b = _subtree_depth(b)
    depth_c = _subtree_depth(c)

    # Only reorder if B and C are both shallower than A
    # → compose the smaller pair (B, C) first
    if depth_b <= _subtree_depth(a) and depth_c <= _subtree_depth(a):
        inner = LogicalNode(op=Op.COMPOSE, inputs=(b, c), params=left.params)
        outer = LogicalNode(op=Op.COMPOSE, inputs=(a, inner), params=node.params)
        return outer, True

    return node, False


# ── R4: Merge Cascade Filters ──────────────────────────────────────


def r4_merge_cascade_filters(node: LogicalNode) -> tuple[LogicalNode, bool]:
    """
    Collapse two consecutive RANKs into one combined RANK.

    RANK(RANK(E, c1), c2)  →  RANK(E, "c1 + c2")

    Also collapses nested DIFFs:
    DIFF(DIFF(E, E1), E2)  →  DIFF(E, UNION(E1, E2))
    """
    # RANK(RANK(...))
    if node.op == Op.RANK and node.inputs[0].op == Op.RANK:
        inner = node.inputs[0]
        merged_criterion = _merge_str(
            inner.params.get("criterion", ""),
            node.params.get("criterion", ""),
        )
        return LogicalNode.rank(inner.inputs[0], criterion=merged_criterion), True

    # DIFF(DIFF(E, E1), E2)
    if node.op == Op.DIFF and node.inputs[0].op == Op.DIFF:
        inner = node.inputs[0]  # DIFF(E, E1)
        e = inner.inputs[0]  # E
        e1 = inner.inputs[1]  # E1
        e2 = node.inputs[1]  # E2
        combined_subtract = LogicalNode.union(e1, e2)
        return LogicalNode.diff(e, combined_subtract), True

    return node, False


# ── R5: Merge Cascade Transforms ──────────────────────────────────


def r5_merge_cascade_transforms(node: LogicalNode) -> tuple[LogicalNode, bool]:
    """
    Collapse two consecutive TRANSFORMs into one.

    TRANSFORM(TRANSFORM(E, s1), s2)  →  TRANSFORM(E, "s1 + s2")

    Only safe when the intermediate representation is not
    referenced elsewhere (true at the logical level).
    """
    if node.op != Op.TRANSFORM:
        return node, False
    if node.inputs[0].op != Op.TRANSFORM:
        return node, False

    inner = node.inputs[0]
    merged_schema = _merge_str(
        inner.params.get("schema", ""),
        node.params.get("schema", ""),
    )
    return LogicalNode.transform(inner.inputs[0], schema=merged_schema), True


# ── R6: Delayed Aggregation ────────────────────────────────────────


def r6_delayed_aggregation(node: LogicalNode) -> tuple[LogicalNode, bool]:
    """
    Ensure AGGREGATE always receives ranked evidence.
    Insert RANK before AGGREGATE if its direct child is not
    already a RANK (or VERIFY, which wraps AGGREGATE).

    AGGREGATE(E)  →  AGGREGATE(RANK(E, "relevance"))

    Does not fire if child is already RANK, DIFF, or TRANSFORM
    (those already reduce the evidence footprint).
    """
    if node.op != Op.AGGREGATE:
        return node, False

    child = node.inputs[0]
    if child.op in (Op.RANK, Op.DIFF, Op.TRANSFORM):
        return node, False  # already filtered

    ranked = LogicalNode.rank(child, criterion="relevance")
    return (
        LogicalNode(op=Op.AGGREGATE, inputs=(ranked,), params=node.params),
        True,
    )


# ── R7: Early Deduplication ────────────────────────────────────────


def r7_early_dedup(node: LogicalNode) -> tuple[LogicalNode, bool]:
    """
    After a UNION, wrap with DIFF to remove overlapping evidence
    before it reaches AGGREGATE or COMPOSE.

    AGGREGATE(UNION(...))
    → AGGREGATE(DIFF(UNION(...), I("overlap sentinel")))

    The overlap sentinel I("__overlap__") is a logical placeholder;
    the physical planner resolves it to a real dedup strategy.
    """
    if node.op != Op.AGGREGATE:
        return node, False

    child = node.inputs[0]

    # Already deduped
    if child.op == Op.DIFF:
        return node, False

    # Child is RANK wrapping a UNION
    if child.op == Op.RANK and child.inputs[0].op == Op.UNION:
        union_node = child.inputs[0]
        overlap_sentinel = LogicalNode.isolate("__overlap__")
        deduped = LogicalNode.diff(union_node, overlap_sentinel)
        new_rank = LogicalNode(
            op=Op.RANK,
            inputs=(deduped,),
            params=child.params,
        )
        return (
            LogicalNode(op=Op.AGGREGATE, inputs=(new_rank,), params=node.params),
            True,
        )

    # Direct UNION child
    if child.op == Op.UNION:
        overlap_sentinel = LogicalNode.isolate("__overlap__")
        deduped = LogicalNode.diff(child, overlap_sentinel)
        return (
            LogicalNode(op=Op.AGGREGATE, inputs=(deduped,), params=node.params),
            True,
        )

    return node, False


# ── R8: Selective Verify ───────────────────────────────────────────


def r8_selective_verify(node: LogicalNode) -> tuple[LogicalNode, bool]:
    """
    Wrap the root AGGREGATE with VERIFY only when the plan
    contains a COMPOSE (multi-hop reasoning = higher quality risk).

    AGGREGATE(... COMPOSE ...)
    → VERIFY(AGGREGATE(...), constraints="grounded in evidence")

    Guards:
    - Only fires on AGGREGATE nodes.
    - Does not fire if already marked __verified__ in params.
    - Uses a param flag so subsequent passes skip this node.
    """
    if node.op != Op.AGGREGATE:
        return node, False

    if node.params.get("__verified__"):
        return node, False

    if not _contains_op(node, Op.COMPOSE):
        return node, False

    # Mark the AGGREGATE as verified so R8 won't re-fire on it
    marked = LogicalNode(
        op=node.op,
        inputs=node.inputs,
        params={**node.params, "__verified__": True},
    )
    return LogicalNode.verify(marked, constraints="grounded in evidence"), True


# ── Rule registry (ordered) ────────────────────────────────────────
# The engine applies rules in this order within each pass.
# Pushdown rules first, placement rules last.

RULES: list[tuple[str, object]] = [
    ("R1_transform_pushdown", r1_transform_pushdown),
    ("R2_filter_pushdown", r2_filter_pushdown),
    ("R4_merge_cascade_filters", r4_merge_cascade_filters),
    ("R5_merge_cascade_transforms", r5_merge_cascade_transforms),
    ("R3_compose_reorder", r3_compose_reorder),
    ("R6_delayed_aggregation", r6_delayed_aggregation),
    ("R7_early_dedup", r7_early_dedup),
    ("R8_selective_verify", r8_selective_verify),
]
