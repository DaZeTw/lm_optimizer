"""
Per-operator cost profilers.

Each profiler estimates token_cost, call_cost, latency_cost, and
quality_risk for one logical operator without making any LLM calls.

Estimation strategies:
  I         — chunk count × avg size from history log
  TRANSFORM — historical compression ratio from history log
  COMPOSE   — embedding cosine distance as semantic gap proxy
  RANK      — lightweight local scoring (no LLM)
  UNION     — sum of inputs
  DIFF      — overlap estimation
  AGGREGATE — saturation ratio vs context window
  VERIFY    — failure probability from history log + task type
  DECOMPOSE — branching factor × downstream multiplier
"""

from __future__ import annotations

from catalog.catalog import SystemCatalog
from cost_model.history import HistoryLog, default_log
from cost_model.vectors import OperatorCostVector
from ir.nodes import LogicalNode
from ir.ops import Op

_DEFAULT_CONTEXT_WINDOW = 128_000  # tokens


def profile_node(
    node: LogicalNode,
    variant: str,
    op_id: str,
    upstream_tokens: float = 0.0,
    corpus=None,
    log: HistoryLog | None = None,
    catalog: SystemCatalog | None = None,
) -> OperatorCostVector:
    """
    Dispatch to the correct profiler for this operator.

    Args:
        node            — the logical node being profiled
        variant         — the physical variant chosen by the planner
        op_id           — unique identifier string for this node
        upstream_tokens — token estimate flowing in from child nodes
        corpus          — optional corpus used for embedding-based estimates
        log             — history log (defaults to module singleton)
        catalog         — optional system catalog with offline statistics

    Returns:
        OperatorCostVector with all four cost dimensions populated.
    """
    log = log or default_log

    dispatch = {
        Op.I: _profile_isolate,
        Op.TRANSFORM: _profile_transform,
        Op.COMPOSE: _profile_compose,
        Op.RANK: _profile_rank,
        Op.UNION: _profile_union,
        Op.DIFF: _profile_diff,
        Op.AGGREGATE: _profile_aggregate,
        Op.VERIFY: _profile_verify,
        Op.DECOMPOSE: _profile_decompose,
    }

    fn = dispatch.get(node.op, _profile_default)
    return fn(node, variant, op_id, upstream_tokens, corpus, log, catalog)


# ── I(q, C) ────────────────────────────────────────────────────────


def _profile_isolate(node, variant, op_id, upstream_tokens, corpus, log, catalog):
    top_k = node.params.get("top_k", log.get("I", "default_top_k", default=10))
    if catalog is not None:
        chunk_toks = catalog.avg_chunk_tokens(default=180.0)
    else:
        chunk_toks = log.get("I", "avg_chunk_tokens", default=180)
    token_cost = top_k * chunk_toks

    if catalog is not None:
        density = catalog.density_for_query(node.params.get("query", ""), default=0.2)
        quality_risk = max(0.0, min(1.0, density))
    else:
        # Selectivity risk: if we have no corpus, assume moderate noise
        quality_risk = 0.2

    return OperatorCostVector(
        op_id=op_id,
        variant=variant,
        token_cost=token_cost,
        call_cost=1.0,  # one retrieval call
        latency_cost=1.0,  # one sequential step
        quality_risk=quality_risk,
        key_metric=f"top_k={top_k} × {chunk_toks} tokens/chunk",
    )


# ── TRANSFORM ──────────────────────────────────────────────────────


def _profile_transform(node, variant, op_id, upstream_tokens, corpus, log, catalog):
    schema = node.params.get("schema", "default")

    # Identity transform: free
    if variant == "IdentityTransform" or not schema.strip():
        return OperatorCostVector(
            op_id=op_id,
            variant=variant,
            token_cost=0.0,
            call_cost=0.0,
            latency_cost=0.0,
            quality_risk=0.0,
            key_metric="identity — no cost",
        )

    # Look up compression ratio from history
    subkey = _schema_subkey(schema)
    ratio = log.get("TRANSFORM", subkey, "avg", default=0.30)

    # ExtractiveCompress: no LLM call
    if variant == "ExtractiveCompress":
        return OperatorCostVector(
            op_id=op_id,
            variant=variant,
            token_cost=upstream_tokens * ratio,
            call_cost=0.0,
            latency_cost=0.5,  # fast local scoring
            quality_risk=max(0.0, 0.5 - ratio),  # aggressive compress → more loss
            key_metric=f"extractive ratio={ratio:.2f}",
        )

    # LLMSummarize: one LLM call
    return OperatorCostVector(
        op_id=op_id,
        variant=variant,
        token_cost=upstream_tokens * ratio,
        call_cost=1.0,
        latency_cost=1.0,
        quality_risk=max(0.0, 0.3 - ratio),  # very aggressive → hallucination risk
        key_metric=f"llm summarize ratio={ratio:.2f}",
    )


# ── COMPOSE ────────────────────────────────────────────────────────


def _profile_compose(node, variant, op_id, upstream_tokens, corpus, log, catalog):
    condition = node.params.get("condition", "")

    # Estimate semantic gap between the two input queries
    if len(node.inputs) == 2 and corpus is not None:
        try:
            q1 = node.inputs[0].params.get("query", condition)
            q2 = node.inputs[1].params.get("query", condition)
            e1 = corpus.embed(q1[:256])
            e2 = corpus.embed(q2[:256])
            gap = max(0.0, 1.0 - _cosine(e1, e2))
        except Exception:
            gap = 0.5
    else:
        gap = 0.5  # conservative default

    if variant == "ConcatCompose":
        return OperatorCostVector(
            op_id=op_id,
            variant=variant,
            token_cost=upstream_tokens,
            call_cost=0.0,
            latency_cost=0.0,
            quality_risk=gap
            * 0.3,  # concat doesn't reason, but large gap = downstream risk
            key_metric=f"concat semantic_gap={gap:.2f}",
        )

    # LLMCompose / KeyMatchCompose
    call_cost = 1.0 if variant == "LLMCompose" else 0.0
    return OperatorCostVector(
        op_id=op_id,
        variant=variant,
        token_cost=upstream_tokens,
        call_cost=call_cost,
        latency_cost=call_cost,
        quality_risk=gap * 0.6,  # high gap = hard join = quality risk
        key_metric=f"semantic_gap={gap:.2f}",
    )


# ── RANK ───────────────────────────────────────────────────────────


def _profile_rank(node, variant, op_id, upstream_tokens, corpus, log, catalog):
    top_k = node.params.get("top_k", 5)

    # After ranking, only top_k chunks remain
    # Estimate output as fraction of input
    keep_ratio = min(1.0, top_k / max(1, upstream_tokens / 180))
    output_toks = upstream_tokens * keep_ratio

    call_cost = 1.0 if variant == "CrossEncoderRank" else 0.0
    latency = 0.5 if variant == "SimilarityRank" else 1.0

    return OperatorCostVector(
        op_id=op_id,
        variant=variant,
        token_cost=output_toks,
        call_cost=call_cost,
        latency_cost=latency,
        quality_risk=max(0.0, 0.1 - keep_ratio * 0.1),  # low risk, rank helps quality
        key_metric=f"top_k={top_k} keep_ratio={keep_ratio:.2f}",
    )


# ── UNION ──────────────────────────────────────────────────────────


def _profile_union(node, variant, op_id, upstream_tokens, corpus, log, catalog):
    # UNION merges N inputs — token cost is additive
    # upstream_tokens here is the sum of all branch estimates
    return OperatorCostVector(
        op_id=op_id,
        variant=variant,
        token_cost=upstream_tokens,
        call_cost=0.0,
        latency_cost=0.0,  # branches run in parallel
        quality_risk=0.1,  # possible redundancy before DIFF
        key_metric=f"merged_tokens={upstream_tokens:.0f}",
    )


# ── DIFF ───────────────────────────────────────────────────────────


def _profile_diff(node, variant, op_id, upstream_tokens, corpus, log, catalog):
    if catalog is not None:
        overlap_ratio = catalog.estimated_overlap_ratio(default=0.20)
    else:
        # Estimate 20% overlap removed on average
        overlap_ratio = 0.20
    output_toks = upstream_tokens * (1.0 - overlap_ratio)

    call_cost = 0.0 if variant == "ExactDiff" else 0.2  # SemanticDiff uses embeddings

    return OperatorCostVector(
        op_id=op_id,
        variant=variant,
        token_cost=output_toks,
        call_cost=call_cost,
        latency_cost=0.2,
        quality_risk=0.05,  # dedup reduces redundancy risk
        key_metric=f"estimated_overlap={overlap_ratio:.0%}",
    )


# ── AGGREGATE ──────────────────────────────────────────────────────


def _profile_aggregate(node, variant, op_id, upstream_tokens, corpus, log, catalog):
    model_id = node.params.get("model_id")
    context_window = node.params.get("context_window")
    if context_window is None and catalog is not None:
        context_window = catalog.context_window(
            model_id=model_id, default=_DEFAULT_CONTEXT_WINDOW
        )
    if context_window is None:
        context_window = _DEFAULT_CONTEXT_WINDOW

    saturation = upstream_tokens / context_window
    output_toks = log.get("AGGREGATE", "avg_output_tokens", default=256)

    # Quality risk rises sharply above 70% context saturation
    if saturation > 0.7:
        quality_risk = min(1.0, (saturation - 0.7) * 3.0)
    else:
        quality_risk = 0.05

    # HierarchicalGenerate: N batch calls + 1 synthesis call
    if variant == "HierarchicalGenerate":
        batch_tokens = node.params.get("batch_tokens", 8_000)
        n_batches = max(1, int(upstream_tokens / batch_tokens))
        call_cost = float(n_batches + 1)
        latency_cost = float(n_batches + 1)  # sequential batches
    else:
        call_cost = 1.0
        latency_cost = 1.0

    return OperatorCostVector(
        op_id=op_id,
        variant=variant,
        token_cost=upstream_tokens + output_toks,
        call_cost=call_cost,
        latency_cost=latency_cost,
        quality_risk=quality_risk,
        key_metric=f"saturation={saturation:.1%}",
        saturation=saturation,
    )


# ── VERIFY ─────────────────────────────────────────────────────────


def _profile_verify(node, variant, op_id, upstream_tokens, corpus, log, catalog):
    # Determine task type from constraints param for failure rate lookup
    constraints = node.params.get("constraints", "").lower()
    if "multi" in constraints or "hop" in constraints or "compose" in constraints:
        task_type = "multi_hop"
    elif "struct" in constraints or "json" in constraints:
        task_type = "structured"
    else:
        task_type = "simple_qa"

    failure_prob = log.get("VERIFY", task_type, "avg", default=0.15)

    # Verify adds one sequential step and may retry
    # But it reduces net quality risk by catching errors
    return OperatorCostVector(
        op_id=op_id,
        variant=variant,
        token_cost=upstream_tokens * 0.5,  # re-reads evidence
        call_cost=1.0 + failure_prob,  # base + expected retries
        latency_cost=1.0 + failure_prob,
        quality_risk=-failure_prob * 0.5,  # negative: verify reduces risk
        key_metric=f"task={task_type} failure_prob={failure_prob:.2f}",
    )


# ── DECOMPOSE ──────────────────────────────────────────────────────


def _profile_decompose(node, variant, op_id, upstream_tokens, corpus, log, catalog):
    query = node.params.get("query", "")
    branching_factor = _estimate_branches(query)

    return OperatorCostVector(
        op_id=op_id,
        variant=variant,
        token_cost=branching_factor * 50.0,  # prompt tokens per sub-query
        call_cost=1.0,
        latency_cost=1.0,
        quality_risk=0.1 * branching_factor,  # over-decomposition risk
        key_metric=f"estimated_branches={branching_factor}",
    )


# ── Default (unknown ops) ──────────────────────────────────────────


def _profile_default(node, variant, op_id, upstream_tokens, corpus, log, catalog):
    return OperatorCostVector(
        op_id=op_id,
        variant=variant,
        token_cost=upstream_tokens,
        call_cost=0.0,
        latency_cost=0.0,
        quality_risk=0.0,
        key_metric="unknown op — passthrough estimate",
    )


# ── Helpers ────────────────────────────────────────────────────────


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    raw = dot / (na * nb) if na and nb else 0.0
    return max(0.0, min(1.0, raw))


def _schema_subkey(schema: str) -> str:
    schema = schema.lower()
    if any(k in schema for k in ("extract", "list", "field", "struct")):
        return "extract"
    if any(k in schema for k in ("summar", "brief", "concis")):
        return "summarize"
    return "default"


def _estimate_branches(query: str) -> int:
    """Rough estimate of sub-queries from conjunctions in the query."""
    markers = ["and", "also", "additionally", "as well as", "furthermore", ","]
    count = sum(1 for m in markers if m in query.lower())
    return max(2, min(count + 1, 6))  # clamp to [2, 6]
