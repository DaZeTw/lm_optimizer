"""System prompt and user message builders for task-level planning.

Task-level planning runs once per task family and produces a Task Strategy
Template (TST) as structured text with three labelled sections:

    LOGICAL SKELETON
        The fixed operator DAG shape, written as an algebraic expression with
        {SLOT_NAME} placeholders where query-specific values will be filled in.

    PHYSICAL POLICY
        One line per operator node:  op_id | op_name | variant | params
        Variants must come from the catalog embedded in the system prompt.

    ADAPTATION POLICY
        Labelled sub-fields (mutable_slots, immutable_slots, mutable_ops,
        immutable_ops, allowed_rewrites, forbidden_rewrites), one value per line.

The LLM output is parsed by ``expr_parser.parse_task_strategy()``, which
returns a plain dict  {"logical_skeleton": ..., "physical_policy": ...,
"adaptation_policy": ...}  with no dataclasses involved.
"""

from __future__ import annotations

from planner.variant_candidates import CANDIDATE_VARIANTS

# ---------------------------------------------------------------------------
# Variant catalog — built at import time from variant_candidates.py so the
# LLM sees exactly what physical variants exist and what the defaults are.
# ---------------------------------------------------------------------------


def _build_variant_catalog() -> str:
    lines: list[str] = []
    for op, variants in CANDIDATE_VARIANTS.items():
        default = variants[0]
        rest = ", ".join(variants[1:]) if len(variants) > 1 else "—"
        lines.append(f"  {op.value:<12} default={default!r:<28} alternatives: {rest}")
    return "\n".join(lines)


_VARIANT_CATALOG = _build_variant_catalog()


# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES: list[dict] = [
    {
        "task": "Single-document question answering over scientific papers",
        "tst": """\
LOGICAL SKELETON
VERIFY(
  AGGREGATE(
    RANK(I({QUERY}), criterion="{RANK_CRITERION}"),
    goal="{AGGREGATION_GOAL}"
  ),
  constraints="{VERIFY_CONSTRAINTS}"
)

PHYSICAL POLICY
I_1        | I        | HybridRetrieve    | {}
RANK_1     | RANK     | CrossEncoderRank  | top_k=5
AGGREGATE_1| AGGREGATE| DirectGenerate    | {}
VERIFY_1   | VERIFY   | CitationVerify    | {}

ADAPTATION POLICY
mutable_slots: QUERY, RANK_CRITERION, AGGREGATION_GOAL
immutable_slots: VERIFY_CONSTRAINTS
mutable_ops: RANK_1
immutable_ops: VERIFY_1
allowed_rewrites: may insert TRANSFORM after I_1 if retrieved chunks are verbose
forbidden_rewrites: must not drop VERIFY""",
    },
    {
        "task": "Multi-document comparative analysis across 10 papers",
        "tst": """\
LOGICAL SKELETON
VERIFY(
  AGGREGATE(
    COMPOSE(
      TRANSFORM(I({QUERY_A}), schema="{SCHEMA}"),
      TRANSFORM(I({QUERY_B}), schema="{SCHEMA}"),
      condition="{COMPOSE_CONDITION}"
    ),
    goal="{AGGREGATION_GOAL}"
  ),
  constraints="{VERIFY_CONSTRAINTS}"
)

PHYSICAL POLICY
I_1        | I        | DenseRetrieve       | {}
I_2        | I        | DenseRetrieve       | {}
TRANSFORM_1| TRANSFORM| ExtractiveCompress  | schema=metrics list
TRANSFORM_2| TRANSFORM| ExtractiveCompress  | schema=metrics list
COMPOSE_1  | COMPOSE  | LLMCompose          | {}
AGGREGATE_1| AGGREGATE| HierarchicalGenerate| {}
VERIFY_1   | VERIFY   | NliVerify           | {}

ADAPTATION POLICY
mutable_slots: QUERY_A, QUERY_B, SCHEMA, COMPOSE_CONDITION, AGGREGATION_GOAL
immutable_slots: VERIFY_CONSTRAINTS
mutable_ops: TRANSFORM_1, TRANSFORM_2
immutable_ops: COMPOSE_1, VERIFY_1
allowed_rewrites: may expand COMPOSE to UNION of more I() branches for >2 documents
allowed_rewrites: may add RANK before AGGREGATE if evidence set is large
forbidden_rewrites: must not drop VERIFY
forbidden_rewrites: must not replace COMPOSE with bare UNION""",
    },
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

TASK_SYSTEM_PROMPT = f"""\
You are a task-level query planner for a long-context reasoning system.

Your job: given a task description, evaluation criteria, and sample queries,
produce a Task Strategy Template (TST) — a reusable strategy that guides all
query-level planning for this task family.

## Logical operators (same set used by the query-level planner)
- I("query")                              — retrieve relevant evidence
- TRANSFORM(child, schema="...")          — compress / extract evidence
- COMPOSE(left, right, condition="...")   — join two evidence sets semantically
- UNION(child1, child2, ...)              — merge multiple evidence sets
- DIFF(base, subtract)                    — remove redundant / conflicting evidence
- RANK(child, criterion="...")            — sort evidence by relevance
- AGGREGATE(child, goal="...")            — synthesise the final answer
- VERIFY(child, constraints="...")        — check grounding and correctness
- DECOMPOSE("query")                      — split into independent sub-tasks (sparingly)

## Physical variants available per operator
(first entry = default / cheapest; use only these — no others)
{_VARIANT_CATALOG}

## Output format
Output exactly three sections, separated by blank lines, with the exact headers shown.
No markdown fences. No explanation. Nothing else.

LOGICAL SKELETON
<algebraic expression using the operators above, with {{SLOT_NAME}} placeholders>
<every leaf must be I({{SLOT_NAME}})>
<must end with AGGREGATE(...) or VERIFY(AGGREGATE(...), ...)>

PHYSICAL POLICY
<one line per operator node in the skeleton>
<format per line: op_id | op_name | variant | params>
<use {{}} for empty params; use key=value pairs for non-empty params>
<variant must come from the catalog above>

ADAPTATION POLICY
mutable_slots: <comma-separated slot names the query planner MUST fill per query>
immutable_slots: <comma-separated slot names locked to their TST value>
mutable_ops: <comma-separated op_ids whose variant/params the query planner may tune>
immutable_ops: <comma-separated op_ids that must stay exactly as specified>
allowed_rewrites: <one structural rewrite rule — repeat key for multiple rules>
forbidden_rewrites: <one prohibited rewrite rule — repeat key for multiple rules>
"""


# ---------------------------------------------------------------------------
# User message builders
# ---------------------------------------------------------------------------


def build_task_user_message(
    task_description: str,
    evaluation_criteria: str,
    sample_queries: list[str] | None = None,
    prior_heuristics: list[str] | None = None,
    context_window: int | None = None,
    avg_chunk_tokens: float | None = None,
) -> str:
    """Build the user message for the initial (cold-start) TST generation call."""
    parts: list[str] = []

    parts.append(f"## Task Description\n{task_description}")
    parts.append(f"## Evaluation Criteria\n{evaluation_criteria}")

    if sample_queries:
        numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(sample_queries))
        parts.append(f"## Sample Queries\n{numbered}")
    else:
        parts.append("## Sample Queries\n(none provided)")

    if prior_heuristics:
        parts.append(
            "## Prior Heuristics\n" + "\n".join(f"- {h}" for h in prior_heuristics)
        )

    if context_window is not None or avg_chunk_tokens is not None:
        stat_lines: list[str] = []
        if context_window is not None:
            stat_lines.append(f"- Model context window : {context_window:,} tokens")
        if avg_chunk_tokens is not None:
            stat_lines.append(f"- Avg chunk size       : {avg_chunk_tokens:.0f} tokens")
        parts.append("## Corpus Context\n" + "\n".join(stat_lines))

    shots = "\n\n".join(
        f"Task: {ex['task']}\nTST:\n{ex['tst']}" for ex in _FEW_SHOT_EXAMPLES
    )
    parts.append(f"## Examples of task → TST\n{shots}")

    parts.append(
        "## Your Job\n"
        "Infer the common reasoning structure across the sample queries and the task.\n"
        "Output a Task Strategy Template using the exact three-section format."
    )

    return "\n\n".join(parts)


def build_task_revise_message(
    task_description: str,
    evaluation_criteria: str,
    prev_tst_text: str,
    feedback_blocks: str,
    num_samples: int,
    sample_queries: list[str] | None = None,
    context_window: int | None = None,
    avg_chunk_tokens: float | None = None,
) -> str:
    """Build the user message for a TST revision call (after multi-sample feedback)."""
    parts: list[str] = []

    parts.append(f"## Task Description\n{task_description}")
    parts.append(f"## Evaluation Criteria\n{evaluation_criteria}")

    if sample_queries:
        numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(sample_queries))
        parts.append(f"## Sample Queries\n{numbered}")

    if context_window is not None or avg_chunk_tokens is not None:
        stat_lines: list[str] = []
        if context_window is not None:
            stat_lines.append(f"- Model context window : {context_window:,} tokens")
        if avg_chunk_tokens is not None:
            stat_lines.append(f"- Avg chunk size       : {avg_chunk_tokens:.0f} tokens")
        parts.append("## Corpus Context\n" + "\n".join(stat_lines))

    parts.append(f"## Current Task Strategy Template\n{prev_tst_text}")
    parts.append(
        f"## Execution Feedback across {num_samples} sample(s)\n{feedback_blocks}"
    )

    parts.append(
        "## Your Job\n"
        "Revise the Task Strategy Template to improve accuracy and/or reduce cost.\n"
        "Keep changes minimal — only adjust what the feedback shows is wrong.\n"
        "Output ONLY the revised TST in the exact three-section format."
    )

    return "\n\n".join(parts)
