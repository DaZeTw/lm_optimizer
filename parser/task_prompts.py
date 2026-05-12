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

from planner.variant_candidates import CANDIDATE_VARIANTS, VARIANT_DESCRIPTIONS
from parser.operator_candidates import LOGICAL_OPERATOR_DESCRIPTIONS

# ---------------------------------------------------------------------------
# Variant catalog — built at import time from variant_candidates.py so the
# LLM sees exactly what physical variants exist and what the defaults are.
# ---------------------------------------------------------------------------


def _build_variant_catalog() -> str:
    lines: list[str] = []

    for op, variants in CANDIDATE_VARIANTS.items():
        lines.append(f"\n{op.value}")
        for i, variant in enumerate(variants):
            meta = VARIANT_DESCRIPTIONS.get(variant, {})
            default_tag = " default" if i == 0 else ""
            description = meta.get("description", "No description provided.")
            use_when = meta.get("use_when", "No usage guidance provided.")
            cost = meta.get("cost", "unknown")

            lines.append(
                f"  - {variant}{default_tag}\n"
                f"    description: {description}\n"
                f"    use_when: {use_when}\n"
                f"    cost: {cost}"
            )

    return "\n".join(lines)


_VARIANT_CATALOG = _build_variant_catalog()

def _build_logical_catalog() -> str:
    lines: list[str] = []

    for op, meta in LOGICAL_OPERATOR_DESCRIPTIONS.items():
        lines.append(f"\n{op.value}")
        lines.append(f"  description: {meta['description']}")
        lines.append(f"  inputs: {meta['inputs']}")
        lines.append(f"  outputs: {meta['outputs']}")
        lines.append(f"  use_when: {meta['use_when']}")

    return "\n".join(lines)


_LOGICAL_CATALOG = _build_logical_catalog()

# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES: list[dict] = [
    {
        "task": "Single-document question answering over scientific papers",
        "tst": """\
LOGICAL SKELETON
AGGREGATE(
  RANK(I({QUERY}), criterion="{RANK_CRITERION}"),
  goal="{AGGREGATION_GOAL}"
)

PHYSICAL POLICY
I_1         | I         | HybridRetrieve    | top_k=10
RANK_1      | RANK      | CrossEncoderRank  | top_k=5
AGGREGATE_1 | AGGREGATE | DirectGenerate    | {}

ADAPTATION POLICY
mutable_slots: QUERY, RANK_CRITERION, AGGREGATION_GOAL
immutable_slots: none
mutable_ops: I_1, RANK_1, AGGREGATE_1
immutable_ops: none
allowed_rewrites: may tune retrieval variant and top_k based on corpus size
allowed_rewrites: may add TRANSFORM before AGGREGATE when the final answer requires structured fields or compact summaries
forbidden_rewrites: must not remove I_1 when external evidence is required
forbidden_rewrites: must not use AGGREGATE when the task asks for exact copied evidence""",
    },

    {
        "task": "Exact evidence retrieval from a scientific paper",
        "tst": """\
LOGICAL SKELETON
RANK(
  I({QUERY}),
  criterion="{RANK_CRITERION}"
)

PHYSICAL POLICY
I_1    | I    | HybridRetrieve    | top_k=20
RANK_1 | RANK | CrossEncoderRank  | top_k=5

ADAPTATION POLICY
mutable_slots: QUERY, RANK_CRITERION
immutable_slots: none
mutable_ops: I_1, RANK_1
immutable_ops: none
allowed_rewrites: may tune retrieval top_k when evidence is too broad or too narrow
allowed_rewrites: may change RANK criterion to sentence-level support when exact evidence is required
forbidden_rewrites: must not add AGGREGATE because exact evidence must not be paraphrased
forbidden_rewrites: must not add TRANSFORM unless the transform preserves verbatim source text""",
    },

    {
        "task": "Structured information extraction from retrieved context",
        "tst": """\
LOGICAL SKELETON
TRANSFORM(
  RANK(I({QUERY}), criterion="{RANK_CRITERION}"),
  schema="{EXTRACTION_SCHEMA}"
)

PHYSICAL POLICY
I_1          | I         | HybridRetrieve    | top_k=10
RANK_1       | RANK      | CrossEncoderRank  | top_k=5
TRANSFORM_1  | TRANSFORM | StructuredExtract | schema={EXTRACTION_SCHEMA}

ADAPTATION POLICY
mutable_slots: QUERY, RANK_CRITERION, EXTRACTION_SCHEMA
immutable_slots: none
mutable_ops: I_1, RANK_1, TRANSFORM_1
immutable_ops: none
allowed_rewrites: may tune retrieval and ranking top_k based on context size
allowed_rewrites: may use LLMSummarize instead of StructuredExtract when the requested output is a short natural-language summary
allowed_rewrites: may add AGGREGATE after TRANSFORM only when extracted fields need narrative explanation
forbidden_rewrites: must not replace terminal TRANSFORM with AGGREGATE when the task asks for structured extraction""",
    },

    {
        "task": "Multi-paper comparison using a single retrieval path",
        "tst": """\
LOGICAL SKELETON
AGGREGATE(
  TRANSFORM(
    RANK(I({QUERY}), criterion="{RANK_CRITERION}"),
    schema="{COMPARISON_SCHEMA}"
  ),
  goal="{AGGREGATION_GOAL}"
)

PHYSICAL POLICY
I_1          | I         | HybridRetrieve        | top_k=20
RANK_1       | RANK      | CrossEncoderRank      | top_k=10
TRANSFORM_1  | TRANSFORM | StructuredExtract     | schema={COMPARISON_SCHEMA}
AGGREGATE_1  | AGGREGATE | HierarchicalGenerate  | {}

ADAPTATION POLICY
mutable_slots: QUERY, RANK_CRITERION, COMPARISON_SCHEMA, AGGREGATION_GOAL
immutable_slots: none
mutable_ops: I_1, RANK_1, TRANSFORM_1, AGGREGATE_1
immutable_ops: none
allowed_rewrites: may use DirectGenerate instead of HierarchicalGenerate when evidence is short
allowed_rewrites: may skip TRANSFORM when the comparison can be directly synthesized from ranked evidence
forbidden_rewrites: must not use AGGREGATE without retrieved evidence
forbidden_rewrites: must not use this template when explicit pairwise source alignment is required""",
    },
]
# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

TASK_SYSTEM_PROMPT = f"""\
You are a task-level query planner for a long-context reasoning system.

Your job: given a task description, evaluation criteria, and sample queries,
produce a Task Strategy Template (TST) — a reusable strategy for this task.

## CORE RULE
The logical skeleton must represent the GENERAL REASONING STRUCTURE of the task,
not the number of queries or documents.

It should capture:
- what information to retrieve
- how to process or extract it
- how to connect evidence (if needed)
- how to produce the final answer

Adaptation should be MINIMAL and LOCAL.

The query-level planner is allowed to:
- fill slots (queries, schema, goals, criteria)
- tune parameters (e.g., top_k)
- switch variants within the same operator
- insert lightweight steps (e.g., TRANSFORM after I, RANK before AGGREGATE)

The query-level planner must NOT:
- change the overall reasoning structure
- remove core operators (e.g., COMPOSE, VERIFY)
- expand the plan into many parallel I(...) branches
- replace structured reasoning (COMPOSE) with flat merging (UNION)

## Logical operators (same set used by the query-level planner)
{_LOGICAL_CATALOG}

## Physical variants available per operator
{_VARIANT_CATALOG}

## Reasoning Instructions

You should think step-by-step and reason freely to design the best Task Strategy Template.

You are encouraged to:
- explore different possible logical structures
- compare alternative operator compositions
- justify why certain operators are used (or not used)
- refine the structure before finalizing

Your reasoning should consider:
- the type of task (retrieval QA, multi-hop, comparison, decomposition, etc.)
- the minimal operator structure needed
- trade-offs between simplicity, accuracy, and cost
- reusability across different queries

You MUST write out your full reasoning before giving the final answer.

## Final Output Requirement

After your reasoning, you MUST output the final Task Strategy Template
in a clearly separated section using the exact delimiter:

=== FINAL TST ===

Only the content AFTER this delimiter will be used by the system.
Do NOT include any explanation after the delimiter.

## Output format (for the FINAL TST only)

The final TST must appear AFTER the delimiter "=== FINAL TST ===".

Output exactly three sections, separated by blank lines, with the exact headers shown.
No markdown fences. No explanation. Nothing else.

LOGICAL SKELETON
<algebraic expression using the operators above, with {{SLOT_NAME}} placeholders>
<every evidence leaf should be I({{SLOT_NAME}}) unless using DECOMPOSE({{QUERY}})>
<do not create one I(...) branch per sample query>
<use a small reusable skeleton with 1-3 main evidence paths>
<include TRANSFORM or COMPOSE when the task requires extraction, comparison, linking, or synthesis>

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
