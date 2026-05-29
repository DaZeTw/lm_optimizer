"""System prompt and user message builders for task-level planning.

Task-level planning runs once per task family and produces a Task Strategy
Template (TST) as structured text with two labelled sections:

    LOGICAL SKELETON
        The fixed operator DAG shape, written as an algebraic expression with
        {SLOT_NAME} placeholders where query-specific values will be filled in.

    ADAPTATION POLICY
        Labelled sub-fields (mutable_slots, immutable_slots, mutable_ops,
        immutable_ops, allowed_rewrites, forbidden_rewrites), one value per line.

The LLM output is parsed by ``expr_parser.parse_task_strategy()``, which
returns a plain dict  {"logical_skeleton": ..., "adaptation_policy": ...}
with no dataclasses involved.
"""

from __future__ import annotations

from parser.operator_candidates import LOGICAL_OPERATOR_DESCRIPTIONS

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
        "task": "Concise answer generation from retrieved scientific context",
        "tst": """\
LOGICAL SKELETON
AGGREGATE(
  TRANSFORM(
    RANK(
      I({QUERY}),
      criterion="{RANK_CRITERION}"
    ),
    schema="{EVIDENCE_NOTES_SCHEMA}"
  ),
  goal="{ANSWER_GOAL}"
)

ADAPTATION POLICY
mutable_slots: QUERY, RANK_CRITERION, EVIDENCE_NOTES_SCHEMA, ANSWER_GOAL
immutable_slots: none
mutable_ops: I_1, RANK_1, TRANSFORM_1, AGGREGATE_1
immutable_ops: none
allowed_rewrites: may specialize QUERY using entities, methods, datasets, metrics, and section cues from the current question
allowed_rewrites: may specialize RANK_CRITERION to prioritize chunks likely to contain the needed evidence
allowed_rewrites: may specialize EVIDENCE_NOTES_SCHEMA to extract only the information needed before answer generation
allowed_rewrites: may remove RANK when retrieval already returns a narrow and low-noise context
forbidden_rewrites: must not remove I_1 when document evidence is required
forbidden_rewrites: must not use RANK as a substitute for extracting information from chunks
forbidden_rewrites: must not use AGGREGATE to answer directly from raw chunks when intermediate extraction is required
forbidden_rewrites: must not invent claims not supported by transformed evidence notes""",
    },

    {
        "task": "Compare methods described in a scientific paper",
        "tst": """\
LOGICAL SKELETON
AGGREGATE(
  TRANSFORM(
    RANK(
      I({QUERY}),
      criterion="{RANK_CRITERION}"
    ),
    schema="{METHOD_COMPARISON_SCHEMA}"
  ),
  goal="{COMPARISON_GOAL}"
)

ADAPTATION POLICY
mutable_slots: QUERY, RANK_CRITERION, METHOD_COMPARISON_SCHEMA, COMPARISON_GOAL
immutable_slots: none
mutable_ops: I_1, RANK_1, TRANSFORM_1, AGGREGATE_1
immutable_ops: none
allowed_rewrites: may specialize QUERY using method names, baselines, datasets, metrics, and comparison cues from the current question
allowed_rewrites: may specialize RANK_CRITERION to prioritize chunks that mention both compared methods or their evaluation results
allowed_rewrites: may specialize METHOD_COMPARISON_SCHEMA with fields such as method_name, input, output, assumption, metric, result, limitation, and difference
allowed_rewrites: may remove AGGREGATE when the extracted comparison table is already the requested final output
forbidden_rewrites: must not remove I_1 when paper evidence is required
forbidden_rewrites: must not use RANK as a substitute for extracting comparison fields from chunks
forbidden_rewrites: must not skip TRANSFORM when the task requires explicit method attributes or comparison fields
forbidden_rewrites: must not compare methods using unsupported details outside the retrieved evidence""",
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

The TST should contain a stable task-level skeleton plus an explicit adaptation
budget. The skeleton is the default reasoning route; the adaptation policy
defines how the query-level planner may expand, specialize, or simplify it for
the current query and evaluation criteria.

The query-level planner is allowed to:
- fill slots (queries, schema, goals, criteria)
- expand the skeleton with operators allowed by the adaptation policy
- specialize retrieval, ranking, extraction, verification, and generation goals
- simplify the skeleton when the current query needs fewer reasoning steps

The query-level planner must NOT:
- ignore the adaptation policy
- remove core operators unless an allowed rewrite permits it
- expand the plan into many parallel I(...) branches
- replace structured reasoning (COMPOSE) with flat merging (UNION)

## Logical operators (same set used by the query-level planner)
{_LOGICAL_CATALOG}

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
- where the query-level planner needs flexibility
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

Output exactly two sections, separated by blank lines, with the exact headers shown.
No markdown fences. No explanation. Nothing else.

LOGICAL SKELETON
<algebraic expression using the operators above, with {{SLOT_NAME}} placeholders>
<every evidence leaf should be I({{SLOT_NAME}}) unless using DECOMPOSE({{QUERY}})>
<do not create one I(...) branch per sample query>
<use a small reusable skeleton with 1-3 main evidence paths>
<include TRANSFORM or COMPOSE when the task requires extraction, comparison, linking, or synthesis>

ADAPTATION POLICY
mutable_slots: <comma-separated slot names the query planner MUST fill per query>
immutable_slots: <comma-separated slot names locked to their TST value>
mutable_ops: <comma-separated logical op_ids the query planner may expand, specialize, remove, or preserve according to allowed_rewrites>
immutable_ops: <comma-separated logical op_ids that should stay structurally stable unless experience later proves otherwise>
allowed_rewrites: <one query-level adaptation or expansion rule — repeat key for multiple rules>
forbidden_rewrites: <one prohibited rewrite or safety boundary — repeat key for multiple rules>
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
        "Output a Task Strategy Template using the exact two-section format."
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
        "Output ONLY the revised TST in the exact two-section format."
    )

    return "\n\n".join(parts)
