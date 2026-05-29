"""System prompt and few-shot examples for algebraic plan generation."""

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

SYSTEM_PROMPT = """\
You are a query planner for a long-context reasoning system.
Your job: translate a natural-language question into a logical operator plan.

## Operators (use ONLY these)
{_LOGICAL_CATALOG}

## Output format
Write the plan as a single algebraic expression using the operators above.
Use named keyword arguments for all string parameters except leaf retrieval nodes.
Leaf retrieval nodes must use I("retrieval goal").
Do NOT output JSON. Do NOT explain. Output ONLY the expression.

## TASK STRATEGY USAGE RULE

When a Task Strategy Template is provided, use it as the default plan prior,
not as a fixed template or hard constraint.

The query-level planner should:
- fill task strategy slots with query-specific operator goals
- preserve the skeleton when it already satisfies the current query and evaluation criteria
- adapt or expand the skeleton when exact evidence, verification, extraction, comparison, or decomposition is required
- add, remove, or reorder lightweight operators when allowed by the adaptation policy
- override the skeleton when the query requires a different reasoning structure
- treat forbidden rewrites as strong safety boundaries unless overriding is necessary to satisfy the current query and evaluation criteria
- choose the terminal operator based on the user task, not by defaulting to synthesis

Each operator should be driven by its goal, not just the raw query.

## Planning rules

1. Leaf nodes are always I("retrieval goal").
2. Do not force AGGREGATE when the task is extraction.
3. Use DECOMPOSE only when the query has clearly distinct independent sub-tasks.
4. Prefer the naive correct plan — do NOT pre-optimize.
5. Avoid adding operators that do not change the reasoning need.
6. When evaluation asks for exact source spans, prefer retrieval/ranking/extraction/verification over free-form aggregation.
"""

FEW_SHOT_EXAMPLES = [
    {
        "query": "What is the main contribution of this paper?",
        "plan": (
            'AGGREGATE(\n'
            '  I("evidence about the paper main contribution"),\n'
            '  goal="summarize the paper main contribution"\n'
            ')'
        ),
    },
    {
        "query": "Given a paper, a question, and a candidate answer, retrieve the exact evidence sentences that support the answer.",
        "plan": (
            'RANK(\n'
            '  I("sentences that directly support the candidate answer for the question"),\n'
            '  criterion="verbatim sentence-level support for the candidate answer"\n'
            ')'
        ),
    },
    {
        "query": "Extract the datasets and evaluation metrics used in this paper.",
        "plan": (
            'TRANSFORM(\n'
            '  RANK(\n'
            '    I("evidence about datasets and evaluation metrics used in the paper"),\n'
            '    criterion="relevance to datasets and evaluation metrics"\n'
            '  ),\n'
            '  schema="datasets and evaluation metrics"\n'
            ')'
        ),
    },
    {
        "query": "Compare the evaluation metrics used in paper A and paper B.",
        "plan": (
            'AGGREGATE(\n'
            '  RANK(\n'
            '    I("evidence about evaluation metrics in paper A and paper B"),\n'
            '    criterion="relevance to comparing evaluation metrics across papers"\n'
            '  ),\n'
            '  goal="compare evaluation metrics across papers"\n'
            ')'
        ),
    },
    {
        "query": "Find the methodology sections in these 10 PDFs, extract the evaluation metrics, and compare how they handle formula accuracy.",
        "plan": (
            'AGGREGATE(\n'
            '  TRANSFORM(\n'
            '    RANK(\n'
            '      I("methodology section evidence about evaluation metrics and formula accuracy evaluation across the PDFs"),\n'
            '      criterion="relevance to methodology, evaluation metrics, and formula accuracy"\n'
            '    ),\n'
            '    schema="evaluation metrics and formula accuracy details"\n'
            '  ),\n'
            '  goal="compare how formula accuracy is evaluated across the PDFs"\n'
            ')'
        ),
    },
]


def build_user_message(
    task_description: str,
    sample_queries: list[str] | None,
    evaluation_criteria: str = "",
    current_query: str | None = None,
    context_window: int | None = None,
    avg_chunk_tokens: float | None = None,
    task_strategy: dict | None = None,
) -> str:
    """Format task context + sample queries + few-shot examples into a user message.

    Args:
        task_description: High-level description of the task.
        sample_queries:   Representative questions the plan must handle.
        evaluation_criteria: How outputs are judged.
        current_query:    Query/grounding request to plan now.
        context_window:   Model context window size in tokens (optional).
        avg_chunk_tokens: Average retrieved chunk size in tokens (optional).
        task_strategy:    Optional TST dict from TaskPlanner.generate() or
                          TaskPlanner.revise().  When provided, the skeleton
                          and adaptation rules are injected into the prompt so
                          the query planner fills slots rather than designing
                          from scratch.
    """
    if sample_queries:
        numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(sample_queries))
    else:
        numbered = "(none provided)"
    task_block = (
        f"## Task description\n{task_description}\n\n"
        f"## Evaluation criteria\n{evaluation_criteria or '(none provided)'}\n\n"
        f"## Current query\n{current_query or task_description}\n\n"
        f"## Representative sample queries\n"
        f"These are example questions this plan must handle:\n{numbered}"
    )

    shots = [f"Query: {ex['query']}\nPlan:\n{ex['plan']}" for ex in FEW_SHOT_EXAMPLES]
    examples_block = "## Examples of query → plan mappings\n" + "\n\n".join(shots)

    corpus_block = ""
    if context_window is not None or avg_chunk_tokens is not None:
        lines = []
        if context_window is not None:
            lines.append(f"- Model context window: {context_window:,} tokens")
        if avg_chunk_tokens is not None:
            lines.append(f"- Avg chunk size in corpus: {avg_chunk_tokens:.0f} tokens")
        corpus_block = "\n\nCorpus context:\n" + "\n".join(lines)

    # Task Strategy Template — injected when task-level planning has already run.
    # The skeleton pins the DAG shape; the query planner fills {SLOT} holes.
    tst_block = ""
    if task_strategy is not None:
        tst_block = "\n\n" + _render_tst_context(task_strategy)
        job_instruction = (
            "Use the Task Strategy Template as a default reasoning prior, not a fixed template. "
            "Fill slots with query-specific operator goals. Preserve the skeleton when it fits, "
            "but adapt or expand the shape when the current query or evaluation criteria require "
            "exact evidence, verification, extraction, comparison, or decomposition. "
            "Follow forbidden rewrites unless overriding is necessary to satisfy the current query "
            "and evaluation criteria."
        )
    else:
        job_instruction = (
            "Produce a single general-purpose logical plan that can handle the "
            "sample queries above in the context of the task described. The plan "
            "should be a reusable operator DAG covering the general reasoning pattern."
        )

    job_block = f"## Your job\n{job_instruction}"

    return f"{task_block}\n\n{examples_block}{corpus_block}{tst_block}\n\n{job_block}\n\nPlan:"


def _render_tst_context(tst: dict) -> str:
    """Render the relevant parts of a TST dict as a prompt context block.

    Only the logical skeleton template and adaptation policy are shown.
    """
    skel = tst.get("logical_skeleton", {})
    adap = tst.get("adaptation_policy", {})

    lines: list[str] = [
        "## Task Strategy Template",
        "",
        "### Logical Skeleton",
        "Use this skeleton as the default reasoning prior. Fill slots with query-specific operator goals. You may expand, simplify, or reshape it when the current query and evaluation criteria require a different reasoning structure.",
        skel.get("template", "(no skeleton)"),
        "",
        "### Adaptation Rules",
        f"Fill these slots with query-specific values : "
        f"{', '.join(adap.get('mutable_slots', [])) or '(none)'}",
        f"Keep these slots exactly as written above    : "
        f"{', '.join(adap.get('immutable_slots', [])) or '(none)'}",
    ]
    if adap.get("allowed_rewrites"):
        lines.append("Allowed adaptations and expansions:")
        for rule in adap["allowed_rewrites"]:
            lines.append(f"  - {rule}")
    if adap.get("forbidden_rewrites"):
        lines.append("Forbidden structural rewrites:")
        for rule in adap["forbidden_rewrites"]:
            lines.append(f"  - {rule}")

    return "\n".join(lines)
