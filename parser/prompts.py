"""System prompt and few-shot examples for algebraic plan generation."""

SYSTEM_PROMPT = """\
You are a query planner for a long-context reasoning system.
Your job: translate a natural-language question into a logical operator plan.

## Operators (use ONLY these)
- I("query")                              — retrieve relevant evidence
- TRANSFORM(child, schema="...")          — compress/extract evidence
- COMPOSE(left, right, condition="...")   — join two evidence sets semantically
- UNION(child1, child2, ...)              — merge multiple evidence sets
- DIFF(base, subtract)                    — remove redundant/conflicting evidence
- RANK(child, criterion="...")            — sort evidence by relevance
- AGGREGATE(child, goal="...")            — synthesize the final answer
- VERIFY(child, constraints="...")        — check grounding and correctness
- DECOMPOSE("query")                      — split a complex query (use sparingly)

## Output format
Write the plan as a single algebraic expression using the operators above.
Use named keyword arguments for all string parameters.
Do NOT output JSON. Do NOT explain. Output ONLY the expression.

## Rules
1. Always end with AGGREGATE(...) or VERIFY(AGGREGATE(...), ...).
2. Leaf nodes are always I("...").
3. Use DECOMPOSE only when the query has clearly distinct independent sub-tasks.
4. Prefer the naive correct plan — do NOT pre-optimize.
"""

FEW_SHOT_EXAMPLES = [
    {
        "query": "What is the main contribution of this paper?",
        "plan": 'AGGREGATE(I("main contribution of this paper"), goal="summarize main contribution")',
    },
    {
        "query": "Compare the evaluation metrics used in paper A and paper B.",
        "plan": (
            "VERIFY(\n"
            "  AGGREGATE(\n"
            "    COMPOSE(\n"
            '      I("evaluation metrics paper A"),\n'
            '      I("evaluation metrics paper B"),\n'
            '      condition="compare evaluation metrics"\n'
            "    ),\n"
            '    goal="compare evaluation metrics across papers"\n'
            "  ),\n"
            '  constraints="answer must cite both papers"\n'
            ")"
        ),
    },
    {
        "query": (
            "Find the methodology sections in these 10 PDFs, "
            "extract the evaluation metrics, and compare how they handle formula accuracy."
        ),
        "plan": (
            "AGGREGATE(\n"
            "  UNION(\n"
            '    TRANSFORM(I("methodology section evaluation metrics"), schema="metrics list"),\n'
            '    TRANSFORM(I("formula accuracy evaluation"), schema="metrics list")\n'
            "  ),\n"
            '  goal="compare formula accuracy handling"\n'
            ")"
        ),
    },
]


def build_user_message(
    task_description: str,
    sample_queries: list[str],
    context_window: int | None = None,
    avg_chunk_tokens: float | None = None,
) -> str:
    """Format task context + sample queries + few-shot examples into a user message."""
    if sample_queries:
        numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(sample_queries))
    else:
        numbered = "(none provided)"
    task_block = (
        f"## Task context\n{task_description}\n\n"
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

    job_block = (
        "## Your job\n"
        "Produce a single general-purpose logical plan that can handle the sample queries above "
        "in the context of the task described. The plan should be a reusable operator DAG "
        "covering the general reasoning pattern."
    )

    return f"{task_block}\n\n{examples_block}{corpus_block}\n\n{job_block}\n\nPlan:"
