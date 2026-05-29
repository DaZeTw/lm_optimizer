"""Registry of logical operators with descriptions and usage guidance."""

from __future__ import annotations

from ir.ops import Op

LOGICAL_OPERATOR_DESCRIPTIONS: dict[Op, dict[str, str]] = {
    Op.I: {
        "description": "Retrieve relevant evidence chunks from the corpus using a query.",
        "inputs": "Query string",
        "outputs": "A set of retrieved evidence chunks",
        "use_when": (
            "Use when external document evidence is required. "
            "The query should describe what evidence to retrieve, not what final answer to generate."
        ),
    },

    Op.RANK: {
        "description": (
            "Score, order, filter, and truncate candidate evidence chunks according to a "
            "relevance criterion."
        ),
        "inputs": "Candidate evidence chunks",
        "outputs": "A filtered and ordered set of candidate evidence chunks",
        "use_when": (
            "Use when retrieved evidence contains noisy, broad, redundant, or competing chunks. "
            "RANK only prioritizes candidate chunks for downstream processing. "
            "It should not be used as the final step when the task requires extraction, "
            "transformation, verification, synthesis, or final answer construction."
        ),
    },

    Op.TRANSFORM: {
        "description": (
            "Change the representation of evidence into another form, such as a summary "
            "or structured fields."
        ),
        "inputs": "Evidence set",
        "outputs": "Transformed evidence representation",
        "use_when": (
            "Use when the evidence needs to be rewritten, summarized, normalized, or converted "
            "into a target schema. "
            "TRANSFORM is not the default operator for evidence filtering or ranking. "
            "Do not use TRANSFORM just to pass evidence through unchanged."
        ),
    },

    # Op.COMPOSE: {
    #     "description": "Connect two evidence sets by reasoning over their relationship.",
    #     "inputs": "Two evidence sets",
    #     "outputs": "Composed evidence set",
    #     "use_when": (
    #         "Use when two evidence sets must be linked, compared, aligned, or jointly interpreted. "
    #         "Do not use COMPOSE when independent evidence only needs merging."
    #     ),
    # },

    # Op.UNION: {
    #     "description": "Merge multiple independent evidence sets into one evidence set.",
    #     "inputs": "Multiple evidence sets",
    #     "outputs": "Combined evidence set",
    #     "use_when": (
    #         "Use when evidence comes from separate retrieval paths or subtopics and does not "
    #         "require pairwise reasoning."
    #     ),
    # },

    # Op.DIFF: {
    #     "description": "Remove evidence from a base evidence set.",
    #     "inputs": "Base evidence set and subtract evidence set",
    #     "outputs": "Filtered evidence set",
    #     "use_when": (
    #         "Use when duplicate, redundant, conflicting, or explicitly unwanted evidence "
    #         "should be removed."
    #     ),
    # },

    Op.AGGREGATE: {
        "description": (
            "Combine selected evidence into the final task output according to the task objective."
        ),
        "inputs": "Evidence set or selected evidence candidates",
        "outputs": "Final task output",
        "use_when": (
            "Use when the task requires a final decision, synthesis, comparison, consolidation, "
            "or output assembly from one or more pieces of evidence. "
            "AGGREGATE may be generative for summary or explanation tasks, but for exact-evidence "
            "tasks it must only assemble and return already-selected verbatim source spans. "
            "It must not introduce unsupported content, paraphrase raw evidence, or change the wording "
            "when the task requires exact text preservation."
        ),
    },

    # Op.VERIFY: {
    #     "description": "Check whether an output satisfies correctness constraints.",
    #     "inputs": "Output evidence or generated answer, optionally with supporting evidence",
    #     "outputs": "Validated output or output with verification metadata",
    #     "use_when": (
    #         "Use when the output must satisfy grounding, exact-copy, citation, schema, "
    #         "or consistency constraints. "
    #         "VERIFY should validate; it should not replace the main extraction or generation step."
    #     ),
    # },

    # Op.DECOMPOSE: {
    #     "description": "Split a complex query into simpler sub-queries.",
    #     "inputs": "Complex query",
    #     "outputs": "Set of simpler sub-queries",
    #     "use_when": (
    #         "Use only when the query contains clearly separable sub-tasks, claims, fields, "
    #         "or evidence needs. Avoid DECOMPOSE for simple retrieval or simple extraction tasks."
    #     ),
    # },
}