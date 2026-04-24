"""Registry of valid physical variants for each logical operator."""

from __future__ import annotations

from ir.ops import Op

# Maps each logical operator to the ordered list of physical variants.
# The first entry is the default / cheapest option.
CANDIDATE_VARIANTS: dict[Op, list[str]] = {
    Op.I: ["BM25Retrieve", "DenseRetrieve", "HybridRetrieve"],
    Op.TRANSFORM: ["IdentityTransform", "ExtractiveCompress", "LLMSummarize"],
    Op.COMPOSE: ["ConcatCompose", "KeyMatchCompose", "LLMCompose"],
    Op.RANK: ["SimilarityRank", "CrossEncoderRank"],
    Op.UNION: ["SimpleUnion"],
    Op.DIFF: ["ExactDiff", "SemanticDiff"],
    Op.AGGREGATE: ["DirectGenerate", "HierarchicalGenerate"],
    Op.VERIFY: ["CitationVerify", "NliVerify"],
    Op.DECOMPOSE: ["IdentityTransform"],
}

VARIANT_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "BM25Retrieve": {
        "description": "Lexical keyword-based retrieval.",
        "use_when": "The query contains clear keywords, entities, or exact terms.",
        "cost": "low",
    },
    "DenseRetrieve": {
        "description": "Embedding-based semantic retrieval.",
        "use_when": "The query is semantic, paraphrased, or does not share exact words with the evidence.",
        "cost": "medium",
    },
    "HybridRetrieve": {
        "description": "Combination of lexical and dense retrieval.",
        "use_when": "Both exact keyword matching and semantic matching are important.",
        "cost": "medium",
    },
    "IdentityTransform": {
        "description": "Pass evidence through without modification.",
        "use_when": "The evidence is already short, clean, or directly usable.",
        "cost": "low",
    },
    "ExtractiveCompress": {
        "description": "Keep only the most relevant spans from the evidence.",
        "use_when": "The evidence is long but contains localized relevant information.",
        "cost": "medium",
    },
    "LLMSummarize": {
        "description": "Use an LLM to summarize or abstract the evidence.",
        "use_when": "The evidence is long, noisy, or needs higher-level synthesis.",
        "cost": "high",
    },
    "ConcatCompose": {
        "description": "Concatenate evidence from multiple branches.",
        "use_when": "Evidence pieces are independent and can simply be placed together.",
        "cost": "low",
    },
    "KeyMatchCompose": {
        "description": "Join evidence using shared keys, entities, sections, or document IDs.",
        "use_when": "Evidence needs to be aligned by a common field or entity.",
        "cost": "medium",
    },
    "LLMCompose": {
        "description": "Use an LLM to synthesize evidence across branches.",
        "use_when": "Evidence requires reasoning, conflict resolution, or complex integration.",
        "cost": "high",
    },
    "SimilarityRank": {
        "description": "Rank evidence using similarity scores.",
        "use_when": "A fast approximate ranking is sufficient.",
        "cost": "low",
    },
    "CrossEncoderRank": {
        "description": "Use a cross-encoder reranker for more precise relevance ranking.",
        "use_when": "High ranking precision is needed for difficult or ambiguous queries.",
        "cost": "high",
    },
    "SimpleUnion": {
        "description": "Merge evidence sets with simple duplicate handling.",
        "use_when": "Multiple branches return complementary evidence.",
        "cost": "low",
    },
    "ExactDiff": {
        "description": "Remove exact duplicates or exact matching items.",
        "use_when": "Duplicates can be identified by ID, text, or exact keys.",
        "cost": "low",
    },
    "SemanticDiff": {
        "description": "Remove semantically similar or redundant evidence.",
        "use_when": "Evidence contains near-duplicates or paraphrased overlaps.",
        "cost": "medium",
    },
    "DirectGenerate": {
        "description": "Generate the final answer directly from the available evidence.",
        "use_when": "The evidence set is small enough to fit into one prompt.",
        "cost": "medium",
    },
    "HierarchicalGenerate": {
        "description": "Generate intermediate summaries before producing the final answer.",
        "use_when": "The evidence set is large and requires multi-stage aggregation.",
        "cost": "high",
    },
    "CitationVerify": {
        "description": "Check whether claims are supported by cited evidence.",
        "use_when": "Grounding and citation correctness are important.",
        "cost": "medium",
    },
    "NliVerify": {
        "description": "Check entailment, contradiction, or unsupported claims using NLI.",
        "use_when": "The system needs stronger factual consistency checking.",
        "cost": "high",
    },
}
