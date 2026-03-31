"""Canonical logical-op to physical-variant candidate mapping."""

from __future__ import annotations

from ir.ops import Op

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
