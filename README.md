# LLM Query Plan Optimizer

## 1. Project Overview

This project designs an **LLM-based query plan optimizer** for complex question answering and evidence-grounded generation tasks. The system treats reasoning over text as a form of query processing, inspired by relational database optimizers.

Instead of directly sending a user query to an LLM, the system first builds a structured plan composed of logical operators such as `DECOMPOSE`, `RETRIEVE`, `TRANSFORM`, `COMPOSE`, `RANK`, `AGGREGATE`, and `VERIFY`. Each logical operator can then be implemented by different physical variants, such as BM25 retrieval, dense retrieval, LLM-based summarization, cross-encoder ranking, or rule-based verification.

The key idea is to separate:

* **Task-level strategy design**: creating a reusable strategy template for a class of tasks.
* **Query-level planning**: adapting the strategy template to a specific query.
* **Execution**: running the selected plan over a corpus or context.
* **Feedback-driven revision**: using evaluation results to improve future plans.

The optimizer therefore acts as a bridge between traditional query optimization and modern LLM reasoning pipelines.

---

## 2. Motivation

Complex LLM tasks often fail because they rely on a single prompt or a fixed retrieval pipeline. This creates several problems:

1. **Weak query understanding**
   A complex question may contain multiple sub-goals, but a flat retrieval query often misses important parts.

2. **Poor evidence selection**
   Retrieval may return irrelevant, duplicated, outdated, or contradictory evidence.

3. **Lack of logical structure**
   Different question types require different reasoning flows. For example, a list question, a yes/no question, and an explanatory question should not use the same pipeline.

4. **High cost and latency**
   Brute-force retrieval, excessive LLM calls, and late filtering can waste tokens and compute.

5. **Weak grounding**
   LLMs may generate final answers that are not fully supported by retrieved evidence.

6. **Non-causal optimization**
   Many systems tune prompts or parameters without knowing which change caused an improvement or failure.

This project addresses these issues by representing LLM reasoning pipelines as explicit query plans that can be optimized, executed, evaluated, and revised.

---

## 3. High-Level Architecture

The system contains three main layers:

1. **Task-Level Layer**
   Builds a general strategy template for a task type.

2. **Query-Level Layer**
   Instantiates the task strategy for a specific query, producing an executable plan.

3. **Revision Layer**
   Aggregates feedback from multiple executions and updates the task-level strategy.

At a high level, the flow is:

```text
Task Definition
      ↓
Task-Level Planner
      ↓
Task Strategy Template
      ↓
Query-Level Parser / Optimizer / Planner
      ↓
Executor
      ↓
Judger
      ↓
Reviser
      ↓
Updated Task Strategy Template
```

The architecture supports an iterative optimization loop. The system does not only answer one query; it learns how to improve the strategy for future queries of the same task type.

---

## 4. Core Concepts

### 4.1 Logical Operators

Logical operators define **what kind of reasoning operation** should happen. They are abstract and implementation-independent.

Examples:

* `DECOMPOSE(q)` splits a complex query into sub-queries.
* `I(q, C)` identifies relevant evidence from a corpus.
* `TRANSFORM(E, s)` compresses or restructures evidence.
* `COMPOSE(E1, E2, c)` links evidence across sources.
* `UNION(E1, E2)` merges evidence sets.
* `DIFF(E1, E2)` removes unwanted evidence.
* `RANK(E, c)` orders evidence by relevance or importance.
* `AGGREGATE(E, g)` generates the final output from evidence.
* `VERIFY(O, E, c)` checks whether the output is valid and grounded.

Logical operators are similar to relational algebra operators in database systems.

---

### 4.2 Physical Operators

Physical operators define **how a logical operator is implemented**.

For example, the logical operator `I(q, C)` may be implemented using:

* `FullScan`
* `BM25Retrieve`
* `DenseRetrieve`
* `HybridRetrieve`

Similarly, `RANK(E, c)` may be implemented using:

* `SimilarityRank`
* `CrossEncoderRank`
* `MetadataRank`
* `LLMRank`

This separation allows the optimizer to first reason about the structure of the plan, then choose efficient implementations.

---

### 4.3 Task Strategy Template

A **Task Strategy Template** is the reusable plan skeleton generated at the task level. It defines:

* The core logical operator sequence.
* Optional operators that may be enabled or disabled.
* Allowed physical variants for each operator.
* Default parameters.
* Adaptation policies.
* Constraints that query-level planning must respect.

The template acts as a prior for query-level execution. It prevents each query from being planned from scratch while still allowing controlled adaptation.

---

### 4.4 Executable Query Plan

An **Executable Query Plan** is the query-specific version of the task strategy template. It contains:

* Instantiated operators.
* Selected physical variants.
* Bound query parameters.
* Enabled or disabled optional nodes.
* Execution order.
* Resource constraints.

For example, a general template may allow `DECOMPOSE`, but the query-level planner may choose `NoDecompose` for a simple query and `LLMDecompose` for a multi-hop query.

---

## 5. Module-by-Module Design

## 5.1 Task Definition

### Purpose

The Task Definition module specifies the problem that the optimizer is expected to solve.

### Input

* Task description
  Example: `Holistic QA over scientific papers`.
* Evaluation goals
  Examples: accuracy, grounding, cost, latency.
* Output requirements
  Examples: paragraph answer, JSON output, cited answer, bullet list.
* Query distribution assumptions
  Examples: broad questions, ambiguous questions, multi-hop questions.

### Output

A structured task specification containing:

* Objective.
* Expected output format.
* Evaluation metrics.
* Corpus assumptions.
* Query assumptions.
* Constraints.

### Role

This module defines what problem the system is solving. Without a clear task definition, the planner may optimize for the wrong objective.

---

## 5.2 Task-Level Planner

### Purpose

The Task-Level Planner builds a reusable strategy for the entire task type.

### Input

* Task specification.
* Optional sample queries.
* Optional dataset statistics.
* Optional prior knowledge or heuristics.

### Output

A **Task Strategy Template**, including:

* Logical skeleton.
* Physical defaults.
* Adaptation policy.
* Tunable parameters.
* Optional operators.

### Role

The Task-Level Planner acts as a strategy builder. It decides the default reasoning structure before any individual query is executed.

For example, for evidence-grounded QA, the planner may produce the following template:

```text
DECOMPOSE(q)
  → I(q_i, C)
  → TRANSFORM(E_i)
  → UNION(E_i)
  → RANK(E)
  → AGGREGATE(E)
  → VERIFY(O, E)
```

This gives the system a reusable baseline pipeline.

---

## 5.3 Parser

### Purpose

The Parser converts the raw query and task context into an initial logical representation.

### Input

* Raw user query.
* Task strategy template.
* Optional corpus metadata.

### Output

* Algebraic expression.
* Initial `LogicalNode` DAG.

### Role

The Parser is responsible for creating a machine-readable structure from natural language. It does not yet optimize the plan; it only builds the first logical representation.

Example:

```text
User query:
"Compare the method and performance of Paper A and Paper B."

Initial logical expression:
COMPOSE(
  I("method of Paper A", C),
  I("performance of Paper B", C),
  condition = paper_entity
)
```

---

## 5.4 Optimizer

### Purpose

The Optimizer improves the logical plan before execution.

### Input

* Initial `LogicalNode` DAG.
* Rewrite rules.
* Task strategy template.
* Cost and quality constraints.

### Output

* Optimized `LogicalNode` DAG.

### Role

The Optimizer applies algebraic rewrite rules and structural improvements. Its goal is to produce a better plan before selecting or executing physical operators.

Examples of optimization:

* Push `RANK` earlier to reduce context size.
* Replace brute-force `UNION(I × 10)` with fewer, higher-quality retrieval branches.
* Add `DECOMPOSE` for multi-hop questions.
* Remove invalid `DIFF` operations when no contradiction semantics exist.
* Add `VERIFY` when grounding is required.

The optimizer should prioritize logical correctness before physical tuning.

---

## 5.5 Query-Level Planner

### Purpose

The Query-Level Planner converts the optimized logical plan into an executable physical plan.

### Input

* Optimized `LogicalNode` DAG.
* Task strategy template.
* Raw query.
* Corpus metadata.

### Output

* Executable Query Plan.
* Selected physical variants.
* Bound parameters.
* Enabled or disabled optional operators.

### Role

This module performs bounded query-level adaptation. It can adjust the plan for a specific query, but only within the constraints defined by the task strategy template.

Examples:

* Use `NoDecompose` for a simple factual query.
* Use `LLMDecompose` for a multi-hop query.
* Use `BM25Retrieve` when the query contains exact entity names.
* Use `DenseRetrieve` when the query is semantic or abstract.
* Use `HybridRetrieve` when both exact terms and semantic similarity matter.
* Enable `VERIFY` for high-risk or citation-required answers.

---

## 5.6 Executor

### Purpose

The Executor runs the physical query plan.

### Input

* Executable Query Plan.
* Corpus or long-context input.
* Runtime configuration.

### Output

* Final answer.
* Retrieved evidence.
* Intermediate outputs.
* Runtime logs.

### Role

The Executor is responsible for actually carrying out the plan. It invokes retrievers, compressors, rankers, LLMs, and verification modules according to the selected physical plan.

Example execution sequence:

```text
1. Generate sub-queries.
2. Retrieve evidence for each sub-query.
3. Compress evidence into key spans.
4. Merge and deduplicate evidence.
5. Rank evidence by relevance.
6. Generate answer from evidence.
7. Verify answer against evidence.
```

---

## 5.7 Judger

### Purpose

The Judger evaluates execution results.

### Input

* Final answer.
* Evidence.
* Ground truth, if available.
* Task constraints.

### Output

* Accuracy score.
* Grounding score.
* Constraint satisfaction result.
* Node-level feedback.
* Failure signals.

### Role

The Judger provides quality signals for optimization. It can evaluate both the final output and intermediate nodes.

Evaluation dimensions may include:

* Correctness.
* Completeness.
* Faithfulness to evidence.
* Citation support.
* Format compliance.
* Latency.
* Token cost.
* Retrieval recall.

---

## 5.8 Reviser

### Purpose

The Reviser uses feedback to improve future task strategies.

### Input

* Sample-level results.
* Node feedback.
* Delta records.
* Aggregated metrics.

### Output

* Updated Task Strategy Template.
* Revised physical defaults.
* Revised adaptation policies.
* Updated rewrite rules.

### Role

The Reviser operates at the task level. It should not overfit to one query. Instead, it aggregates feedback across many samples and promotes reliable improvements into the base strategy.

Example revisions:

* If many queries fail because evidence is missing, improve query rewriting or retrieval strategy.
* If many answers hallucinate, add mandatory `VERIFY`.
* If cost is too high, add early `RANK` or reduce retrieval branches.
* If explanatory questions are weak, introduce a separate explanation-specific template.

---

## 6. Logical Operators and Physical Variants

## 6.1 `DECOMPOSE(q)`

### Purpose

Split a complex query into smaller sub-queries.

### Logical Meaning

`DECOMPOSE` breaks a broad or multi-hop question into manageable parts so that each part can be retrieved and reasoned over separately.

### Physical Variants

#### `NoDecompose`

Uses the original query directly.

Best for:

* Simple factual questions.
* Short queries with one clear intent.
* Cases where decomposition may introduce noise.

#### `LLMDecompose`

Uses an LLM to generate sub-questions.

Best for:

* Multi-hop reasoning.
* Comparative questions.
* Broad analytical questions.
* Questions with multiple entities or dimensions.

#### `TemplateDecompose`

Applies predefined decomposition templates.

Best for:

* Repeated task patterns.
* Structured domains.
* Cheaper and more deterministic decomposition.

Example:

```text
Query:
"Compare the revenue growth and margin trend of Company A and Company B."

Sub-queries:
1. Revenue growth of Company A.
2. Revenue growth of Company B.
3. Margin trend of Company A.
4. Margin trend of Company B.
5. Comparison between both companies.
```

---

## 6.2 `I(q, C)`

### Purpose

Identify relevant evidence from a corpus or context.

### Logical Meaning

`I(q, C)` selects evidence from corpus `C` that is relevant to query `q`.

### Physical Variants

#### `FullScan`

Scans all documents or chunks.

Best for:

* Small corpora.
* High-recall settings.
* Cases where indexing is unavailable.

#### `BM25Retrieve`

Uses sparse keyword retrieval.

Best for:

* Exact entity names.
* Technical terms.
* Numeric identifiers.
* Queries where lexical overlap matters.

#### `DenseRetrieve`

Uses embedding similarity.

Best for:

* Semantic search.
* Paraphrased queries.
* Abstract concepts.
* Cases with low keyword overlap.

#### `HybridRetrieve`

Combines sparse and dense retrieval.

Best for:

* General-purpose retrieval.
* Queries with both exact terms and semantic intent.
* Robust retrieval across heterogeneous corpora.

---

## 6.3 `TRANSFORM(E, s)`

### Purpose

Reshape, compress, or structure evidence.

### Logical Meaning

`TRANSFORM` reduces evidence size or converts evidence into a more usable representation.

### Physical Variants

#### `IdentityTransform`

Passes evidence unchanged.

Best for:

* Already concise evidence.
* Low-cost pipelines.
* Cases where compression may lose information.

#### `ExtractiveCompress`

Selects key sentences or spans.

Best for:

* Grounded QA.
* Citation-sensitive tasks.
* Reducing token cost while preserving original wording.

#### `LLMSummarize`

Uses an LLM to summarize evidence.

Best for:

* Long documents.
* Explanatory answers.
* Evidence that needs synthesis.

Risk:

* May introduce hallucination or lose fine details.

#### `StructuredExtract`

Converts evidence into structured fields such as JSON.

Best for:

* Tables.
* Fact extraction.
* Comparison tasks.
* Financial or scientific information extraction.

Example output:

```json
{
  "company": "Company A",
  "metric": "revenue growth",
  "period": "2023-2025",
  "value": "12% CAGR",
  "source_span": "..."
}
```

---

## 6.4 `COMPOSE(E1, E2, c)`

### Purpose

Link evidence across sources.

### Logical Meaning

`COMPOSE` combines related evidence sets using a condition `c`, similar to a join.

### Physical Variants

#### `ConcatCompose`

Concatenates evidence sets.

Best for:

* Simple synthesis.
* Low-cost composition.
* Cases where explicit joining is not needed.

#### `LLMCompose`

Uses an LLM to combine evidence through reasoning.

Best for:

* Multi-hop reasoning.
* Explanatory answers.
* Cases requiring inference.

#### `KeyMatchCompose`

Joins evidence using shared entities or fields.

Best for:

* Structured extraction.
* Comparisons.
* Evidence with common keys such as company, date, product, or metric.

---

## 6.5 `UNION(E1, E2)`

### Purpose

Merge evidence sets from multiple branches.

### Physical Variants

#### `SimpleUnion`

Combines two evidence lists without deduplication.

Best for:

* Preserving all retrieved candidates.
* Early-stage retrieval.

#### `DedupUnion`

Merges while removing exact duplicates.

Best for:

* Repeated chunks.
* Same document retrieved by multiple queries.

#### `SemanticUnion`

Merges while removing semantically similar evidence.

Best for:

* Reducing redundant paraphrases.
* Managing context budget.
* Multi-branch retrieval.

---

## 6.6 `DIFF(E1, E2)`

### Purpose

Remove unwanted evidence.

### Physical Variants

#### `ExactDiff`

Removes exact duplicates or exact matches.

#### `SemanticDiff`

Removes semantically redundant evidence.

#### `ContradictionDiff`

Removes evidence contradicted by another evidence set using NLI or LLM judgment.

### Caution

`DIFF` should only be used when the semantics are clear. Invalid use of `DIFF` can remove useful evidence and damage answer quality.

---

## 6.7 `RANK(E, c)`

### Purpose

Order evidence by importance or relevance.

### Physical Variants

#### `SimilarityRank`

Ranks evidence by embedding similarity.

#### `CrossEncoderRank`

Uses a cross-encoder model to score query-evidence relevance.

Best for:

* High-quality reranking.
* Moderate evidence set sizes.

#### `MetadataRank`

Uses metadata such as recency, source quality, citation count, or document type.

Best for:

* Time-sensitive tasks.
* Financial, legal, or research domains.

#### `LLMRank`

Asks an LLM to score relevance.

Best for:

* Complex relevance judgments.
* Small candidate sets.

Risk:

* Higher cost and latency.

---

## 6.8 `AGGREGATE(E, g)`

### Purpose

Consolidate evidence into the final output.

### Physical Variants

#### `DirectGenerate`

Uses one LLM generation step from evidence.

Best for:

* Simple answers.
* Small evidence sets.

#### `HierarchicalGenerate`

Summarizes parts first, then combines them.

Best for:

* Long documents.
* Many evidence chunks.
* Multi-section reports.

#### `VoteAggregate`

Generates multiple answers and selects or votes among them.

Best for:

* Uncertain answers.
* Robustness testing.

#### `StructuredAggregate`

Aggregates into structured fields before final generation.

Best for:

* Comparison tables.
* Financial analysis.
* Scientific extraction.
* Output formats requiring consistency.

---

## 6.9 `VERIFY(O, E, c)`

### Purpose

Validate output correctness and grounding.

### Physical Variants

#### `CitationVerify`

Checks whether claims have supporting evidence.

#### `NliVerify`

Uses natural language inference to check whether evidence entails the answer.

#### `ConstraintVerify`

Checks structural rules such as JSON validity, required fields, length, or format.

#### `SelfConsistencyVerify`

Generates multiple outputs and checks agreement.

### Role

`VERIFY` is especially important for high-stakes or evidence-grounded tasks. It reduces hallucination and improves trustworthiness.

---

## 7. Database Analogy

The project maps LLM reasoning operators to database query operators.

| LLM Operator         | Database Analogy                       | Meaning                                                |
| -------------------- | -------------------------------------- | ------------------------------------------------------ |
| `DECOMPOSE(q)`       | Query rewrite / subquery decomposition | Break complex query into simpler query paths           |
| `I(q, C)`            | `FROM` + `WHERE` + `SELECT`            | Retrieve relevant evidence from corpus                 |
| `TRANSFORM(E, s)`    | `PROJECT`                              | Keep only required fields or compressed representation |
| `COMPOSE(E1, E2, c)` | `JOIN`                                 | Link evidence across sources or contexts               |
| `UNION(E1, E2)`      | `UNION` / `UNION ALL`                  | Merge evidence sets                                    |
| `DIFF(E1, E2)`       | `EXCEPT` / anti-join                   | Remove unwanted evidence                               |
| `RANK(E, c)`         | `ORDER BY` + `LIMIT`                   | Prioritize top-k evidence                              |
| `AGGREGATE(E, g)`    | `GROUP BY` + aggregate / reduce        | Consolidate evidence into output                       |
| `VERIFY(O, E, c)`    | `HAVING` / constraint check            | Validate result after aggregation                      |

This analogy helps structure the optimizer around known principles from database systems:

* Logical planning before physical execution.
* Operator pushdown.
* Cost-aware physical selection.
* Rewrite rules.
* Execution feedback.
* Iterative optimization.

---

## 8. Logical Planning vs Physical Planning

## 8.1 Logical Planning

Logical planning decides **what operations should happen**.

Example logical plan:

```text
DECOMPOSE(q)
  → I(q_i, C)
  → TRANSFORM(E_i)
  → UNION(E_i)
  → RANK(E)
  → AGGREGATE(E)
  → VERIFY(O, E)
```

Logical planning should answer:

* Does the query require decomposition?
* Is evidence retrieval needed?
* Should evidence be joined across sources?
* Is verification required?
* What is the reasoning structure?

---

## 8.2 Physical Planning

Physical planning decides **how each operation is implemented**.

Example physical plan:

```text
LLMDecompose(q)
  → HybridRetrieve(q_i, top_k = 8)
  → ExtractiveCompress(E_i, max_spans = 3)
  → SemanticUnion(E_i)
  → CrossEncoderRank(E, top_k = 10)
  → StructuredAggregate(E)
  → CitationVerify(O, E)
```

Physical planning should answer:

* Which retriever should be used?
* How many chunks should be retrieved?
* Should compression be extractive or abstractive?
* Which ranker should rerank evidence?
* What token budget should be used?
* Which verification method is cost-effective?

---

## 8.3 Why the Separation Matters

A common failure is tuning physical parameters while the logical structure is wrong.

For example:

```text
UNION(I × 10) → RANK → AGGREGATE
```

This plan may fail because it retrieves too much irrelevant evidence without understanding the query structure. Increasing `top_k` or changing the ranker may not fix the core problem.

A better approach is:

```text
DECOMPOSE(q)
  → I(q_i, C)
  → TRANSFORM(E_i)
  → UNION(E_i)
  → RANK(E)
  → AGGREGATE(E)
  → VERIFY(O, E)
```

The optimizer should first repair the logical plan, then tune the physical implementations.

---

## 9. Query-Type Awareness

Different query types require different logical templates.

## 9.1 List Queries

Example:

```text
"List the main risks mentioned in the report."
```

Recommended template:

```text
I(q, C)
  → TRANSFORM(E, StructuredExtract)
  → RANK(E)
  → AGGREGATE(E, StructuredAggregate)
  → VERIFY(O, ConstraintVerify)
```

Goal:

* Extract multiple items.
* Deduplicate similar items.
* Preserve structure.

---

## 9.2 Yes/No Queries

Example:

```text
"Does the report say revenue declined in 2024?"
```

Recommended template:

```text
I(q, C)
  → TRANSFORM(E, ExtractiveCompress)
  → AGGREGATE(E, DirectGenerate)
  → VERIFY(O, NliVerify)
```

Goal:

* Find direct evidence.
* Avoid unsupported inference.
* Return yes/no with explanation.

---

## 9.3 Explanation Queries

Example:

```text
"Why did margins decline in 2024?"
```

Recommended template:

```text
DECOMPOSE(q)
  → I(q_i, C)
  → COMPOSE(E_i)
  → RANK(E)
  → AGGREGATE(E, HierarchicalGenerate)
  → VERIFY(O, CitationVerify)
```

Goal:

* Gather multiple causes.
* Connect evidence across sections.
* Produce coherent reasoning.

---

## 9.4 Comparison Queries

Example:

```text
"Compare Company A and Company B on revenue growth and profitability."
```

Recommended template:

```text
TemplateDecompose(q)
  → I(q_i, C)
  → StructuredExtract(E_i)
  → KeyMatchCompose(E_i, key = metric)
  → StructuredAggregate(E)
  → VERIFY(O, ConstraintVerify)
```

Goal:

* Extract comparable fields.
* Align entities and metrics.
* Produce a structured comparison.

---

## 10. Rewrite and Optimization Rules

Rewrite rules modify the logical plan before execution.

## 10.1 Decomposition Rule

### Trigger

The query contains multiple entities, multiple metrics, causal language, or comparison language.

### Rewrite

```text
I(q, C)
```

becomes:

```text
DECOMPOSE(q)
  → I(q_i, C)
  → UNION(E_i)
```

### Benefit

Improves recall and reasoning structure for complex queries.

---

## 10.2 Early Ranking Pushdown

### Trigger

Retrieved evidence volume is large or token budget is exceeded.

### Rewrite

```text
I(q_i, C)
  → UNION(E_i)
  → RANK(E)
```

becomes:

```text
I(q_i, C)
  → RANK(E_i)
  → UNION(E_i)
  → RANK(E)
```

### Benefit

Reduces cost before expensive downstream operations.

---

## 10.3 Evidence Compression Pushdown

### Trigger

Evidence chunks are long and contain irrelevant text.

### Rewrite

```text
I(q, C)
  → AGGREGATE(E)
```

becomes:

```text
I(q, C)
  → TRANSFORM(E, ExtractiveCompress)
  → AGGREGATE(E)
```

### Benefit

Improves grounding and reduces context size.

---

## 10.4 Verification Insertion

### Trigger

Task requires citations, high factual accuracy, or evidence grounding.

### Rewrite

```text
AGGREGATE(E)
```

becomes:

```text
AGGREGATE(E)
  → VERIFY(O, E)
```

### Benefit

Reduces hallucination and unsupported claims.

---

## 10.5 Invalid Operator Removal

### Trigger

An operator has unclear semantics or no valid input.

### Rewrite

```text
DIFF(E1, E2)
```

is removed or replaced with:

```text
DedupUnion(E1, E2)
```

when the goal is duplicate removal.

### Benefit

Prevents logical errors from propagating to execution.

---

## 11. Failure Modes and Fixes

## 11.1 Task-Level Strategy Design Failure

### Problem

The strategy is too generic, such as:

```text
UNION(I × 10) → RANK → AGGREGATE
```

This brute-force plan lacks task-specific reasoning structure.

### Fix

* Add `DECOMPOSE`.
* Define task-specific logical templates.
* Separate core operators from optional operators.
* Add query-type routing.

---

## 11.2 Ineffective Retrieval Formulation

### Problem

Queries are generic and not grounded to the original question.

### Fix

* Use question-aware rewriting.
* Preserve key entities and intent.
* Limit retrieval branches to 2–4 high-quality queries.
* Add diversity control across queries.

---

## 11.3 Missing Query-Type Awareness

### Problem

The same plan is used for list, yes/no, comparison, and explanation queries.

### Fix

Add a query classifier and route to different templates:

* List query → retrieve + extract + structured aggregate.
* Yes/no query → retrieve + verify.
* Explanation query → retrieve + summarize + verify.
* Comparison query → decompose + structured extract + key-match compose.

---

## 11.4 Broken Logical–Physical Optimization Flow

### Problem

The system tunes parameters before fixing the logical plan.

### Fix

Apply hierarchical optimization:

1. Validate logical correctness.
2. Apply logical rewrites.
3. Select physical operators.
4. Tune parameters.

---

## 11.5 Ineffective Rewrite Rules

### Problem

Rewrite rules are applied blindly and may introduce invalid operators.

### Fix

* Use condition-based rewrite triggers.
* Restrict rewrites to a valid operator set.
* Log before/after metrics.
* Keep rewrite rules interpretable.

---

## 11.6 Weak Feedback Utilization

### Problem

Failure signals are collected but not translated into actionable improvements.

### Fix

Map error types to strategy changes:

| Error Type                   | Suggested Action                                          |
| ---------------------------- | --------------------------------------------------------- |
| No evidence found            | Improve query rewriting or switch retriever               |
| Too much irrelevant evidence | Add early ranking or reduce retrieval branches            |
| Hallucinated answer          | Add `VERIFY` or extractive compression                    |
| Output format mismatch       | Add `ConstraintVerify`                                    |
| Missing comparison fields    | Use `StructuredExtract` and `KeyMatchCompose`             |
| High latency                 | Reduce LLM-based ranking or use cheaper physical variants |

---

## 11.7 Lack of Evidence–Answer Grounding

### Problem

The aggregator generates answers without sufficient evidence support.

### Fix

* Enforce evidence-first aggregation.
* Extract supporting spans before generation.
* Add `CitationVerify` or `NliVerify`.
* Constrain generation to retrieved evidence.

---

## 11.8 Cost Explosion

### Problem

Too many retrieval calls, LLM calls, or ranking operations cause high cost and latency.

### Fix

* Reduce retrieval branches.
* Push ranking and compression earlier.
* Use lightweight retrievers before expensive rerankers.
* Replace `LLMRank` with `CrossEncoderRank` or `SimilarityRank` when possible.

---

## 12. Feedback and Revision Loop

The system improves by repeatedly executing plans, evaluating results, and revising the task strategy.

## 12.1 Delta Logging

A delta record tracks how the query-level plan differs from the base template.

Example delta record:

```json
{
  "query_id": "q_001",
  "base_template": "evidence_grounded_qa_v1",
  "changes": {
    "decompose": "enabled",
    "retriever": "HybridRetrieve",
    "top_k": 8,
    "verify": "CitationVerify"
  },
  "reason": "query classified as multi-hop explanation"
}
```

Delta logging makes adaptation interpretable.

---

## 12.2 Metrics Collection

The Judger collects metrics such as:

* Accuracy.
* Retrieval recall.
* Evidence relevance.
* Faithfulness.
* Citation coverage.
* Format compliance.
* Token cost.
* Latency.

---

## 12.3 Feedback Aggregation

The Feedback Aggregator combines results across many queries.

It identifies:

* Which templates work best.
* Which adaptations improve performance.
* Which query types fail often.
* Which physical operators are too costly.
* Which rewrite rules have positive or negative impact.

---

## 12.4 Task-Level Revision

The Task-Level Reviser updates the strategy template based on aggregated feedback.

Examples:

* Promote `HybridRetrieve` to default if it consistently improves recall.
* Make `VERIFY` mandatory if hallucination is frequent.
* Add a comparison-specific template if comparison queries perform poorly.
* Reduce default `top_k` if cost is high without accuracy improvement.

---

## 13. Example End-to-End Plan

## 13.1 User Query

```text
"Compare the demand drivers and margin outlook of condensed milk and liquid milk in Vietnam."
```

## 13.2 Query Classification

```json
{
  "query_type": "comparison + explanation",
  "entities": ["condensed milk", "liquid milk", "Vietnam"],
  "dimensions": ["demand drivers", "margin outlook"],
  "requires_decomposition": true,
  "requires_structured_output": true
}
```

## 13.3 Logical Plan

```text
DECOMPOSE(q)
  → I(q_i, C)
  → TRANSFORM(E_i, structured_fields)
  → COMPOSE(E_i, key = product_segment)
  → RANK(E)
  → AGGREGATE(E, comparison_table)
  → VERIFY(O, E)
```

## 13.4 Physical Plan

```text
TemplateDecompose(q)
  → HybridRetrieve(q_i, top_k = 6)
  → StructuredExtract(fields = [segment, driver, margin_factor, evidence])
  → KeyMatchCompose(key = segment)
  → CrossEncoderRank(top_k = 10)
  → StructuredAggregate(format = table + commentary)
  → CitationVerify
```

## 13.5 Expected Output

```markdown
| Segment | Demand Drivers | Margin Outlook | Evidence Strength |
|---|---|---|---|
| Condensed milk | Coffee/F&B usage, affordability, traditional consumption | Stable to slightly improving if input costs ease | Medium |
| Liquid milk | Health awareness, school milk, premiumization | Sensitive to raw milk and packaging costs | Medium |
```

---

## 14. Recommended Data Structures

## 14.1 Task Strategy Template

```json
{
  "template_id": "evidence_grounded_qa_v1",
  "task_type": "evidence_grounded_qa",
  "logical_skeleton": [
    "DECOMPOSE",
    "I",
    "TRANSFORM",
    "UNION",
    "RANK",
    "AGGREGATE",
    "VERIFY"
  ],
  "core_operators": ["I", "RANK", "AGGREGATE"],
  "optional_operators": ["DECOMPOSE", "TRANSFORM", "VERIFY"],
  "allowed_physical_variants": {
    "DECOMPOSE": ["NoDecompose", "LLMDecompose", "TemplateDecompose"],
    "I": ["BM25Retrieve", "DenseRetrieve", "HybridRetrieve"],
    "TRANSFORM": ["IdentityTransform", "ExtractiveCompress", "StructuredExtract"],
    "RANK": ["SimilarityRank", "CrossEncoderRank", "MetadataRank"],
    "AGGREGATE": ["DirectGenerate", "HierarchicalGenerate", "StructuredAggregate"],
    "VERIFY": ["CitationVerify", "ConstraintVerify", "NliVerify"]
  },
  "default_parameters": {
    "retrieval_top_k": 8,
    "rank_top_k": 10,
    "max_subqueries": 4
  },
  "adaptation_policy": {
    "allow_decompose_for_multi_hop": true,
    "require_verify_for_citation_tasks": true,
    "enable_structured_extract_for_comparison": true
  }
}
```

---

## 14.2 Logical Node

```json
{
  "node_id": "n_001",
  "operator": "I",
  "inputs": ["q_001", "corpus_main"],
  "outputs": ["evidence_candidates"],
  "constraints": {
    "top_k": 8,
    "source_filter": "all"
  }
}
```

---

## 14.3 Physical Node

```json
{
  "node_id": "p_001",
  "logical_node_id": "n_001",
  "physical_variant": "HybridRetrieve",
  "parameters": {
    "bm25_weight": 0.45,
    "dense_weight": 0.55,
    "top_k": 8
  }
}
```

---

## 14.4 Execution Result

```json
{
  "query_id": "q_001",
  "final_answer": "...",
  "evidence": [
    {
      "doc_id": "doc_01",
      "span": "...",
      "score": 0.87
    }
  ],
  "metrics": {
    "latency_ms": 4200,
    "token_cost": 3100,
    "grounding_score": 0.91
  }
}
```

---

## 15. Implementation Roadmap

## Phase 1: Baseline Planner

* Define operator schema.
* Implement task strategy template.
* Implement parser from query to logical DAG.
* Implement basic physical variants:

  * `NoDecompose`
  * `BM25Retrieve`
  * `DenseRetrieve`
  * `SimpleUnion`
  * `DirectGenerate`

## Phase 2: Query-Level Optimization

* Add query classifier.
* Add rewrite rules.
* Add physical selection logic.
* Implement early ranking and compression.
* Add delta logging.

## Phase 3: Evaluation and Feedback

* Implement Judger.
* Track answer quality, grounding, cost, and latency.
* Implement node-level feedback.
* Build feedback aggregator.

## Phase 4: Task-Level Revision

* Implement Reviser.
* Update task strategy templates based on aggregate metrics.
* Promote successful adaptations into defaults.
* Add rollback when revisions hurt performance.

## Phase 5: Advanced Optimization

* Add cost model.
* Add query cluster detection.
* Add learned routing policies.
* Add contradiction detection.
* Add multi-objective optimization for quality, cost, and latency.

---

## 16. Key Design Principles

1. **Logical correctness before physical tuning**
   A bad logical plan cannot be fixed only by changing parameters.

2. **Evidence-first generation**
   The system should retrieve and structure evidence before generating the final answer.

3. **Bounded adaptation**
   Query-level planning should adapt within task-level constraints.

4. **Interpretable changes**
   Every adaptation should be logged as a delta from the base template.

5. **Feedback should be causal**
   The system should connect performance changes to specific plan changes.

6. **Cost matters**
   Optimization should consider token cost, latency, and API calls, not only accuracy.

7. **Different queries need different plans**
   List, yes/no, comparison, and explanation queries should route to different logical templates.

---

## 17. Summary

This project builds an optimizer for LLM reasoning pipelines by treating them as query plans. It separates task-level strategy design from query-level execution, distinguishes logical operators from physical implementations, and introduces feedback-driven revision.

The system is designed to improve evidence-grounded question answering by making reasoning pipelines:

* More structured.
* More interpretable.
* More adaptable.
* More cost-efficient.
* More grounded in evidence.
* More capable of continuous improvement.

The final goal is not just to answer individual queries, but to learn better task strategies over time.
