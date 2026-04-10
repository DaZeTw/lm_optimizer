# LM-Optimizer: Query Optimization for Long-Context LLM Reasoning

## Overview

LM-Optimizer is a query optimization system that applies classical database-style optimization techniques to LLM reasoning pipelines. Given a task description and sample queries, it parses the problem into a logical operator DAG, rewrites the DAG with algebraic rules, selects physical execution variants, runs the plan against a corpus, judges the output, and iteratively refines the plan using execution feedback.

The core idea: treat LLM reasoning as a query plan that can be optimized before execution, not a black-box prompt.

---

## End-to-End Flow

```text
Task Description + Sample Queries
         │
         ▼
   ┌─────────────┐
   │   PARSER    │  SemanticParser
   │             │  LLM → algebraic expression → LogicalNode DAG
   └──────┬──────┘
          │ LogicalNode
          ▼
   ┌─────────────┐
   │  OPTIMIZER  │  OptimizerEngine
   │             │  8 algebraic rewrite rules (fixpoint)
   └──────┬──────┘
          │ Optimized LogicalNode
          ▼
   ┌─────────────┐
   │   PLANNER   │  LLMPhysicalPlanner (init)
   │             │  LLM selects physical variants → PhysicalNode DAG
   └──────┬──────┘
          │ PhysicalNode
          ▼
   ┌─────────────┐
   │  EXECUTOR   │  PlanRunner
   │             │  Async bottom-up DAG walk, parallel branches
   └──────┬──────┘
          │ ExecutionResult + NodeFeedback (per sample)
          ▼
   ┌─────────────┐
   │    JUDGE    │  AccuracyJudge
   │             │  LLM-as-judge: result vs gold → accuracy 0.0–1.0
   └──────┬──────┘
          │ Feedback
          ▼
   ┌─────────────┐
   │   REVISE    │  LLMPhysicalPlanner (revise)
   │  (optional) │  LLM refines plan given feedback from all samples
   └──────┬──────┘
          │ (loop back to EXECUTOR for next iteration)
          ▼
      PipelineResult
```

---

## Directory Structure

```text
lm_optimizer/
├── ir/                        # Intermediate representation
│   ├── ops.py                 # Op enum (9 logical operators)
│   ├── nodes.py               # LogicalNode, PhysicalNode, CostVector
│   ├── evidence.py            # Chunk, EvidenceSet (runtime data)
│   └── feedback.py            # NodeFeedback, Feedback
│
├── parser/                    # Natural language → LogicalNode
│   ├── prompts.py             # System prompt, few-shot examples, build_user_message()
│   ├── expr_parser.py         # Recursive descent parser for algebraic expressions
│   └── semantic_parser.py     # SemanticParser: LLM + parser with retry loop
│
├── optimizer/                 # Logical plan rewrite engine
│   ├── engine.py              # OptimizerEngine: fixpoint rewriter
│   └── rules.py               # 8 algebraic rewrite rules
│
├── planner/                   # LogicalNode → PhysicalNode
│   ├── variant_candidates.py  # CANDIDATE_VARIANTS: op → [variant, ...]
│   ├── plan_parser.py         # JSON dict → PhysicalNode
│   └── llm_planner.py         # LLMPhysicalPlanner: init + revise
│
├── executor/                  # Physical plan execution
│   ├── corpus.py              # Corpus + LLM protocols; InMemoryCorpus, OpenAILLM
│   ├── registry.py            # @register("VariantName") decorator + REGISTRY
│   ├── runner.py              # PlanRunner: async DAG walker
│   └── ops/                   # Physical variant implementations
│       ├── retrieve.py        # BM25Retrieve, DenseRetrieve, HybridRetrieve
│       ├── transform.py       # IdentityTransform, ExtractiveCompress, LLMSummarize
│       ├── rank.py            # SimilarityRank, CrossEncoderRank
│       ├── compose.py         # ConcatCompose, KeyMatchCompose, LLMCompose
│       ├── aggregate.py       # DirectGenerate, HierarchicalGenerate
│       ├── union_diff.py      # SimpleUnion, ExactDiff, SemanticDiff
│       └── verify.py          # CitationVerify, NliVerify
│
├── catalog/                   # System metadata
│   ├── catalog.py             # SystemCatalog: doc_stats + model_stats
│   ├── doc_stats.py           # DocStats: per-document chunk/token counts
│   ├── model_stats.py         # ModelStats: context_window, pricing, latency
│   └── indexer.py             # build_system_catalog(), save/load_catalog()
│
├── cost_model/                # Cost estimation and quality scoring
│   ├── vectors.py             # OperatorCostVector, PlanCostReport
│   ├── telemetry.py           # TelemetryStore: observed (op, variant) → cost
│   ├── scorer.py              # PlanScorer: cost report for PhysicalNode DAG
│   ├── judge.py               # AccuracyJudge: LLM-as-judge (0.0–1.0)
│   └── cost_aware_planner.py  # CostAwarePlanner: variant selection via scoring
│
├── pipeline.py                # LmOptimizerPipeline: end-to-end orchestration
├── tests/
│   └── test_qasper.py         # Integration tests + CLI runner (QASPER dataset)
└── pyproject.toml
```

---

## Core Modules

### IR — Intermediate Representation

The IR defines the two-level plan representation used throughout the system.

**`ir/ops.py`** — `Op` enum with 9 logical operators:

| Operator | Purpose |
| -------- | ------- |
| `I(query)` | Retrieve relevant evidence from corpus |
| `TRANSFORM(child, schema)` | Compress or extract structured information |
| `COMPOSE(left, right, condition)` | Join two evidence sets semantically |
| `UNION(child1, child2, ...)` | Merge multiple evidence sets |
| `DIFF(base, subtract)` | Remove redundant/conflicting evidence |
| `RANK(child, criterion)` | Sort evidence by relevance |
| `AGGREGATE(child, goal)` | Synthesize the final answer |
| `VERIFY(child, constraints)` | Check grounding and correctness |
| `DECOMPOSE(query)` | Split a complex query into sub-tasks |

**`ir/nodes.py`** — Plan nodes:

```python
@dataclass(frozen=True)
class LogicalNode:
    op: Op
    inputs: tuple[LogicalNode, ...]
    params: dict[str, Any]

@dataclass(frozen=True)
class PhysicalNode:
    variant: str                   # e.g. "HybridRetrieve"
    logical_ref: LogicalNode       # back-reference to logical op
    inputs: tuple[PhysicalNode, ...]
    params: dict[str, Any]
    cost: CostVector
```

**`ir/evidence.py`** — Runtime data flowing between operators:

- `Chunk`: Text snippet with `doc_id`, `section`, `score`, and metadata
- `EvidenceSet`: Ordered list of chunks; carries `op_trace` (execution history) and `query_ref`

**`ir/feedback.py`** — Execution feedback:

```python
@dataclass
class NodeFeedback:
    op_id: str          # e.g. "AGGREGATE_0"
    variant: str        # e.g. "DirectGenerate"
    token_cost: int
    latency_ms: float
    output_summary: str # first ~200 chars of output

@dataclass
class Feedback:
    items: list[NodeFeedback]
    accuracy: float     # 0.0–1.0 from AccuracyJudge
    result: str         # predicted answer
    gold_ans: str
```

---

### Parser

Translates task description + sample queries into a `LogicalNode` DAG.

**`parser/prompts.py`**

- `SYSTEM_PROMPT`: Defines the 9 operators, output format rules (single algebraic expression, not JSON), and constraints (must end with `AGGREGATE(...)` or `VERIFY(AGGREGATE(...))`, leaves are `I("...")`).
- `FEW_SHOT_EXAMPLES`: 3 examples showing simple to complex plan structures.
- `build_user_message(task_description, sample_queries, ...)`: Builds the user message with task context, numbered sample queries, few-shot examples, and optional corpus stats.

**`parser/expr_parser.py`** — Recursive descent parser for algebraic expressions:

```text
expr := OP_NAME '(' arglist ')'
      | STRING
arglist := (positional | kwarg)*
```

On `ParseError`, the error is fed back to the LLM for self-correction.

**`parser/semantic_parser.py`** — `SemanticParser`:

```python
def parse(self, task_description: str, sample_queries: list[str]) -> LogicalNode:
    # 1. Build prompt with task + samples
    # 2. Send to LLM
    # 3. Parse algebraic expression → LogicalNode
    # 4. On ParseError: append error to conversation, retry (up to max_retries=3)
```

---

### Optimizer

Rewrites the `LogicalNode` DAG using algebraic rules until no further changes apply (fixpoint).

**`optimizer/engine.py`** — `OptimizerEngine`:

- Post-order tree walk (children first)
- Applies at most one rule per node per pass
- Repeats until fixpoint (max 20 passes)
- Returns `(optimized_root, list[RewriteEntry])`

**`optimizer/rules.py`** — 8 rewrite rules:

| Rule | Pattern | Transformation |
| ---- | ------- | -------------- |
| R1 Transform Pushdown | `COMPOSE(E1, E2)` | Insert `TRANSFORM` on each input to compress before joining |
| R2 Filter Pushdown | `AGGREGATE(I(q))` | Insert `RANK` to filter noisy results before aggregation |
| R3 Compose Reorder | `COMPOSE(COMPOSE(A,B), C)` | Reorder to minimize intermediate evidence size |
| R4 Cascade Rank Merge | `RANK(RANK(E, c1), c2)` | Merge into single `RANK` with combined criterion |
| R5 Cascade Transform Merge | `TRANSFORM(TRANSFORM(E, s1), s2)` | Merge into single `TRANSFORM` |
| R6 Delayed Aggregation | `AGGREGATE(E)` without RANK | Insert `RANK` before aggregation |
| R7 Early Dedup | `AGGREGATE(UNION(...))` | Insert `DIFF` to deduplicate before synthesis |
| R8 Selective Verify | `AGGREGATE(... COMPOSE ...)` | Wrap with `VERIFY` for grounding |

---

### Planner

Maps an optimized `LogicalNode` to a `PhysicalNode` with concrete variant choices.

**`planner/variant_candidates.py`** — `CANDIDATE_VARIANTS`:

| Op | Variants (cheapest → most capable) |
| -- | ----------------------------------- |
| `I` | BM25Retrieve → HybridRetrieve → DenseRetrieve |
| `TRANSFORM` | IdentityTransform → ExtractiveCompress → LLMSummarize |
| `COMPOSE` | ConcatCompose → KeyMatchCompose → LLMCompose |
| `RANK` | SimilarityRank → CrossEncoderRank |
| `AGGREGATE` | DirectGenerate → HierarchicalGenerate |
| `DIFF` | ExactDiff → SemanticDiff |
| `VERIFY` | CitationVerify → NliVerify |
| `UNION` | SimpleUnion |

**`planner/llm_planner.py`** — `LLMPhysicalPlanner`:

- **`init(task_description, logical)`**: Cold-start. LLM receives the logical plan JSON and corpus/model stats, returns a physical plan JSON.
- **`revise(task_description, logical, prev_physical, feedbacks: list[Feedback])`**: Warm-start. LLM receives the previous physical plan plus feedback from all samples (one `### Sample N` block per sample), returns a refined physical plan JSON.
- Retries on JSON parse errors (max 2 attempts).

**`planner/plan_parser.py`** — `parse_physical_plan(data)`: Validates and converts JSON dict → `PhysicalNode` tree.

---

### Executor

Executes a `PhysicalNode` DAG asynchronously, bottom-up.

**`executor/registry.py`** — Variant registration:

```python
@register("BM25Retrieve")
async def bm25_retrieve(inputs, params, corpus, llm) -> EvidenceSet:
    ...
```

All variants in `executor/ops/` self-register on import.

**`executor/runner.py`** — `PlanRunner`:

1. Recursively resolve children in parallel (`asyncio.gather`)
2. Look up variant in `REGISTRY`, call it
3. Record `NodeFeedback` (tokens, latency, output summary)
4. Return `(ExecutionResult, list[NodeFeedback])`

`ExecutionResult` carries: `answer`, `trace`, `token_counts`, `errors`.

**`executor/ops/`** — Physical variant implementations:

- **Retrieve**: BM25 (keyword), Dense (embedding), Hybrid (weighted combination)
- **Transform**: Identity (pass-through), Extractive (top-k sentences by relevance), LLM (summarize via LLM call)
- **Rank**: Similarity (cosine to query embedding), CrossEncoder (NLI model for precise relevance)
- **Compose**: Concat (concatenate), KeyMatch (align by keyword overlap), LLMCompose (multi-hop synthesis via LLM)
- **Aggregate**: DirectGenerate (single LLM call with saturation heuristic), HierarchicalGenerate (map-reduce for large evidence)
- **Union/Diff**: SimpleUnion, ExactDiff (hash-based), SemanticDiff (cosine similarity dedup)
- **Verify**: CitationVerify (keyword grounding check), NliVerify (NLI model)

---

### Catalog

Offline-built metadata store for corpus and model statistics.

**`catalog/catalog.py`** — `SystemCatalog`:
- `doc_stats: dict[str, DocStats]` — per-document chunk count, avg chunk tokens, total tokens
- `model_stats: dict[str, ModelStats]` — context window, input/output pricing per 1k tokens, latency
- Methods: `avg_chunk_tokens()`, `context_window()`, `get_model(model_id)`

**`catalog/indexer.py`**:
- `build_system_catalog(corpus, model_registry, default_model_id)` — offline scan
- `save_catalog(catalog, path)` / `load_catalog(path)` — JSON persistence

The catalog feeds the planner (context_window, avg_chunk_tokens) and the cost scorer (pricing, saturation).

---

### Cost Model

**`cost_model/scorer.py`** — `PlanScorer`:

Walks the `PhysicalNode` DAG and produces a `PlanCostReport` with:
- **Token cost**: Sum of estimated tokens per variant (from telemetry)
- **Call cost**: Number of LLM API calls
- **Latency**: Critical path (sequential) + parallel paths (max-of-branches for UNION)
- **Quality risk**: High for unobserved (op, variant) pairs
- **Warnings**: Saturation > 70%, quality_risk > 50%, semantic_gap > 70%

**`cost_model/telemetry.py`** — `TelemetryStore`:

Stores and averages observed runtime metrics per `(op, variant)` pair. Backed by `telemetry.json`.

**`cost_model/judge.py`** — `AccuracyJudge`:

```python
async def score(self, result: str, gold_ans: str) -> float:
    # Prompt: "Rate how accurately result answers gold_ans. Score: 0.0 to 1.0"
    # Returns float parsed from LLM response
```

---

### Pipeline

**`pipeline.py`** — `LmOptimizerPipeline` is the top-level entry point that wires all stages together.

**`PipelineResult`** (frozen dataclass):

| Field | Type | Description |
| ----- | ---- | ----------- |
| `task_description` | `str` | The task label used for parsing |
| `sample_queries` | `list[str]` | Queries extracted from samples |
| `logical_plan` | `LogicalNode` | Raw parse output |
| `optimized_plan` | `LogicalNode` | After optimizer |
| `physical_plan` | `PhysicalNode` | Final physical plan (last iteration) |
| `rewrite_log` | `list[RewriteEntry]` | Rules fired during optimization |
| `executions` | `list[ExecutionResult]` | One per sample (last iteration) |
| `feedbacks` | `list[Feedback]` | One per sample (last iteration) |
| `.execution` | property | Last execution (convenience) |
| `.feedback` | property | Last feedback (convenience) |

**Entry points**:

```python
# Async
result = await pipeline.run_with_samples(
    task_description="QA over scientific papers",
    samples=[("What is the method?", "gold answer"), ...],
    iterations=2,
    log_path="logs/run.json",
)

# Sync wrapper (handles Windows event loop)
result = pipeline.run_sync_with_samples(
    task_description="...",
    samples=[...],
    iterations=2,
)
```

**Constructor parameters**:

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `corpus` | required | Corpus implementation |
| `llm` | required | LLM for execution (answer generation) |
| `planning_client` | `None` | Separate LLM client for parser, planner, judge |
| `parser` | auto | `SemanticParser` instance |
| `optimizer` | auto | `OptimizerEngine` |
| `catalog` | `None` | `SystemCatalog` (or load from `catalog_path`) |
| `planning_model` | `"gpt-4o-mini"` | Model for planner |
| `judge_model` | `"gpt-4o-mini"` | Model for judge |

---

## Key Data Flows

### Parse Phase

```text
task_description + sample_queries
    → build_user_message() → LLM → algebraic expression string
    → expr_parser.parse_expression() → LogicalNode DAG
```

### Optimize Phase

```text
LogicalNode
    → OptimizerEngine.run()
    → post-order tree walk, apply rules until fixpoint
    → (optimized_LogicalNode, list[RewriteEntry])
```

### Plan + Execute + Revise Loop

```python
for iteration in range(iterations):
    if iteration == 0:
        physical = planner.init(task_description, optimized)

    run_results = await asyncio.gather(runner.run(physical) for each sample)

    feedbacks = [
        Feedback(items=node_feedbacks, accuracy=judge.score(result, gold), ...)
        for (execution, node_feedbacks), gold in zip(run_results, gold_answers)
    ]

    if not last_iteration:
        physical = planner.revise(task_description, optimized, physical, feedbacks)
```

---

## Dependencies

| Package | Version | Use |
| ------- | ------- | --- |
| `openai` | ≥2.29.0 | Chat completions for parser, planner, judge, executor LLM |
| `sentence-transformers` | ≥5.3.0 | Dense embeddings, CrossEncoderRank, NliVerify |
| `rank-bm25` | ≥0.2.2 | BM25Retrieve sparse search |
| `python-dotenv` | ≥1.2.2 | Load `OPENAI_API_KEY` from `.env` |
| `pydantic` | ≥2.12.5 | Data validation |
| `pytest` / `pytest-asyncio` | dev | Test runner |

---

## Testing

Tests live in `tests/test_qasper.py` and run against the real OpenAI API (requires `OPENAI_API_KEY`).

```bash
# Run all tests (skipped if no API key)
pytest tests/test_qasper.py -s

# CLI script: evaluate on N samples, two rounds (no feedback → with feedback)
python tests/test_qasper.py --n 5
```

The CLI script runs two rounds and prints a recall/accuracy comparison table per sample.

**Dataset**: `qasper_top_100_context.json` — top 100 answerable QA pairs from the QASPER scientific paper QA benchmark, each with full paper text as corpus.

---

## Design Principles

1. **Algebra first**: The query plan is a first-class data structure, not a string prompt. Optimization is deterministic and inspectable.
2. **Separation of concerns**: Parsing, optimization, planning, execution, and judging are independent stages with clean interfaces.
3. **Cheap before expensive**: Variant lists are ordered cheapest-first; the LLM planner defaults to fast variants and upgrades only when needed.
4. **Feedback-driven refinement**: Execution metrics (tokens, latency, accuracy) flow back to the planner to guide variant selection in subsequent iterations.
5. **Modular variants**: Adding a new physical variant requires only implementing one async function and decorating it with `@register("Name")`.
