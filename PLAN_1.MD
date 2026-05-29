# Clean Planning/Execution Separation Refactor

## Summary
Refactor the pipeline so TST, logical plan, physical plan, and executor have distinct responsibilities. Execution will depend only on `PhysicalNode.params` plus runtime resources supplied by `PlanRunner`. Physical planning will receive task/evaluation context and a concrete variant parameter schema, then produce self-contained executable params.

## Key Changes
- Executor/runtime boundary:
  - Change `PlanRunner` so it calls variants with only `node.params`; remove `{**node.logical_ref.params, **node.params}`.
  - Add a small runtime context object, e.g. `ExecutionContext(catalog=...)`, passed separately to physical variants.
  - Update registered variant signatures to accept runtime context separately from params.
  - Move `LLMSummarize` catalog cache access from `params["catalog"]` to runtime context.
  - Keep `corpus` and `llm` as executor-supplied runtime resources, not physical params.

- Physical variant param schemas:
  - Add a schema registry, likely near `planner/variant_candidates.py`, for active physical variants.
  - Render schemas in `LLMPhysicalPlanner` prompt alongside variant names.
  - Add recursive validation after physical plan JSON parsing.
  - Reject unsupported params and runtime-resource params such as `corpus`, `llm`, `catalog`, `index`, `client`, `model_client`, `embedder`.
  - Required executable intent params:
    - `BM25Retrieve`, `DenseRetrieve`, `HybridRetrieve`: `query`; optional `top_k`; `HybridRetrieve` optional `alpha`.
    - `SimilarityRank`: `query`, `criterion`; optional `top_k`.
    - `CrossEncoderRank`: `query`; optional `top_k`, `model`.
    - `LLMSummarize`: `schema`; optional `query`, `summary_key`.
    - `DirectGenerate`: `goal`; optional `max_tokens`, `context_window`.
    - `HierarchicalGenerate`: `goal`; optional `batch_tokens`, `max_tokens`.
  - Keep schemas aligned to params actually read by executor functions; do not invent `bm25_weight`, `dense_weight`, `unit`, or `output_constraints` unless the variant implementations are extended to consume them.

- Planner context flow:
  - Update pipeline calls so `_build_logical` receives `task_description`, `evaluation_criteria`, `grounding_query`, and `tst`.
  - Update `_build_physical` to receive `task_description`, `evaluation_criteria`, `grounding_query`, and optimized logical plan.
  - Update logical planner user prompt to separate:
    - task description
    - evaluation criteria
    - current query / grounding query
    - representative samples / TST context
  - Update physical planner prompt to include task description and evaluation criteria, and to state that any logical intent needed at execution must be copied into physical params.
  - Preserve current logical expression output format and physical JSON output format.

- Default/no-LLM physical planning path:
  - Replace the current fallback `parse_physical_plan(optimized.to_dict())`, which cannot satisfy variant requirements, with a deterministic default physical-plan builder.
  - It should choose the first active candidate variant per op and copy only schema-supported logical intent params into physical params.
  - It must produce a valid self-contained `PhysicalNode` tree.

## Interfaces
- `SemanticParser.parse(...)` should gain `evaluation_criteria` and `query`/`grounding_query` context without changing its returned `LogicalNode`.
- `LLMPhysicalPlanner.plan(...)` should become:
  - `task_description`
  - `evaluation_criteria`
  - `query`
  - `logical`
  - `corpus_stats`
- Variant functions should receive runtime context separately from params, e.g.:
  - `fn(inputs, params, corpus, llm, context)`
- Physical plan validation should be callable independently and used by `LLMPhysicalPlanner` after JSON parsing.

## Test Plan
- Executor tests:
  - A fake variant proves `PlanRunner` receives only `node.params`, not `node.logical_ref.params`.
  - A fake or real transform variant proves `catalog` is available through runtime context, not params.

- Schema/validation tests:
  - Valid physical plans pass for active variants.
  - Missing required params fail early.
  - Unsupported params fail early.
  - Runtime-resource params fail early.
  - CrossEncoder `model` string remains allowed because it is an operator-control model name, not a model client.

- Planner prompt/API tests:
  - Logical planner user message includes task description, evaluation criteria, and grounding query.
  - Physical planner user message includes task description, evaluation criteria, logical JSON, and variant parameter schemas.
  - Physical planner retry path includes validation errors.

- Pipeline tests:
  - `_build_logical` and `_build_physical` receive task/evaluation context.
  - No-LLM fallback physical planning produces executable self-contained params.
  - Existing focused TST tests still pass.

## Assumptions
- Runtime context is the chosen way to pass `catalog` and future executor-only resources.
- Only active variants in `CANDIDATE_VARIANTS` need to be rendered and validated for physical planning.
- Logical and physical output formats remain unchanged; only prompt inputs, schema validation, and execution param handling change.
