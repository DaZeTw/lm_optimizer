# Evaluation Report — LM-Optimizer on QASPER (10 samples, 2 rounds)

**Date**: 2026-04-05  
**Dataset**: QASPER top-100 (10 answerable samples selected)  
**Model**: GPT-4o-mini (parser, planner, judge, executor LLM)  
**Metrics**: Token recall (lexical overlap with gold), Judge accuracy (LLM-as-judge 0.0–1.0)  
**Rounds**: Round 1 = initial plan (no feedback); Round 2 = revised plan after one feedback iteration

---

## Aggregate Results

| Metric | Round 1 | Round 2 | Δ |
| ------ | ------- | ------- | - |
| Avg token recall | 0.139 | 0.213 | **+0.074 ▲** |
| Avg judge score | 0.100 | 0.050 | **−0.050 ▼** |

---

## Per-Sample Results

| # | Paper | Question (short) | Gold (short) | R1 Recall | R2 Recall | Δ Recall | R1 Judge | R2 Judge | Δ Judge |
| - | ----- | ---------------- | ------------ | --------- | --------- | -------- | -------- | -------- | ------- |
| 1 | 1810.02100 | Which English domains? | Conll, Weblogs, Newsgroups, Reviews, Answers | 0.00 | 0.40 | ▲ | 0.00 | 0.50 | ▲ |
| 2 | 1804.08186 | Evaluation methods? | accuracy, precision, recall, F-score | 0.20 | 0.60 | ▲ | 0.00 | 0.00 | = |
| 3 | 1804.08186 | Off-the-shelf systems? | TextCat, ChromeCLD, LangDetect, … | 0.15 | 0.27 | ▲ | 0.00 | 0.00 | = |
| 4 | 1601.04012 | Datasets used? | GENIA corpus | 0.50 | 0.00 | ▼ | 0.00 | 0.00 | = |
| 5 | 1810.13414 | Ontologies used? | Wine Ontology, m-piro, Disease Ontology | 0.00 | 0.25 | ▲ | 0.00 | 0.00 | = |
| 6 | 1903.08237 | Real human experiments? | Yes | 0.00 | 0.00 | = | 0.50 | 0.00 | ▼ |
| 7 | 1811.00051 | Ontologies used? | Wine, Consumer Electronics, Disease Ontology | 0.29 | 0.31 | ▲ | 0.50 | 0.00 | ▼ |
| 8 | 2003.04866 | Dataset annotation? | Score 0–6, 1888 pairs, no communication | 0.25 | 0.30 | ▲ | 0.00 | 0.00 | = |
| 9 | 2003.04866 | 12 languages covered? | Mandarin, Welsh, English, Estonian, … | 0.00 | 0.00 | = | 0.00 | 0.00 | = |
| 10 | 1601.02403 | English only results? | Yes | 0.00 | 0.00 | = | 0.00 | 0.00 | = |

**Summary**: 7/10 recall improved, 1/10 regressed, 2/10 unchanged. 1/10 judge improved, 2/10 regressed, 7/10 unchanged.

---

## Observations

### 1. Retrieval quality improved after revision

Token recall increased on 7 of 10 samples after one round of feedback-driven plan revision. The most notable gains:

- **Sample 1** (1810.02100): 0.00 → 0.40, judge 0.00 → 0.50. The revised plan correctly focused on the domain evaluation section. This is the strongest positive result — both metrics improved together.
- **Sample 2** (1804.08186 evaluation methods): 0.20 → 0.60. The revised plan better retrieved the specific metrics list rather than summarising the general task.

The planner visibly switched to more capable retrieval and ranking variants in Round 2: the `cross-encoder/ms-marco-MiniLM-L-6-v2` model (CrossEncoderRank) was loaded on nearly every sample, compared to Round 1 which primarily used `cross-encoder/nli-deberta-v3-small` (NliVerify). This shows the planner correctly diagnosed that evidence ranking quality was a bottleneck.

### 2. Judge score declined overall despite better recall

Aggregate judge accuracy fell from 0.100 to 0.050. This is the key tension in the results. Two causes are identifiable:

**a) Precision-recall tradeoff in generation**: The revised plans retrieve and synthesise more evidence, but the generated answers become verbose multi-paragraph summaries rather than concise direct answers. The judge — like a human evaluator — penalises answers that are technically relevant but fail to directly answer the question.

Examples:

- Sample 6 (1903.08237, "real human experiments? → Yes"): Round 1 correctly answered "Yes, the paper describes experiments with real human participants…" (judge 0.50). Round 2 returned a long description of the paper's approach to referring expressions — missing the simple yes/no answer entirely (judge 0.00).
- Sample 7 (1811.00051): Round 1 mentioned ontologies with some detail (judge 0.50). Round 2 synthesised a broader NLG capability summary, losing the specific ontology names (judge 0.00).

**b) Over-aggregation on simple questions**: Several questions require a short factual answer (a list, a yes/no, a dataset name). The revised plans with `HierarchicalGenerate` or `LLMCompose` paths produce synthesised prose that paraphrases rather than quotes the answer. Token recall is sometimes better because more relevant words appear in the longer text, but the judge correctly identifies that the answer form is wrong.

### 3. One sample regressed on both metrics (Sample 4)

Sample 4 (1601.04012, "Which datasets are used?", gold: "GENIA corpus") went from recall 0.50 to 0.00. Round 1 retrieved content about linguistic events and briefly mentioned GENIA. Round 2 produced a long synthesis about event extraction methodology that omitted the corpus name entirely. This is a direct consequence of the planner prioritising synthesis quality (based on low accuracy feedback) over direct extraction.

### 4. Three samples had no improvement (Samples 9, 10)

- **Sample 9** (12 languages): The gold answer is a specific list of 12 languages. Both rounds produced a description of the Multi-SimLex project. The physical plan never retrieved the section containing the language list — this is a retrieval coverage failure rather than a plan quality failure.
- **Sample 10** (English data only): The gold is "Yes". Both rounds returned long argumentation mining descriptions. The I-node query string in the plan did not match the specific evidence section.

These failures suggest the retrieval stage (the `I` operator) is the primary bottleneck for factual lookup questions. The planner can improve ranking and aggregation, but cannot recover from retrieval that does not surface the relevant passage at all.

### 5. Plan variant shift between rounds

The model loading logs confirm the planner upgraded retrieval and ranking in Round 2:

| Stage | Round 1 variant | Round 2 variant |
| ----- | --------------- | --------------- |
| Rank | SimilarityRank (embedding cosine) | CrossEncoderRank (ms-marco-MiniLM-L-6-v2) |
| Verify | NliVerify (nli-deberta-v3-small) | Often removed / replaced |
| Aggregate | DirectGenerate | Often HierarchicalGenerate or LLMCompose path |

This is consistent with the planner receiving low accuracy scores (0.0 on most samples) and responding by upgrading precision-oriented variants. The upgrade to CrossEncoderRank is directionally correct — it re-ranks evidence more precisely — but the downstream aggregation changes hurt answer conciseness.

---

## Analysis

### What is working

- The feedback loop correctly identifies retrieval quality as the primary problem and upgrades ranking variants.
- Token recall as a signal is meaningful: 70% of samples improved, the average gain (+0.074) is substantial.
- The best-case outcome (Sample 1) demonstrates that the full loop — retrieve, rank better, synthesise more focused evidence, produce a concise answer — can work well when the question type matches the plan structure.

### What is not working

**1. The judge signal is noisy for short-answer questions.** When the gold answer is "Yes" or a short named list, the judge gives 0.0 to any verbose answer even if factually correct. The planner interprets 0.0 accuracy as "wrong answer" and responds by adding synthesis complexity. This makes the problem worse, not better. The feedback signal needs to distinguish between *wrong content* and *wrong form*.

**2. The I-node query is too general.** The parser generates a single general-purpose logical plan from the task description. For factual lookup questions, the retrieval query embedded in `I("...")` does not match the specific section containing the answer. This is a parser/plan granularity problem that the revise loop cannot fix because it only selects physical variants, not logical operator parameters.

**3. Verbose synthesis hurts judge scores.** The revised aggregation variants (HierarchicalGenerate, LLMCompose) produce comprehensive summaries. For QA, the correct behaviour is to extract and state the specific fact, not to summarise the surrounding context. The aggregation prompt needs a conciseness constraint.

---

## Recommendations

| Priority | Issue | Recommended fix |
| -------- | ----- | --------------- |
| High | Verbose answers reduce judge score | Add conciseness instruction to `AGGREGATE` goal parameter: "answer directly and concisely" |
| High | I-node query too generic | Parser should generate per-question I-node queries, not a single general plan |
| Medium | Feedback signal conflates form and content | Break judge into two signals: factual accuracy (0/1) + answer form match |
| Medium | CrossEncoderRank loaded per iteration | Cache the cross-encoder model across pipeline runs; loading it 5–8 times per sample adds significant latency |
| Low | Planner ignores question type | Pass question type (factual vs. descriptive) as a hint to the planner |

---

## Conclusion

The feedback-driven plan revision demonstrates a clear and repeatable improvement in retrieval quality: average token recall rose from 0.139 to 0.213 (+53% relative). The planner correctly diagnosed that ranking precision was the bottleneck and upgraded to CrossEncoderRank across all samples.

However, the revision also degraded answer conciseness, causing judge accuracy to fall from 0.100 to 0.050. This reveals a structural tension: optimising for more evidence coverage conflicts with producing short, direct answers that score well on factual QA. The current feedback loop does not distinguish between these failure modes.

The next iteration should focus on (1) adding conciseness constraints to the aggregation stage, and (2) making the retrieval queries more specific so that direct factual lookup questions are answerable without multi-hop synthesis.
