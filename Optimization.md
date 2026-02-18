# RAG Evaluation — Cost & Latency Optimization Guide

> **Context:** This guide is based on hands-on experience running RAGAS evaluations
> against an 18-question test dataset using OpenAI GPT-3.5-Turbo as the judge LLM.
> The core formula driving all cost and latency:
>
> ```
> Total LLM Calls = Questions × Metrics
> 18 questions × 4 metrics = 72 OpenAI API calls per full run (~3 minutes, ~$0.05–0.10)
> ```
> At 100 questions × 4 metrics = 400 calls. At 500 questions = 2,000 calls.
> Every optimization below attacks either the call count or the cost per call.

---

## Table of Contents

1. [Understanding the Cost Structure](#1-understanding-the-cost-structure)
2. [Lever 1 — Reduce Question Count with Smart Subsets](#2-lever-1--reduce-question-count-with-smart-subsets)
3. [Lever 2 — Run Only Relevant Metrics](#3-lever-2--run-only-relevant-metrics)
4. [Lever 3 — Use a Cheaper Judge Model](#4-lever-3--use-a-cheaper-judge-model)
5. [Lever 4 — Cache Pipeline Results](#5-lever-4--cache-pipeline-results)
6. [Lever 5 — Tiered Evaluation Strategy](#6-lever-5--tiered-evaluation-strategy)
7. [Lever 6 — Incremental Evaluation](#7-lever-6--incremental-evaluation)
8. [Lever 7 — RunConfig Tuning](#8-lever-7--runconfig-tuning)
9. [Observed Issues in This Project](#9-observed-issues-in-this-project)
10. [What to Do Next](#10-what-to-do-next)

---

## 1. Understanding the Cost Structure

RAGAS evaluation has **two cost layers**, both hitting your OpenAI bill:

### Layer 1 — RAG Pipeline (Retrieval + Generation)
Runs once per question during `build_ragas_dataset()`.

| Step | Model Used | Cost Driver |
|---|---|---|
| Embedding the query | `text-embedding-ada-002` | Tiny — negligible |
| Generating the answer | `gpt-3.5-turbo` or `gpt-4o` | Moderate — one call per question |

### Layer 2 — RAGAS Judge (The Expensive Part)
Runs once per metric per question during `run_evaluation()`.

| Metric | What the Judge Does | Calls |
|---|---|---|
| Faithfulness | Extracts claims from answer → checks each against context | 1 per question |
| Answer Relevancy | Generates reverse questions → computes embedding similarity | 1 per question + embeddings |
| Context Precision | Checks each chunk for relevance using ground truth | 1 per question |
| Context Recall | Checks each ground truth sentence against retrieved chunks | 1 per question |

**Total: 4 judge calls per question.** All four use your judge LLM (currently `gpt-3.5-turbo`).

The "LLM returned 1 generation instead of requested 3" warning you saw means RAGAS
requested multiple completions for statistical robustness but your model returned only one.
This reduces judgment quality slightly but is expected with standard chat models.

---

## 2. Lever 1 — Reduce Question Count with Smart Subsets

This is the highest-impact lever. You do **not** need to evaluate all questions on every run.

### The Principle
Your 18 questions can be grouped by what they test:

```
Group A — Retrieval stress tests (one per document topic)
  → "What are the three types of machine learning?"
  → "What is the primary cause of modern climate change?"
  → "Who was the first human in space?"

Group B — Faithfulness stress tests (previously scored < 1.0)
  → "What is deep learning and how does it relate to machine learning?"
  → "What is the greenhouse effect?"

Group C — Out-of-context canaries (always include these)
  → "What is the current price of Bitcoin?"
  → "Who won the FIFA World Cup in 2022?"
```

A **dev subset** of 7 questions (Groups A + B + C) costs 28 LLM calls instead of 72.
That is a **61% cost reduction** with no loss of diagnostic signal.

### Recommended Test Dataset Structure

```python
# In test_dataset.py, tag each case with a category
TEST_CASES = [
    {
        "question": "...",
        "ground_truth": "...",
        "category": "retrieval",   # or "faithfulness", "canary", "full_only"
    },
    ...
]

# Then in evaluator.py, filter by category:
# DEV_CATEGORIES  = ["retrieval", "faithfulness", "canary"]
# FULL_CATEGORIES = ["retrieval", "faithfulness", "canary", "full_only"]
```

---

## 3. Lever 2 — Run Only Relevant Metrics

Not all metrics change together. Match the metrics you run to the change you made.

| What You Changed | Metrics to Run | Metrics to Skip |
|---|---|---|
| Generator prompt | Faithfulness, Answer Relevancy | Context Precision, Context Recall |
| Generator model | Faithfulness, Answer Relevancy | Context Precision, Context Recall |
| Chunk size | All four | None |
| top_k (retriever) | Context Precision, Context Recall | Faithfulness, Answer Relevancy |
| Embedding model | All four | None |

### Cost Impact
Running 2 metrics instead of 4 cuts your call count in half.

```
18 questions × 2 metrics = 36 calls  (~90 seconds, ~half the cost)
18 questions × 4 metrics = 72 calls  (~3 minutes, full cost)
```

---

## 4. Lever 3 — Use a Cheaper Judge Model

Your **generator** and your **RAGAS judge** are independent — they can be different models.

### Current Setup
```
Generator : gpt-3.5-turbo   (produces answers)
Judge     : gpt-3.5-turbo   (scores them)
```

### Recommended Split
```
Generator : gpt-4o-mini     (better instruction following → higher faithfulness)
Judge     : gpt-3.5-turbo   (scoring task is simpler, cheaper model is fine)
```

### Model Cost Reference (approximate, as of early 2026)

| Model | Input (per 1M tokens) | Best For |
|---|---|---|
| gpt-4o | ~$2.50 | High-quality generation |
| gpt-4o-mini | ~$0.15 | Good generation, great value |
| gpt-3.5-turbo | ~$0.50 | Judging, simple generation |

> **Key insight:** The judging task (NLI verdicts, claim extraction) does not require
> a frontier model. GPT-3.5-Turbo handles it well. Save your budget for generation quality.

---

## 5. Lever 4 — Cache Pipeline Results

The most overlooked optimization. If you are **only changing evaluation settings**
(not the pipeline itself), you are wastefully re-running retrieval and generation every time.

### What to Cache

After `build_ragas_dataset()` runs, save the collected results to disk:

```
evaluation_cache/
  pipeline_results_2026_02_18.json    ← question + answer + contexts + ground_truth
```

On subsequent runs, load from cache instead of calling the pipeline again.
This skips all retrieval embedding calls and all generation LLM calls.

### When to Invalidate the Cache

Invalidate and re-run the pipeline when you change:
- Documents (re-ingest)
- Chunk size or overlap
- Embedding model
- Retriever top_k
- Generator model or prompt

### Expected Savings
If your pipeline run costs ~$0.02 in embeddings + generation calls, caching saves that
on every subsequent evaluation-only run. Small individually, but significant over dozens
of iteration cycles in a day.

---

## 6. Lever 5 — Tiered Evaluation Strategy

The most practical framework for day-to-day development. Run different evaluation
depths at different points in your workflow.

```
┌─────────────────────────────────────────────────────────────────┐
│  TIER 1 — Dev Run (every code change)                           │
│  Questions : 7 (representative subset)                          │
│  Metrics   : 2 (faithfulness + answer_relevancy)                │
│  LLM Calls : 14                                                  │
│  Duration  : ~30 seconds                                         │
│  Cost      : ~$0.005                                             │
│  Purpose   : Fast feedback, catch regressions early              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  TIER 2 — Change Run (after significant modification)           │
│  Questions : 18 (full test dataset)                             │
│  Metrics   : 2–4 (depending on what changed)                    │
│  LLM Calls : 36–72                                              │
│  Duration  : 1.5–3 minutes                                       │
│  Cost      : ~$0.02–0.05                                         │
│  Purpose   : Verify change improved scores, no regressions       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  TIER 3 — Full Run (before any deployment or weekly)            │
│  Questions : 18 (full test dataset)                             │
│  Metrics   : 4 (all metrics)                                     │
│  LLM Calls : 72                                                  │
│  Duration  : ~3 minutes                                          │
│  Cost      : ~$0.05–0.10                                         │
│  Purpose   : Complete picture, save to CSV, compare baselines    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Lever 6 — Incremental Evaluation

Once you have a baseline CSV, you only need to re-evaluate questions that
are **relevant to your specific change**.

### Example
You improved the system prompt to fix faithfulness on the "deep learning" question.

Instead of re-running all 18 questions, run only:
- "What is deep learning..." (the question you fixed)
- "What is the greenhouse effect..." (the other faithfulness failure)
- One canary question (sanity check)

That is 3 questions × 2 metrics = 6 LLM calls instead of 72. A **92% reduction.**

### How to Implement This
Keep your `evaluation_results.csv` as a **baseline**. When you run a targeted
re-evaluation, merge the new scores with the baseline for the unchanged rows.
This gives you a complete picture without re-running everything.

---

## 8. Lever 7 — RunConfig Tuning

RAGAS exposes a `RunConfig` object that controls timeout and retry behaviour.
Default settings are conservative — designed for reliability over speed.

### Key Parameters

| Parameter | Default | Recommended for Dev |
|---|---|---|
| `timeout` | 60s | 30s (fail fast, don't wait on slow calls) |
| `max_retries` | 3 | 1 (retries add latency on failures) |
| `max_wait` | 60s | 10s |

Pass `RunConfig` into `evaluate()`:

```python
from ragas.run_config import RunConfig

run_config = RunConfig(timeout=30, max_retries=1, max_wait=10)
results = evaluate(dataset=dataset, metrics=[...], run_config=run_config)
```

This alone can reduce latency on runs where some judge calls are slow.

---

## 9. Observed Issues in This Project

Based on the actual evaluation run, here are the specific problems and their causes:

### Faithfulness Failures

| Question | Score | Root Cause |
|---|---|---|
| "What is deep learning..." | 0.6 | GPT-3.5 added extra detail not in the 500-char chunk |
| "What is the greenhouse effect..." | 0.5 | Document chunk is too brief; LLM fills gaps from training data |

**Fix options:**
1. Increase chunk size to 800 characters so more context is captured per chunk
2. Switch generator to `gpt-4o-mini` which follows the "only use context" instruction more strictly
3. Strengthen the system prompt with explicit instruction: "If the context does not contain enough information, say so rather than elaborating."

### Context Precision < 1.0 on 2 Questions (Score: 0.833)

| Question | Issue |
|---|---|
| "What are the three types of machine learning?" | One of the 3 retrieved chunks was off-topic |
| "Who was the first human in space?" | Same — third chunk less relevant |

**Fix:** Reduce `top_k` from 3 to 2. You already have perfect recall, so you don't need the third chunk.

### "LLM returned 1 generation instead of requested 3" Warnings

RAGAS requests multiple completions for robustness but chat models return one.
This is expected behaviour with `gpt-3.5-turbo` — not an error, just a warning.
It slightly reduces the statistical confidence of Answer Relevancy scores.

---

## 10. What to Do Next

You now have a working, evaluated RAG pipeline with a solid baseline. Here is a
structured roadmap of experiments and improvements, ordered by priority.

### Phase 1 — Fix the Known Failures (Immediate)

**Step 1: Tighten the generator system prompt**
The two faithfulness failures (deep learning, greenhouse effect) are caused by the
LLM adding knowledge beyond the context. Add an explicit rule to the system prompt
in `generator.py`:

> "If the provided context does not contain sufficient information to answer fully,
> state what you can from the context and explicitly say the context is insufficient
> for the rest. Never elaborate beyond what the context supports."

Re-run Tier 1 evaluation on just those two questions. Compare faithfulness scores.

**Step 2: Try `gpt-4o-mini` as the generator**
Swap `gpt-3.5-turbo` → `gpt-4o-mini` in `generator.py`. GPT-4 family models are
significantly better at following strict grounding instructions. Re-run and compare.

**Step 3: Reduce top_k from 3 to 2**
Your context recall is perfect at 1.0, which means you don't need the third chunk.
Reducing top_k to 2 will push context precision to 1.0 on those two borderline questions.

---

### Phase 2 — Implement the Tiered Evaluation System

Add a `--tier` flag to your evaluator so you can run `uv run python -m rag_evaluation.evaluation.evaluator --tier dev` for fast runs and `--tier full` for complete runs.

Implement question category tags in `test_dataset.py` as described in Lever 1.
Implement pipeline result caching as described in Lever 4.

---

### Phase 3 — Experiment with Chunking Strategy

Your current settings: `chunk_size=500`, `chunk_overlap=50`.

Run these experiments and compare all four RAGAS scores:

| Experiment | chunk_size | chunk_overlap | Hypothesis |
|---|---|---|---|
| Baseline (current) | 500 | 50 | — |
| Larger chunks | 800 | 100 | Faithfulness improves, recall stays 1.0 |
| Smaller chunks | 300 | 30 | Precision improves, recall may drop |
| More overlap | 500 | 150 | Boundary question recall improves |

For each experiment: re-run `ingest.py` → re-run full evaluation → save CSV → compare.

---

### Phase 4 — Add Answer Correctness Metric

You currently measure 4 metrics. RAGAS also offers `AnswerCorrectness` which
compares your generated answer directly against the `ground_truth` using both
semantic similarity and factual overlap. This is a more holistic end-to-end metric.

Add it to your evaluator and observe which questions score low. These are cases
where the pipeline retrieves correctly and generates faithfully, but the answer
is still missing key information from the ground truth.

---

### Phase 5 — Expand the Test Dataset

Your current 18 questions are sufficient for learning but limited for real confidence.
A production-grade evaluation dataset should have:

- 50+ questions
- Multi-hop questions that require information from multiple chunks
- Paraphrased questions (same meaning, different wording) to test robustness
- Edge cases: very long questions, ambiguous questions, questions with multiple valid answers

Consider using RAGAS's `TestsetGenerator` to automatically generate questions
from your documents — this can create hundreds of test cases quickly.

---

### Phase 6 — Build a Comparison Dashboard

Once you have multiple evaluation runs saved as CSVs, build a simple comparison
view in the `notebooks/explore_results.ipynb` that:

- Plots each metric over time (x = run date, y = score)
- Highlights regressions (score dropped vs baseline)
- Shows per-question score heatmap across runs

This turns your evaluation from a one-time check into a continuous quality signal.

---

## Summary Cheat Sheet

```
FASTEST  → 7 questions × 2 metrics = 14 calls  (~30s)
BALANCED → 18 questions × 2 metrics = 36 calls  (~90s)
FULL     → 18 questions × 4 metrics = 72 calls  (~3min)

CHEAPEST JUDGE  → gpt-3.5-turbo
BEST GENERATOR  → gpt-4o-mini
CACHE PIPELINE  → save dataset JSON, skip re-running retrieval+generation

TOP FIXES RIGHT NOW:
  1. Tighten system prompt for faithfulness
  2. Reduce top_k to 2 for better precision
  3. Implement tiered evaluation for dev speed
```