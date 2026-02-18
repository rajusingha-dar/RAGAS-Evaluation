# RAG Evaluation Without LLM — Traditional Methods Guide

> **Why this matters:** RAGAS LLM-as-judge costs money and takes 2-3 minutes per run.
> Traditional evaluation methods run in **milliseconds**, cost nearly **nothing**, and
> are perfect for fast dev-tier feedback loops. This guide covers every non-LLM
> evaluation method, how each works mathematically, and how they integrate into
> this project alongside RAGAS.

---

## Table of Contents

1. [The Two Worlds of RAG Evaluation](#1-the-two-worlds-of-rag-evaluation)
2. [Method 1 — Exact Match](#2-method-1--exact-match)
3. [Method 2 — String Presence](#3-method-2--string-presence)
4. [Method 3 — ROUGE Score](#4-method-3--rouge-score)
5. [Method 4 — BLEU Score](#5-method-4--bleu-score)
6. [Method 5 — Semantic Similarity (Embeddings)](#6-method-5--semantic-similarity-embeddings)
7. [Method 6 — Context Relevance via Embeddings](#7-method-6--context-relevance-via-embeddings)
8. [How They Map to RAG Metrics](#8-how-they-map-to-rag-metrics)
9. [Project Integration — non_llm_evaluator.py](#9-project-integration--non_llm_evaluatorpy)
10. [Required Dependencies](#10-required-dependencies)
11. [Hybrid Strategy — When to Use What](#11-hybrid-strategy--when-to-use-what)
12. [Score Interpretation Guide](#12-score-interpretation-guide)

---

## 1. The Two Worlds of RAG Evaluation

```
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│       LLM-as-Judge (RAGAS)      │    │    Traditional / Non-LLM        │
├─────────────────────────────────┤    ├─────────────────────────────────┤
│ ✅ Handles paraphrasing well    │    │ ✅ Near-zero cost                │
│ ✅ Understands semantic nuance  │    │ ✅ Runs in milliseconds          │
│ ✅ Faithfulness via NLI         │    │ ✅ Fully deterministic           │
│ ❌ Expensive (72 API calls)     │    │ ✅ No API dependency             │
│ ❌ 2–3 minutes per run          │    │ ❌ Sensitive to exact wording    │
│ ❌ Non-deterministic            │    │ ❌ Less nuanced judgment         │
└─────────────────────────────────┘    └─────────────────────────────────┘
```

The mindmap you shared classifies RAG evaluation into two branches:

**With LLM** — Text Model (Context Precision, Context Recall, Noise Sensitivity,
Context Entities Recall, Response Relevancy, Faithfulness) and Multi Model
(MultiModel Faithfulness, MultiModel Relevancy).

**Without LLM** — Factual Correctness branch covering Exact Match, String Presence,
ROUGE Score, Semantic Similarity, Non-LLM String Similarity, and BLEU Score.

This document covers the **Without LLM** branch entirely.

---

## 2. Method 1 — Exact Match

### What It Is
The strictest possible metric. Checks whether the generated answer is character-for-character
identical to the ground truth (after normalisation like lowercasing and stripping whitespace).

### The Math
```
Exact Match Score = 1  if normalise(answer) == normalise(ground_truth)
                  = 0  otherwise
```

### When It Works Well
- Short factual answers: names, dates, numbers, single words
- Questions with only one correct phrasing ("1.1 degrees Celsius")
- Controlled test environments where you wrote the ground truth to match expected output

### When It Fails
- Any answer longer than a sentence
- Correct answers phrased differently ("Yuri Gagarin" vs "Gagarin")
- Answers with extra correct context ("Yuri Gagarin, born 1934" — technically wrong by this metric)

### Practical Example From This Project
```
Question     : "By how much have global average temperatures risen?"
Ground Truth : "approximately 1.1 degrees Celsius"
Answer       : "Global temperatures have risen by approximately 1.1 degrees Celsius"
Exact Match  : 0  ← fails despite being correct
```

---

## 3. Method 2 — String Presence

### What It Is
A softer version of exact match. You define a set of key terms or phrases that
**must appear** in the answer. Score = fraction of required terms found.

### The Math
```
String Presence = (number of required terms found in answer)
                  ÷ (total number of required terms)
```

### How to Define Required Terms
Extract the key factual terms from your ground truth:
- Proper nouns (Yuri Gagarin, Apollo 11, Paris Agreement)
- Numbers (1.1 degrees, 90 percent, 1961)
- Domain terms (carbon dioxide, supervised learning, Falcon 9)

### Practical Example From This Project
```
Question      : "What causes climate change?"
Ground Truth  : "increase of greenhouse gases, carbon dioxide, burning fossil fuels"
Required Terms: ["greenhouse gases", "carbon dioxide", "fossil fuels"]
Answer        : "Climate change is caused by greenhouse gases like CO2 from burning fossil fuels"

greenhouse gases → FOUND  ✓
carbon dioxide   → not found literally (CO2 used instead) ✗
fossil fuels     → FOUND  ✓

String Presence = 2/3 = 0.667
```

This is where String Presence has a known weakness — it doesn't handle abbreviations
or synonyms. "CO2" ≠ "carbon dioxide" by exact string matching, even though they mean
the same thing. Semantic similarity (Method 5) handles this better.

---

## 4. Method 3 — ROUGE Score

### What It Is
Recall-Oriented Understudy for Gisting Evaluation. Originally developed at USC/ISI
for automatic summarisation evaluation. Measures the overlap of word sequences
(n-grams) between the generated text and the reference text.

### The Three Variants

**ROUGE-1** — overlap of individual words (unigrams)
```
Precision = (word overlap) ÷ (words in generated answer)
Recall    = (word overlap) ÷ (words in ground truth)
F1        = 2 × (Precision × Recall) ÷ (Precision + Recall)
```

**ROUGE-2** — overlap of two-word phrases (bigrams)
Same formula but counts two-word sequences instead of individual words.
Stricter than ROUGE-1. Captures phrase-level similarity.

**ROUGE-L** — Longest Common Subsequence (LCS)
Finds the longest sequence of words that appears in both texts in the same order,
but not necessarily contiguously. Better at capturing sentence structure similarity.

### Worked Example From This Project
```
Ground Truth : "The three types of machine learning are supervised learning,
                unsupervised learning, and reinforcement learning."

Answer       : "The three types are supervised learning, unsupervised learning,
                and reinforcement learning."

Shared words : "The", "three", "types", "supervised", "learning", "unsupervised",
               "learning", "and", "reinforcement", "learning"  (10 words)

ROUGE-1 Recall    = 10 ÷ 14 = 0.714   (14 words in ground truth)
ROUGE-1 Precision = 10 ÷ 13 = 0.769   (13 words in answer)
ROUGE-1 F1        = 0.741
```

### ROUGE Strengths and Weaknesses
ROUGE is great for measuring how much of the ground truth content made it into
the answer. It rewards coverage. However it penalises different-but-correct phrasings,
is sensitive to word choice, and doesn't understand meaning — "not good" and "bad"
would score 0 similarity despite meaning the same thing.

---

## 5. Method 4 — BLEU Score

### What It Is
Bilingual Evaluation Understudy. Originally from machine translation (IBM, 2002).
Where ROUGE is recall-oriented (did we cover the reference?), BLEU is
precision-oriented (is what we generated accurate to the reference?).

### The Math
BLEU computes precision for n-grams of length 1 through 4, then combines them
using a geometric mean, with a brevity penalty for short answers:

```
BLEU = BP × exp( Σ wₙ × log(precisionₙ) )
         n=1 to 4

where BP (Brevity Penalty) = 1 if answer length ≥ reference length
                            = exp(1 - ref_length/answer_length) otherwise
```

### Why BLEU is Less Suited for RAG Than ROUGE
BLEU was designed to evaluate whether a **translation** is accurate to a reference.
RAG answers are generative and often longer than ground truth. The brevity penalty
can hurt scores for thorough correct answers. ROUGE-L is generally preferred for
RAG evaluation.

BLEU is included here because your mindmap lists it as a valid non-LLM option,
and it remains widely used as a benchmark comparison metric.

---

## 6. Method 5 — Semantic Similarity (Embeddings)

### What It Is
The most powerful non-LLM method. Both the generated answer and the ground truth
are converted to embedding vectors using the same embedding model, then the
cosine similarity between the two vectors is computed.

This captures **meaning** rather than word overlap. Two sentences can share no
words but still score high similarity if they mean the same thing.

### The Math
```
Step 1: embed_answer    = EmbeddingModel(answer)        → vector of 1536 dimensions
Step 2: embed_reference = EmbeddingModel(ground_truth)  → vector of 1536 dimensions

Step 3: Cosine Similarity = (embed_answer · embed_reference)
                            ÷ (||embed_answer|| × ||embed_reference||)

Range: -1 to 1  (in practice 0 to 1 for text)
```

### Why Cosine Similarity Works for Semantic Comparison
The embedding space is structured so that semantically similar texts cluster
together. "Supervised learning trains on labeled data" and "Labeled datasets
are used to train supervised ML models" will have embeddings pointing in nearly
the same direction in 1536-dimensional space — hence high cosine similarity.

### Practical Example From This Project
```
Ground Truth : "Yuri Gagarin was the first human in space on April 12, 1961"
Answer       : "The first person to travel to space was Yuri Gagarin in 1961"

Word overlap : low (different sentence structure)
ROUGE-1      : ~0.45  (misses many shared words)
Semantic Sim : ~0.94  (correctly identifies they mean the same thing)
```

### Cost Consideration
Embedding calls are vastly cheaper than LLM judge calls.
`text-embedding-ada-002` costs ~$0.0001 per 1K tokens.
18 questions × 2 embedding calls = ~$0.0001 total. Essentially free.

---

## 7. Method 6 — Context Relevance via Embeddings

### What It Is
Measures how relevant each retrieved chunk is to the query, without using an LLM.
Embed both the query and each retrieved chunk, then compute cosine similarity.
High similarity = relevant chunk. This approximates RAGAS Context Precision cheaply.

### The Math
```
For each retrieved chunk i:
  relevance_score[i] = cosine_similarity(embed(query), embed(chunk[i]))

Context Relevance = average(relevance_score[1], ..., relevance_score[k])

Precision Proxy   = fraction of chunks with relevance_score > threshold (e.g. 0.75)
```

### Why This Matters
This is essentially what your vector store already does during retrieval —
ChromaDB ranks chunks by embedding similarity. But you can surface these scores
explicitly during evaluation to understand how strong your retrieval signal is,
and to diagnose cases where a low-relevance chunk was retrieved at position 1.

---

## 8. How They Map to RAG Metrics

| RAGAS Metric | Non-LLM Approximation | Quality of Approximation |
|---|---|---|
| Faithfulness | String Presence (context terms in answer) | Weak — misses semantic matches |
| Answer Relevancy | Semantic Similarity (answer vs question) | Good |
| Answer Correctness | Semantic Similarity (answer vs ground truth) | Very Good |
| Context Precision | Embedding similarity (query vs each chunk) | Good |
| Context Recall | ROUGE-L (ground truth coverage in chunks) | Moderate |

No single non-LLM method perfectly replicates LLM-as-judge judgment. The best
non-LLM approximation combines semantic similarity for answer quality and
embedding-based context relevance for retrieval quality.

---

## 9. Project Integration — non_llm_evaluator.py

Place this file at `src/rag_evaluation/evaluation/non_llm_evaluator.py`.

```python
"""
non_llm_evaluator.py
────────────────────
Traditional ML evaluation — no LLM API calls required.

Metrics implemented:
  - Exact Match
  - String Presence
  - ROUGE-1, ROUGE-2, ROUGE-L  (via rouge-score library)
  - BLEU Score                  (via nltk)
  - Semantic Similarity         (via OpenAI embeddings — cheap)
  - Context Relevance           (embedding similarity, query vs chunks)

Run with:
    uv run python -m rag_evaluation.evaluation.non_llm_evaluator
"""

import os
import re
import json
import math
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from openai import OpenAI

from rag_evaluation.pipeline import ask
from rag_evaluation.evaluation.test_dataset import TEST_CASES

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


# ── Utility ───────────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def get_embedding(text: str) -> list:
    """Get OpenAI embedding vector for a text string."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


# ── Metric 1: Exact Match ─────────────────────────────────────────────────────

def exact_match(answer: str, ground_truth: str) -> float:
    """
    Returns 1.0 if normalised answer == normalised ground truth, else 0.0.
    Best for: short factual answers, numbers, names.
    """
    return 1.0 if normalise(answer) == normalise(ground_truth) else 0.0


# ── Metric 2: String Presence ─────────────────────────────────────────────────

def string_presence(answer: str, required_terms: list) -> float:
    """
    Fraction of required_terms found in the answer (case-insensitive).
    required_terms: list of key phrases extracted from ground truth.
    Best for: factual term coverage checking.
    """
    if not required_terms:
        return 0.0
    answer_lower = answer.lower()
    found = sum(1 for term in required_terms if term.lower() in answer_lower)
    return found / len(required_terms)


def extract_key_terms(ground_truth: str) -> list:
    """
    Auto-extract key terms from ground truth by keeping words > 4 chars
    and filtering common stop words.
    For production, define required_terms manually per test case.
    """
    stop_words = {
        "that", "this", "with", "from", "they", "have", "been", "were",
        "which", "their", "there", "about", "also", "more", "than", "into",
        "some", "such", "when", "what", "where", "would", "could", "should",
    }
    words = re.findall(r'\b[a-zA-Z]{4,}\b', ground_truth.lower())
    return list(set(w for w in words if w not in stop_words))


# ── Metric 3: ROUGE Scores ────────────────────────────────────────────────────

def rouge_scores(answer: str, ground_truth: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
    Uses the rouge-score library (pip install rouge-score).

    Returns dict with keys: rouge1, rouge2, rougeL
    All values are F1 scores (harmonic mean of precision and recall).
    Best for: measuring content coverage in longer answers.
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True  # stems words so "running" matches "run"
    )
    scores = scorer.score(ground_truth, answer)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


# ── Metric 4: BLEU Score ──────────────────────────────────────────────────────

def bleu_score(answer: str, ground_truth: str) -> float:
    """
    Compute sentence-level BLEU score.
    Uses NLTK with smoothing to handle zero n-gram counts.

    BLEU is precision-oriented — measures how much of the answer
    is present in the reference. Penalises answers shorter than reference.
    Best for: translation-style evaluation, precise factual answers.
    """
    reference = [normalise(ground_truth).split()]
    hypothesis = normalise(answer).split()

    # Smoothing prevents score=0 when higher n-gram counts are zero
    smoothing = SmoothingFunction().method1
    score = sentence_bleu(reference, hypothesis, smoothing_function=smoothing)
    return round(float(score), 4)


# ── Metric 5: Semantic Similarity ────────────────────────────────────────────

def semantic_similarity(answer: str, ground_truth: str) -> float:
    """
    Cosine similarity between answer and ground_truth embeddings.
    Uses text-embedding-ada-002 (1536-dim vectors).

    This is the most accurate non-LLM metric for RAG answer evaluation.
    Handles paraphrasing, synonyms, and different sentence structures.
    Cost: ~$0.0001 per 1K tokens — negligible.
    Best for: answer correctness evaluation.
    """
    embed_answer = get_embedding(answer)
    embed_reference = get_embedding(ground_truth)
    return round(cosine_similarity(embed_answer, embed_reference), 4)


# ── Metric 6: Context Relevance ───────────────────────────────────────────────

def context_relevance(query: str, contexts: list, threshold: float = 0.75) -> dict:
    """
    Measures how relevant each retrieved chunk is to the query using embeddings.
    Approximates RAGAS Context Precision without any LLM call.

    Returns:
      avg_relevance  : mean cosine similarity across all chunks
      precision_proxy: fraction of chunks above the relevance threshold
      chunk_scores   : individual relevance score per chunk
    """
    query_embed = get_embedding(query)
    chunk_scores = []

    for chunk in contexts:
        chunk_embed = get_embedding(chunk)
        score = cosine_similarity(query_embed, chunk_embed)
        chunk_scores.append(round(score, 4))

    avg = round(sum(chunk_scores) / len(chunk_scores), 4) if chunk_scores else 0.0
    precision = round(
        sum(1 for s in chunk_scores if s >= threshold) / len(chunk_scores), 4
    ) if chunk_scores else 0.0

    return {
        "avg_relevance": avg,
        "precision_proxy": precision,
        "chunk_scores": chunk_scores,
    }


# ── Full Non-LLM Evaluator ────────────────────────────────────────────────────

def evaluate_non_llm(test_cases: list = None, use_semantic: bool = True) -> list:
    """
    Run all non-LLM metrics across the test dataset.

    Args:
        test_cases   : list of test cases (defaults to TEST_CASES)
        use_semantic : whether to include semantic similarity (requires embedding API)
                       Set False for completely free evaluation (ROUGE + BLEU only)

    Returns list of result dicts, one per question.
    """
    test_cases = test_cases or TEST_CASES

    print("=" * 60)
    print("  RAG Evaluation — Non-LLM Metrics")
    print(f"  Semantic similarity: {'enabled' if use_semantic else 'disabled (ROUGE + BLEU only)'}")
    print("=" * 60)

    results = []

    for i, case in enumerate(test_cases, 1):
        question     = case["question"]
        ground_truth = case["ground_truth"]

        print(f"\n  [{i:02d}/{len(test_cases)}] {question[:55]}...")

        # Run pipeline
        pipeline_result = ask(question)
        answer   = pipeline_result["answer"]
        contexts = pipeline_result["contexts"]

        # ── Run all metrics ───────────────────────────────────────────────────

        # 1. Exact Match
        em = exact_match(answer, ground_truth)

        # 2. String Presence
        key_terms = extract_key_terms(ground_truth)
        sp = string_presence(answer, key_terms)

        # 3. ROUGE
        rouge = rouge_scores(answer, ground_truth)

        # 4. BLEU
        bleu = bleu_score(answer, ground_truth)

        # 5. Semantic Similarity (optional — requires embedding API)
        sem_sim = semantic_similarity(answer, ground_truth) if use_semantic else None

        # 6. Context Relevance via embeddings (optional)
        ctx_rel = context_relevance(question, contexts) if use_semantic else None

        result = {
            "question":          question,
            "ground_truth":      ground_truth,
            "answer":            answer,
            "exact_match":       em,
            "string_presence":   round(sp, 4),
            "rouge1":            rouge["rouge1"],
            "rouge2":            rouge["rouge2"],
            "rougeL":            rouge["rougeL"],
            "bleu":              bleu,
            "semantic_sim":      sem_sim,
            "ctx_avg_relevance": ctx_rel["avg_relevance"] if ctx_rel else None,
            "ctx_precision_proxy": ctx_rel["precision_proxy"] if ctx_rel else None,
        }
        results.append(result)

        # Quick print
        print(f"      ROUGE-L: {rouge['rougeL']:.3f} | "
              f"BLEU: {bleu:.3f} | "
              f"{'SemanticSim: ' + str(sem_sim) if sem_sim else 'SemanticSim: skipped'}")

    return results


def print_non_llm_report(results: list):
    """Print summary report and save CSV."""
    import csv

    print("\n" + "=" * 60)
    print("  Non-LLM Evaluation Summary")
    print("=" * 60)

    metrics = ["exact_match", "string_presence", "rouge1", "rouge2", "rougeL", "bleu"]
    if results[0].get("semantic_sim") is not None:
        metrics += ["semantic_sim", "ctx_avg_relevance"]

    # Compute averages (skip None values)
    for metric in metrics:
        vals = [r[metric] for r in results if r.get(metric) is not None]
        avg = sum(vals) / len(vals) if vals else 0.0
        bar    = "█" * int(avg * 20)
        spacer = "░" * (20 - int(avg * 20))
        print(f"  {metric:<22} {avg:.3f}  [{bar}{spacer}]")

    # Save to CSV
    output_path = "non_llm_evaluation_results.csv"
    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  💾  Results saved to: {output_path}\n")


if __name__ == "__main__":
    # Set use_semantic=False to run with zero API calls (ROUGE + BLEU only)
    results = evaluate_non_llm(use_semantic=True)
    print_non_llm_report(results)
```

---

## 10. Required Dependencies

Add these to your project with UV:

```bash
uv add rouge-score nltk numpy
```

Then run a one-time NLTK data download (needed for BLEU tokenisation):

```bash
uv run python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

Your full dependency list in `pyproject.toml` will now include:

```toml
dependencies = [
    "langchain",
    "langchain-openai",
    "langchain-chroma",
    "langchain-community",
    "langchain-text-splitters",
    "chromadb",
    "ragas",
    "python-dotenv",
    "openai",
    "rouge-score",      # ← new
    "nltk",             # ← new
    "numpy",            # ← new
]
```

Run the non-LLM evaluator with:

```bash
uv run python -m rag_evaluation.evaluation.non_llm_evaluator
```

For completely free evaluation (zero API calls — ROUGE and BLEU only):

```bash
# Edit the last line in non_llm_evaluator.py:
results = evaluate_non_llm(use_semantic=False)
```

---

## 11. Hybrid Strategy — When to Use What

This is the recommended evaluation workflow combining both approaches:

```
┌──────────────────────────────────────────────────────────────────┐
│  TIER 1 — Dev Run (every code change, ~10 seconds, free)         │
│  Tool    : non_llm_evaluator.py  with use_semantic=False         │
│  Metrics : ROUGE-L + BLEU                                        │
│  Cost    : $0.00 (no API calls)                                  │
│  Use for : Catching obvious regressions during active coding     │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  TIER 2 — Smart Dev Run (~30 seconds, near-free)                 │
│  Tool    : non_llm_evaluator.py  with use_semantic=True          │
│  Metrics : ROUGE-L + BLEU + Semantic Similarity + Ctx Relevance  │
│  Cost    : ~$0.001 (embedding calls only)                        │
│  Use for : Better signal when iterating on prompts or chunking   │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  TIER 3 — Full RAGAS Run (~3 minutes, moderate cost)             │
│  Tool    : evaluator.py  (RAGAS, LLM-as-judge)                  │
│  Metrics : Faithfulness + Answer Relevancy + Ctx Precision/Recall│
│  Cost    : ~$0.05–0.10                                           │
│  Use for : Before significant changes, weekly baseline, releases │
└──────────────────────────────────────────────────────────────────┘
```

---

## 12. Score Interpretation Guide

### ROUGE-L (most useful ROUGE variant for RAG)

| Score | Interpretation |
|---|---|
| 0.8 – 1.0 | Answer closely mirrors ground truth wording |
| 0.6 – 0.8 | Good content coverage, some phrasing differences |
| 0.4 – 0.6 | Partial coverage — may be missing key information |
| 0.0 – 0.4 | Low word overlap — either very different phrasing or wrong answer |

### BLEU

| Score | Interpretation |
|---|---|
| 0.6 – 1.0 | High precision — answer phrases closely match reference |
| 0.3 – 0.6 | Moderate match — some phrases align |
| 0.0 – 0.3 | Low precision — common for paraphrased correct answers |

> BLEU scores tend to be lower than ROUGE for RAG because generative answers
> are often longer and more varied than ground truth. A BLEU of 0.3 can still
> indicate a correct answer.

### Semantic Similarity

| Score | Interpretation |
|---|---|
| 0.90 – 1.00 | Near-identical meaning |
| 0.80 – 0.90 | Very similar meaning — likely correct |
| 0.70 – 0.80 | Related but potentially missing details |
| 0.60 – 0.70 | Weak relationship — may be off-topic or incorrect |
| Below 0.60  | Likely wrong or completely unrelated |

### Context Relevance (Embedding-based Precision Proxy)

| Score | Interpretation |
|---|---|
| 0.85 – 1.00 | All retrieved chunks are highly relevant |
| 0.75 – 0.85 | Most chunks relevant — one may be marginal |
| 0.65 – 0.75 | Mixed retrieval — consider reducing top_k |
| Below 0.65  | Poor retrieval — review embedding model or chunk size |

### Exact Match
Binary — 1.0 or 0.0. Only meaningful for short factual answers.
A score of 0.0 does NOT mean the answer is wrong — it may just be phrased differently.
Always cross-check with Semantic Similarity before concluding an answer is incorrect.

---

## Expected Results for This Project

Based on your pipeline's current performance (from RAGAS evaluation), here is
what to expect from non-LLM metrics:

| Question Type | Expected ROUGE-L | Expected SemanticSim |
|---|---|---|
| Factual lookup (dates, names) | 0.55 – 0.75 | 0.88 – 0.96 |
| Definition questions | 0.45 – 0.65 | 0.82 – 0.92 |
| Out-of-context (Bitcoin, FIFA) | 0.05 – 0.20 | 0.40 – 0.60 |
| Multi-detail questions | 0.40 – 0.65 | 0.80 – 0.90 |

The out-of-context questions should score very low on both ROUGE and Semantic
Similarity since "I don't have enough context" shares little with the ground truth
"The documents do not contain information about...". This is correct behaviour.