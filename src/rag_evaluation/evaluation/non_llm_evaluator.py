"""
non_llm_evaluator.py
────────────────────
Traditional ML evaluation — no LLM API calls required.

Metrics implemented:
  - Exact Match
  - String Presence
  - ROUGE-1, ROUGE-2, ROUGE-L  (via rouge-score library)
  - BLEU Score                  (via nltk)
  - Semantic Similarity         (via OpenAI embeddings — cheap, not LLM)
  - Context Relevance           (embedding similarity, query vs chunks)

Run with:
    uv run python -m rag_evaluation.evaluation.non_llm_evaluator

For completely free evaluation (zero API calls):
    Set use_semantic=False in the __main__ block below.
"""

import csv
import os
import re

import numpy as np
from dotenv import load_dotenv
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from openai import OpenAI
from rouge_score import rouge_scorer

from rag_evaluation.evaluation.test_dataset import TEST_CASES
from rag_evaluation.pipeline import ask

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
    """
    Get OpenAI embedding vector for a text string.
    Uses text-embedding-ada-002 — NOT an LLM, just a vector encoder.
    Cost: ~$0.0001 per 1K tokens. 18 questions ≈ $0.001 total.
    """
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
    )
    return response.data[0].embedding


def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """
    Cosine similarity between two embedding vectors.
    Formula: dot(a, b) / (||a|| × ||b||)
    Range: 0.0 (unrelated) to 1.0 (identical meaning)
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


# ── Metric 1: Exact Match ─────────────────────────────────────────────────────

def exact_match(answer: str, ground_truth: str) -> float:
    """
    Returns 1.0 if normalised answer == normalised ground truth, else 0.0.

    Best for  : short factual answers, numbers, names
    Weakness  : any phrasing difference = score 0, even if answer is correct
    """
    return 1.0 if normalise(answer) == normalise(ground_truth) else 0.0


# ── Metric 2: String Presence ─────────────────────────────────────────────────

def extract_key_terms(ground_truth: str) -> list:
    """
    Auto-extract meaningful terms from ground truth.
    Filters stop words and short words.
    For best results, define required_terms manually per test case.
    """
    stop_words = {
        "that", "this", "with", "from", "they", "have", "been", "were",
        "which", "their", "there", "about", "also", "more", "than", "into",
        "some", "such", "when", "what", "where", "would", "could", "should",
        "these", "those", "then", "than", "will", "does",
    }
    words = re.findall(r'\b[a-zA-Z]{4,}\b', ground_truth.lower())
    return list(set(w for w in words if w not in stop_words))


def string_presence(answer: str, required_terms: list) -> float:
    """
    Fraction of required_terms found in the answer (case-insensitive).

    Score = found_terms / total_required_terms
    Best for  : checking factual term coverage
    Weakness  : "CO2" ≠ "carbon dioxide" — use semantic_similarity for synonyms
    """
    if not required_terms:
        return 0.0
    answer_lower = answer.lower()
    found = sum(1 for term in required_terms if term.lower() in answer_lower)
    return round(found / len(required_terms), 4)


# ── Metric 3: ROUGE Scores ────────────────────────────────────────────────────

def rouge_scores(answer: str, ground_truth: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    ROUGE-1  : unigram (single word) overlap — broad coverage signal
    ROUGE-2  : bigram (two-word phrase) overlap — stricter, captures phrasing
    ROUGE-L  : longest common subsequence — best for RAG (handles word reordering)

    All scores are F1 = harmonic mean of Precision and Recall.
    use_stemmer=True normalises "learning" / "learned" / "learns" → same stem.

    Best for  : measuring content coverage in longer answers
    Weakness  : "not good" and "bad" score 0 similarity despite same meaning
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
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
    Sentence-level BLEU score using NLTK.

    BLEU is precision-oriented: how much of the answer appears in the reference?
    Uses SmoothingFunction.method1 to prevent score=0 when higher n-grams are absent.

    Best for  : precise short answers, translation-style evaluation
    Weakness  : penalises correct answers that are longer than ground truth (brevity penalty)
                scores tend to be lower than ROUGE for generative RAG answers
    """
    reference  = [normalise(ground_truth).split()]
    hypothesis = normalise(answer).split()
    smoothing  = SmoothingFunction().method1
    score = sentence_bleu(reference, hypothesis, smoothing_function=smoothing)
    return round(float(score), 4)


# ── Metric 5: Semantic Similarity ────────────────────────────────────────────

def semantic_similarity(answer: str, ground_truth: str) -> float:
    """
    Cosine similarity between embedding vectors of answer and ground_truth.

    This is the most powerful non-LLM metric for RAG evaluation.
    Captures meaning, handles paraphrasing and synonyms.

    How it works:
      1. Both texts → 1536-dimensional embedding vectors via ada-002
      2. Cosine similarity between the two vectors
      3. High similarity = same meaning, regardless of exact wording

    Example:
      "Yuri Gagarin was first in space on April 12 1961"
      "The first human in space was Gagarin in 1961"
      → cosine similarity ≈ 0.94 (despite different structure)

    Cost    : ~$0.0001 per 1K tokens (NOT an LLM, just an encoder)
    Best for: answer correctness evaluation, especially paraphrased answers
    """
    embed_answer    = get_embedding(answer)
    embed_reference = get_embedding(ground_truth)
    return round(cosine_similarity(embed_answer, embed_reference), 4)


# ── Metric 6: Context Relevance ───────────────────────────────────────────────

def context_relevance(query: str, contexts: list, threshold: float = 0.75) -> dict:
    """
    Measures how relevant each retrieved chunk is to the query using embeddings.
    Approximates RAGAS Context Precision without any LLM call.

    How it works:
      1. Embed the query → query_vector
      2. Embed each retrieved chunk → chunk_vector[i]
      3. Compute cosine_similarity(query_vector, chunk_vector[i]) per chunk
      4. avg_relevance = mean of all chunk scores
      5. precision_proxy = fraction of chunks scoring above threshold

    Returns:
      avg_relevance    : mean cosine similarity across all chunks
      precision_proxy  : fraction of chunks above threshold (default 0.75)
      chunk_scores     : individual relevance score per chunk
    """
    query_embed  = get_embedding(query)
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
        "avg_relevance":   avg,
        "precision_proxy": precision,
        "chunk_scores":    chunk_scores,
    }


# ── Full Non-LLM Evaluator ────────────────────────────────────────────────────

def evaluate_non_llm(test_cases: list = None, use_semantic: bool = True) -> list:
    """
    Run all non-LLM metrics across the test dataset.

    Args:
        test_cases   : defaults to TEST_CASES from test_dataset.py
        use_semantic : include semantic similarity + context relevance
                       Set False for completely free evaluation (ROUGE + BLEU only)
    """
    test_cases = test_cases or TEST_CASES

    print("=" * 60)
    print("  RAG Evaluation — Non-LLM Metrics")
    if use_semantic:
        print("  Mode: ROUGE + BLEU + Semantic Similarity + Context Relevance")
        print("  Cost: ~$0.001 (embedding calls only, no LLM judge)")
    else:
        print("  Mode: ROUGE + BLEU only (zero API calls)")
        print("  Cost: $0.00")
    print("=" * 60)

    results = []

    for i, case in enumerate(test_cases, 1):
        question     = case["question"]
        ground_truth = case["ground_truth"]

        print(f"\n  [{i:02d}/{len(test_cases)}] {question[:55]}...")

        # Run the full RAG pipeline to get answer + contexts
        pipeline_result = ask(question)
        answer   = pipeline_result["answer"]
        contexts = pipeline_result["contexts"]

        # 1. Exact Match
        em = exact_match(answer, ground_truth)

        # 2. String Presence
        key_terms = extract_key_terms(ground_truth)
        sp = string_presence(answer, key_terms)

        # 3. ROUGE
        rouge = rouge_scores(answer, ground_truth)

        # 4. BLEU
        bleu = bleu_score(answer, ground_truth)

        # 5 + 6. Semantic metrics (optional — requires embedding API)
        sem_sim = None
        ctx_rel = None
        if use_semantic:
            sem_sim = semantic_similarity(answer, ground_truth)
            ctx_rel = context_relevance(question, contexts)

        result = {
            "question":            question,
            "ground_truth":        ground_truth,
            "answer":              answer,
            "exact_match":         em,
            "string_presence":     sp,
            "rouge1":              rouge["rouge1"],
            "rouge2":              rouge["rouge2"],
            "rougeL":              rouge["rougeL"],
            "bleu":                bleu,
            "semantic_sim":        sem_sim,
            "ctx_avg_relevance":   ctx_rel["avg_relevance"]   if ctx_rel else None,
            "ctx_precision_proxy": ctx_rel["precision_proxy"] if ctx_rel else None,
        }
        results.append(result)

        # Per-question summary line
        line = (f"      ROUGE-L: {rouge['rougeL']:.3f}  "
                f"BLEU: {bleu:.3f}")
        if sem_sim is not None:
            line += f"  SemanticSim: {sem_sim:.3f}"
        if ctx_rel is not None:
            line += f"  CtxRelevance: {ctx_rel['avg_relevance']:.3f}"
        print(line)

    return results


def print_non_llm_report(results: list):
    """Print averaged summary and save per-question CSV."""

    print("\n" + "=" * 60)
    print("  Non-LLM Evaluation — Summary")
    print("=" * 60)

    base_metrics = ["exact_match", "string_presence", "rouge1", "rouge2", "rougeL", "bleu"]
    sem_metrics  = ["semantic_sim", "ctx_avg_relevance", "ctx_precision_proxy"]

    all_metrics = base_metrics + (
        [m for m in sem_metrics if results[0].get(m) is not None]
    )

    for metric in all_metrics:
        vals = [r[metric] for r in results if r.get(metric) is not None]
        if not vals:
            continue
        avg    = sum(vals) / len(vals)
        bar    = "█" * int(avg * 20)
        spacer = "░" * (20 - int(avg * 20))
        print(f"  {metric:<24} {avg:.3f}  [{bar}{spacer}]")

    print("=" * 60)

    # Save to CSV
    output_path = "non_llm_evaluation_results.csv"
    fieldnames  = list(results[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  💾  Results saved to: {output_path}\n")


if __name__ == "__main__":
    # use_semantic=True  → ROUGE + BLEU + Semantic Similarity (~$0.001)
    # use_semantic=False → ROUGE + BLEU only                  ($0.00)
    results = evaluate_non_llm(use_semantic=True)
    print_non_llm_report(results)