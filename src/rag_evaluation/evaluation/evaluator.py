"""
evaluator.py
────────────
Run RAGAS metrics against the full pipeline using the ground truth test dataset.

Run with:
    uv run python -m rag_evaluation.evaluation.evaluator
"""

import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
# Use the legacy metric classes that are compatible with evaluate()
# The new ragas.metrics.collections classes use a different base class
# that evaluate() does not accept — this is confirmed in ragas/evaluation.py
from ragas.metrics._faithfulness import faithfulness
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from rag_evaluation.pipeline import ask
from rag_evaluation.evaluation.test_dataset import TEST_CASES

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def build_ragas_dataset() -> Dataset:
    """
    Run every question in TEST_CASES through the pipeline and collect:
      - question      : the input question
      - answer        : the LLM generated answer
      - contexts      : list of retrieved chunk texts
      - ground_truth  : the ideal answer from our test dataset

    This dict structure is exactly what RAGAS expects as input.
    RAGAS uses an LLM-as-judge internally to score each metric.
    """
    print("\n🔄  Running all test questions through the pipeline...")
    print(f"    Total questions: {len(TEST_CASES)}\n")

    questions     = []
    answers       = []
    contexts_list = []
    ground_truths = []

    for i, test_case in enumerate(TEST_CASES, 1):
        question     = test_case["question"]
        ground_truth = test_case["ground_truth"]

        print(f"  [{i:02d}/{len(TEST_CASES)}] {question[:60]}...")

        # Run the full RAG pipeline
        result = ask(question)

        questions.append(question)
        answers.append(result["answer"])
        contexts_list.append(result["contexts"])   # list of strings
        ground_truths.append(ground_truth)

    print(f"\n    ✅  Collected {len(questions)} results — building RAGAS dataset\n")

    # RAGAS requires a Hugging Face Dataset object
    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts_list,
        "ground_truth": ground_truths,
    })

    return dataset


def run_evaluation(dataset: Dataset):
    """
    Pass the dataset to RAGAS and compute the four core RAG metrics.

    What each metric measures:
    ──────────────────────────
    Faithfulness        : Is every claim in the answer supported by the retrieved contexts?
                          Score 1.0 = fully grounded, 0.0 = hallucinated
                          Computed by: LLM extracts claims from answer, checks each against contexts

    Answer Relevancy    : Does the answer actually address the question asked?
                          Score 1.0 = directly answers, 0.0 = off-topic
                          Computed by: LLM generates reverse questions from the answer,
                          measures embedding similarity to original question

    Context Precision   : Are the most relevant chunks ranked at the top?
                          Score 1.0 = relevant chunks all at top, 0.0 = buried at bottom
                          Computed by: Average Precision formula — rewards relevant chunks
                          ranked higher

    Context Recall      : Did we retrieve all information needed to answer correctly?
                          Score 1.0 = all ground truth info found in context,
                          0.0 = key information missing from retrieved chunks
                          Computed by: LLM checks if each sentence in ground_truth
                          can be attributed to the retrieved contexts
    """
    print("=" * 55)
    print("  Running RAGAS Evaluation...")
    print("  (Each question is scored by an LLM-as-judge — this may take 1-2 minutes)")
    print("=" * 55 + "\n")

    # Pass LangChain LLM directly to evaluate() — it wraps it internally
    # (confirmed in ragas/evaluation.py line 158-161)
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    llm        = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    return results


def print_report(results):
    """
    Print a clean summary of RAGAS scores.

    EvaluationResult has:
      .scores  — list of per-row dicts: [{"faithfulness": 0.9, ...}, ...]
      .to_pandas() — full DataFrame
    """
    print("\n" + "=" * 55)
    print("  RAGAS Evaluation Results")
    print("=" * 55)

    # Collect all metric keys from the per-row score dicts
    all_keys = set()
    for row in results.scores:
        all_keys.update(row.keys())

    def avg(key):
        vals = [row[key] for row in results.scores if row.get(key) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    def find_key(substring):
        for k in all_keys:
            if substring in k.lower():
                return k
        return None

    metric_map = {
        "Faithfulness":      find_key("faithfulness"),
        "Answer Relevancy":  find_key("answer_relevancy") or find_key("answerrelevancy"),
        "Context Precision": find_key("context_precision") or find_key("contextprecision"),
        "Context Recall":    find_key("context_recall") or find_key("contextrecall"),
    }

    scores = {name: avg(key) if key else float("nan") for name, key in metric_map.items()}

    for metric, score in scores.items():
        if score != score:  # nan
            print(f"  {metric:<22}  N/A")
        else:
            bar    = "\u2588" * int(score * 20)
            spacer = "\u2591" * (20 - int(score * 20))
            print(f"  {metric:<22} {score:.3f}  [{bar}{spacer}]")

    print("=" * 55)
    valid = [s for s in scores.values() if s == s]
    overall = sum(valid) / len(valid) if valid else 0.0
    print(f"\n  Overall Average: {overall:.3f}")
    print("\n  Score Guide:")
    print("    0.8 - 1.0  Excellent")
    print("    0.6 - 0.8  Good, room to improve")
    print("    0.4 - 0.6  Needs tuning")
    print("    0.0 - 0.4  Significant issues\n")

    # Per-question breakdown
    print("=" * 55)
    print("  Per-Question Breakdown")
    print("=" * 55)
    df = results.to_pandas()
    print("  DataFrame columns:", list(df.columns))

    def find_col(substring):
        for c in df.columns:
            if substring in c.lower():
                return c
        return None

    col_q  = find_col("user_input") or find_col("question")
    col_f  = find_col("faithfulness")
    col_ar = find_col("answer_relevancy") or find_col("answerrelevancy")
    col_cp = find_col("context_precision") or find_col("contextprecision")
    col_cr = find_col("context_recall") or find_col("contextrecall")

    cols = [c for c in [col_q, col_f, col_ar, col_cp, col_cr] if c]
    df_display = df[cols].copy()
    if col_q:
        df_display[col_q] = df_display[col_q].astype(str).str[:45] + "..."

    print(df_display.to_string(index=False))
    print()

    output_path = "evaluation_results.csv"
    df.to_csv(output_path, index=False)
    print(f"  Results saved to: {output_path}\n")


if __name__ == "__main__":
    dataset = build_ragas_dataset()
    results = run_evaluation(dataset)
    print_report(results)










# """
# evaluator.py
# ────────────
# Run RAGAS metrics against the full pipeline using the ground truth test dataset.

# Run with:
#     uv run python -m rag_evaluation.evaluation.evaluator
# """

# import os
# from dotenv import load_dotenv
# from datasets import Dataset
# from ragas import evaluate
# # Use the legacy metric classes that are compatible with evaluate()
# # The new ragas.metrics.collections classes use a different base class
# # that evaluate() does not accept — this is confirmed in ragas/evaluation.py
# from ragas.metrics._faithfulness import faithfulness
# from ragas.metrics._answer_relevance import answer_relevancy
# from ragas.metrics._context_precision import context_precision
# from ragas.metrics._context_recall import context_recall
# from rag_evaluation.pipeline import ask
# from rag_evaluation.evaluation.test_dataset import TEST_CASES

# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# def build_ragas_dataset() -> Dataset:
#     """
#     Run every question in TEST_CASES through the pipeline and collect:
#       - question      : the input question
#       - answer        : the LLM generated answer
#       - contexts      : list of retrieved chunk texts
#       - ground_truth  : the ideal answer from our test dataset

#     This dict structure is exactly what RAGAS expects as input.
#     RAGAS uses an LLM-as-judge internally to score each metric.
#     """
#     print("\n🔄  Running all test questions through the pipeline...")
#     print(f"    Total questions: {len(TEST_CASES)}\n")

#     questions     = []
#     answers       = []
#     contexts_list = []
#     ground_truths = []

#     for i, test_case in enumerate(TEST_CASES, 1):
#         question     = test_case["question"]
#         ground_truth = test_case["ground_truth"]

#         print(f"  [{i:02d}/{len(TEST_CASES)}] {question[:60]}...")

#         # Run the full RAG pipeline
#         result = ask(question)

#         questions.append(question)
#         answers.append(result["answer"])
#         contexts_list.append(result["contexts"])   # list of strings
#         ground_truths.append(ground_truth)

#     print(f"\n    ✅  Collected {len(questions)} results — building RAGAS dataset\n")

#     # RAGAS requires a Hugging Face Dataset object
#     dataset = Dataset.from_dict({
#         "question":     questions,
#         "answer":       answers,
#         "contexts":     contexts_list,
#         "ground_truth": ground_truths,
#     })

#     return dataset


# def run_evaluation(dataset: Dataset):
#     """
#     Pass the dataset to RAGAS and compute the four core RAG metrics.

#     What each metric measures:
#     ──────────────────────────
#     Faithfulness        : Is every claim in the answer supported by the retrieved contexts?
#                           Score 1.0 = fully grounded, 0.0 = hallucinated
#                           Computed by: LLM extracts claims from answer, checks each against contexts

#     Answer Relevancy    : Does the answer actually address the question asked?
#                           Score 1.0 = directly answers, 0.0 = off-topic
#                           Computed by: LLM generates reverse questions from the answer,
#                           measures embedding similarity to original question

#     Context Precision   : Are the most relevant chunks ranked at the top?
#                           Score 1.0 = relevant chunks all at top, 0.0 = buried at bottom
#                           Computed by: Average Precision formula — rewards relevant chunks
#                           ranked higher

#     Context Recall      : Did we retrieve all information needed to answer correctly?
#                           Score 1.0 = all ground truth info found in context,
#                           0.0 = key information missing from retrieved chunks
#                           Computed by: LLM checks if each sentence in ground_truth
#                           can be attributed to the retrieved contexts
#     """
#     print("=" * 55)
#     print("  Running RAGAS Evaluation...")
#     print("  (Each question is scored by an LLM-as-judge — this may take 1-2 minutes)")
#     print("=" * 55 + "\n")

#     # Pass LangChain LLM directly to evaluate() — it wraps it internally
#     # (confirmed in ragas/evaluation.py line 158-161)
#     from langchain_openai import ChatOpenAI, OpenAIEmbeddings

#     llm        = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
#     embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

#     results = evaluate(
#         dataset=dataset,
#         metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
#         llm=llm,
#         embeddings=embeddings,
#     )

#     return results


# def print_report(results):
#     """
#     Print a clean summary of RAGAS scores with interpretation guidance.
#     """
#     print("\n" + "=" * 55)
#     print("  RAGAS Evaluation Results")
#     print("=" * 55)

#     def to_float(val):
#         """RAGAS may return a list or a float depending on version — normalise to float."""
#         if isinstance(val, list):
#             val = [v for v in val if v is not None]
#             return sum(val) / len(val) if val else 0.0
#         return float(val) if val is not None else 0.0

#     # RAGAS result keys can vary by version — find them dynamically
#     result_keys = list(results.keys()) if hasattr(results, "keys") else []

#     def find_key(candidates):
#         for k in candidates:
#             if k in result_keys:
#                 return k
#         return None

#     scores = {
#         "Faithfulness":      to_float(results[find_key(["faithfulness", "Faithfulness"])]),
#         "Answer Relevancy":  to_float(results[find_key(["answer_relevancy", "AnswerRelevancy"])]),
#         "Context Precision": to_float(results[find_key(["context_precision", "ContextPrecision"])]),
#         "Context Recall":    to_float(results[find_key(["context_recall", "ContextRecall"])]),
#     }

#     for metric, score in scores.items():
#         bar    = "█" * int(score * 20)
#         spacer = "░" * (20 - int(score * 20))
#         print(f"  {metric:<22} {score:.3f}  [{bar}{spacer}]")

#     print("=" * 55)

#     overall = sum(scores.values()) / len(scores)
#     print(f"\n  Overall Average: {overall:.3f}")

#     print("\n  Score Guide:")
#     print("    0.8 – 1.0  🟢  Excellent")
#     print("    0.6 – 0.8  🟡  Good, room to improve")
#     print("    0.4 – 0.6  🟠  Needs tuning")
#     print("    0.0 – 0.4  🔴  Significant issues\n")

#     # Per-question breakdown
#     print("=" * 55)
#     print("  Per-Question Breakdown")
#     print("=" * 55)
#     df = results.to_pandas()

#     # Find actual column names dynamically
#     def find_col(candidates):
#         for c in candidates:
#             if c in df.columns:
#                 return c
#         return None

#     col_q   = find_col(["question"])
#     col_f   = find_col(["faithfulness", "Faithfulness"])
#     col_ar  = find_col(["answer_relevancy", "AnswerRelevancy"])
#     col_cp  = find_col(["context_precision", "ContextPrecision"])
#     col_cr  = find_col(["context_recall", "ContextRecall"])

#     cols = [c for c in [col_q, col_f, col_ar, col_cp, col_cr] if c]
#     df_display = df[cols].copy()
#     df_display[col_q] = df_display[col_q].str[:45] + "..."

#     print(df_display.to_string(index=False))
#     print()

#     # Save to CSV
#     output_path = "evaluation_results.csv"
#     df.to_csv(output_path, index=False)
#     print(f"  💾  Full results saved to: {output_path}\n")


# if __name__ == "__main__":
#     # Step 1 — Run all questions through the pipeline and collect results
#     dataset = build_ragas_dataset()

#     # Step 2 — Score with RAGAS
#     results = run_evaluation(dataset)

#     # DEBUG: print actual keys returned by RAGAS
#     print("RAGAS result keys:", list(results.keys()))
#     print("RAGAS result scores sample:", results.scores[:1])

#     # Step 3 — Print and save the report
#     print_report(results)





# """
# evaluator.py
# ────────────
# Run RAGAS metrics against the full pipeline using the ground truth test dataset.

# Run with:
#     uv run python -m rag_evaluation.evaluation.evaluator
# """

# import os
# from dotenv import load_dotenv
# from datasets import Dataset
# from ragas import evaluate
# # Use the legacy metric classes that are compatible with evaluate()
# # The new ragas.metrics.collections classes use a different base class
# # that evaluate() does not accept — this is confirmed in ragas/evaluation.py
# from ragas.metrics._faithfulness import faithfulness
# from ragas.metrics._answer_relevance import answer_relevancy
# from ragas.metrics._context_precision import context_precision
# from ragas.metrics._context_recall import context_recall
# from rag_evaluation.pipeline import ask
# from rag_evaluation.evaluation.test_dataset import TEST_CASES

# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# def build_ragas_dataset() -> Dataset:
#     """
#     Run every question in TEST_CASES through the pipeline and collect:
#       - question      : the input question
#       - answer        : the LLM generated answer
#       - contexts      : list of retrieved chunk texts
#       - ground_truth  : the ideal answer from our test dataset

#     This dict structure is exactly what RAGAS expects as input.
#     RAGAS uses an LLM-as-judge internally to score each metric.
#     """
#     print("\n🔄  Running all test questions through the pipeline...")
#     print(f"    Total questions: {len(TEST_CASES)}\n")

#     questions     = []
#     answers       = []
#     contexts_list = []
#     ground_truths = []

#     for i, test_case in enumerate(TEST_CASES, 1):
#         question     = test_case["question"]
#         ground_truth = test_case["ground_truth"]

#         print(f"  [{i:02d}/{len(TEST_CASES)}] {question[:60]}...")

#         # Run the full RAG pipeline
#         result = ask(question)

#         questions.append(question)
#         answers.append(result["answer"])
#         contexts_list.append(result["contexts"])   # list of strings
#         ground_truths.append(ground_truth)

#     print(f"\n    ✅  Collected {len(questions)} results — building RAGAS dataset\n")

#     # RAGAS requires a Hugging Face Dataset object
#     dataset = Dataset.from_dict({
#         "question":     questions,
#         "answer":       answers,
#         "contexts":     contexts_list,
#         "ground_truth": ground_truths,
#     })

#     return dataset


# def run_evaluation(dataset: Dataset):
#     """
#     Pass the dataset to RAGAS and compute the four core RAG metrics.

#     What each metric measures:
#     ──────────────────────────
#     Faithfulness        : Is every claim in the answer supported by the retrieved contexts?
#                           Score 1.0 = fully grounded, 0.0 = hallucinated
#                           Computed by: LLM extracts claims from answer, checks each against contexts

#     Answer Relevancy    : Does the answer actually address the question asked?
#                           Score 1.0 = directly answers, 0.0 = off-topic
#                           Computed by: LLM generates reverse questions from the answer,
#                           measures embedding similarity to original question

#     Context Precision   : Are the most relevant chunks ranked at the top?
#                           Score 1.0 = relevant chunks all at top, 0.0 = buried at bottom
#                           Computed by: Average Precision formula — rewards relevant chunks
#                           ranked higher

#     Context Recall      : Did we retrieve all information needed to answer correctly?
#                           Score 1.0 = all ground truth info found in context,
#                           0.0 = key information missing from retrieved chunks
#                           Computed by: LLM checks if each sentence in ground_truth
#                           can be attributed to the retrieved contexts
#     """
#     print("=" * 55)
#     print("  Running RAGAS Evaluation...")
#     print("  (Each question is scored by an LLM-as-judge — this may take 1-2 minutes)")
#     print("=" * 55 + "\n")

#     # Pass LangChain LLM directly to evaluate() — it wraps it internally
#     # (confirmed in ragas/evaluation.py line 158-161)
#     from langchain_openai import ChatOpenAI, OpenAIEmbeddings

#     llm        = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
#     embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

#     results = evaluate(
#         dataset=dataset,
#         metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
#         llm=llm,
#         embeddings=embeddings,
#     )

#     return results


# def print_report(results):
#     """
#     Print a clean summary of RAGAS scores with interpretation guidance.
#     """
#     print("\n" + "=" * 55)
#     print("  RAGAS Evaluation Results")
#     print("=" * 55)

#     def to_float(val):
#         """RAGAS may return a list or a float depending on version — normalise to float."""
#         if isinstance(val, list):
#             val = [v for v in val if v is not None]
#             return sum(val) / len(val) if val else 0.0
#         return float(val) if val is not None else 0.0

#     # RAGAS result keys can vary by version — find them dynamically
#     result_keys = list(results.keys()) if hasattr(results, "keys") else []

#     def find_key(candidates):
#         for k in candidates:
#             if k in result_keys:
#                 return k
#         return None

#     scores = {
#         "Faithfulness":      to_float(results[find_key(["faithfulness", "Faithfulness"])]),
#         "Answer Relevancy":  to_float(results[find_key(["answer_relevancy", "AnswerRelevancy"])]),
#         "Context Precision": to_float(results[find_key(["context_precision", "ContextPrecision"])]),
#         "Context Recall":    to_float(results[find_key(["context_recall", "ContextRecall"])]),
#     }

#     for metric, score in scores.items():
#         bar    = "█" * int(score * 20)
#         spacer = "░" * (20 - int(score * 20))
#         print(f"  {metric:<22} {score:.3f}  [{bar}{spacer}]")

#     print("=" * 55)

#     overall = sum(scores.values()) / len(scores)
#     print(f"\n  Overall Average: {overall:.3f}")

#     print("\n  Score Guide:")
#     print("    0.8 – 1.0  🟢  Excellent")
#     print("    0.6 – 0.8  🟡  Good, room to improve")
#     print("    0.4 – 0.6  🟠  Needs tuning")
#     print("    0.0 – 0.4  🔴  Significant issues\n")

#     # Per-question breakdown
#     print("=" * 55)
#     print("  Per-Question Breakdown")
#     print("=" * 55)
#     df = results.to_pandas()

#     # Find actual column names dynamically
#     def find_col(candidates):
#         for c in candidates:
#             if c in df.columns:
#                 return c
#         return None

#     col_q   = find_col(["question"])
#     col_f   = find_col(["faithfulness", "Faithfulness"])
#     col_ar  = find_col(["answer_relevancy", "AnswerRelevancy"])
#     col_cp  = find_col(["context_precision", "ContextPrecision"])
#     col_cr  = find_col(["context_recall", "ContextRecall"])

#     cols = [c for c in [col_q, col_f, col_ar, col_cp, col_cr] if c]
#     df_display = df[cols].copy()
#     df_display[col_q] = df_display[col_q].str[:45] + "..."

#     print(df_display.to_string(index=False))
#     print()

#     # Save to CSV
#     output_path = "evaluation_results.csv"
#     df.to_csv(output_path, index=False)
#     print(f"  💾  Full results saved to: {output_path}\n")


# if __name__ == "__main__":
#     # Step 1 — Run all questions through the pipeline and collect results
#     dataset = build_ragas_dataset()

#     # Step 2 — Score with RAGAS
#     results = run_evaluation(dataset)

#     # Step 3 — Print and save the report
#     print_report(results)