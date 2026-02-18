"""Orchestrate retrieve → generate. Returns answer + contexts."""
# TODO: wire up pipeline
"""
pipeline.py
───────────
Orchestrates retrieve → generate in a single ask() call.
Returns both the answer AND the retrieved contexts (required by RAGAS).

Run a quick manual test with:
    uv run python -m rag_evaluation.pipeline
"""

from rag_evaluation.generator import generate
from rag_evaluation.retriever import retrieve


def ask(question: str, top_k: int = 3) -> dict:
    """
    Full RAG pipeline: question → retrieve → generate → return result.

    Returns a dict shaped exactly for RAGAS evaluation:
    {
        "question"  : str,               # original question
        "answer"    : str,               # LLM generated answer
        "contexts"  : list[str],         # retrieved chunk texts
        "sources"   : list[str],         # source filenames
    }

    RAGAS needs question + answer + contexts to compute all four metrics:
      - Faithfulness        : is the answer grounded in contexts?
      - Answer Relevance    : does the answer address the question?
      - Context Precision   : are the top-ranked chunks actually relevant?
      - Context Recall      : did we retrieve all necessary information?
    """
    # Step 1 — Retrieve relevant chunks from ChromaDB
    docs = retrieve(question, top_k=top_k)

    # Step 2 — Generate a grounded answer from those chunks
    result = generate(question, docs)

    return result


# ── Manual test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    test_questions = [
        "What is reinforcement learning?",
        "What are the effects of climate change on sea levels?",
        "What is the International Space Station?",
    ]

    print("=" * 55)
    print("  RAG Evaluation — Full Pipeline Test")
    print("=" * 55)

    for question in test_questions:
        print(f"\n❓  Question : {question}")
        result = ask(question)
        print(f"💬  Answer   : {result['answer']}")
        print(f"📄  Sources  : {', '.join(set(result['sources']))}")
        print(f"🧩  Chunks   : {len(result['contexts'])} retrieved")
        print("-" * 55)