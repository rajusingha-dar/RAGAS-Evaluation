"""Generate answers from retrieved context using OpenAI."""
# TODO: implement generator
"""
generator.py
────────────
Take retrieved chunks + a question, call OpenAI, return a grounded answer.

Run a quick manual test with:
    uv run python -m rag_evaluation.generator
"""

import os

from dotenv import load_dotenv
# from langchain.schema import Document
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ── LLM config ────────────────────────────────────────────────────────────────
# temperature=0 is critical for evaluation — deterministic answers mean
# repeated runs of the same question produce consistent scores.
LLM_MODEL   = "gpt-3.5-turbo"
TEMPERATURE = 0


# ── Prompt template ───────────────────────────────────────────────────────────
# This prompt is intentionally strict: "only use the context provided".
# This is what makes RAGAS Faithfulness meaningful — if the LLM answers
# from its own training knowledge and ignores context, faithfulness drops.
SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly \
based on the provided context.

Rules:
- Only use information from the context below to answer the question.
- If the answer is not found in the context, say: "I don't have enough context to answer this question."
- Do not use any prior knowledge outside of the context.
- Keep your answer concise and factual.

Context:
{context}
"""

HUMAN_PROMPT = "Question: {question}"


def build_chain():
    """
    Build a simple LangChain chain:
        prompt → LLM → string output parser

    Using LCEL (LangChain Expression Language) with the pipe operator |
    This is the modern LangChain way to compose chains.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  HUMAN_PROMPT),
    ])

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        openai_api_key=OPENAI_API_KEY,
    )

    # pipe: prompt → llm → parse output to plain string
    chain = prompt | llm | StrOutputParser()

    return chain


def format_context(docs: list[Document]) -> str:
    """
    Flatten the list of retrieved Document objects into a single context string.

    Each chunk is separated by a divider and labelled with its source file
    so the LLM knows which document each piece of context comes from.
    """
    sections = []
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        sections.append(f"[{i}] Source: {source}\n{doc.page_content}")

    return "\n\n---\n\n".join(sections)


def generate(question: str, docs: list[Document]) -> dict:
    """
    Generate an answer grounded in the retrieved documents.

    Returns a dict with:
      - question  : original question
      - answer    : LLM generated answer
      - contexts  : list of chunk texts (needed by RAGAS)
      - sources   : list of source filenames
    """
    context_str = format_context(docs)
    chain       = build_chain()

    answer = chain.invoke({
        "context":  context_str,
        "question": question,
    })

    return {
        "question": question,
        "answer":   answer,
        "contexts": [doc.page_content for doc in docs],   # RAGAS expects a list of strings
        "sources":  [os.path.basename(doc.metadata.get("source", "")) for doc in docs],
    }


# ── Manual test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from rag_evaluation.retriever import retrieve

    test_questions = [
        "What are the three types of machine learning?",
        "What causes climate change?",
        "Who was the first human in space and when did it happen?",
        "What is the greenhouse effect?",
    ]

    print("=" * 55)
    print("  RAG Evaluation — Generator Test")
    print("=" * 55)

    for question in test_questions:
        print(f"\n❓  Question : {question}")

        # Step 1: retrieve relevant chunks
        docs = retrieve(question)

        # Step 2: generate grounded answer
        result = generate(question, docs)

        print(f"💬  Answer   : {result['answer']}")
        print(f"📄  Sources  : {', '.join(result['sources'])}")
        print("-" * 55)