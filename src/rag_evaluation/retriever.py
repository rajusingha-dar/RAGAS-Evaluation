"""Query ChromaDB and return relevant chunks."""
# TODO: implement retriever
"""
retriever.py
────────────
Connect to ChromaDB and retrieve relevant chunks for a given query.

Run a quick manual test with:
    uv run python -m rag_evaluation.retriever
"""

import os

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
CHROMA_HOST     = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT     = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_docs")

# How many chunks to retrieve per query — this is a key parameter for evaluation.
# Higher k = more context for the LLM but more noise for the retriever metrics.
TOP_K = 3


def get_vector_store():
    """
    Connect to the existing ChromaDB collection and return a LangChain
    Chroma vector store instance.

    NOTE: We use Chroma(...) not Chroma.from_documents() here because
    we are connecting to an EXISTING collection, not creating a new one.
    """
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-ada-002",
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        client=client,
    )

    return vector_store


def get_retriever(top_k: int = TOP_K):
    """
    Return a LangChain Retriever object from the vector store.

    Using as_retriever() gives us a standard LangChain Retriever interface
    which plugs directly into chains, pipelines, and RAGAS evaluation later.

    search_type="similarity" uses cosine similarity between the query
    embedding and stored chunk embeddings to rank results.
    """
    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    return retriever


def retrieve(query: str, top_k: int = TOP_K) -> list:
    """
    Retrieve the top-k most relevant chunks for a given query.

    Returns a list of LangChain Document objects, each containing:
      - page_content : the chunk text
      - metadata     : {"source": "path/to/file.txt"}

    This is the function the pipeline and evaluator will call.
    """
    retriever = get_retriever(top_k=top_k)
    docs = retriever.invoke(query)
    return docs


# ── Manual test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    test_queries = [
        "What are the types of machine learning?",
        "What causes climate change?",
        "Who was the first human in space?",
    ]

    print("=" * 55)
    print("  RAG Evaluation — Retriever Test")
    print("=" * 55)

    for query in test_queries:
        print(f"\n🔍  Query: {query}")
        print("-" * 55)

        results = retrieve(query)

        for i, doc in enumerate(results, 1):
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            preview = doc.page_content[:200].replace("\n", " ")
            print(f"  Chunk {i} [{source}]")
            print(f"  {preview}...")
            print()