"""Load documents → chunk → embed → store in ChromaDB."""
# TODO: implement ingestion pipeline
"""
ingest.py
─────────
Load documents → chunk → embed → store in ChromaDB.

Run with:
    uv run python -m rag_evaluation.ingest
"""

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings

# ── Load environment variables from .env ─────────────────────────────────────
load_dotenv()

# ── Config (all values come from .env) ───────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
CHROMA_HOST      = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT      = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME  = os.getenv("COLLECTION_NAME", "rag_docs")

# Documents folder is two levels up from this file: src/rag_evaluation/ → project root
DOCUMENTS_DIR = Path(__file__).resolve().parents[2] / "documents"


def load_documents():
    """
    Use LangChain's DirectoryLoader to read every .txt file in documents/.
    Each file becomes a LangChain Document with:
      - page_content : raw text of the file
      - metadata     : {"source": "path/to/file.txt"}
    """
    print(f"\n📂  Loading documents from: {DOCUMENTS_DIR}")

    loader = DirectoryLoader(
        path=str(DOCUMENTS_DIR),
        glob="**/*.txt",            # pick up all .txt files recursively
        loader_cls=TextLoader,      # use TextLoader for plain text files
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )

    documents = loader.load()
    print(f"    ✅  Loaded {len(documents)} document(s)")
    for doc in documents:
        print(f"        • {Path(doc.metadata['source']).name}")

    return documents


def split_documents(documents):
    """
    Use RecursiveCharacterTextSplitter to break each document into chunks.

    Why RecursiveCharacterTextSplitter?
    It tries to split on natural boundaries in this order:
      paragraphs (\\n\\n) → sentences (\\n) → words ( ) → characters
    This preserves semantic meaning better than a naive fixed-size split.

    chunk_size=500   : each chunk is at most 500 characters
    chunk_overlap=50 : 50 characters are shared between adjacent chunks
                       so context isn't lost at boundaries
    """
    print("\n✂️   Splitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],  # explicit split priority
    )

    chunks = splitter.split_documents(documents)

    print(f"    ✅  Created {len(chunks)} chunk(s) from {len(documents)} document(s)")
    print(f"        avg chunk size ≈ {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

    return chunks


def get_chroma_client():
    """
    Create an HTTP client pointing at the ChromaDB Docker container.
    LangChain's Chroma integration accepts this client directly.
    """
    print(f"\n🔌  Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}...")

    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
    )

    # Quick heartbeat check — will raise if Chroma is not reachable
    client.heartbeat()
    print("    ✅  ChromaDB is reachable")

    return client


def reset_collection(client):
    """
    Delete the collection if it already exists so re-running ingest.py
    doesn't duplicate chunks.
    """
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"    🗑️   Deleted existing collection '{COLLECTION_NAME}'")


def store_in_chroma(chunks, client):
    """
    Embed every chunk with OpenAI and store in ChromaDB using LangChain's
    Chroma.from_documents() — this handles embedding + storage in one call.

    OpenAIEmbeddings uses 'text-embedding-ada-002' by default.
    Each chunk gets:
      - its embedding vector stored in Chroma
      - its page_content stored as the document
      - its metadata (source filename) stored alongside
    """
    print("\n🧠  Generating embeddings and storing in ChromaDB...")
    print("    (This calls the OpenAI Embeddings API — may take a few seconds)")

    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-ada-002",
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        client=client,
    )

    print(f"    ✅  Stored {len(chunks)} chunks in collection '{COLLECTION_NAME}'")

    return vector_store


def ingest_documents():
    """
    Main ingestion pipeline:
      load → split → connect → reset → embed + store
    """
    print("=" * 55)
    print("  RAG Evaluation — Document Ingestion Pipeline")
    print("=" * 55)

    # 1. Load raw documents
    documents = load_documents()

    # 2. Split into chunks
    chunks = split_documents(documents)

    # 3. Connect to ChromaDB
    client = get_chroma_client()

    # 4. Reset collection (idempotent re-runs)
    reset_collection(client)

    # 5. Embed and store
    vector_store = store_in_chroma(chunks, client)

    print("\n" + "=" * 55)
    print("  ✅  Ingestion complete!")
    print(f"      Documents : {len(documents)}")
    print(f"      Chunks    : {len(chunks)}")
    print(f"      Collection: {COLLECTION_NAME}")
    print(f"      ChromaDB  : {CHROMA_HOST}:{CHROMA_PORT}")
    print("=" * 55 + "\n")

    return vector_store


if __name__ == "__main__":
    ingest_documents()