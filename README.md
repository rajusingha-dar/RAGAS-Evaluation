# RAG Evaluation Project

A practical hands-on project for evaluating RAG pipelines using:
- **LangChain** — RAG orchestration
- **OpenAI** — Embeddings + LLM
- **ChromaDB** — Vector store (Docker)
- **RAGAS** — Evaluation framework

## Quick Start
```bash
# 1. Start ChromaDB
cd docker && docker compose up -d

# 2. Install dependencies
uv sync

# 3. Copy and fill in your env
cp .env.example .env

# 4. Ingest documents
uv run python -m rag_evaluation.ingest

# 5. Run evaluation
uv run python -m rag_evaluation.evaluation.evaluator
```
