# E-Library — RAG Query System & Thematic Analysis Pipeline

A personal research tool built around a **22-volume academic library** authored by Gerry Stahl,
primarily on computer-supported collaborative learning (CSCL) and related topics (philosophy of
mind, dynamic geometry, collaborative design).

The project has two components:

1. **RAG query system** — ask free-form questions about the library in natural language and receive
   grounded, cited answers. Uses hybrid retrieval (FAISS dense search + BM25 keyword search +
   Reciprocal Rank Fusion + cross-encoder reranking) backed by Anthropic Claude or local Ollama
   models.

2. **Thematic analysis pipeline** — k-means clustering of the 17,441 base text chunks into 20
   semantic clusters, self-citation extraction (2,852 inline Stahl citations), and cluster
   publication timeline visualisations tracing thematic development across 1970–2026.

---

## What is in this repo

| Path | Contents |
|---|---|
| `scripts/` | ~115 Python scripts — RAG pipeline, build tools, analysis, reporting |
| `chunkers/`, `embedders/`, `parsers/`, `summarizers/` | Core pipeline modules |
| `reports/` | CSV, TXT, JSON analysis outputs (cluster assignments, citation data, timelines) |
| `documents/` | Full project guide, planning docs, chapter/book metrics |
| `summarizers/book summaries/` | Claude-generated summaries for all 22 books |
| `webpages/` | Generated HTML pages for the e-library |
| `cache/` | Python source files and prompt templates (not the data files) |
| `memory/elibrary_memory.jsonl` | MCP knowledge graph (42 entities, 27 relations) |
| `.github/copilot-instructions.md` | Copilot Chat context instructions |

## What is NOT in this repo

Large or copyrighted files are excluded via `.gitignore` and backed up separately:

- `sourcepdfs/` — the 22 source books (copyright)
- `cache/elibrary_cache.pkl` (86 MB) — parsed text + Claude summaries; backed up to local disk
- `vector_store/*.faiss` — FAISS vector indexes (recreatable in ~20 min)
- `.venv/` — Python virtual environment (recreatable via `pip install -r requirements.txt`)
- Ollama models (~22 GB in `~/.ollama/`) and HuggingFace model cache (~18 GB in `~/.cache/huggingface/`)

---

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com) (for local models): `ollama pull qwen2.5:14b`
- Anthropic API key in environment: `export ANTHROPIC_API_KEY=sk-...`
- Node.js ≥18 (for MCP memory server): `brew install node`

---

## Quick start

```bash
# 1. Clone and create venv
git clone https://github.com/GerryStahl/E-Library.git elibrary
cd elibrary
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Restore cache from backup
# copy cache/elibrary_cache.pkl from backup disk

# 3. Rebuild vector indexes (~20 min total)
.venv/bin/python scripts/build_vector_store.py
.venv/bin/python scripts/build_chapter_summary_vectors.py
.venv/bin/python scripts/build_bm25_index.py

# 4. Launch the web chat interface
.venv/bin/python scripts/query_app.py
# → opens http://localhost:7860
```

---

## Full documentation

See [`documents/GUIDE_TO_GERRY_RAG_QUERY_SYSTEM.md`](documents/GUIDE_TO_GERRY_RAG_QUERY_SYSTEM.md)
for complete documentation covering:

- System architecture (chunking, embeddings, FAISS, BM25, RRF, cross-encoder reranking, Claude/Ollama generation)
- Retrieval parameters and usage tips
- Analysis pipeline steps (clustering, self-citation extraction, timeline visualisation)
- Key findings from the cluster analysis
- Script inventory (~115 scripts)
- Disaster recovery procedure
- GitHub backup and access management
