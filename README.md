# E-Library — RAG Query System & Thematic Analysis Pipeline

A personal research tool built around a **22-volume academic library including ** authored by Gerry Stahl,
primarily on computer-supported collaborative learning (CSCL) and related topics (philosophy of
mind, dynamic geometry, collaborative design), but also including social philosophy, sculpture and ecology. 
The E-Library is available at: https://gerrystahl.net/elibrary .

The project has several components:

1. **RAG query system** — ask free-form questions about the library in natural language and receive
   grounded, cited answers. Uses hybrid retrieval (FAISS dense search + BM25 keyword search +
   Reciprocal Rank Fusion + cross-encoder reranking) backed by multiple versions of Anthropic Claude or local Ollama
   models, using a large semantic vector space of the e-library texts at various semantic levels.

2. **A custom, local query interface for analyzing the e-library** — Allows choice of LMM model and provides
	long-term memory context for generating meaningful responses to searches of the texts.

3. **Summaries of the books and chapters based on chunk semantics** — Summaries of each of the 22 books 
	and each of their 337 chapters were generated and are displayed on the webpage for the e-library.

4. **Thematic analysis pipeline** — k-means clustering of the 17,441 base text chunks into 20
   semantic clusters, self-citation extraction (2,852 inline Stahl citations), and cluster
   publication timeline visualisations tracing thematic development across period 1970–2026.

5. **Long-term memory for the chat Copilot** — supplemental context for queries through several coordinated resources:
- An overview Guide -- which is computer-readable (in MD format) as well as human-readable -- with technical detail and instructions.
- Github/copilot-instructions.md -- provides a variety of instructions to the chat Copilot on how to respond to queries.
- MCP Knowledge Graph (`memory/elibrary_memory.jsonl`) -- a JSONL queryable memory structure about the project.
- Auto-generated Project Status (`documents/project_status.md`) -- summarizes the current status of the ongoing project.
- In addition, query_history.json -- is a repository of previous queries; semantically similar queries are also used for context and deambiguity.

6. **GitHub repository for E-Library** — This public repo includes all the files of the project 
	except for the large vector spaces (which can be recreated) 
	and the LLM models (which can be downloaded). 

7. **Timeline analysis of litereary style**— The narrative text of the elibrary is analyzed in terms of many measures of literary style.
The various factors are then graphed to compare their historical variation (1967-2026) and their differences in thematic clusters of the writings.

---

## What is in this repo

| Path | Contents |
|---|---|
| `scripts/` | ~115 Python scripts — RAG pipeline, build tools, analysis, reporting |
| `chunkers/`, `embedders/`, `parsers/`, `summarizers/` | Core pipeline modules |
| `reports/` | CSV, TXT, JSON analysis outputs (cluster assignments, citation data, timelines) |
| `documents/` | Full project guide, planning docs, chapter/book metrics |
| `summarizers/book summaries/` | Claude-generated summaries for all 22 books |
| `webpages/` | Generated HTML pages for the e-library webpages  |
| `cache/` | Python source files and prompt templates (not the data files) |
| `memory/elibrary_memory.jsonl` | MCP knowledge graph (42 entities, 27 relations) |
| `.github/copilot-instructions.md` | Copilot Chat context instructions |

## What is NOT in this repo

Large or copyrighted files are excluded via `.gitignore` and backed up separately:

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
