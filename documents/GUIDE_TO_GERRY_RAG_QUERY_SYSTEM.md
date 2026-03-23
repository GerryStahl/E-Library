# Gerry's E-Library — Full Project Memory

**Last updated:** March 2026  
**Workspace:** `/Users/GStahl2/AI/elibrary/`  
**Library:** 22 books · 337 chapters · 36,089 chunks  
**Python venv:** `.venv/bin/python` (never bare `python` or `python3`)

This Guide document is the **authoritative reference** for the elibrary project. It covers the major components:
1. **The RAG query system** — hybrid retrieval + Claude/Ollama generation
2. **A custom, local query interface for analyzing the e-library
3. **Summaries of the books and chapters based on chunk semantics
4. **The citation/cluster analysis pipeline** — thematic clustering and self-citation timelines
5. **Long-term memory for the chat Copilot

This Guide document is designed to be attached as context in Copilot Chat sessions and to also serve as a human-readable project overview and record.

---

## Overview

### RAG. The RAG (Retrieval-Augmented Generation) system lets you ask free-form questions about the 22-book e-library and receive grounded, cited answers written by Claude or another selected LMM model. Rather than searching by keyword, you can ask questions in natural language — *"What does Stahl say about intersubjectivity?"* or *"How does group cognition differ from individual cognition?"* — and the system finds the most relevant passages across all 22 books, then generates a scholarly answer with inline citations.

The system uses a **three-stage hybrid retrieval** pipeline (coarse → fine → rerank) combining **dense semantic search** (neural embeddings + FAISS), **sparse keyword search** (BM25), merged by **Reciprocal Rank Fusion**, and finally scored by a **cross-encoder reranker** (`BAAI/bge-reranker-base`). This consistently outperforms either method alone, especially for an academic library where both semantic nuance and precise terminology matter.

The query answers are based on LMM summaries of selected chunks of text from the e-library of 337 book chapters, journal articles and essays. The text chunks were created by dividing each chapter into 1000-character-long chunks of text (overlapping about 200 characters) and then merging consequtive chunks that were semantically related/similar in up to 5 iterative vector matches. The chapter texts, the original 1000 character chunks ("level-0") and the merged chunks were all embedded into a 768-dimensional semantic vector space for determining semantic similarities.

The main pipeline for extracting the chapters in the books, chunking them and embedding semantic vectors of the chunks and texts in the vector space are done in main.py and run_complete_pipeline.py


### Querying. The query interface for human users permits natural-language questions to be submitted. One of 6 different LMM models can be selected: Anthropic's Claude models of varying size, power and cost (Haiku / Sonnet / Opus) or smaller models running on the local computer without requiring costs or Internet access (Qwen2.5 14b / Llama3.1 8b / Gemma3 4b).


### Summaries. Summaries of each of the 22 books and each of their 337 chapters were created and displayed on the webpage for the e-library. 
The semantic vectors were also analyzed to cluster the books, chapters and their component chunks to define 20 cluster topics to sort the e-library writings into different topic areas.


### Citation Analysis.
The 2,852 inline Stahl self-citations extracted from the library chunks were used to explore how ideas in the writings reference earlier work. An attempt was made to resolve each citation pair to a specific cited chapter (so that the cited chapter's semantic vector could be compared with the citing chunk's vector). A pipeline was built that searched each citing chapter's reference list and book-level consolidated bibliographies (found in Books 3, 4, 5, 6, 10, 15, 18) for the matching APA entry, then matched the title against the chapter index. The resolution rate was only 14.8% (422/2,852), mainly because 38.7% of citations had ambiguous years (multiple Stahl works in the same year) and 25.1% had title-matching failures. After reviewing the diagnostic report, it was decided that chapter-level vectors are probably too coarse for the kind of inter-chapter semantic comparison this approach would require, and the citation-pair resolution work was paused.

Instead, the focus shifted to **cluster publication timelines**: visualising the chunk (and chapter) count per cluster per year across the full 1970–2026 publication range. Two visualisations were produced (chapter-based and chunk-based) along with a CSV export of the underlying data. These show the thematic distribution and temporal evolution of Gerry's writing directly from cluster assignments, without needing citation resolution.


### Memory. This Guide document is part of a long-term memory aspect of the RAG project. Several mechanisms have been created to provide long-term memory. They attach information to each new query to provide context for responding to the query:
* This MD Guide document -- which is computer-readable (with its MD structure) as well as human-readable -- provides a technical overview.
* Github/copilot-instructions.md -- provides a variety of instructions to the chat Copilot on how to respond to queries.
* MCP Knowledge Graph (`memory/elibrary_memory.jsonl`) -- a JSONL queryable memory structure about the project.
* Auto-generated Project Status (`documents/project_status.md`) -- summarizes the current status of the ongoing project.
* In addition, query_history.json -- is a repository of previous queries; semantically similar queries are also used for context and deambiguity.

--

## How to Use

### Option 1 — Web Chat Interface (recommended)

```bash
cd /Users/GStahl2/AI/elibrary
.venv/bin/python scripts/query_app.py
```

Opens a browser window at `http://localhost:7860` with:
- A chat window for multi-turn conversation
- Radio buttons to switch model — **Claude** (Haiku / Sonnet / Opus) or **local Ollama** models (Qwen2.5 14b / Llama3.1 8b / Gemma3 4b, free after initial download)
- Sliders to adjust how many chapters and passages are retrieved
- A **"Cross-encoder reranker"** checkbox — enabled by default; uncheck to skip reranking for faster (slightly lower quality) responses
- A "Show sources" toggle to see which chapters and passages were used
- A "Clear" button to start a new conversation

### Option 2 — Terminal REPL

```bash
cd /Users/GStahl2/AI/elibrary
.venv/bin/python scripts/query_library.py
# With options:
.venv/bin/python scripts/query_library.py \
    --model claude-sonnet-4-20250514 \
    --chapters 7 \
    --chunks 10 \
    --verbose
```

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `claude-haiku-4-5-20251001` | Model for generation. Claude options: `claude-haiku-4-5-20251001`, `claude-sonnet-4-20250514`, `claude-opus-4-20250514`. Local (Ollama) options: `qwen2.5:14b`, `llama3.1:8b`, `gemma3:4b` |
| `--chapters` | `5` | Number of chapters retrieved in stage 1 |
| `--chunks` | `8` | Number of passages sent to the LLM |
| `--no-rerank` | off | Disable cross-encoder reranking (faster) |
| `--verbose` | off | Print retrieved chapters + passage snippets |
| `--user` | `Gerry` | Name tag saved with each query in the history log |

During a session, type `clear` to reset conversation memory while keeping the persistent history log intact. Type `quit` or `exit` to stop.

---

## System Architecture

### Bird's-eye view

```
Your question
     │
     │  ─ if prior conversation exists ──────────────────────────────────────────
     │  │  Claude Haiku rewrites question as self-contained retrieval query      │
     │  ────────────────────────────────────────────────────────────────────────
     │
  retrieval query
     │
     ├─── Dense embedding ──► chapter_summaries.faiss (335 vectors) ──┐
     └─── BM25 keyword ─────► summaries_bm25/                         ┤ RRF → top-N chapters
                                                                       ┘
                                         │
                    ┌────────────────────┴─────────────────────────┐
                    │  Filter to top-N chapters                     │
                    ├─ Dense embedding ──► elibrary.faiss          ─┤
                    └─ BM25 keyword ─────► chunks_bm25/            ─┘
                                         │ RRF → 20 candidates
                                         │
                              bge-reranker-base (cross-encoder)
                              scores all 20 as (query, passage) pairs
                                         │ top-M passages
                                         │
                              Claude: original question
                                     + last 4 Q&A turns  (conversation memory)
                                     + retrieved passages
                                     → answer + citations
```

---

## Technology Stack

### 1. Chunking — `chunkers/semantic_hierarchical_chunker.py`

Before retrieval can work, each chapter's text must be split into manageable pieces called **chunks**.

| Parameter | Value | Meaning |
|---|---|---|
| `chunk_size` | 1,000 chars | Target size of a base chunk |
| `overlap` | 200 chars | Character overlap between adjacent chunks |
| `SIM_THRESHOLD` | 0.70 | Cosine similarity required to merge two chunks |
| `MIN_CHUNKS` | 3 | Minimum chunks before merging begins |
| `MAX_LEVELS` | 5 | Depth of the merge hierarchy |
| Merging model | `all-MiniLM-L6-v2` | 384-dim model used *only* during chunk merging |

The chunker produces a **hierarchy** of chunks:

- **Level 0** — base chunks (~17,441 across the library). These are the finest-grained passages used for BM25 and the primary retrieval target.
- **Levels 1–5** — progressively merged chunks (~18,648 total). Adjacent semantically similar level-0 chunks are merged into broader "topic chunks". These are included in the FAISS index and help dense retrieval find broader context that smaller chunks might miss.

Total: **36,089 Chunk objects** stored in `cache/elibrary_cache.pkl`.

### 2. Dense Embeddings — `intfloat/e5-base-v2`

All passages and queries are converted to 768-dimensional vectors using the **e5-base-v2** model from Hugging Face. This model is instruction-tuned for retrieval and requires a text prefix:

| Text type | Prefix |
|---|---|
| Indexed passages (chunks, summaries) | `"passage: "` |
| Queries at search time | `"query: "` |

The prefix distinction tells the model which role the text is playing, significantly improving retrieval accuracy over using the same prefix for both.

Vectors are **L2-normalised** before indexing, which converts inner-product similarity to cosine similarity (values range from −1 to 1, higher = more similar).

### 3. FAISS Vector Indexes

**FAISS** (Facebook AI Similarity Search) stores the vectors and finds nearest neighbours extremely fast even at 36,089 vectors.

| Index | Vectors | Contents |
|---|---|---|
| `vector_store/elibrary.faiss` | 36,089 | All chunk texts (all levels) |
| `vector_store/chapter_summaries.faiss` | 335 | Claude chapter summary texts |

Index type: `IndexFlatIP` — exact inner product search over L2-normalised vectors (= exact cosine similarity). No approximation is used because at this scale (<100k vectors) exact search is still fast (<10ms per query).

Each index has a **metadata sidecar** (`.json` file) that maps integer vector IDs to human-readable fields (book number, chapter number, chunk index, page, word count, etc.).

### 4. BM25 Keyword Search — `bm25s`

**BM25** (Best Match 25) is the classic probabilistic keyword search algorithm — the same one that powers most search engines before neural retrieval became mainstream. It scores documents by:

$$\text{BM25}(d, q) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

where $f(t,d)$ is the term frequency in document $d$, $\text{IDF}$ is inverse document frequency, and $k_1$, $b$ are tuning constants. In plain English: terms that appear often in a document but rarely in the full corpus score highest.

BM25 is essential alongside dense retrieval because it handles **exact matches** that semantic embeddings sometimes miss: author names (*Vygotsky, Heidegger*), acronyms (*CSCL, VMT, GeoGebra*), book-specific jargon, and chapter titles.

Library used: **`bm25s`** (fast, supports disk serialisation). Queries and documents are tokenised with English stopword removal and Porter stemming via **`PyStemmer`**.

| Index | Documents | Contents |
|---|---|---|
| `vector_store/chunks_bm25/` | 17,441 | Level-0 chunk texts |
| `vector_store/summaries_bm25/` | 335 | Chapter summary texts |

Each BM25 index has a `sidecar.json` file that maps BM25 document IDs back to chunk metadata (book, chapter, page, the raw text itself). This means the query pipeline does **not** need to load the 400 MB cache PKL at runtime.

### 5. Reciprocal Rank Fusion (RRF)

Dense retrieval and BM25 produce ranked lists that can't be directly combined by score (their score scales are incomparable). RRF merges them by rank position only:

$$\text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}, \quad k = 60$$

where $k=60$ is a standard constant that dampens the influence of very top-ranked items. A document appearing in rank 1 from one ranker and rank 3 from another beats a document appearing in rank 1 from only one ranker. No tuning is required.

This is applied twice:
1. **Coarse stage** — merging dense + BM25 chapter rankings → top-N chapters
2. **Fine stage** — merging dense + BM25 chunk rankings (filtered to top-N chapters) → 20 candidates for the reranker

### 6. Cross-Encoder Reranking — `BAAI/bge-reranker-base`

After RRF produces 20 candidates, a **cross-encoder** reranker scores each one against the query and re-orders them. The top-M are then passed to Claude.

Unlike bi-encoders (which embed query and passage *separately* and compare vectors), a cross-encoder feeds the query and passage **together** into the model and computes a single relevance score with full attention over both texts. This is slower but significantly more accurate — it can catch relevance signals that bi-encoders miss, such as a passage that uses different vocabulary from the query but directly answers it.

**Why `bge-reranker-base` over MS-MARCO models:**  
The common MS-MARCO rerankers (`cross-encoder/ms-marco-MiniLM-L-*`) are trained on web search queries — short factual lookups. They generalise poorly to academic philosophical prose. `BAAI/bge-reranker-base` (XLM-RoBERTa, 278M params) is trained on a broader mix including scholarly text and generalises much better to the kinds of passages in this library.

| Parameter | Value | Notes |
|---|---|---|
| Model | `BAAI/bge-reranker-base` | XLM-RoBERTa based, 278M params |
| Candidates scored | 20 | Fetched by RRF; reranked to `TOP_CHUNKS` |
| Latency (M5) | ~50ms per query | Runs on MPS (Metal) backend |
| Toggle | `--no-rerank` / UI checkbox | Can be disabled for faster responses |

**Example where reranking makes a difference:** Asking *"How does Stahl define intersubjectivity?"* — the reranker promotes passages that explicitly *define* the term to the top, over passages that merely *use* the word in passing. The resulting answer is more definitional and precise than RRF alone would produce.

### 7. Generation — Claude (Anthropic API) or Ollama (local)

The top-M retrieved passages are formatted into a numbered context block and sent to the LLM with a system prompt instructing it to:
- Answer in scholarly prose
- Cite each passage with `[Book N, Ch M]` inline
- Not invent information beyond the passages

Two backends are supported; the routing is automatic based on the model name:

**Cloud — Anthropic Claude** (requires `ANTHROPIC_API_KEY`)

| Model | Speed | Cost | Best for |
|---|---|---|---|
| `claude-haiku-4-5-20251001` | Fast | Low | Quick lookups, exploration |
| `claude-sonnet-4-20250514` | Medium | Medium | Detailed analysis, writing |
| `claude-opus-4-20250514` | Slow | High | Complex synthesis, nuanced questions |

**Local — Ollama** (free, runs on-device; Ollama must be running)

| Model | RAM | Quality | Best for |
|---|---|---|---|
| `qwen2.5:14b` | ~9 GB | High | Best local option for long, nuanced answers |
| `llama3.1:8b` | ~5 GB | Good | Fast local responses |
| `gemma3:4b` | ~3 GB | Adequate | Minimal RAM footprint |

Ollama is called via its `/api/chat` endpoint (`http://localhost:11434`). No API key or internet connection is required. The system prompt is passed as a `system` role message, which Ollama supports natively.

### 8. Conversation Memory and Persistent Query History

Every Q&A exchange is saved automatically to two files in `vector_store/`:

| File | Contents |
|---|---|
| `query_history.json` | Human-readable log: id, timestamp, user, question, retrieval query used, full answer, Claude model, list of chapters retrieved, 200-char snippet of each passage |
| `query_history_embeddings.npy` | Float32 `(N, 768)` matrix — row `i` is the L2-normalised embedding of record `i`'s question |

When you ask a new question, the system embeds it and runs a cosine similarity search over all past query embeddings. Any past record with similarity ≥ 0.78 is prepended to the Claude prompt as `RELEVANT PAST QUERIES FROM THIS LIBRARY:`, so Claude can build on and cross-reference prior answers across sessions.

Within a single session, the last `HISTORY_TURNS` (4) Q&A turns are also prepended as a multi-turn `messages` list, giving the model awareness of the conversation thread. If a follow-up question uses pronouns or references ("How does *he* define it?"), the system first rewrites it as a self-contained query before retrieval runs. Rewriting uses the same backend as generation — Ollama when a local model is selected, or Claude Haiku otherwise — so a fully local session makes no cloud calls at all.

---

## Vector Store File Layout

```
vector_store/
├── elibrary.faiss                 110.9 MB   36,089 chunk vectors (all levels)
├── elibrary_meta.json              12.2 MB   chunk sidecar (book/ch/level/page)
├── elibrary_build.json                       build stats
│
├── chapter_summaries.faiss          1.0 MB   335 chapter summary vectors
├── chapter_summaries_meta.json      0.2 MB   chapter sidecar (incl. book_keywords)
├── chapter_summaries_build.json               build stats
│
├── chunks_bm25/                               BM25 over 17,441 level-0 chunks
│   ├── *.npz / *.json                         bm25s index files
│   └── sidecar.json                           doc_id → {book, chapter, text, …}
│
├── summaries_bm25/                            BM25 over 335 chapter summaries
│   ├── *.npz / *.json                         bm25s index files
│   └── sidecar.json                           doc_id → {book, chapter, text, …}
│
├── query_history.json                         Human-readable Q&A audit log
└── query_history_embeddings.npy               Float32 (N, 768) question embeddings
```

---

## Retrieval Parameters

| Parameter | Default | Notes |
|---|---|---|
| `TOP_CHAPTERS` | 5 | Increase if you want broader coverage across books |
| `TOP_CHUNKS` | 8 | Final passages sent to Claude; context window allows up to ~20 |
| `RERANKER_CANDIDATES` | 20 | RRF fetches this many; reranker scores all 20, keeps `TOP_CHUNKS` |
| `DENSE_SEARCH_WIDE` | 400 | FAISS over-fetch before chapter filtering; rarely needs changing |
| `RRF_K` | 60 | Standard constant; no need to tune |
| `HISTORY_TURNS` | 4 | Prior conversation turns passed to Claude alongside the new passages |
| `REWRITE_MODEL` | `claude-haiku-4-5-20251001` | Cloud fallback for query rewriting when a Claude model is selected; local models rewrite via Ollama instead |
| `SIM_THRESHOLD` | 0.78 | Cosine similarity required for a past query to be injected as context |
| `MAX_PAST` | 3 | Maximum past Q&A records injected per turn |

### Tips for good queries
- Ask questions, not just keywords: *"How does Stahl define the joint problem space?"* works better than *"joint problem space definition"*
- Include author names when relevant: *"What does Vygotsky say about zone of proximal development?"* — BM25 will catch the exact name
- **To scope a search to one book, just say so in the question**: *"In Book 11, how does Stahl describe group practices?"* — BM25 picks up book numbers and title keywords naturally. No separate metadata filter is needed.
- Use `--chapters 7 --chunks 12` for broad comparative questions across many books
- Use `--model claude-sonnet-4-20250514` when you want a detailed, nuanced answer
- Use `--no-rerank` (terminal) or uncheck the reranker checkbox (web app) if you want the fastest possible response and can accept slightly lower precision
- In the terminal REPL, type `clear` to reset conversation memory for a new topic while keeping the persistent history log intact
- Browse `vector_store/query_history.json` to review all past queries, their answers, and which chapters were retrieved

---

## Building / Rebuilding Indexes

If the cache changes (new chapters, re-chunking, etc.), rebuild in this order:

```bash
# 1. Re-chunk (if chapter texts changed)
python scripts/chunk_all.py

# 2. Re-export JSON
python scripts/export_cache_json.py

# 3. Rebuild FAISS chunk index
python scripts/build_vector_store.py          # ~17 min, uses GPU if available

# 4. Rebuild FAISS chapter summary index
python scripts/build_chapter_summary_vectors.py   # ~9s

# 5. Rebuild BM25 indexes
python scripts/build_bm25_index.py               # ~1s
```

Step 3 is the only slow step. Steps 4 and 5 are trivially fast.

---

## Is the RAG System Complete?

**Core pipeline: Yes.** The system is fully operational:
- ✅ 22 books / 337 chapters chunked (36,089 chunks)
- ✅ Hybrid retrieval (dense + BM25 + RRF) at both chapter and chunk level
- ✅ Cross-encoder reranking (`BAAI/bge-reranker-base`) after RRF
- ✅ Claude generation with inline citations
- ✅ Web chat interface (Gradio) with model selector, sliders, reranker toggle, and sources panel
- ✅ Book and chapter keywords populated
- ✅ **Conversation memory** — two-part implementation:
  - *Query rewriting (Part A):* when there is prior history, the LLM rewrites the follow-up question as a self-contained retrieval query before retrieval runs (resolves pronouns like "it", "him", "that approach"). Uses the same backend as generation — Ollama for local models, Claude Haiku for cloud models.
  - *Generation history (Part B):* the last `HISTORY_TURNS` (4) Q&A turns are prepended to the Claude generation request, so the answer can reference and build on prior exchanges.
  - In the terminal REPL, type `clear` to reset the conversation.
- ✅ **Persistent query history** — every exchange saved to `vector_store/query_history.json` (human-readable audit log) with a parallel embeddings file enabling semantic search over all past queries across sessions.
- ✅ **User name tagging** — each history record includes the name of the person who asked (editable in the web UI sidebar; set via `--user` in the terminal).
- ✅ **Local Ollama backend** — Qwen2.5 14b, Llama3.1 8b, and Gemma3 4b available as free, fully offline alternatives to Claude. Both query rewriting and generation route to Ollama when a local model is selected; no internet or API key required.

**Design decisions:**
- **Metadata filtering by book/author was not implemented** — it is not needed. Because BM25 indexes the raw text, any mention of a book number, title, or keyword in your question naturally focuses retrieval on the right books without a separate filter mechanism. For example, *"In Book 11, what does Stahl say about group practices?"* works correctly as written.

**Possible future enhancements:**
- 🔲 `chunk_token_count` — currently 0 for all chunks; filling this enables tighter context window packing
- 🔲 Batch query script — `query_library_batch.py` for running many questions from a file and saving results to CSV

---

## System Requirements at Query Time

The query system loads into RAM at startup (~2 GB total):

| Component | RAM | Load time |
|---|---|---|
| e5-base-v2 embedding model | ~300 MB | ~5s |
| bge-reranker-base reranker | ~300 MB | ~3s |
| `elibrary.faiss` index | ~110 MB | <1s |
| `elibrary_meta.json` | ~50 MB | <1s |
| `chapter_summaries.faiss` | ~1 MB | <1s |
| BM25 indexes + sidecars | ~80 MB | <1s |
| **Total** | **~850 MB** | **~10s** |

The cache PKL (elibrary_cache.pkl, ~400 MB) is **not** loaded at query time — all needed data is in the BM25 sidecars and FAISS metadata files.

---

## Long-Term Memory Mechanisms for Copilot Chat

Three complementary mechanisms preserve project context across Copilot Chat sessions.

---

### 1. Auto-injected Instructions (`.github/copilot-instructions.md`)

Automatically loaded into **every** Copilot Chat session in this workspace — no `#file` attachment needed.

Contains:
- Personal & relational rules (call the user Gerry, collegial tone, no filler)
- Project identity and library description
- Technical conventions: venv path, file naming, code style, API key handling
- Key file paths with schema annotations
- Analysis pipeline status (Completed / Pending)
- Established key findings
- Behavioural guardrails (check scripts before writing new ones, prefer targeted edits, etc.)
- Full cluster reference table (k=20)

**Updating:** edit `.github/copilot-instructions.md` directly when the pipeline state, key findings, or conventions change. The "wrap up / clean up" guardrail triggers a reminder to keep it current.

---

### 2. Auto-generated Project Status (`documents/project_status.md`)

A compact (~3 KB) snapshot generated by `scripts/generate_project_status.py`.  
**No API calls — reads only the local file system.**

Attach with `#file:documents/project_status.md` at the start of a Copilot Chat session for an up-to-date current-state briefing. Contains:
- Pipeline step status (✅/❌) inferred from file existence + timestamps
- Cluster table with self-citation counts
- Top cited years summary
- Report file inventory with sizes and dates
- Open Work Items section (free-form, preserved across runs — edit manually)

**Regenerate:** run `.venv/bin/python scripts/generate_project_status.py` at the end of each working session (automatically triggered by the "wrap up" guardrail).

---

### 3. MCP Knowledge Graph (`memory/elibrary_memory.jsonl`)

A **structured, queryable knowledge graph** served by the `elibrary-memory` MCP server, available as tools in Copilot Chat agent mode.

**Configuration:** `.vscode/mcp.json`  
```json
{
  "servers": {
    "elibrary-memory": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"],
      "env": {
        "MEMORY_FILE_PATH": "${workspaceFolder}/memory/elibrary_memory.jsonl"
      }
    }
  }
}
```

**Storage:** `memory/elibrary_memory.jsonl` — JSONL knowledge graph persisted to disk.

**Seeding:** `scripts/seed_mcp_memory.py` — one-time population from `cluster_summaries.json` and `self_citations.csv`. Re-run after major pipeline changes to add new entities. Produces 42 entities and 27 relations covering:

| Entity type | Examples |
|---|---|
| `project` | `elibrary_project` — workspace, venv, library size |
| `pipeline_step` | Steps 1–4 with scripts, inputs, outputs, status |
| `semantic_cluster` | All 20 clusters — label, summary, self-citation count, noise/non-CSCL flags |
| `findings` | `key_findings` — 10 established analysis results |
| `file` | 13 key data files with schema annotations |
| `system` | `rag_query_system` — RAG architecture |
| `technical_reference` | `citation_regex_patterns` — verified regex strings |

**Available tools in Copilot Chat:**
- `search_nodes` — semantic search across all entities
- `open_nodes` — retrieve specific entities by name
- `read_graph` — return full graph
- `create_entities` / `add_observations` / `create_relations` — add new knowledge
- `delete_entities` / `delete_relations` / `delete_observations` — remove stale entries

**Requirements:** Node.js ≥18 (installed via `brew install node`; v25.8.1). VS Code discovers the server automatically from `.vscode/mcp.json` — trust the server when prompted.

---

### When to use each mechanism

| Situation | Use |
|---|---|
| Every session, automatically | `copilot-instructions.md` (always active) |
| Start of a session, current state | Attach `documents/project_status.md` with `#file` |
| Deep context — architecture, scripts, findings | Attach `documents/GUIDE_TO_GERRY_RAG_QUERY_SYSTEM.md` with `#file` |
| Ad-hoc query: "What do we know about cluster X?" | Ask Copilot to search the `elibrary-memory` |
| Adding a new finding mid-session | Ask Copilot to `add_observations` to `key_findings` |

---

## Analysis Pipeline — Thematic Clustering & Self-Citation Timelines

### Goal

Trace the **thematic development of Gerry's work** across the 22-volume library by:
1. Grouping the 17,441 base chunks into semantic clusters (k-means on FAISS embeddings)
2. Extracting inline Stahl self-citations from each chunk and recording the citing year + cited year
3. Visualising per-cluster citation timelines: how each thematic cluster's writing looks back over time
4. Building a cross-cluster citation matrix: which themes cite work originating in which other themes

This is distinct from the RAG query system — it is a **research analysis** of the library corpus itself, not a retrieval tool.

---

### Step 1 — Chunk Clustering (`scripts/cluster_chunks.py`) ✅

Clusters all 17,441 level-0 vectors from `elibrary.faiss` using **k-means (k=20)**.

**Key parameters:**
- `KMeans(n_clusters=20, random_state=42, n_init=10)`
- Vectors L2-normalised before clustering
- **Breadth** = mean cosine distance of member chunks from cluster centroid
- **Relative breadth** = breadth / global mean breadth; >1.0 = broader than average
- Global mean distance to centroid: **0.0972**
- 5 exemplar passages (nearest to centroid) sent to **Claude Haiku** for a label + one-sentence explanation

**Outputs:**
- `reports/chunk_clusters.csv` — 17,441 rows: `vector_id, book_number, chapter_number, book_title, chapter_title, chunk_index, chunk_page, pub_year, cluster_id, cluster_label`
- `reports/cluster_report.txt` — full labelled cluster listing with sizes, breadth, exemplars, top books

---

### Step 1b — Expanded Summaries (`scripts/expand_cluster_summaries.py`) ✅

Re-calls Claude Haiku with 10 exemplars per cluster and a richer prompt requesting a **3–5 sentence paragraph** covering: core theme, key concepts, distinctive characteristics, and role in Stahl's broader work.

**Outputs:**
- `reports/cluster_report.txt` — overwritten with paragraph summaries (750 lines)
- `reports/cluster_summaries.json` — `{cluster_id: {"label": …, "summary": …}}`

---

### Step 2 — Self-Citation Extraction (`scripts/extract_self_citations.py`) ✅

Scans all 17,441 level-0 chunks for **inline Stahl self-citations** using three regex patterns:

```python
_PAT_STAHL_FIRST = re.compile(r'\bStahl(?:\s+et\s+al\.?|\s+&\s+[^,()]{1,40})?,\s*(\d{4}[a-z]?)\b')
_PAT_NARRATIVE   = re.compile(r'\bStahl\s+\((\d{4}[a-z]?)\b')
_PAT_PAREN_BROAD = re.compile(r'\(([^()]*\bStahl\b[^()]*)\)')
```

Reference list entries (`Stahl, G. (YYYY). Title…`) are **not** matched — the initial `G.` between name and year breaks all three patterns. This was verified by sampling.

**Results:** 2,852 citation pairs from 1,085 chunks.

**Outputs:**
- `reports/self_citations.csv` — fields: `vector_id, book_number, chapter_number, book_title, chapter_title, cluster_id, cluster_label, pub_year, cited_year, snippet`
- `reports/self_citation_report.txt` — top cited years, per-cluster summary table, per-cluster timelines

---

### Step 3 — Visualisation (`scripts/visualize_self_citations.py`) ✅

Generates two matplotlib figures from `self_citations.csv`:

1. **`reports/cluster_citation_scatter.png`** — 20-panel grid, one panel per cluster sorted by citation count. x = citing year (pub_year), y = cited year, bubble size ∝ frequency. Dashed diagonal = same year. CSCL clusters in deep blue, non-CSCL in light blue, bibliography noise in red diamonds.

2. **`reports/global_citation_heatmap.png`** — full heatmap of citing year × cited year with log colour scale. Top marginal = total citations per citing year. Right marginal = total citations per cited year. Cells ≥ 20 annotated with raw count.

---

### Step 2b — Self-Citation Pair Resolution (`scripts/build_citation_pairs.py`) ⏸ Paused

Attempted to resolve each of the 2,852 extracted self-citations to a **specific cited chapter** so that the cited chapter's semantic vector could be compared with the citing chunk's vector — enabling proper cross-cluster citation analysis.

**Resolution strategy:**
1. Search the citing chapter's own chunks for an APA reference-list entry matching the cited year
2. If not found (Books 3, 4, 5, 6, 10, 15, 18 have consolidated end-of-book bibliographies), fall back to a book-level search across all chapters
3. Parse the APA entry to extract title + publication type (monograph / book chapter / journal article)
4. Match title against the chapter index to find the cited chapter
5. Look up the cited chapter's FAISS vector

**Result:** 422 resolved (14.8%), 2,430 unresolved

**Unresolved breakdown:**

| Reason | Count | % |
|---|---|---|
| `ambiguous_year` | 1,104 | 38.7% — year cited without letter suffix; multiple Stahl works in that year |
| `chapter_title_not_in_library` | 717 | 25.1% — title matching failure |
| `year_not_in_ref_list` | 330 | 11.6% — no APA entry found for that year |
| `cited_whole_book` | 174 | 6.1% — citation is to a whole book, not a chapter |
| `cited_article_not_in_library` | 92 | 3.2% — article not in the 22-volume library |
| `no_ref_list_in_book` | 13 | 0.5% — consolidated bibliography not located |

**Decision:** Paused. Chapter-level vectors are too coarse for meaningful inter-chapter comparison, and the ambiguous-year problem (38.7%) would require heuristics with uncertain accuracy. The diagnostic report is preserved for potential future resumption.

**Outputs:**
- `reports/citation_pairs.jsonl` — 2,852 records with resolution status and fields
- `reports/citation_vectors.npz` — 422 resolved pairs × 768 dim (citing chunk vector + cited chapter vector)
- `reports/citation_pairs_report.txt` — resolution summary by category
- `reports/citation_pairs_diagnostic.txt` — 600-line diagnostic with examples from each failure category

---

### Step 3b — Cluster Publication Timelines ✅

Visualises the **temporal distribution of writing across thematic clusters**, using chunk and chapter counts per year as a proxy for how much of Gerry's annual output belongs to each theme. This provides a direct, vector-based picture of thematic evolution that does not require citation resolution.

Two variants were built:

**Chapter-level (`scripts/visualize_cluster_timelines.py`):**
- Each chapter is assigned to its **plurality cluster** (the cluster_id that appears most often among that chapter's level-0 chunks)
- 336 chapters assigned; 18 active clusters (excl. 6 & 12 noise)
- Two-panel figure: raw chapter counts (top) + share % (bottom), stacked area, 1970–2026
- Output: `reports/cluster_timelines.png`

**Chunk-level (`scripts/visualize_chunk_timelines.py`):**
- Each level-0 chunk counted once in its own cluster, aggregated by `pub_year`
- 15,629 chunks; 18 active clusters
- Same two-panel layout
- Output: `reports/chunk_timelines.png`
- *Preferred for analysis* — shows thematic composition within chapters, not just the dominant topic

**CSV export (`scripts/export_timeline_data.py`):**
- Exports the data underlying both figures to a single spreadsheet
- 40 year rows × 74 columns: raw count + share % per cluster for both chapters and chunks
- Output: `reports/timeline_data.csv`

Both plots use **EXCLUDE_CLUSTERS = {6, 12}** (noise only). Non-CSCL clusters 8, 9, 15 are included.

---

### Step 4 — Cross-Cluster Citation Matrix ❌ Pending

Goal: determine which clusters' body chunks cite works that originated in which other clusters, by matching `cited_year` back to the cluster(s) whose chapters were published in that year.

---

### Key Findings (treat as established)

| Finding | Detail |
|---|---|
| Most-cited year | **2006** — 773 hits — *Group Cognition* (MIT Press) |
| Second most cited | **2009** — 378 hits — *Studying Virtual Math Teams* |
| Most citations produced | **Cluster 4** (VMT) — 638 pairs; 2021 alone = 219 (retrospective book) |
| Retrospective survey | **Cluster 5**, citing year 2010 → cited year 1999: 124 hits |
| Bibliography noise | **Clusters 6 & 12** — exemplars show raw reference lists, not prose; exclude from content analysis |
| Early-career clusters | **Cluster 10** (Tacit Knowledge) and **Cluster 14** (HERMES) — cited years top out at 1993–1995 |
| Non-CSCL clusters | **9** (salt marsh), **8** (electronic music), **15** (sculpture) — peripheral to analysis |
| Partially off-topic | **18** (Marx/Heidegger/ontology) — philosophically adjacent but not CSCL proper |

---

### Cluster Reference (k=20, sorted by size)

| ID | Label | Chunks | Rel. Breadth | Notes |
|---|---|---|---|---|
| 11 | Group Cognition in Collaborative Learning | 1,619 | 1.02x | Core CSCL |
| 1 | Group Processes and Collaborative Knowledge Building | 1,510 | 0.94x | Core CSCL |
| 5 | Perspectives and Knowledge Construction Systems | 1,336 | 1.14x | Early software/CSCL |
| 3 | Threading and Response Structure in Chat | 1,300 | 1.07x | VMT methodology |
| 7 | Group Discourse in Mathematics Problem-Solving | 1,207 | 0.95x | Math CSCL |
| 4 | Virtual Math Teams Project and Online Collaboration | 1,185 | 0.83x | VMT project |
| 19 | Collaborative Mathematical Meaning-Making | 1,096 | 1.05x | Math discourse |
| 0 | Heidegger's Philosophy of Understanding and Being | 1,059 | 1.17x | Philosophy |
| 10 | Representing Tacit Knowledge in Design Systems | 894 | 1.07x | Early design |
| 6 | Joint Problem Space and Collaborative Knowledge Construction | 873 | 0.91x | ⚠ Bibliography noise |
| 12 | Group Cognition and Collaborative Knowledge Building | 823 | 0.75x | ⚠ Bibliography noise (tightest) |
| 2 | Group Cognition in Dynamic Geometry Learning | 703 | 0.92x | DG learning |
| 17 | Foundations and Evolution of CSCL | 679 | 1.18x | Broadest real cluster |
| 13 | Dynamic Geometry Construction and Drag Testing | 645 | 0.91x | DG tutorials |
| 18 | Marx, Heidegger, and Historical Ontology | 588 | 1.09x | Philosophy/Marx |
| 14 | HERMES Language Design and Features | 529 | 1.06x | Early software |
| 16 | Collaborative Problem-Solving in Dynamic Geometry | 408 | 0.70x | Tightest real cluster |
| 9 | Salt Marsh Conservation and Climate Resilience | 407 | 1.00x | Non-CSCL |
| 8 | Electronic Music and Sound Technology | 294 | 1.05x | Non-CSCL |
| 15 | Three-Dimensional Sculptural Forms and Techniques | 286 | 1.14x | Non-CSCL |

---

## Script Inventory

### RAG Query System

| Script | Purpose |
|---|---|
| `scripts/main.py` | Gradio web app (legacy; superseded by `query_app.py`) |
| `scripts/query_app.py` | **Primary web chat interface** — Gradio at `localhost:7860`; model selector, sliders, reranker toggle |
| `scripts/query_library.py` | **Terminal REPL** — `--model`, `--chapters`, `--chunks`, `--no-rerank`, `--verbose`, `--user` |
| `scripts/query_history.py` | Browse/search persistent query history |

### Build & Index Pipeline

| Script | Purpose |
|---|---|
| `scripts/build_cache.py` | Parse PDFs → `ElibraryCache` dataclass → `cache/elibrary_cache.pkl` |
| `scripts/chunk_all.py` | Chunk all chapters; uses `chunkers/semantic_hierarchical_chunker.py` |
| `scripts/chunk_range.py` | Chunk a subset of books (useful for incremental updates) |
| `scripts/build_vector_store.py` | Embed all chunks → `elibrary.faiss` (~17 min) |
| `scripts/build_chapter_summary_vectors.py` | Embed chapter summaries → `chapter_summaries.faiss` (~9s) |
| `scripts/build_bm25_index.py` | Build BM25 indexes for chunks and summaries (~1s) |
| `scripts/regenerate_all_vector_stores.py` | Runs build pipeline end-to-end |
| `scripts/run_complete_pipeline.py` | Full pipeline from cache to indexes |
| `scripts/export_cache_json.py` | Export cache metadata to JSON for inspection |

### Cache & Metadata Maintenance

| Script | Purpose |
|---|---|
| `scripts/fill_cache_metadata.py` | Fill missing metadata fields in cache |
| `scripts/fill_cache_wordcounts.py` | Populate word count fields |
| `scripts/migrate_cache_fields.py` | Field renaming/migration across cache versions |
| `scripts/verify_cache.py` | Validate cache integrity |
| `scripts/generate_book_keywords.py` | Generate book-level keywords with Claude |
| `scripts/generate_book_summaries_2016_2025.py` | Generate summaries for later books |
| `scripts/fill_missing_summaries.py` | Fill any chapter summaries that are empty |
| `scripts/sync_wordcounts_from_pkl.py` | Sync word count data from PKL to JSON |
| `scripts/strip_author_digits.py` | Clean author name formatting |

### Editorials Cache (separate)

| Script | Purpose |
|---|---|
| `scripts/build_editorials_cache.py` | Build separate cache for ijCSCL editorial summaries |
| `scripts/download_editorials.py` | Download editorial PDFs |
| `scripts/fill_editorial_metadata.py` | Populate editorial metadata |
| `scripts/generate_editorial_summaries.py` | Summarise editorials with Claude |

### Analysis Pipeline

| Script | Purpose |
|---|---|
| `scripts/cluster_chunks.py` | **Step 1** — k=20 KMeans on level-0 vectors; label with Claude Haiku; outputs `chunk_clusters.csv` + `cluster_report.txt` |
| `scripts/expand_cluster_summaries.py` | **Step 1b** — rewrite cluster summaries as 3–5 sentence paragraphs; outputs `cluster_summaries.json` |
| `scripts/extract_self_citations.py` | **Step 2** — extract inline Stahl citations from chunk text; outputs `self_citations.csv` + `self_citation_report.txt` |
| `scripts/visualize_self_citations.py` | **Step 3** — 20-panel scatter grid + global heatmap PNG |
| `scripts/build_citation_pairs.py` | **Step 2b** (paused) — resolve 2,852 inline citations to specific cited chapters; chapter-level ref search + book-level fallback; JSONL + NPZ output |
| `scripts/visualize_cluster_timelines.py` | **Step 3b** — chapter-level publication timeline: plurality-cluster assignment per chapter, 2-panel stacked area, `reports/cluster_timelines.png` |
| `scripts/visualize_chunk_timelines.py` | **Step 3b** — chunk-level publication timeline: each level-0 chunk in its own cluster, `reports/chunk_timelines.png` |
| `scripts/export_timeline_data.py` | Export timeline raw counts + share % for all clusters (both chapter and chunk) to `reports/timeline_data.csv` |
| `scripts/seed_mcp_memory.py` | One-time seed of MCP knowledge graph from `cluster_summaries.json` + `self_citations.csv`; re-run after major pipeline changes |
| `scripts/generate_project_status.py` | Auto-generate `documents/project_status.md` status snapshot (no API calls); run at end of each session |

### Report Generation

| Script | Purpose |
|---|---|
| `scripts/generate_claude_report.py` | Generate a Claude-authored narrative report |
| `scripts/generate_total_history.py` | Overall intellectual history of the library |
| `scripts/generate_total_history_book.py` | Book-by-book history |
| `scripts/generate_annual_summaries.py` | Year-by-year summary of output |
| `scripts/generate_comparison_report.py` | Compare model outputs |

### Probe / Debug Scripts (prefixed `_`)

One-off diagnostic and repair scripts. Safe to ignore unless debugging a specific issue:
`_audit_empty_slots.py`, `_check_book_summaries.py`, `_check_truncation.py`, `_fill_appendix_summary.py`, `_fill_chapter_references.py`, `_fix_chapter_prompts.py`, `_probe_article_page.py`, `_probe_book16.py`, `_read_b3ch11.py`, `_store_b3ch11_summary.py`, `_verify_wc.py`, etc.

Citation pair analysis probe scripts (all prefixed `_`):
- `_survey_citation_refs.py` — found 53 chapters with no reference list (books with consolidated bibliographies)
- `_survey_citation_refs2.py` — confirmed consolidated bibliographies exist in Books 3, 4, 5, 6, 10, 15, 18
- `_survey_citation_refs3.py` — mapped the specific chapter number of each book's consolidated ref section
- `_report_citation_pairs.py` — generated `citation_pairs_diagnostic.txt` with examples of each failure category
- `_count_page_refs.py` — confirmed only 108/2852 (3.8%) citations include page numbers (not useful for disambiguation)

---

## Reports & Output Files

| File | Rows / Size | Contents |
|---|---|---|
| `reports/chunk_clusters.csv` | 17,441 rows | Cluster assignment per level-0 chunk: `vector_id, book/chapter, pub_year, cluster_id, cluster_label` |
| `reports/cluster_report.txt` | ~750 lines | Full cluster listing: sizes, breadth, paragraph summaries, exemplar passages, top books |
| `reports/cluster_summaries.json` | 20 entries | `{cluster_id: {label, summary}}` — structured metadata for programmatic use |
| `reports/self_citations.csv` | 2,852 rows | One row per (chunk, cited_year) pair: `vector_id, cluster_id, pub_year, cited_year, snippet` |
| `reports/self_citation_report.txt` | — | Top cited years bar chart; per-cluster summary table; per-cluster timelines |
| `reports/cluster_citation_scatter.png` | — | 20-panel scatter: citing year × cited year, bubble size ∝ frequency |
| `reports/global_citation_heatmap.png` | — | Full heatmap with marginal bar charts |
| `reports/citation_pairs.jsonl` | 2,852 records | Per-citation resolution record: status, cited_year, resolved chapter, failure reason |
| `reports/citation_vectors.npz` | 422 pairs × 768 dim | Embedding pairs for resolved citations (citing chunk + cited chapter vectors) |
| `reports/citation_pairs_report.txt` | — | Summary of resolution outcomes by category |
| `reports/citation_pairs_diagnostic.txt` | ~600 lines | Diagnostic report with examples of each resolution failure category |
| `reports/cluster_timelines.png` | — | Chapter-level publication timeline: plurality-cluster per chapter, 18 clusters, 1970–2026 |
| `reports/chunk_timelines.png` | — | Chunk-level publication timeline: each level-0 chunk in its own cluster, 15,629 chunks, 1970–2026 |
| `reports/timeline_data.csv` | 40 year rows × 74 cols | Raw counts + share % per cluster for both chapter and chunk timelines; spreadsheet-ready |
| `reports/annual_summaries.txt` | — | Year-by-year narrative summaries of Gerry's output |
| `reports/annual_summaries_book.txt` | — | Book-organised version of annual summaries |
| `reports/total_history.txt` | — | Full intellectual history of the library (Claude-generated) |
| `reports/total_history_book.txt` | — | Book-by-book history |
| `reports/editorial_summaries.txt` | — | ijCSCL editorial summaries |
| `reports/claude_editorial_summaries.txt` | — | Claude-generated editorial summaries |
| `reports/empty_slots_report.txt` | — | Audit of missing metadata/summaries |
| `vector_store/query_history.json` | grows over time | All past Q&A sessions: timestamp, user, question, answer, chapters retrieved, passage snippets |

---

## Cache Data Structures

```
ElibraryCache
└── books: List[Book]
    ├── book_number: int
    ├── book_name: str
    ├── book_title: str
    ├── book_author: str
    ├── book_keywords: List[str]
    ├── book_reference: str          # APA string; year extractable via \(\d{4}\)
    ├── book_text: str
    ├── book_number_of_pages: int
    ├── book_kind: str
    ├── book_summaries: dict
    └── book_chapters: List[Chapter]   # NOTE: NOT .chapters
        ├── chapter_number: int
        ├── chapter_title: str
        ├── chapter_author: str
        ├── chapter_keywords: List[str]
        ├── chapter_reference: str     # APA string; most precise pub year source
        ├── chapter_text: str
        ├── chapter_number_of_pages: int
        ├── chapter_number_of_words: int
        ├── chapter_number_of_tokens: int
        ├── chapter_number_of_symbols: int
        ├── chapter_summaries: dict
        └── chapter_chunks: List[Chunk]
```

**Publication year extraction** (use `chapter_reference` for most precision; fall back to `book_reference`):
```python
import re
def extract_year(reference: str) -> int | None:
    m = re.search(r'\((\d{4}[a-z]?)\)', reference or '')
    return int(m.group(1)[:4]) if m else None
```

**Important field naming:** `book.book_chapters` (not `.chapters`), `book.book_title` (not `.title`).

---

## Project Status & Roadmap

### Completed ✅

- Full RAG query pipeline (FAISS + BM25 + RRF + cross-encoder + Claude/Ollama)
- Gradio web app (`query_app.py`) and terminal REPL (`query_library.py`)
- Conversation memory (query rewriting + multi-turn history)
- Persistent query history with semantic search across sessions
- Local Ollama backend (Qwen2.5 14b, Llama3.1 8b, Gemma3 4b)
- Analysis Step 1: Chunk clustering (k=20, Claude-labelled, breadth-measured)
- Analysis Step 1b: Expanded paragraph summaries per cluster
- Analysis Step 2: Self-citation extraction (2,852 pairs)
- Analysis Step 1: Chunk clustering (k=20, Claude-labelled, breadth-measured)
- Analysis Step 1b: Expanded paragraph summaries per cluster
- Analysis Step 2: Self-citation extraction (2,852 pairs)
- Analysis Step 3: Visualisation (scatter grid + global heatmap)
- Analysis Step 2b: Self-citation pair resolution pipeline built (422/2852 = 14.8% resolved); paused after review
- Analysis Step 3b: Cluster publication timelines — chapter-level and chunk-level, plus CSV export
- Long-term memory: `copilot-instructions.md`, `generate_project_status.py`, MCP knowledge graph seeded (42 entities, 27 relations)

### Pending / Paused ❌

- **Analysis Step 4:** Cross-cluster citation matrix — which cluster's chunks cite works from which other clusters (match `cited_year` → cluster of that year's chapters)
- **Analysis Step 2b (resumption):** Citation pair resolution — improving the 14.8% resolution rate, e.g. by filtering monographs from `ambiguous_year` candidates; only worthwhile if a use case for chapter-level vector comparison is identified
- `chunk_token_count` field — currently 0 for all chunks; filling this enables tighter context window packing
- Batch query script — `query_library_batch.py` for running many questions from a file and saving results to CSV

### Design Decisions (record for continuity)

- **No metadata filtering by book/author in RAG** — BM25 indexing of raw text handles this naturally; a mention of a book number or title in the question focuses retrieval correctly without a separate filter mechanism.
- **Clusters 6 & 12 excluded from citation content analysis and timelines** — confirmed bibliography/reference-list noise by exemplar inspection.
- **`chapter_reference` preferred over `book_reference` for pub years** — more precise for chapters published before the book's official year.
- **Inline citations only, not reference list entries** — reference list format `Stahl, G. (YYYY)` is intentionally excluded from citation extraction; the `G.` initial between name and year breaks all three regex patterns.
- **Citation pair resolution paused at 14.8%** — chapter-level vectors considered too coarse for meaningful inter-chapter semantic comparison; the dominant failure mode (ambiguous_year, 38.7%) would require year-suffix disambiguation heuristics.
- **Chunk-level timeline preferred over chapter-level** — chunk counts per cluster per year show the thematic composition *within* chapters, not just the dominant topic; this is a more granular and honest picture of thematic evolution.
- **Non-CSCL clusters (8, 9, 15) included in timelines** — included per Gerry's instruction; they are part of the library even if peripheral to CSCL analysis.
