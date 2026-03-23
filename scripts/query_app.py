"""
scripts/query_app.py
~~~~~~~~~~~~~~~~~~~~
Gradio web chat interface for the e-library RAG system.

Usage:
    python scripts/query_app.py
    # then open http://localhost:7860 in your browser

Features:
  - Multi-turn chat with conversation memory (last 4 Q&A turns sent to the LLM)
  - Query rewriting — resolves pronouns/references before retrieval; uses the same
    backend as generation (Ollama for local models, Claude Haiku for cloud models)
  - Persistent query history — every exchange saved to vector_store/query_history.json
    with semantic search so past relevant Q&As surface as context for new questions
  - Model selector: Claude (Haiku / Sonnet / Opus) or local Ollama models
    (Qwen2.5 14b / Llama3.1 8b / Gemma3 4b) — all free after initial download
  - Chapter/passage sliders, reranker toggle
  - Sources panel — shows which chapters and passages were retrieved
  - User name field — name tag saved with each query in the history log
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
ROOT    = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

import gradio as gr
from query_history import QueryHistory
import numpy as np
import faiss
import bm25s
import Stemmer
from sentence_transformers import SentenceTransformer, CrossEncoder
import anthropic

# ── Config ─────────────────────────────────────────────────────────────────────
EMBED_MODEL          = "intfloat/e5-base-v2"
RERANKER_MODEL       = "BAAI/bge-reranker-base"
QUERY_PFX            = "query: "
RRF_K                = 60
DENSE_WIDE           = 400
RERANKER_CANDIDATES  = 20   # fetch this many via RRF, then rerank to top_m
HISTORY_TURNS        = 4    # prior conversation turns passed to Claude for context
REWRITE_MODEL        = "claude-haiku-4-5-20251001"  # cheap model for query rewriting
DEFAULT_USER         = "Gerry"   # name tag written to query history records

MODELS = {
    "Haiku  (fast · cheap)":        "claude-haiku-4-5-20251001",
    "Sonnet (balanced)":            "claude-sonnet-4-20250514",
    "Opus   (thorough · slow)":     "claude-opus-4-20250514",
    "Qwen2.5 14b  (local · free)":  "qwen2.5:14b",
    "Llama3.1 8b  (local · free)":  "llama3.1:8b",
    "Gemma3 4b   (local · free)":   "gemma3:4b",
}

# ── Ollama local backend ───────────────────────────────────────────────────────
OLLAMA_URL          = "http://localhost:11434"    # default Ollama server
OLLAMA_LOCAL_MODELS = {"qwen2.5:14b", "llama3.1:8b", "gemma3:4b"}  # route locally

VS_DIR         = ROOT / "vector_store"
CH_FAISS       = VS_DIR / "chapter_summaries.faiss"
CH_META_FILE   = VS_DIR / "chapter_summaries_meta.json"
CH_BM25_DIR    = VS_DIR / "summaries_bm25"
CHUNK_FAISS     = VS_DIR / "elibrary.faiss"
CHUNK_META_FILE = VS_DIR / "elibrary_meta.json"
CHUNK_BM25_DIR  = VS_DIR / "chunks_bm25"

SYSTEM_PROMPT = """\
You are an expert research librarian with deep knowledge of a 22-volume \
academic library. Answer the user's question based solely on the provided \
passage excerpts from the library.

Guidelines:
- Answer in clear, scholarly prose.
- Cite each passage you draw on using the format [Book N, Ch M] inline.
- If multiple passages support a point, cite all of them.
- If the passages do not contain enough information, say so plainly.
- Do not invent information beyond what the passages state.
"""

# ── Load all indexes once at module load ───────────────────────────────────────
print("Loading e-library indexes…", flush=True)
stemmer = Stemmer.Stemmer("english")

print("  · Embedding model…", flush=True)
_embed_model = SentenceTransformer(EMBED_MODEL)

print("  · Reranker…", flush=True)
_reranker = CrossEncoder(RERANKER_MODEL)

print("  · Chapter indexes…", flush=True)
_ch_faiss = faiss.read_index(str(CH_FAISS))
with open(CH_META_FILE) as f:
    _ch_meta = json.load(f)
_ch_bm25 = bm25s.BM25.load(str(CH_BM25_DIR), load_corpus=False)
with open(CH_BM25_DIR / "sidecar.json") as f:
    _ch_sidecar = json.load(f)

print("  · Chunk indexes…", flush=True)
_chunk_faiss = faiss.read_index(str(CHUNK_FAISS))
with open(CHUNK_META_FILE) as f:
    _chunk_meta = json.load(f)
_chunk_bm25 = bm25s.BM25.load(str(CHUNK_BM25_DIR), load_corpus=False)
with open(CHUNK_BM25_DIR / "sidecar.json") as f:
    _chunk_sidecar = json.load(f)

_vid_to_chunk: dict[int, dict] = {
    int(e["vector_id"]): e
    for e in _chunk_sidecar.values()
    if e["vector_id"] is not None
}

print("  All indexes ready.", flush=True)
print("  · Query history…", flush=True)
_query_history = QueryHistory()
print(f"    {len(_query_history)} past queries loaded.\n", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Retrieval helpers
# ══════════════════════════════════════════════════════════════════════════════

def _embed(text: str) -> np.ndarray:
    """Return a (1, 768) L2-normalised float32 query vector."""
    return _embed_model.encode(
        [QUERY_PFX + text], normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)


def _rrf(ranked_lists: list[list], k: int = RRF_K) -> dict:
    """Merge ranked lists with Reciprocal Rank Fusion; returns {item: score}."""
    scores: dict = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return scores


def retrieve_chapters(query: str, top_n: int) -> list[tuple[int, int]]:
    """Return top-N (book_number, chapter_number) pairs ranked by RRF."""
    q_vec = _embed(query)
    K = min(top_n * 6, _ch_faiss.ntotal)
    _, idxs = _ch_faiss.search(q_vec, K)
    dense_keys = []
    for vid in (int(i) for i in idxs[0] if i >= 0):
        m = _ch_meta.get(str(vid))
        if m:
            dense_keys.append((int(m["book_number"]), int(m["chapter_number"])))

    q_tok = bm25s.tokenize([query], stopwords="en", stemmer=stemmer, show_progress=False)
    bm25_res, _ = _ch_bm25.retrieve(q_tok, k=min(top_n * 6, len(_ch_sidecar)))
    bm25_keys = []
    for did in (int(i) for i in bm25_res[0]):
        s = _ch_sidecar.get(str(did))
        if s:
            bm25_keys.append((int(s["book_number"]), int(s["chapter_number"])))

    merged = _rrf([dense_keys, bm25_keys])
    return sorted(merged, key=lambda c: merged[c], reverse=True)[:top_n]


def _rerank(query: str, candidates: list[dict], top_m: int) -> list[dict]:
    """Score (query, chunk_text) pairs with the cross-encoder; return top_m."""
    if not candidates:
        return candidates
    pairs  = [(query, c["chunk_text"]) for c in candidates]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_m]]


def retrieve_chunks(
    query: str, top_chapters: list[tuple[int, int]], top_m: int,
    use_reranker: bool = True,
) -> list[dict]:
    chapter_set  = set(top_chapters)
    n_candidates = max(top_m, RERANKER_CANDIDATES) if use_reranker else top_m
    q_vec = _embed(query)

    _, idxs = _chunk_faiss.search(q_vec, min(DENSE_WIDE, _chunk_faiss.ntotal))
    dense_vids = []
    for vid in (int(i) for i in idxs[0] if i >= 0):
        m = _chunk_meta.get(str(vid))
        if m and m["chunk_level"] == 0:
            if (int(m["book_number"]), int(m["chapter_number"])) in chapter_set:
                dense_vids.append(vid)

    q_tok = bm25s.tokenize([query], stopwords="en", stemmer=stemmer, show_progress=False)
    bm25_res, _ = _chunk_bm25.retrieve(q_tok, k=min(200, len(_chunk_sidecar)))
    bm25_vids = []
    for did in (int(i) for i in bm25_res[0]):
        s = _chunk_sidecar.get(str(did))
        if s and (int(s["book_number"]), int(s["chapter_number"])) in chapter_set:
            vid = s.get("vector_id")
            if vid is not None:
                bm25_vids.append(int(vid))

    merged = _rrf([dense_vids, bm25_vids])
    top_vids = sorted(merged, key=lambda v: merged[v], reverse=True)[:n_candidates]
    candidates = [e for vid in top_vids if (e := _vid_to_chunk.get(vid))]

    if use_reranker and len(candidates) > top_m:
        return _rerank(query, candidates, top_m)
    return candidates[:top_m]


# ══════════════════════════════════════════════════════════════════════════════
# Generation
# ══════════════════════════════════════════════════════════════════════════════

def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as numbered context blocks for the Claude prompt."""
    parts = []
    for i, c in enumerate(chunks, 1):
        section = f" — {c['section']}" if c.get("section") else ""
        header  = (f"[Passage {i}] Book {c['book_number']}, "
                   f"Ch {c['chapter_number']}: \"{c['chapter_title']}{section}\" "
                   f"(p. {c['chunk_page']}, ~{c['chunk_words']} words)")
        parts.append(f"{header}\n{c['chunk_text'].strip()}")
    return "\n\n---\n\n".join(parts)


def _format_sources_md(
    top_chapters: list[tuple[int, int]], chunks: list[dict]
) -> str:
    """Return a Markdown string listing retrieved chapters and passages."""
    lines = ["### 📚 Retrieved sources\n"]
    lines.append("**Chapters searched:**")
    for bn, cn in top_chapters:
        m = next(
            (v for v in _ch_meta.values()
             if int(v["book_number"]) == bn and int(v["chapter_number"]) == cn),
            None,
        )
        title = m["chapter_title"] if m else f"Ch {cn}"
        book  = m["book_title"]    if m else f"Book {bn}"
        lines.append(f"- Book {bn}, Ch {cn}: *{title}* — {book}")
    lines.append("\n**Passages used:**")
    for i, c in enumerate(chunks, 1):
        snippet = c["chunk_text"].replace("\n", " ").strip()[:100]
        lines.append(
            f"{i}. Book {c['book_number']} Ch {c['chapter_number']} "
            f"p.{c['chunk_page']} ({c['chunk_words']}w) — {snippet}…"
        )
    return "\n".join(lines)


def _is_local_model(model_id: str) -> bool:
    """Return True if *model_id* should be routed to the local Ollama server."""
    return model_id in OLLAMA_LOCAL_MODELS


def _call_ollama(
    messages: list[dict],
    model_id: str,
    max_tokens: int = 1500,
) -> str:
    """
    Low-level call to the Ollama /api/chat endpoint.
    *messages* must follow the OpenAI roles format (system/user/assistant).
    """
    payload = json.dumps({
        "model": model_id,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": max_tokens},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    return result["message"]["content"]


def rewrite_query(question: str, history: list[dict], model_id: str = REWRITE_MODEL) -> str:
    """
    If there is conversation history, rewrite *question* as a self-contained
    search query with all pronouns and references resolved.
    Uses the local Ollama backend when *model_id* is a local model, otherwise
    falls back to Claude Haiku — so rewriting and generation always use the
    same backend.
    Returns the rewritten query (used only for retrieval; the original question
    is shown to the user and sent to the LLM for generation).
    """
    if not history:
        return question
    # Build compact conversation snippet (last HISTORY_TURNS turns = 2×N dicts)
    turns = history[-(HISTORY_TURNS * 2):]
    convo = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}"
        for m in turns
    )
    prompt = (
        f"Conversation so far:\n{convo}\n\n"
        f"Follow-up question: {question}\n\n"
        "Rewrite the follow-up as a self-contained search query with all "
        "pronouns and references resolved. Output ONLY the rewritten query."
    )
    if _is_local_model(model_id):
        return _call_ollama(
            [{"role": "user", "content": prompt}],
            model_id=model_id,
            max_tokens=120,
        ).strip()
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=REWRITE_MODEL,
        max_tokens=120,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def ask_claude(
    question: str,
    chunks: list[dict],
    model_id: str,
    history: list[dict] | None = None,
    past_records: list[dict] | None = None,
) -> str:
    """Generate an answer from the LLM.  Routes to Ollama for local models."""
    context  = _format_context(chunks)
    past_ctx = (QueryHistory.format_for_context(past_records) + "\n\n---\n\n"
                if past_records else "")
    user_msg = f"{past_ctx}LIBRARY PASSAGES:\n\n{context}\n\n---\n\nQUESTION: {question}"

    # Build message list: prior turns (Q&A, no passages) + current turn with passages
    messages: list[dict] = []
    if history:
        for m in history[-(HISTORY_TURNS * 2):]:
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_msg})

    if _is_local_model(model_id):
        # Prepend system prompt as a system role message (Ollama supports it)
        return _call_ollama(
            [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            model_id=model_id,
        )

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=model_id,
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return msg.content[0].text


# ══════════════════════════════════════════════════════════════════════════════
# Gradio callback
# ══════════════════════════════════════════════════════════════════════════════

def respond(
    message: str,
    history: list[dict],
    model_label: str,
    n_chapters: int,
    n_chunks: int,
    show_sources: bool,
    use_reranker: bool,
    user_name: str,
) -> tuple[list[dict], str]:
    """
    Gradio chatbot callback.
    Returns (updated_history, sources_markdown).
    """
    if not message.strip():
        return history, ""

    model_id = MODELS[model_label]

    # Embed original question for persistent history search (query: prefix applied)
    q_vec        = _embed(message)[0]             # (768,) L2-normalised
    past_records = _query_history.search(q_vec)

    # Rewrite query for retrieval when there is prior conversation
    retrieval_query = rewrite_query(message, history, model_id=model_id)

    top_chapters = retrieve_chapters(retrieval_query, top_n=n_chapters)
    top_chunks   = retrieve_chunks(retrieval_query, top_chapters, top_m=n_chunks,
                                   use_reranker=use_reranker)

    if not top_chunks:
        reply = (
            "I couldn't find relevant passages for that question. "
            "Try rephrasing or broadening your query."
        )
        sources_md = ""
    else:
        reply      = ask_claude(message, top_chunks, model_id=model_id,
                                history=history, past_records=past_records)
        sources_md = _format_sources_md(top_chapters, top_chunks) if show_sources else ""
        _query_history.add(
            question=message,
            retrieval_query=retrieval_query,
            answer=reply,
            model=model_id,
            top_chapters=top_chapters,
            top_chunks=top_chunks,
            embedding=q_vec,
            ch_meta=_ch_meta,
            user=user_name,
        )

    history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": reply},
    ]
    return history, sources_md


# ══════════════════════════════════════════════════════════════════════════════
# Gradio UI
# ══════════════════════════════════════════════════════════════════════════════

DESCRIPTION = """
## 📖 E-Library Research Assistant

Ask any question about Gerry Stahl's 22-volume academic library. The system uses
**hybrid retrieval** (semantic search + BM25 keyword search) to find the most
relevant passages, then generates a grounded answer with inline citations.

**Examples to try:**
- *What is group cognition and how does it differ from individual cognition?*
- *How does Stahl use Heidegger's concept of concealment?*
- *What role does GeoGebra play in collaborative mathematics learning?*
- *How do virtual math teams build a joint problem space?*
"""

FOOTER = (
    "**Tip:** Use **Sonnet** or **Opus** for detailed cloud synthesis. "
    "**Qwen2.5 14b** is a strong free local alternative. "
    "Increase *Chapters* to 7–10 for broad comparative questions."
)

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="E-Library Research Assistant",
        theme=gr.themes.Soft(primary_hue="slate"),
    ) as app:

        gr.Markdown(DESCRIPTION)

        with gr.Row():
            # ── Left column: controls ────────────────────────────────────────
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### ⚙️ Settings")

                user_name = gr.Textbox(
                    value=DEFAULT_USER,
                    label="Your name",
                    max_lines=1,
                    info="Saved with each query in the history log",
                )
                model_radio = gr.Radio(
                    choices=list(MODELS.keys()),
                    value=list(MODELS.keys())[0],   # Haiku default
                    label="Model",
                )
                n_chapters = gr.Slider(
                    minimum=1, maximum=15, value=5, step=1,
                    label="Chapters retrieved",
                    info="How many chapters to focus the search on",
                )
                n_chunks = gr.Slider(
                    minimum=2, maximum=20, value=8, step=1,
                    label="Passages sent to Claude",
                    info="More passages = richer context, higher cost",
                )
                show_sources = gr.Checkbox(
                    value=False,
                    label="Show retrieved sources",
                    info="Display which chapters and passages were used",
                )
                use_reranker = gr.Checkbox(
                    value=True,
                    label="Cross-encoder reranker (bge-reranker-base)",
                    info="Reranks top-20 candidates for higher precision. Adds ~50ms.",
                )
                clear_btn = gr.Button("🗑️ Clear conversation", variant="secondary")

                gr.Markdown("---")
                gr.Markdown(FOOTER)

            # ── Right column: chat ───────────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Research Assistant",
                    height=520,
                    buttons=["copy", "copy_all"],
                    layout="bubble",
                    placeholder=(
                        "Ask a question about the e-library. "
                        "Answers will include inline citations like [Book N, Ch M]."
                    ),
                )

                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Ask a question…",
                        show_label=False,
                        scale=5,
                        container=False,
                        autofocus=True,
                    )
                    send_btn = gr.Button("Send ➤", variant="primary", scale=1)

                sources_box = gr.Markdown(visible=True, label="Sources")

        # ── Wiring ────────────────────────────────────────────────────────────
        def _submit(msg, history, model, nc, nk, ss, ur, un):
            new_history, sources = respond(msg, history, model, nc, nk, ss, ur, un)
            return new_history, sources, ""   # clear textbox

        send_btn.click(
            fn=_submit,
            inputs=[msg_box, chatbot, model_radio, n_chapters, n_chunks, show_sources, use_reranker, user_name],
            outputs=[chatbot, sources_box, msg_box],
        )
        msg_box.submit(
            fn=_submit,
            inputs=[msg_box, chatbot, model_radio, n_chapters, n_chunks, show_sources, use_reranker, user_name],
            outputs=[chatbot, sources_box, msg_box],
        )
        clear_btn.click(
            fn=lambda: ([], ""),
            outputs=[chatbot, sources_box],
            queue=False,
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,       # auto-open browser tab
        share=False,
    )
