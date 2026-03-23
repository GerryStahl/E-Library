"""
scripts/query_library.py
~~~~~~~~~~~~~~~~~~~~~~~~
Interactive hybrid RAG query system for the e-library (terminal REPL).

Retrieval pipeline:
  Stage 1 — Chapter coarse retrieval
    Dense:  chapter_summaries.faiss  (e5-base-v2, 768-dim, cosine)
    Sparse: summaries_bm25/          (BM25 with English stemming)
    Merge:  Reciprocal Rank Fusion   → top-N chapters

  Stage 2 — Chunk fine retrieval (within top-N chapters)
    Dense:  elibrary.faiss           (level-0 chunks filtered by chapter)
    Sparse: chunks_bm25/             (BM25 level-0 chunks, filtered by chapter)
    Merge:  Reciprocal Rank Fusion   → RERANKER_CANDIDATES chunks

  Stage 3 — Cross-encoder reranking (optional, default ON)
    Model:  BAAI/bge-reranker-base   (278M params, runs on MPS/CPU)
    Input:  (query, chunk_text) pairs for top candidates
    Output: reranked top-M chunks sent to Claude

  Generation — Claude (Anthropic API) or Ollama (local)
    Model is configurable (defaults to Haiku for speed; pass a local model ID for
    free offline generation).  Conversation history (last HISTORY_TURNS turns)
    is prepended to the prompt.  Semantically similar past queries from
    query_history.json are also injected.  Each retrieved chunk is cited;
    the model answers with inline [Book N, Ch M] refs.

Usage:
    python scripts/query_library.py [--model MODEL] [--chapters N] [--chunks M]
                                    [--user NAME] [--verbose] [--no-rerank]

    MODEL can be a Claude model ID (default: claude-haiku-4-5-20251001) or a
    local Ollama model (qwen2.5:14b, llama3.1:8b, gemma3:4b).

Session commands:
    clear   Reset conversation history for this session
    quit / exit / q   Exit the program

Query conventions (must match build scripts):
    Passage embeddings: "passage: " + text  (already in FAISS indexes)
    Query  embeddings:  "query: "   + text  (applied here at runtime)
"""

from __future__ import annotations

import argparse
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

import numpy as np
import faiss
import bm25s
import Stemmer
from sentence_transformers import SentenceTransformer, CrossEncoder
import anthropic
from query_history import QueryHistory

# ── Default config ─────────────────────────────────────────────────────────────
DEFAULT_MODEL        = "claude-haiku-4-5-20251001"    # fast; swap for sonnet/opus
EMBED_MODEL          = "intfloat/e5-base-v2"
RERANKER_MODEL       = "BAAI/bge-reranker-base"       # cross-encoder reranker
QUERY_PFX            = "query: "
TOP_CHAPTERS         = 5    # coarse retrieval: chapters to focus on
TOP_CHUNKS           = 8    # final chunks sent to Claude
RERANKER_CANDIDATES  = 20   # candidates fetched before reranking (>= TOP_CHUNKS)
RRF_K                = 60   # standard RRF constant
DENSE_SEARCH_WIDE    = 400  # FAISS over-fetch before chapter filtering
USE_RERANKER         = True # set False to skip reranking (faster, slightly lower quality)
HISTORY_TURNS        = 4    # prior conversation turns passed to Claude for context
REWRITE_MODEL        = "claude-haiku-4-5-20251001"  # cheap model for query rewriting

# ── Ollama local backend ───────────────────────────────────────────────────────
OLLAMA_URL          = "http://localhost:11434"    # default Ollama server
OLLAMA_LOCAL_MODELS = {"qwen2.5:14b", "llama3.1:8b", "gemma3:4b"}  # route locally

# ── Paths ──────────────────────────────────────────────────────────────────────
VS_DIR            = ROOT / "vector_store"
CH_FAISS          = VS_DIR / "chapter_summaries.faiss"
CH_META_FILE      = VS_DIR / "chapter_summaries_meta.json"
CH_BM25_DIR       = VS_DIR / "summaries_bm25"
CHUNK_FAISS       = VS_DIR / "elibrary.faiss"
CHUNK_META_FILE   = VS_DIR / "elibrary_meta.json"
CHUNK_BM25_DIR    = VS_DIR / "chunks_bm25"


# ══════════════════════════════════════════════════════════════════════════════
# Reciprocal Rank Fusion
# ══════════════════════════════════════════════════════════════════════════════

def rrf(ranked_lists: list[list], k: int = RRF_K) -> dict:
    """
    Merge *ranked_lists* with Reciprocal Rank Fusion.

    Each element of ranked_lists is a list of item IDs ordered best-first.
    Returns {item_id: rrf_score}, highest score = best.
    """
    scores: dict = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# Indexes (loaded once at startup)
# ══════════════════════════════════════════════════════════════════════════════

class Indexes:
    """Holds all loaded indexes and provides retrieval helpers."""

    def __init__(self) -> None:
        self.stemmer = Stemmer.Stemmer("english")

        print("Loading embedding model…", flush=True)
        self.embed_model = SentenceTransformer(EMBED_MODEL)

        print("Loading reranker…", flush=True)
        self.reranker = CrossEncoder(RERANKER_MODEL)

        print("Loading chapter indexes…", flush=True)
        self.ch_faiss  = faiss.read_index(str(CH_FAISS))
        with open(CH_META_FILE) as f:
            self.ch_meta: dict[str, dict] = json.load(f)
        self.ch_bm25 = bm25s.BM25.load(str(CH_BM25_DIR), load_corpus=False)
        with open(CH_BM25_DIR / "sidecar.json") as f:
            self.ch_sidecar: dict[str, dict] = json.load(f)

        print("Loading chunk indexes…", flush=True)
        self.chunk_faiss = faiss.read_index(str(CHUNK_FAISS))
        with open(CHUNK_META_FILE) as f:
            self.chunk_meta: dict[str, dict] = json.load(f)
        self.chunk_bm25 = bm25s.BM25.load(str(CHUNK_BM25_DIR), load_corpus=False)
        with open(CHUNK_BM25_DIR / "sidecar.json") as f:
            self.chunk_sidecar: dict[str, dict] = json.load(f)

        # Fast lookup: vector_id (int) → chunk_sidecar entry
        self.vid_to_chunk: dict[int, dict] = {
            int(entry["vector_id"]): entry
            for entry in self.chunk_sidecar.values()
            if entry["vector_id"] is not None
        }

        print("All indexes ready.\n", flush=True)

    # ── Embed ─────────────────────────────────────────────────────────────────

    def embed_query(self, text: str) -> np.ndarray:
        """Return L2-normalised 768-dim query vector."""
        vec = self.embed_model.encode(
            [QUERY_PFX + text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        return vec   # shape (1, 768)

    # ── Stage 1: Chapter retrieval ────────────────────────────────────────────

    def retrieve_chapters(
        self, query: str, top_n: int = TOP_CHAPTERS
    ) -> list[tuple[int, int]]:
        """
        Return a list of (book_number, chapter_number) pairs ranked by RRF,
        representing the most relevant chapters for *query*.
        """
        q_vec = self.embed_query(query)

        # Dense: search chapter_summaries.faiss
        K_dense = min(top_n * 6, self.ch_faiss.ntotal)
        scores, indices = self.ch_faiss.search(q_vec, K_dense)
        dense_order = [int(i) for i in indices[0] if i >= 0]

        # BM25: search summary corpus
        q_tok = bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer,
                               show_progress=False)
        K_bm25 = min(top_n * 6, len(self.ch_sidecar))
        bm25_results, _ = self.ch_bm25.retrieve(q_tok, k=K_bm25)
        bm25_order = [int(i) for i in bm25_results[0]]

        # Convert FAISS vector_ids to chapter keys (book_num, ch_num)
        def ch_key_from_faiss(vid: int) -> tuple[int, int] | None:
            m = self.ch_meta.get(str(vid))
            if m:
                return (int(m["book_number"]), int(m["chapter_number"]))
            return None

        def ch_key_from_bm25(doc_id: int) -> tuple[int, int] | None:
            s = self.ch_sidecar.get(str(doc_id))
            if s:
                return (int(s["book_number"]), int(s["chapter_number"]))
            return None

        dense_keys = [k for vid in dense_order
                      if (k := ch_key_from_faiss(vid)) is not None]
        bm25_keys  = [k for did in bm25_order
                      if (k := ch_key_from_bm25(did)) is not None]

        merged = rrf([dense_keys, bm25_keys])
        top_chapters = sorted(merged, key=lambda c: merged[c], reverse=True)[:top_n]
        return top_chapters

    # ── Reranker ──────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_m: int,
    ) -> list[dict]:
        """
        Score each (query, chunk_text) pair with the cross-encoder and return
        the top-*top_m* candidates re-ordered by reranker score.
        """
        if not candidates:
            return candidates
        pairs  = [(query, c["chunk_text"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_m]]

    # ── Stage 2: Chunk retrieval ───────────────────────────────────────────────

    def retrieve_chunks(
        self,
        query: str,
        top_chapters: list[tuple[int, int]],
        top_m: int = TOP_CHUNKS,
        use_reranker: bool = USE_RERANKER,
    ) -> list[dict]:
        """
        Return top-M chunk sidecar entries relevant to *query* within *top_chapters*.
        If use_reranker=True, fetches RERANKER_CANDIDATES via RRF then reranks with
        bge-reranker-base before trimming to top_m.
        Each entry has keys: book_number, book_title, chapter_number, chapter_title,
        chunk_page, section, chunk_text, chunk_words.
        """
        n_candidates = max(top_m, RERANKER_CANDIDATES) if use_reranker else top_m
        chapter_set = set(top_chapters)
        q_vec = self.embed_query(query)

        # ── Dense: full FAISS search, filter to top chapters + level-0 ─────────
        K_wide = min(DENSE_SEARCH_WIDE, self.chunk_faiss.ntotal)
        scores, indices = self.chunk_faiss.search(q_vec, K_wide)

        dense_order: list[int] = []   # vector_ids ordered by score
        for vid in (int(i) for i in indices[0] if i >= 0):
            m = self.chunk_meta.get(str(vid))
            if m is None:
                continue
            if m["chunk_level"] != 0:
                continue
            if (int(m["book_number"]), int(m["chapter_number"])) not in chapter_set:
                continue
            dense_order.append(vid)

        # ── BM25: search chunk corpus, filter to top chapters ──────────────────
        q_tok = bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer,
                               show_progress=False)
        K_bm25 = min(200, len(self.chunk_sidecar))
        bm25_results, _ = self.chunk_bm25.retrieve(q_tok, k=K_bm25)

        bm25_order: list[int] = []   # vector_ids ordered by BM25 score
        for doc_id in (int(i) for i in bm25_results[0]):
            s = self.chunk_sidecar.get(str(doc_id))
            if s is None:
                continue
            if (int(s["book_number"]), int(s["chapter_number"])) not in chapter_set:
                continue
            vid = s.get("vector_id")
            if vid is not None:
                bm25_order.append(int(vid))

        # ── RRF merge by vector_id ─────────────────────────────────────────────
        merged = rrf([dense_order, bm25_order])
        top_vids = sorted(merged, key=lambda v: merged[v], reverse=True)[:n_candidates]

        # ── Fetch sidecar entries ──────────────────────────────────────────────
        candidates = []
        for vid in top_vids:
            entry = self.vid_to_chunk.get(vid)
            if entry:
                candidates.append(entry)

        # ── Rerank then trim to top_m ──────────────────────────────────────────
        if use_reranker and len(candidates) > top_m:
            return self.rerank(query, candidates, top_m)
        return candidates[:top_m]


# ══════════════════════════════════════════════════════════════════════════════
# Generation
# ══════════════════════════════════════════════════════════════════════════════

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


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as numbered context blocks for the Claude prompt."""
    parts = []
    for i, c in enumerate(chunks, 1):
        section = f" — {c['section']}" if c.get("section") else ""
        header  = (f"[Passage {i}] Book {c['book_number']}, "
                   f"Ch {c['chapter_number']}: \"{c['chapter_title']}{section}\" "
                   f"(p. {c['chunk_page']}, ~{c['chunk_words']} words)")
        parts.append(f"{header}\n{c['chunk_text'].strip()}")
    return "\n\n---\n\n".join(parts)


def _is_local_model(model_id: str) -> bool:
    """Return True if *model_id* should be routed to the local Ollama server."""
    return model_id in OLLAMA_LOCAL_MODELS


def _call_ollama(
    messages: list[dict],
    model_id: str,
    max_tokens: int = 1024,
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
    falls back to Claude Haiku.
    Returns the rewritten query (used only for retrieval).
    """
    if not history:
        return question
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
    model: str = DEFAULT_MODEL,
    history: list[dict] | None = None,
    past_records: list[dict] | None = None,
) -> str:
    """Generate an answer from the LLM.  Routes to Ollama for local models."""
    context  = format_context(chunks)
    past_ctx = (QueryHistory.format_for_context(past_records) + "\n\n---\n\n"
                if past_records else "")
    user_msg = f"{past_ctx}LIBRARY PASSAGES:\n\n{context}\n\n---\n\nQUESTION: {question}"

    # Build message list: prior turns (Q&A, no passages) + current turn with passages
    messages: list[dict] = []
    if history:
        for m in history[-(HISTORY_TURNS * 2):]:
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_msg})

    if _is_local_model(model):
        return _call_ollama(
            [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            model_id=model,
        )

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return msg.content[0].text


# ══════════════════════════════════════════════════════════════════════════════
# Pretty output helpers
# ══════════════════════════════════════════════════════════════════════════════

def print_chapter_hits(chapters: list[tuple[int, int]], ch_meta: dict) -> None:
    """Print the top-N retrieved chapters."""
    print("\n── Top chapters ────────────────────────────────────")
    for rank, (bn, cn) in enumerate(chapters, 1):
        # Find matching entry in ch_meta
        entry = next(
            (m for m in ch_meta.values()
             if int(m["book_number"]) == bn and int(m["chapter_number"]) == cn),
            None,
        )
        if entry:
            print(f"  {rank}. Book {bn}, Ch {cn}: \"{entry['chapter_title']}\"")
            print(f"       {entry['book_title']}")
        else:
            print(f"  {rank}. Book {bn}, Ch {cn}")


def print_chunk_hits(chunks: list[dict]) -> None:
    """Print the top-M retrieved chunks (snippet only)."""
    print("\n── Retrieved passages ──────────────────────────────")
    for i, c in enumerate(chunks, 1):
        snippet = c["chunk_text"].replace("\n", " ").strip()[:120]
        section = f" [{c['section']}]" if c.get("section") else ""
        print(f"  {i}. Book {c['book_number']} Ch {c['chapter_number']}"
              f"{section} p.{c['chunk_page']}  ({c['chunk_words']}w)")
        print(f"     {snippet}…")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid RAG query over the e-library.")
    p.add_argument("--model",    default=DEFAULT_MODEL,
                   help=(f"Model to use for generation (default: {DEFAULT_MODEL}). "
                         "Claude options: claude-haiku-4-5-20251001, claude-sonnet-4-20250514, "
                         "claude-opus-4-20250514. "
                         "Local (Ollama) options: qwen2.5:14b, llama3.1:8b, gemma3:4b"))
    p.add_argument("--chapters", type=int, default=TOP_CHAPTERS,
                   help=f"Number of chapters to retrieve (default: {TOP_CHAPTERS})")
    p.add_argument("--chunks",   type=int, default=TOP_CHUNKS,
                   help=f"Number of chunks to send to Claude (default: {TOP_CHUNKS})")
    p.add_argument("--no-rerank", action="store_true",
                   help="Disable cross-encoder reranking (faster, slightly lower quality)")
    p.add_argument("--verbose",  action="store_true",
                   help="Show retrieved chapters and passage snippets")
    p.add_argument("--user",     default="Gerry",
                   help="Name tag written to query history records (default: Gerry)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    use_reranker = not args.no_rerank
    rerank_label = f"reranker={RERANKER_MODEL.split('/')[-1]}" if use_reranker else "no reranker"
    print(f"\n{'='*60}")
    print(f"  E-Library RAG  |  {args.model}  |  user: {args.user}")
    print(f"  chapters={args.chapters}  chunks={args.chunks}  hybrid+RRF+{rerank_label}")
    print(f"{'='*60}\n")

    idx = Indexes()

    print("Loading query history…", flush=True)
    _hist = QueryHistory()
    print(f"  {len(_hist)} past queries loaded.\n", flush=True)

    print("Type a question and press Enter.")
    print("Commands: 'clear' to reset conversation  |  'quit' or 'exit' to stop.\n")
    conversation_history: list[dict] = []
    while True:
        try:
            question = input("❓ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break
        if question.lower() == "clear":
            conversation_history = []
            print("[Conversation cleared.]\n")
            continue

        print("\nRetrieving…", flush=True)

        # Embed question for persistent history search
        q_vec        = idx.embed_query(question)[0]   # (768,) L2-normalised
        past_records = _hist.search(q_vec)

        # Rewrite query for retrieval if there is prior conversation
        retrieval_query = rewrite_query(question, conversation_history, model_id=args.model)
        if retrieval_query != question and args.verbose:
            print(f"  [Rewritten query: {retrieval_query}]")

        # Stage 1: chapters
        top_chapters = idx.retrieve_chapters(retrieval_query, top_n=args.chapters)
        if args.verbose:
            print_chapter_hits(top_chapters, idx.ch_meta)

        # Stage 2: chunks (+ reranking)
        top_chunks = idx.retrieve_chunks(
            retrieval_query, top_chapters, top_m=args.chunks, use_reranker=use_reranker
        )
        if args.verbose:
            print_chunk_hits(top_chunks)

        if not top_chunks:
            print("\n[No relevant passages found — try rephrasing your question.]\n")
            continue

        # Generation
        print("\nGenerating answer…", flush=True)
        answer = ask_claude(question, top_chunks, model=args.model,
                            history=conversation_history, past_records=past_records)

        print("\n" + "─" * 60)
        print(answer)
        print("─" * 60 + "\n")

        # Update persistent history + conversation memory
        _hist.add(
            question=question,
            retrieval_query=retrieval_query,
            answer=answer,
            model=args.model,
            top_chapters=top_chapters,
            top_chunks=top_chunks,
            embedding=q_vec,
            ch_meta=idx.ch_meta,
            user=args.user,
        )
        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
