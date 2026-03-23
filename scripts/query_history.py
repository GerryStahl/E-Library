"""
scripts/query_history.py
~~~~~~~~~~~~~~~~~~~~~~~~
Persistent semantic Q&A history for the e-library RAG system.

Every Q&A exchange is saved to a human-readable JSON file and a parallel numpy
embedding array, enabling two capabilities:

  1. Audit log — scan vector_store/query_history.json to browse all past queries:
       date, original question, retrieval query used, answer, chapters retrieved,
       and a 200-char snippet of each passage chunk.

  2. Cross-session context injection — when you ask a new question, the system
       finds past Q&A entries with high cosine similarity to your question and
       prepends them to the Claude generation prompt, so Claude can say "as noted
       in a previous query..." and build on prior answers.

Files written to vector_store/:
  query_history.json            — all records (human-readable, append-only)
  query_history_embeddings.npy  — float32 (N, 768), row i = embedding of record i
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT         = Path(__file__).resolve().parent.parent
HISTORY_JSON = ROOT / "vector_store" / "query_history.json"
HISTORY_EMB  = ROOT / "vector_store" / "query_history_embeddings.npy"

SIM_THRESHOLD = 0.78   # minimum cosine similarity to surface a past Q&A as context
MAX_PAST      = 3      # maximum past records injected per turn


class QueryHistory:
    """
    Persistent store of past Q&A exchanges with cosine-similarity search.

    Typical usage in a RAG query loop:
        hist         = QueryHistory()
        q_vec        = embed(question)          # (768,) L2-normalised
        past         = hist.search(q_vec)
        ctx_str      = QueryHistory.format_for_context(past)
        # ... generate answer ...
        hist.add(question, retrieval_query, answer, model,
                 top_chapters, top_chunks, q_vec, ch_meta=ch_meta)
    """

    def __init__(self) -> None:
        self._records: list[dict] = []
        self._emb: np.ndarray = np.empty((0, 768), dtype=np.float32)
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if HISTORY_JSON.exists():
            with open(HISTORY_JSON, encoding="utf-8") as f:
                self._records = json.load(f)
        if HISTORY_EMB.exists():
            self._emb = np.load(str(HISTORY_EMB)).astype(np.float32)

    def _save(self) -> None:
        HISTORY_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_JSON, "w", encoding="utf-8") as f:
            json.dump(self._records, f, indent=2, ensure_ascii=False)
        np.save(str(HISTORY_EMB), self._emb)

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(
        self,
        question:        str,
        retrieval_query: str,
        answer:          str,
        model:           str,
        top_chapters:    list[tuple[int, int]],
        top_chunks:      list[dict],
        embedding:       np.ndarray,   # (768,) or (1, 768), L2-normalised
        ch_meta:         dict | None = None,
        user:            str = "Gerry",
    ) -> None:
        """Append a Q&A record to the history and persist to disk."""
        chapters_info: list[dict] = []
        for bn, cn in top_chapters:
            entry: dict = {"book_number": bn, "chapter_number": cn}
            if ch_meta:
                meta = next(
                    (m for m in ch_meta.values()
                     if int(m["book_number"]) == bn and int(m["chapter_number"]) == cn),
                    None,
                )
                if meta:
                    entry["chapter_title"] = meta.get("chapter_title", "")
                    entry["book_title"]    = meta.get("book_title", "")
            chapters_info.append(entry)

        chunks_info = [
            {
                "book_number":    c["book_number"],
                "chapter_number": c["chapter_number"],
                "chunk_page":     c.get("chunk_page", ""),
                "snippet":        c["chunk_text"].replace("\n", " ").strip()[:200],
            }
            for c in top_chunks
        ]

        record: dict = {
            "id":              len(self._records) + 1,
            "timestamp":       datetime.now().isoformat(timespec="seconds"),
            "user":            user,
            "question":        question,
            "retrieval_query": retrieval_query,
            "answer":          answer,
            "model":           model,
            "chapters":        chapters_info,
            "chunks":          chunks_info,
        }
        self._records.append(record)

        vec = np.asarray(embedding, dtype=np.float32).ravel().reshape(1, -1)  # (1, 768)
        self._emb = vec if len(self._emb) == 0 else np.vstack([self._emb, vec])
        self._save()

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        embedding: np.ndarray,
        k: int = MAX_PAST,
        min_sim: float = SIM_THRESHOLD,
    ) -> list[dict]:
        """
        Return up to *k* past records whose question embedding has cosine
        similarity >= *min_sim* with *embedding*.  Embeddings must be
        L2-normalised so that dot-product equals cosine similarity.
        """
        n = min(len(self._records), len(self._emb))
        if n == 0:
            return []

        q    = np.asarray(embedding, dtype=np.float32).ravel()  # (768,)
        sims = self._emb[:n] @ q                                 # (n,) cosine sims
        order = np.argsort(sims)[::-1]

        results: list[dict] = []
        for idx in order:
            if len(results) >= k:
                break
            sim = float(sims[idx])
            if sim < min_sim:
                break
            results.append({**self._records[idx], "_sim": round(sim, 3)})
        return results

    # ── Formatting ────────────────────────────────────────────────────────────

    @staticmethod
    def format_for_context(past: list[dict]) -> str:
        """
        Format a list of past Q&A records as a prompt block for Claude.
        Returns an empty string if *past* is empty.
        """
        if not past:
            return ""
        parts = []
        for rec in past:
            date_str    = rec["timestamp"][:10]
            ans         = rec["answer"]
            ans_snippet = ans[:600].rstrip() + ("…" if len(ans) > 600 else "")
            parts.append(
                f"[Past query — {date_str}]\n"
                f"Q: {rec['question']}\n"
                f"A: {ans_snippet}"
            )
        return (
            "RELEVANT PAST QUERIES FROM THIS LIBRARY:\n\n"
            + "\n\n---\n\n".join(parts)
        )

    def __len__(self) -> int:
        return len(self._records)
