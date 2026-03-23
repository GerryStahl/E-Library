"""
scripts/build_bm25_index.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Build BM25 indices for hybrid retrieval.

Outputs (written to vector_store/):
  chunks_bm25/         — BM25 index over ~17k level-0 chunk texts
    ├─ *.npz / *.json  — bm25s index files
    └─ sidecar.json    — str(doc_id) → {vector_id, book, chapter, text, ...}

  summaries_bm25/      — BM25 index over 335 chapter summaries
    ├─ *.npz / *.json  — bm25s index files
    └─ sidecar.json    — str(doc_id) → {faiss_vid, book, chapter, text, ...}

The sidecar files embed the raw text so query_library.py needs NO cache PKL
at runtime — only the FAISS indexes, BM25 indexes, their sidecars, and the
embedding model.

Usage:
    python scripts/build_bm25_index.py
"""

from __future__ import annotations
import json, sys, time, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bm25s
import Stemmer

from cache.elibrary_cache import ElibraryCache

# ── Paths ─────────────────────────────────────────────────────────────────────
PKL_PATH        = ROOT / "cache" / "elibrary_cache.pkl"
CHUNK_META_PATH = ROOT / "vector_store" / "elibrary_meta.json"
CH_META_PATH    = ROOT / "vector_store" / "chapter_summaries_meta.json"
VS_DIR          = ROOT / "vector_store"

CHUNKS_DIR    = VS_DIR / "chunks_bm25"
SUMMARIES_DIR = VS_DIR / "summaries_bm25"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_and_save(texts: list[str], save_dir: Path, stemmer) -> None:
    """Tokenize *texts*, build a BM25 index, and persist to *save_dir*."""
    save_dir.mkdir(parents=True, exist_ok=True)
    tok = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer,
                         show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tok)
    retriever.save(str(save_dir))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    stemmer = Stemmer.Stemmer("english")

    # ── Load cache ────────────────────────────────────────────────────────────
    print("Loading cache…")
    cache = ElibraryCache.load(str(PKL_PATH))

    # ── Load FAISS chunk meta (for vector_id look-up) ─────────────────────────
    print("Loading elibrary_meta.json…")
    with open(CHUNK_META_PATH) as f:
        chunk_meta = json.load(f)   # str(vid) → {book_number, …, chunk_level}

    # Reverse lookup: (book_number, chapter_number, chunk_index) → vector_id
    # Only level-0 chunks are indexed in BM25; higher levels are dense-only.
    coord_to_vid: dict[tuple[int, int, int], int] = {
        (int(m["book_number"]), int(m["chapter_number"]), int(m["chunk_index"])): int(vs)
        for vs, m in chunk_meta.items()
        if m["chunk_level"] == 0
    }

    # ── 1. Chunk BM25 (level-0 only) ─────────────────────────────────────────
    print("\n[1/2] Building chunk BM25 index (level-0 only)…")

    chunk_texts: list[str] = []
    chunk_sidecar: dict[str, dict] = {}

    for book in cache.books:
        for ch in book.book_chapters:
            level0 = sorted(
                [c for c in ch.chapter_chunks if c.chunk_level == 0],
                key=lambda c: c.chunk_index,
            )
            for chunk in level0:
                coord  = (book.book_number, ch.chapter_number, chunk.chunk_index)
                vid    = coord_to_vid.get(coord)          # None if mismatch
                text   = chunk.chunk_text.strip()
                doc_id = len(chunk_texts)
                chunk_texts.append(text)
                chunk_sidecar[str(doc_id)] = {
                    "vector_id":      vid,
                    "book_number":    book.book_number,
                    "book_title":     book.book_title,
                    "chapter_number": ch.chapter_number,
                    "chapter_title":  ch.chapter_title,
                    "chunk_index":    chunk.chunk_index,
                    "chunk_page":     chunk.chunk_page_number,
                    "chunk_words":    chunk.chunk_word_count,
                    "section":        chunk.chunk_section_heading,
                    "chunk_text":     text,
                }

    print(f"  {len(chunk_texts):,} level-0 chunks collected")
    _build_and_save(chunk_texts, CHUNKS_DIR, stemmer)
    with open(CHUNKS_DIR / "sidecar.json", "w", encoding="utf-8") as f:
        json.dump(chunk_sidecar, f, ensure_ascii=False)
    print(f"  Saved → {CHUNKS_DIR}/")

    # ── 2. Summary BM25 ───────────────────────────────────────────────────────
    print("\n[2/2] Building chapter summary BM25 index…")

    with open(CH_META_PATH) as f:
        ch_meta_map = json.load(f)   # str(faiss_vid) → chapter metadata

    books_by_num = {b.book_number: b for b in cache.books}

    summary_texts: list[str]   = []
    summary_sidecar: dict[str, dict] = {}

    for faiss_vid_str, m in ch_meta_map.items():
        book = books_by_num.get(m["book_number"])
        if book is None:
            continue
        ch = next(
            (c for c in book.book_chapters if c.chapter_number == m["chapter_number"]),
            None,
        )
        if ch is None or not ch.chapter_summaries:
            continue
        summary = next(
            (s for s in ch.chapter_summaries
             if "claude" in s.chapter_summary_author.lower()),
            ch.chapter_summaries[0],
        )
        text = summary.chapter_summary_text.strip()
        if not text:
            continue

        doc_id = len(summary_texts)
        summary_texts.append(text)
        summary_sidecar[str(doc_id)] = {
            "faiss_vid":        int(faiss_vid_str),
            "book_number":      m["book_number"],
            "book_title":       m["book_title"],
            "book_keywords":    m["book_keywords"],
            "chapter_number":   m["chapter_number"],
            "chapter_title":    m["chapter_title"],
            "chapter_author":   m["chapter_author"],
            "chapter_keywords": m["chapter_keywords"],
            "summary_text":     text,
        }

    print(f"  {len(summary_texts)} chapter summaries collected")
    _build_and_save(summary_texts, SUMMARIES_DIR, stemmer)
    with open(SUMMARIES_DIR / "sidecar.json", "w", encoding="utf-8") as f:
        json.dump(summary_sidecar, f, ensure_ascii=False)
    print(f"  Saved → {SUMMARIES_DIR}/")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"  Chunk BM25:   {len(chunk_texts):,} docs  → {CHUNKS_DIR.name}/")
    print(f"  Summary BM25: {len(summary_texts)} docs  → {SUMMARIES_DIR.name}/")
    print(f"  Total time:   {elapsed:.1f}s")
    print(f"{'='*55}")
    print("Done.")


if __name__ == "__main__":
    main()
