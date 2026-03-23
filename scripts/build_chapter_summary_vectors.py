"""
scripts/build_chapter_summary_vectors.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Embed the Claude summary text for every chapter (337 total) using
intfloat/e5-base-v2 and store in a FAISS IndexFlatIP index.

Outputs (written to vector_store/):
  chapter_summaries.faiss      — 337-vector cosine index
  chapter_summaries_meta.json  — vector_id → chapter metadata
  chapter_summaries_build.json — build stats

Query-time convention (same as chunk index):
  Passages → "passage: " prefix + L2-normalise
  Queries  → "query: "   prefix + L2-normalise
  Search   → inner-product (= cosine on normalised vecs)

Usage:
    python scripts/build_chapter_summary_vectors.py
"""

from __future__ import annotations
import json, sys, time, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from cache.elibrary_cache import ElibraryCache

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "intfloat/e5-base-v2"
PASSAGE_PFX = "passage: "
DIM         = 768

PKL_PATH  = ROOT / "cache" / "elibrary_cache.pkl"
VS_DIR    = ROOT / "vector_store"
VS_DIR.mkdir(exist_ok=True)

FAISS_PATH = VS_DIR / "chapter_summaries.faiss"
META_PATH  = VS_DIR / "chapter_summaries_meta.json"
BUILD_PATH = VS_DIR / "chapter_summaries_build.json"


def main():
    t0 = time.time()

    # 1. Load cache
    print("Loading cache…")
    cache = ElibraryCache.load(str(PKL_PATH))

    # 2. Collect one summary per chapter
    print("Collecting chapter summaries…")
    records = []  # list of (summary_text, book, chapter)
    skipped = []

    for book in cache.books:
        for ch in book.book_chapters:
            if not ch.chapter_summaries:
                skipped.append((book.book_number, ch.chapter_number,
                                ch.chapter_title))
                continue
            # Use Claude summary (only one exists); fall back to first if needed
            summary = next(
                (s for s in ch.chapter_summaries
                 if "claude" in s.chapter_summary_author.lower()),
                ch.chapter_summaries[0]
            )
            text = summary.chapter_summary_text.strip()
            if not text:
                skipped.append((book.book_number, ch.chapter_number,
                                ch.chapter_title))
                continue
            records.append((text, book, ch, summary))

    n = len(records)
    print(f"  {n} chapters with summaries"
          + (f"  ({len(skipped)} skipped — no text)" if skipped else ""))
    if skipped:
        for bn, cn, t in skipped:
            print(f"    Book {bn} Ch {cn}: {t[:50]}")

    # 3. Load model (may already be cached locally)
    print(f"\nLoading model: {MODEL_NAME}…")
    model = SentenceTransformer(MODEL_NAME)
    print("  Model ready.")

    # 4. Embed all summaries in one shot (only 337 — fits easily)
    print(f"\nEmbedding {n} chapter summaries…")
    texts = [PASSAGE_PFX + text for text, *_ in records]
    matrix = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=64,
    ).astype(np.float32)
    print(f"  Done. Matrix shape: {matrix.shape}")

    # 5. Build FAISS index
    print("\nBuilding FAISS index…")
    index = faiss.IndexFlatIP(DIM)
    index.add(matrix)
    faiss.write_index(index, str(FAISS_PATH))
    print(f"  Index saved → {FAISS_PATH}  ({index.ntotal} vectors)")

    # 6. Metadata sidecar
    print("Writing metadata sidecar…")
    meta = {}
    for vid, (text, book, ch, summary) in enumerate(records):
        meta[vid] = {
            "book_number":          book.book_number,
            "book_title":           book.book_title,
            "book_keywords":        book.book_keywords,
            "chapter_number":       ch.chapter_number,
            "chapter_title":        ch.chapter_title,
            "chapter_author":       ch.chapter_author,
            "chapter_keywords":     ch.chapter_keywords,
            "chapter_pages":        ch.chapter_number_of_pages,
            "chapter_words":        ch.chapter_number_of_words,
            "summary_author":       summary.chapter_summary_author,
            "summary_word_count":   summary.chapter_summary_number_of_words,
        }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  Metadata saved → {META_PATH}")

    # 7. Build stats
    elapsed = time.time() - t0
    build_info = {
        "model":          MODEL_NAME,
        "dimension":      DIM,
        "index_type":     "IndexFlatIP (cosine via L2-normalised vectors)",
        "passage_prefix": PASSAGE_PFX,
        "query_prefix":   "query: ",
        "total_vectors":  n,
        "skipped":        len(skipped),
        "build_seconds":  round(elapsed, 1),
        "faiss_path":     str(FAISS_PATH),
        "meta_path":      str(META_PATH),
    }
    with open(BUILD_PATH, "w") as f:
        json.dump(build_info, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Vectors:    {n}  (one per chapter summary)")
    print(f"  Dimension:  {DIM}")
    print(f"  Model:      {MODEL_NAME}")
    print(f"  Index:      IndexFlatIP (cosine)")
    print(f"  FAISS:      {FAISS_PATH}")
    print(f"  Metadata:   {META_PATH}")
    print(f"  Build time: {elapsed:.1f}s")
    print(f"{'='*55}")
    print("Done.")


if __name__ == "__main__":
    main()
