"""
scripts/build_vector_store.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Embed all Chunk objects from the PKL using intfloat/e5-base-v2 (768-dim)
and store them in a FAISS flat index.

Outputs (written to vector_store/):
  elibrary.faiss         — FAISS IndexFlatIP index (inner-product / cosine)
  elibrary_meta.json     — sidecar: vector_id → chunk metadata
  elibrary_build.json    — build statistics & parameters

Usage:
    python scripts/build_vector_store.py

Query-time convention (MUST match what the RAG pipeline does):
  Chunks  → prepend "passage: " before embedding
  Queries → prepend "query: "   before embedding
  After encoding, L2-normalise; then use inner-product search = cosine sim.
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
MODEL_NAME   = "intfloat/e5-base-v2"
DIM          = 768
PASSAGE_PFX  = "passage: "
BATCH_SIZE   = 256          # chunks per encode() call
LEVELS       = None         # None = all levels; or e.g. {0} for base only

PKL_PATH = ROOT / "cache" / "elibrary_cache.pkl"
VS_DIR   = ROOT / "vector_store"
VS_DIR.mkdir(exist_ok=True)

FAISS_PATH = VS_DIR / "elibrary.faiss"
META_PATH  = VS_DIR / "elibrary_meta.json"
BUILD_PATH = VS_DIR / "elibrary_build.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

def batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # 1. Load cache
    print("Loading cache…")
    cache = ElibraryCache.load(str(PKL_PATH))
    print(f"  {cache.total_books} books / {cache.total_chapters} chapters")

    # 2. Collect all Chunk objects
    print("Collecting chunks…")
    records = []   # list of (chunk_obj, book_title, chapter_title)
    for book in cache.books:
        for chapter in book.book_chapters:
            for chunk in chapter.chapter_chunks:
                if LEVELS is not None and chunk.chunk_level not in LEVELS:
                    continue
                records.append((chunk, book.book_title, chapter.chapter_title))

    n = len(records)
    print(f"  {n:,} chunks to embed (levels: {'all' if LEVELS is None else sorted(LEVELS)})")

    # 3. Load model
    print(f"\nLoading model: {MODEL_NAME}…")
    model = SentenceTransformer(MODEL_NAME)
    print("  Model ready.")

    # 4. Embed in batches
    print(f"\nEmbedding {n:,} chunks in batches of {BATCH_SIZE}…")
    texts = [PASSAGE_PFX + r[0].chunk_text for r in records]

    all_vecs = []
    for i, batch in enumerate(batched(texts, BATCH_SIZE)):
        vecs = model.encode(batch, normalize_embeddings=True,
                            show_progress_bar=False, convert_to_numpy=True)
        all_vecs.append(vecs.astype(np.float32))
        done = min((i + 1) * BATCH_SIZE, n)
        print(f"  {done:>6,}/{n:,}  [{time.time()-t0:.0f}s]", end="\r")

    print(f"  {n:,}/{n:,} — embedding done.           ")
    matrix = np.vstack(all_vecs)   # shape (n, DIM)

    # 5. Build FAISS index (inner product on L2-normalised vecs = cosine)
    print("\nBuilding FAISS index…")
    index = faiss.IndexFlatIP(DIM)
    index.add(matrix)
    faiss.write_index(index, str(FAISS_PATH))
    print(f"  Index saved → {FAISS_PATH}  ({index.ntotal:,} vectors)")

    # 6. Write metadata sidecar
    print("Writing metadata sidecar…")
    meta = {}
    for vid, (chunk, book_title, chapter_title) in enumerate(records):
        meta[vid] = {
            "book_number":     chunk.chunk_book_number,
            "chapter_number":  chunk.chunk_chapter_number,
            "chunk_index":     chunk.chunk_index,
            "chunk_level":     chunk.chunk_level,
            "chunk_page":      chunk.chunk_page_number,
            "chunk_start":     chunk.chunk_start_offset,
            "chunk_end":       chunk.chunk_end_offset,
            "chunk_words":     chunk.chunk_word_count,
            "book_title":      book_title,
            "chapter_title":   chapter_title,
        }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  Metadata saved → {META_PATH}")

    # 7. Build stats
    elapsed = time.time() - t0
    level_counts = {}
    for chunk, _, _ in records:
        lv = chunk.chunk_level
        level_counts[lv] = level_counts.get(lv, 0) + 1

    build_info = {
        "model":          MODEL_NAME,
        "dimension":      DIM,
        "index_type":     "IndexFlatIP (cosine via L2-normalised vectors)",
        "passage_prefix": PASSAGE_PFX,
        "query_prefix":   "query: ",
        "total_vectors":  n,
        "levels_included": "all" if LEVELS is None else sorted(LEVELS),
        "vectors_per_level": {str(k): v for k, v in sorted(level_counts.items())},
        "build_seconds":  round(elapsed, 1),
        "faiss_path":     str(FAISS_PATH),
        "meta_path":      str(META_PATH),
    }
    with open(BUILD_PATH, "w") as f:
        json.dump(build_info, f, indent=2)

    # 8. Summary
    total_elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Vectors:     {n:,}")
    print(f"  Dimension:   {DIM}")
    print(f"  Model:       {MODEL_NAME}")
    print(f"  Index:       IndexFlatIP (cosine)")
    print(f"  Levels:")
    for lv, cnt in sorted(level_counts.items()):
        label = "base" if lv == 0 else f"merge-{lv}"
        print(f"    {label:<10} {cnt:>7,} vectors")
    print(f"  FAISS:       {FAISS_PATH}")
    print(f"  Metadata:    {META_PATH}")
    print(f"  Build time:  {total_elapsed/60:.1f} min")
    print(f"{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
