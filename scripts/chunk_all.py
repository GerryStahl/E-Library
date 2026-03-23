"""
scripts/chunk_all.py
~~~~~~~~~~~~~~~~~~~~~
Clear any existing chapter_chunks from the cache, then chunk every
chapter in every book using SemanticHierarchicalChunker.

Usage:
    python scripts/chunk_all.py

Parameters are identical to chunk_range.py / chunk_sample.py.
Progress is printed book-by-book; a grand summary is shown at the end.
PKL is saved once at the very end.
"""

from __future__ import annotations
import sys, time, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from cache.elibrary_cache import Chunk, ElibraryCache
from chunkers.semantic_chunker import TextChunker
from chunkers.semantic_hierarchical_chunker import SemanticHierarchicalChunker
from embedders.embedder import Embedder

# ── Parameters ────────────────────────────────────────────────────────────────
BASE_CHUNK_SIZE = 1_000
BASE_OVERLAP    = 200
SIM_THRESHOLD   = 0.70
MIN_CHUNKS      = 3
MAX_LEVELS      = 5

PKL_PATH = ROOT / "cache" / "elibrary_cache.pkl"


# ── Helpers ───────────────────────────────────────────────────────────────────

def approx_page(offset: int, text_len: int, n_pages: int) -> int:
    if text_len == 0 or n_pages == 0:
        return 0
    return 1 + int(offset / text_len * n_pages)


def chunk_chapter(book, chapter, chunker, embedder, hier_chunker):
    """Chunk one chapter; returns (hierarchy, chunk_objects) or (None, [])."""
    text = chapter.chapter_text
    if not text or not text.strip():
        return None, []

    # Base chunks
    base_dicts = chunker.chunk_chapter(
        text=text,
        book_number=book.book_number,
        chapter_number=chapter.chapter_number,
        extra_metadata={"book_title": book.book_title,
                        "chapter_title": chapter.chapter_title},
    )
    if not base_dicts:
        return None, []

    # Embed
    texts      = [d["text"] for d in base_dicts]
    embeddings = embedder.embed_texts(texts)

    # Tag offsets onto dicts for easy access
    for d in base_dicts:
        d["char_start"] = d["metadata"]["char_start"]
        d["char_end"]   = d["metadata"]["char_end"]

    # Build hierarchy
    base_hier = hier_chunker.create_base_chunks(base_dicts, embeddings)
    for hc, d in zip(base_hier, base_dicts):
        hc.metadata["char_start"] = d["char_start"]
        hc.metadata["char_end"]   = d["char_end"]

    hierarchy = hier_chunker.build_hierarchy(base_hier, max_levels=MAX_LEVELS)

    # Convert to Chunk dataclass objects
    chapter.chapter_chunks = []
    for level, level_chunks in enumerate(hierarchy):
        for idx, hc in enumerate(level_chunks):
            start = hc.metadata.get("char_start", 0)
            end   = hc.metadata.get("char_end", start + len(hc.text))
            chapter.chapter_chunks.append(Chunk(
                chunk_index           = idx,
                chunk_book_number     = book.book_number,
                chunk_chapter_number  = chapter.chapter_number,
                chunk_page_number     = approx_page(start, len(text),
                                                    chapter.chapter_number_of_pages),
                chunk_text            = hc.text,
                chunk_section_heading = hc.metadata.get("section_heading", ""),
                chunk_level           = level,
                chunk_start_offset    = start,
                chunk_end_offset      = end,
                chunk_word_count      = len(hc.text.split()),
                chunk_token_count     = 0,
            ))

    return hierarchy, chapter.chapter_chunks


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    # 1. Load cache
    print("Loading cache…")
    cache = ElibraryCache.load(str(PKL_PATH))

    # 2. Clear ALL existing chunks
    total_cleared = 0
    for book in cache.books:
        for chapter in book.book_chapters:
            if chapter.chapter_chunks:
                total_cleared += len(chapter.chapter_chunks)
                chapter.chapter_chunks = []
    print(f"Cleared {total_cleared:,} existing Chunk objects from cache.\n")

    # 3. Initialise shared tools (load embedder model once)
    print("Loading embedding model…")
    chunker      = TextChunker(chunk_size=BASE_CHUNK_SIZE, overlap=BASE_OVERLAP)
    embedder     = Embedder()
    hier_chunker = SemanticHierarchicalChunker(
        embedder             = embedder,
        similarity_threshold = SIM_THRESHOLD,
        min_chunks_per_level = MIN_CHUNKS,
        max_text_length      = BASE_CHUNK_SIZE * 20,
    )
    print("Model ready.\n")

    # 4. Chunk every book / chapter
    grand_total_chunks   = 0
    grand_total_chapters = 0
    skipped              = 0
    book_summaries       = []   # (book_number, title, n_chapters, n_chunks, secs)

    for book in cache.books:
        chapters = book.content_chapters   # skips front-matter / TOC chapters
        if not chapters:
            continue

        b_start      = time.time()
        b_chunks     = 0
        b_chunked    = 0

        print(f"Book {book.book_number:>2}: {book.book_title[:60]}")

        for chapter in chapters:
            t0 = time.time()
            hierarchy, stored = chunk_chapter(
                book, chapter, chunker, embedder, hier_chunker
            )
            elapsed = time.time() - t0

            if hierarchy is None:
                skipped += 1
                print(f"  Ch {chapter.chapter_number:>3}: (no text — skipped)")
                continue

            base_count = len(hierarchy[0]) if hierarchy else 0
            n_levels   = len(hierarchy)
            n_stored   = len(stored)
            b_chunks  += n_stored
            b_chunked += 1
            grand_total_chapters += 1
            grand_total_chunks   += n_stored

            print(f"  Ch {chapter.chapter_number:>3}: "
                  f"{chapter.chapter_number_of_words:>6,}w  "
                  f"base={base_count:>4}  levels={n_levels}  "
                  f"stored={n_stored:>5}  [{elapsed:.1f}s]")

        b_elapsed = time.time() - b_start
        book_summaries.append((book.book_number, book.book_title,
                                b_chunked, b_chunks, b_elapsed))
        print(f"  → Book {book.book_number} done: "
              f"{b_chunked} chapters, {b_chunks:,} chunks  [{b_elapsed:.1f}s]\n")

    # 5. Save PKL
    print("Saving PKL…")
    cache.save(str(PKL_PATH))
    total_elapsed = time.time() - t_start
    print(f"PKL saved → {PKL_PATH}\n")

    # 6. Grand summary table
    print("=" * 72)
    print(f"  {'Book':>4}  {'Title':<40}  {'Chs':>4}  {'Chunks':>7}  {'Time':>6}")
    print(f"  {'-'*4}  {'-'*40}  {'-'*4}  {'-'*7}  {'-'*6}")
    for bn, title, nc, nch, elapsed in book_summaries:
        print(f"  {bn:>4}  {title[:40]:<40}  {nc:>4}  {nch:>7,}  {elapsed:>5.0f}s")
    print(f"  {'':>4}  {'TOTAL':<40}  "
          f"{grand_total_chapters:>4}  {grand_total_chunks:>7,}  "
          f"{total_elapsed:>5.0f}s")
    print("=" * 72)
    if skipped:
        print(f"\n  ({skipped} chapters skipped — no text)")
    print(f"\nAll done in {total_elapsed/60:.1f} min.")


if __name__ == "__main__":
    main()
