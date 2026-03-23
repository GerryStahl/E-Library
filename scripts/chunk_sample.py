"""
scripts/chunk_sample.py
~~~~~~~~~~~~~~~~~~~~~~~~
Chunk a single chapter using SemanticHierarchicalChunker and display
chunk counts and character lengths at every merge level.

Usage:
    python scripts/chunk_sample.py [book_number] [chapter_number]
    python scripts/chunk_sample.py          # defaults: book 1, chapter 1

Embedding model: all-MiniLM-L6-v2 (384-dim, already installed).
  → For the production vector store, switch to intfloat/e5-base-v2
    (768-dim) or intfloat/e5-large-instruct with "passage: " prefix.

The script stores the resulting Chunk objects (all levels) back into
the Chapter and saves the PKL so you can inspect them later.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from cache.elibrary_cache import Chunk, ElibraryCache
from chunkers.semantic_chunker import TextChunker
from chunkers.semantic_hierarchical_chunker import (
    SemanticHierarchicalChunker,
    SemanticHierarchicalChunk,
)
from embedders.embedder import Embedder

# ── Parameters ────────────────────────────────────────────────────────────────
BOOK_NUMBER    = int(sys.argv[1]) if len(sys.argv) > 1 else 1
CHAPTER_NUMBER = int(sys.argv[2]) if len(sys.argv) > 2 else 1

BASE_CHUNK_SIZE = 1_000   # characters
BASE_OVERLAP    = 200     # characters

SIM_THRESHOLD        = 0.70   # level-1 merge threshold
MIN_CHUNKS_PER_LEVEL = 3      # stop merging when ≤ this many chunks remain
MAX_LEVELS           = 5

PKL_PATH = ROOT / "cache" / "elibrary_cache.pkl"


# ── Helpers ───────────────────────────────────────────────────────────────────

def approx_page(offset: int, text_len: int, n_pages: int) -> int:
    """Estimate page number (1-based, relative to chapter start)."""
    if text_len == 0 or n_pages == 0:
        return 0
    return 1 + int(offset / text_len * n_pages)


def hier_chunk_to_cache_chunk(
    hc: SemanticHierarchicalChunk,
    idx: int,
    level: int,
    book_number: int,
    chapter_number: int,
    chapter_text: str,
    n_pages: int,
) -> Chunk:
    """Convert a SemanticHierarchicalChunk to a cache Chunk dataclass."""
    start  = hc.metadata.get("char_start", 0)
    end    = hc.metadata.get("char_end",   start + len(hc.text))
    return Chunk(
        chunk_index           = idx,
        chunk_book_number     = book_number,
        chunk_chapter_number  = chapter_number,
        chunk_page_number     = approx_page(start, len(chapter_text), n_pages),
        chunk_text            = hc.text,
        chunk_section_heading = hc.metadata.get("section_heading", ""),
        chunk_level           = level,
        chunk_start_offset    = start,
        chunk_end_offset      = end,
        chunk_word_count      = len(hc.text.split()),
        chunk_token_count     = 0,   # fill with tokenizer later
    )


def print_level_table(hierarchy: list[list[SemanticHierarchicalChunk]]) -> None:
    """Print a table showing chunk stats at every hierarchy level."""
    header = (
        f"  {'Level':<7} {'Chunks':>6}  {'Avg chars':>9}  "
        f"{'Min chars':>9}  {'Max chars':>9}  {'Avg words':>9}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for lvl, chunks in enumerate(hierarchy):
        lengths = [len(c.text) for c in chunks]
        words   = [len(c.text.split()) for c in chunks]
        label   = f"base" if lvl == 0 else f"merge-{lvl}"
        print(
            f"  {label:<7} {len(chunks):>6}  "
            f"{int(np.mean(lengths)):>9,}  "
            f"{min(lengths):>9,}  "
            f"{max(lengths):>9,}  "
            f"{int(np.mean(words)):>9,}"
        )
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Load cache
    print(f"Loading cache…")
    cache   = ElibraryCache.load(str(PKL_PATH))
    book    = cache.get_book(BOOK_NUMBER)
    if book is None:
        sys.exit(f"Book {BOOK_NUMBER} not found.")
    chapter = book.get_chapter(CHAPTER_NUMBER)
    if chapter is None:
        sys.exit(f"Chapter {CHAPTER_NUMBER} not found in book {BOOK_NUMBER}.")

    print(f"\nBook {book.book_number}: {book.book_title}")
    print(f"Chapter {chapter.chapter_number}: {chapter.chapter_title}")
    print(f"  {chapter.chapter_number_of_words:,} words  |  "
          f"{chapter.chapter_number_of_symbols:,} chars  |  "
          f"{chapter.chapter_number_of_pages} pages\n")

    # 2. Base chunks
    print(f"Step 1 — Base chunking  "
          f"(chunk_size={BASE_CHUNK_SIZE}, overlap={BASE_OVERLAP})")
    chunker     = TextChunker(chunk_size=BASE_CHUNK_SIZE, overlap=BASE_OVERLAP)
    base_dicts  = chunker.chunk_chapter(
        text           = chapter.chapter_text,
        book_number    = BOOK_NUMBER,
        chapter_number = CHAPTER_NUMBER,
        extra_metadata = {
            "book_title":    book.book_title,
            "chapter_title": chapter.chapter_title,
        },
    )
    print(f"  {len(base_dicts)} base chunks\n")

    # 3. Embed base chunks
    print(f"Step 2 — Embedding  (all-MiniLM-L6-v2, 384-dim)…")
    embedder   = Embedder()
    texts      = [d["text"] for d in base_dicts]
    embeddings = embedder.embed_texts(texts)
    print(f"  {len(embeddings)} vectors, dim={len(embeddings[0])}\n")

    # 4. Build SemanticHierarchicalChunk base level
    # Add char_start / char_end into the metadata dict so they survive into Chunk
    for d in base_dicts:
        d["char_start"] = d["metadata"]["char_start"]
        d["char_end"]   = d["metadata"]["char_end"]

    hier_chunker = SemanticHierarchicalChunker(
        embedder             = embedder,
        similarity_threshold = SIM_THRESHOLD,
        min_chunks_per_level = MIN_CHUNKS_PER_LEVEL,
        max_text_length      = BASE_CHUNK_SIZE * 20,   # generous ceiling for merges
    )
    base_hier = hier_chunker.create_base_chunks(base_dicts, embeddings)

    # Patch char_start / char_end into each base chunk's metadata
    for hc, d in zip(base_hier, base_dicts):
        hc.metadata["char_start"] = d["char_start"]
        hc.metadata["char_end"]   = d["char_end"]

    # 5. Build hierarchy (iterative merging)
    print(f"Step 3 — Hierarchical merging  "
          f"(threshold={SIM_THRESHOLD}, max_levels={MAX_LEVELS})\n")
    hierarchy = hier_chunker.build_hierarchy(base_hier, max_levels=MAX_LEVELS)

    print_level_table(hierarchy)

    # 6. Convert all levels → Chunk objects and store in chapter
    print("Step 4 — Storing chunks in cache…")
    chapter.chapter_chunks = []
    for level, level_chunks in enumerate(hierarchy):
        for idx, hc in enumerate(level_chunks):
            chunk = hier_chunk_to_cache_chunk(
                hc             = hc,
                idx            = idx,
                level          = level,
                book_number    = BOOK_NUMBER,
                chapter_number = CHAPTER_NUMBER,
                chapter_text   = chapter.chapter_text,
                n_pages        = chapter.chapter_number_of_pages,
            )
            chapter.chapter_chunks.append(chunk)

    total_stored = len(chapter.chapter_chunks)
    print(f"  {total_stored} Chunk objects stored across {len(hierarchy)} levels")

    cache.save(str(PKL_PATH))
    print(f"  PKL saved → {PKL_PATH}\n")

    # 7. Sample: show first 3 chunks at each level
    print("Sample — first 2 chunks per level:")
    print("=" * 70)
    for level, level_chunks in enumerate(hierarchy):
        label = "base" if level == 0 else f"merge-{level}"
        print(f"\n  [{label}]  {len(level_chunks)} chunks total")
        for hc in level_chunks[:2]:
            preview = hc.text[:120].replace("\n", " ")
            print(f"    {len(hc.text):>6,} chars | {preview!r}…")


if __name__ == "__main__":
    main()
