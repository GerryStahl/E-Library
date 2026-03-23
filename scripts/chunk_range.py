"""
scripts/chunk_range.py
~~~~~~~~~~~~~~~~~~~~~~~
Chunk a range of chapters in one book and print a combined summary table.

Usage:
    python scripts/chunk_range.py <book> <ch_from> <ch_to>
    python scripts/chunk_range.py 11 2 11
"""

from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from cache.elibrary_cache import Chunk, ElibraryCache
from chunkers.semantic_chunker import TextChunker
from chunkers.semantic_hierarchical_chunker import SemanticHierarchicalChunker
from embedders.embedder import Embedder

BOOK_NUMBER  = int(sys.argv[1]) if len(sys.argv) > 1 else 11
CH_FROM      = int(sys.argv[2]) if len(sys.argv) > 2 else 2
CH_TO        = int(sys.argv[3]) if len(sys.argv) > 3 else 11

BASE_CHUNK_SIZE = 1_000
BASE_OVERLAP    = 200
SIM_THRESHOLD   = 0.70
MIN_CHUNKS      = 3
MAX_LEVELS      = 5

PKL_PATH = ROOT / "cache" / "elibrary_cache.pkl"


def approx_page(offset, text_len, n_pages):
    if text_len == 0 or n_pages == 0:
        return 0
    return 1 + int(offset / text_len * n_pages)


def run_chapter(book, chapter, chunker, embedder, hier_chunker):
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

    # Tag offsets
    for d in base_dicts:
        d["char_start"] = d["metadata"]["char_start"]
        d["char_end"]   = d["metadata"]["char_end"]

    # Build hierarchy
    base_hier = hier_chunker.create_base_chunks(base_dicts, embeddings)
    for hc, d in zip(base_hier, base_dicts):
        hc.metadata["char_start"] = d["char_start"]
        hc.metadata["char_end"]   = d["char_end"]

    hierarchy = hier_chunker.build_hierarchy(base_hier, max_levels=MAX_LEVELS)

    # Store Chunk objects
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


def fmt_row(label, chunks):
    lengths = [len(c.text) for c in chunks]
    words   = [len(c.text.split()) for c in chunks]
    return (f"  {label:<10} {len(chunks):>6}  "
            f"{int(np.mean(lengths)):>9,}  "
            f"{min(lengths):>9,}  "
            f"{max(lengths):>9,}  "
            f"{int(np.mean(words)):>9,}")


def main():
    print(f"Loading cache…")
    cache = ElibraryCache.load(str(PKL_PATH))
    book  = cache.get_book(BOOK_NUMBER)
    if book is None:
        sys.exit(f"Book {BOOK_NUMBER} not found.")

    print(f"\nBook {book.book_number}: {book.book_title}")
    print(f"Chunking chapters {CH_FROM}–{CH_TO}\n")

    chunker      = TextChunker(chunk_size=BASE_CHUNK_SIZE, overlap=BASE_OVERLAP)
    embedder     = Embedder()
    hier_chunker = SemanticHierarchicalChunker(
        embedder             = embedder,
        similarity_threshold = SIM_THRESHOLD,
        min_chunks_per_level = MIN_CHUNKS,
        max_text_length      = BASE_CHUNK_SIZE * 20,
    )

    chapters = [ch for ch in book.content_chapters
                if CH_FROM <= ch.chapter_number <= CH_TO]

    if not chapters:
        sys.exit(f"No chapters found in range {CH_FROM}–{CH_TO}.")

    col_w = 69
    hdr = (f"  {'Level':<10} {'Chunks':>6}  {'Avg chars':>9}  "
           f"{'Min chars':>9}  {'Max chars':>9}  {'Avg words':>9}")

    for chapter in chapters:
        print(f"{'='*col_w}")
        print(f"  Ch {chapter.chapter_number}: {chapter.chapter_title}")
        print(f"  {chapter.chapter_number_of_words:,} words | "
              f"{chapter.chapter_number_of_pages} pages")
        print(f"{'='*col_w}")
        print(hdr)
        print("  " + "-" * (col_w - 2))

        hierarchy, stored = run_chapter(book, chapter, chunker, embedder, hier_chunker)
        if hierarchy is None:
            print("  (no text)")
            continue

        for lvl, lchunks in enumerate(hierarchy):
            label = "base" if lvl == 0 else f"merge-{lvl}"
            print(fmt_row(label, lchunks))

        print(f"\n  → {len(stored)} Chunk objects stored "
              f"across {len(hierarchy)} levels\n")

    cache.save(str(PKL_PATH))
    print(f"PKL saved → {PKL_PATH}")


if __name__ == "__main__":
    main()
