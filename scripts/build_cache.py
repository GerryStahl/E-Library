"""
build_cache.py — Reads chapters.csv and creates elibrary_cache.pkl.

Run from the workspace root:

    python scripts/build_cache.py

Or supply custom paths:

    python scripts/build_cache.py \\
        --csv    documents/chapters.csv \\
        --output elibrary_cache.pkl

What it does
------------
1. Reads every row of chapters.csv.
2. Rows where chapter_number == 0 define a Book (author, reference, kind, …).
3. Rows where chapter_number  > 0 define a Chapter within that book.
4. Assembles an ElibraryCache and serialises it with pickle.

CSV columns (positional, 0-indexed):
  0  book number
  1  book name          (PDF filename)
  2  book title
  3  book kind
  4  chapter number
  5  book author / chapter author
  6  chapter title
  7  book reference / chapter reference

Author : Gerry Stahl
Created: March 2026
"""

import argparse
import csv
import sys
from pathlib import Path

# ── allow running from either workspace root or scripts/ sub-directory ──────
_HERE = Path(__file__).resolve().parent          # scripts/
_ROOT      = _HERE.parent                        # elibrary/
_CACHE_DIR = _ROOT / "cache"                     # elibrary/cache/
if str(_CACHE_DIR) not in sys.path:
    sys.path.insert(0, str(_CACHE_DIR))

from elibrary_cache import (                     # noqa: E402  (after sys.path fix)
    Book,
    BookSummary,
    Chapter,
    ChapterSummary,
    ElibraryCache,
)


# ---------------------------------------------------------------------------
# CSV column indices
# ---------------------------------------------------------------------------
COL_BOOK_NUMBER    = 0
COL_BOOK_NAME      = 1
COL_BOOK_TITLE     = 2
COL_BOOK_KIND      = 3
COL_CHAPTER_NUMBER = 4
COL_AUTHOR         = 5   # book author when ch==0, chapter author when ch>0
COL_CHAPTER_TITLE  = 6
COL_REFERENCE      = 7   # book reference when ch==0, chapter reference when ch>0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(value: str) -> str:
    """Strip whitespace; return empty string for missing values."""
    return (value or "").strip()


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(_clean(value))
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_cache(csv_path: Path) -> ElibraryCache:
    """
    Parse *csv_path* and return a fully populated ElibraryCache.

    The CSV must have a header row followed by data rows.
    Books are identified by rows where chapter_number == 0.
    """
    cache = ElibraryCache()
    books_by_number: dict[int, Book] = {}   # quick lookup while building

    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)

        # ── skip header row ─────────────────────────────────────────────
        header = next(reader, None)
        if header is None:
            raise ValueError(f"CSV file is empty: {csv_path}")

        for line_num, row in enumerate(reader, start=2):   # 1-based, header=1
            # Pad short rows to avoid index errors
            while len(row) < 8:
                row.append("")

            book_number    = _safe_int(row[COL_BOOK_NUMBER])
            chapter_number = _safe_int(row[COL_CHAPTER_NUMBER])
            author         = _clean(row[COL_AUTHOR])
            reference      = _clean(row[COL_REFERENCE])

            if book_number == 0:
                print(f"  [line {line_num}] WARNING: book_number is 0, skipping.")
                continue

            # ── Book-level row (chapter 0) ───────────────────────────────
            if chapter_number == 0:
                book = Book(
                    book_number    = book_number,
                    book_name      = _clean(row[COL_BOOK_NAME]),
                    book_title     = _clean(row[COL_BOOK_TITLE]),
                    book_author    = author,
                    book_reference = reference,
                    book_kind      = _clean(row[COL_BOOK_KIND]),
                )
                books_by_number[book_number] = book
                cache.add_book(book)

                print(
                    f"  Book {book.book_number:>2}: {book.book_name:<25}  "
                    f"({book.book_kind})  '{book.book_title}'"
                )

            # ── Chapter-level row ────────────────────────────────────────
            else:
                # Find (or lazily create) the parent book
                if book_number not in books_by_number:
                    # Rare: chapter row appeared before the book-level row
                    print(
                        f"  [line {line_num}] WARNING: chapter before book "
                        f"(book {book_number}); creating placeholder book."
                    )
                    book = Book(book_number=book_number)
                    books_by_number[book_number] = book
                    cache.add_book(book)
                else:
                    book = books_by_number[book_number]

                chapter = Chapter(
                    chapter_number    = chapter_number,
                    chapter_title     = _clean(row[COL_CHAPTER_TITLE]),
                    chapter_author    = author,
                    chapter_reference = reference,
                )
                book.add_chapter(chapter)

    return cache


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build elibrary_cache.pkl from chapters.csv."
    )
    parser.add_argument(
        "--csv",
        default=str(_ROOT / "documents" / "chapters.csv"),
        help="Path to chapters.csv  (default: documents/chapters.csv)",
    )
    parser.add_argument(
        "--output",
        default=str(_CACHE_DIR / "elibrary_cache.pkl"),
        help="Output path for the pickle file  (default: cache/elibrary_cache.pkl)",
    )
    args = parser.parse_args()

    csv_path    = Path(args.csv)
    output_path = Path(args.output)

    print(f"\nBuilding e-library cache")
    print(f"  Source : {csv_path}")
    print(f"  Output : {output_path}\n")

    if not csv_path.exists():
        sys.exit(f"ERROR: CSV file not found: {csv_path}")

    cache = build_cache(csv_path)

    saved = cache.save(str(output_path))
    print(f"\nCache saved → {saved}\n")
    print(cache.summary_stats())

    # Quick sanity check
    total_ch = cache.total_chapters
    if total_ch == 0:
        print("\nWARNING: no chapters were loaded — check the CSV format.")
    else:
        print(f"\n✓  {cache.total_books} books, {total_ch} chapters loaded successfully.")


if __name__ == "__main__":
    main()
