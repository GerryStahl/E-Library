"""
migrate_cache_fields.py — One-time migration of elibrary_cache.pkl to use
the new prefixed field names defined in cache/elibrary_cache.py.

When loaded, the updated elibrary_cache.py triggers __setstate__ on every
Book, Chapter, BookSummary and ChapterSummary object, automatically mapping
old field names (number, name, title, …) to the new prefixed names
(book_number, book_name, book_title, …).  This script simply loads and
re-saves the cache so the pkl is written with the new names.

Run from the workspace root:

    python3 scripts/migrate_cache_fields.py
"""

import sys
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
sys.path.insert(0, str(CACHE_DIR))

from elibrary_cache import ElibraryCache

PKL = CACHE_DIR / "elibrary_cache.pkl"

print(f"Loading  {PKL} …")
cache = ElibraryCache.load(str(PKL))

# Spot-check that migration worked
b1 = cache.get_book(1)
if b1 is None:
    print("ERROR: book 1 not found after load — aborting")
    sys.exit(1)

print(f"  book 1 → book_number={b1.book_number!r}  book_name={b1.book_name!r}")
print(f"  chapters: {len(b1.book_chapters)}")
if b1.book_chapters:
    ch = b1.book_chapters[0]
    print(f"  first chapter → chapter_number={ch.chapter_number!r}  chapter_title={ch.chapter_title!r}")
if b1.book_summaries:
    bs = b1.book_summaries[0]
    print(f"  book summary  → author={bs.book_summary_author!r}  words={bs.book_summary_number_of_words}")

print(f"\nSaving   {PKL} …")
cache.save(str(PKL))
print("Done — pkl re-saved with new prefixed field names.")
print(cache.summary_stats())
