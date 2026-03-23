"""
add_marx_chapter_summaries.py — Store chapter-level summaries for 1.marx in the cache.

Parses reports/marx_summary.txt, extracts the 12 chapter summaries in order,
creates a ChapterSummary for each cache chapter of book 1, and saves the pkl.

Run from the workspace root:

    python3 scripts/add_marx_chapter_summaries.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
sys.path.insert(0, str(CACHE_DIR))

from elibrary_cache import ElibraryCache, ChapterSummary

CACHE_PKL    = CACHE_DIR / "elibrary_cache.pkl"
SUMMARY_FILE = ROOT / "reports" / "marx_summary.txt"

SEPARATOR = "=" * 70

# ---------------------------------------------------------------------------
# Parse marx_summary.txt into ordered list of (title, text) blocks
# ---------------------------------------------------------------------------
# File format:
#   ======...======\n
#   CHAPTER TITLE\n
#   ======...======\n
#   body text\n
#   (repeat)
#
# We match each occurrence of: separator → title-line(s) → separator → body
# and collect (title, body) pairs, stopping before the book-level summary.

def parse_summary_blocks(path: Path) -> list[tuple[str, str]]:
    """
    Scan for the pattern  SEP + title + SEP + body  throughout the file and
    return a list of (title, body) pairs.
    """
    raw = path.read_text(encoding="utf-8")

    # Each chapter block: one separator, a short title section, another
    # separator, then body text up to the next separator.
    pattern = re.compile(
        r"={60,}[ \t]*\n"          # opening separator
        r"((?:(?!={60,}).)+?)\n"   # title text (one or more non-sep lines)
        r"={60,}[ \t]*\n"          # closing separator
        r"(.*?)"                   # body text (non-greedy)
        r"(?=\n={60,}|\Z)",        # up to next separator or end of file
        re.DOTALL,
    )

    blocks: list[tuple[str, str]] = []
    for m in pattern.finditer(raw):
        title = m.group(1).strip()
        body  = m.group(2).strip()
        if title and body:
            blocks.append((title, body))

    return blocks


blocks = parse_summary_blocks(SUMMARY_FILE)

# Drop the book-level summary (already stored via add_marx_summary.py)
chapter_blocks = [b for b in blocks
                  if "SUMMARY OF THE DISSERTATION" not in b[0]]

# ---------------------------------------------------------------------------
# Load cache and match blocks to chapters by position
# ---------------------------------------------------------------------------

cache = ElibraryCache.load(str(CACHE_PKL))
book1 = cache.get_book(1)
if book1 is None:
    raise RuntimeError("Book 1 not found in cache")

content_chapters = book1.content_chapters  # chapters 1–12

if len(chapter_blocks) != len(content_chapters):
    print(f"WARNING: {len(chapter_blocks)} summary blocks found, "
          f"but {len(content_chapters)} chapters in cache.")
    print("Parsed block titles:")
    for i, (t, _) in enumerate(chapter_blocks):
        print(f"  {i+1:02d}: {t}")
    print("Cache chapter titles:")
    for ch in content_chapters:
        print(f"  {ch.chapter_number:02d}: {ch.chapter_title}")

# Store summaries — match by position
stored = 0
for ch, (blk_title, blk_text) in zip(content_chapters, chapter_blocks):
    word_count = len(blk_text.split())
    summary = ChapterSummary(
        chapter_summary_author="Claude agent",
        chapter_summary_date="March 1, 2026",
        chapter_summary_prompt="",
        chapter_summary_number_of_words=word_count,
        chapter_summary_text=blk_text,
    )
    ch.chapter_summaries = [summary]
    stored += 1
    print(f"  ch{ch.chapter_number:02d} ({word_count:>4}w)  {ch.chapter_title[:55]!r}")

cache.save()
print(f"\nStored {stored} chapter summaries → {CACHE_PKL}")
