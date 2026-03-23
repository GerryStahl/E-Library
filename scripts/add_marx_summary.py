"""
add_marx_summary.py — Store the book-level summary for 1.marx in the cache.

Reads the book summary from reports/marx_summary.txt, creates a BookSummary
object, attaches it to book 1 (1.marx.pdf), and saves the updated pickle.

Run from the workspace root:

    python3 scripts/add_marx_summary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
sys.path.insert(0, str(CACHE_DIR))

from elibrary_cache import ElibraryCache, BookSummary

CACHE_PKL    = CACHE_DIR / "elibrary_cache.pkl"
SUMMARY_FILE = ROOT / "reports" / "marx_summary.txt"

PROMPT = (
    "1.marx is a PhD dissertation in philosophy. It analyzes the philosophies of Marx and Heidegger "
    "and suggests what each could learn from the other. You are a philosophy teacher summarizing the "
    "arguments of this dissertation for graduate students studying philosophy. Write an approximately "
    "300-word-long summary of each chapter analyzing its main arguments. Then write an approximately "
    "500-word-long summary of the entire dissertation, based on the chapter summaries. Produce a txt "
    "report listing the chapter (or book) titles followed by their summaries. Do not use names or "
    "terminology that do not appear in the book."
)

# ---------------------------------------------------------------------------
# Extract the book-level summary section from the txt file
# ---------------------------------------------------------------------------

TITLE = "MARX AND HEIDEGGER — SUMMARY OF THE DISSERTATION"

full_text = SUMMARY_FILE.read_text(encoding="utf-8")
idx = full_text.rfind(TITLE)
if idx == -1:
    raise RuntimeError(f"Could not find book summary title in {SUMMARY_FILE}")

# Slice from the title to the end of file
section = full_text[idx:].strip()

# Strip separator lines (lines whose non-whitespace characters are all "=")
lines = section.split("\n")
cleaned = [
    line for line in lines
    if not (line.strip() and all(c == "=" for c in line.strip()))
]
book_summary_text = "\n".join(cleaned).strip()

word_count = len(book_summary_text.split())

# ---------------------------------------------------------------------------
# Load cache, replace book 1's summary list, save
# ---------------------------------------------------------------------------

cache = ElibraryCache.load(str(CACHE_PKL))
book1 = cache.get_book(1)
if book1 is None:
    raise RuntimeError("Book 1 not found in cache")

summary = BookSummary(
    book_summary_author="Claude agent",
    book_summary_date="March 1, 2026",
    book_summary_prompt=PROMPT,
    book_summary_number_of_words=word_count,
    book_summary_text=book_summary_text,
)

book1.book_summaries = [summary]
cache.save()

print(f"Book summary stored for book 1 — {word_count} words")
print(f"Saved → {CACHE_PKL}")
