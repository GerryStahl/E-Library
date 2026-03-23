"""
insert_summary_prompts.py — For every book that has no BookSummary (or has
a summary with an empty book_summary_prompt), create/update a BookSummary
whose prompt field is set from cache/prompts_for_summaries.txt.

The prompt file has one paragraph per book; each paragraph begins with the
book stem name (e.g. "1.marx", "10.science") — which is the book_name
without the ".pdf" extension.

Run from the workspace root:
    python3 scripts/insert_summary_prompts.py
"""

import sys
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
sys.path.insert(0, str(CACHE_DIR))

from elibrary_cache import ElibraryCache, BookSummary

PKL           = CACHE_DIR / "elibrary_cache.pkl"
PROMPTS_FILE  = CACHE_DIR / "prompts_for_summaries.txt"

# ── Parse prompts_for_summaries.txt into {stem: prompt_text} ─────────────────
# Each paragraph is separated by one or more blank lines.
raw = PROMPTS_FILE.read_text(encoding="utf-8")
paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]

prompts_by_stem: dict[str, str] = {}
for para in paragraphs:
    # The stem is the first word / token before the first space
    first_word = para.split()[0]          # e.g. "1.marx", "10.science,"
    stem = first_word.rstrip(",.")        # strip trailing punctuation
    prompts_by_stem[stem] = para

print(f"Loaded {len(prompts_by_stem)} prompts from {PROMPTS_FILE.name}")
for stem in sorted(prompts_by_stem):
    print(f"  {stem:20s} {prompts_by_stem[stem][:60]!r}…")

# ── Load cache ────────────────────────────────────────────────────────────────
cache = ElibraryCache.load(str(PKL))

# ── Insert / update prompts ───────────────────────────────────────────────────
updated = 0
skipped = 0

for book in cache.books:
    stem = book.book_name.replace(".pdf", "")   # e.g. "1.marx"
    prompt_text = prompts_by_stem.get(stem)

    if prompt_text is None:
        print(f"\nWARNING: no prompt found for stem '{stem}' ({book.book_name}) — skipping")
        skipped += 1
        continue

    # Check if a summary with a non-empty prompt already exists
    if book.book_summaries and book.book_summaries[0].book_summary_prompt.strip():
        print(f"  Book {book.book_number:>2} ({book.book_name}) — prompt already set, skipping")
        skipped += 1
        continue

    if book.book_summaries:
        # Summary exists but prompt is empty — fill it in
        book.book_summaries[0].book_summary_prompt = prompt_text
        print(f"  Book {book.book_number:>2} ({book.book_name}) — updated existing summary's prompt")
    else:
        # No summary at all — create a stub with just the prompt
        stub = BookSummary(
            book_summary_author="",
            book_summary_date="",
            book_summary_prompt=prompt_text,
            book_summary_number_of_words=0,
            book_summary_text="",
        )
        book.book_summaries = [stub]
        print(f"  Book {book.book_number:>2} ({book.book_name}) — created stub summary with prompt")
    updated += 1

print(f"\nUpdated {updated} books, skipped {skipped}")

# ── Save ──────────────────────────────────────────────────────────────────────
cache.save(str(PKL))
print(f"Saved → {PKL}")
