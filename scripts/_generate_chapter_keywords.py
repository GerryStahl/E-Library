"""
_generate_chapter_keywords.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate ~5 search keywords for every chapter whose chapter_keywords list is
currently empty, using the OpenAI API (gpt-4o-mini).

Keywords are based on:
  • chapter title
  • chapter summary (if available)
  • book title (for context)

Saves PKL and re-exports JSON when done.

Author  : Gerry Stahl
Created : March 4, 2026
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = Path('/Users/GStahl2/AI/elibrary')
CACHE_DIR = BASE_DIR / 'cache'
PKL_PATH  = CACHE_DIR / 'elibrary_cache.pkl'

sys.path.insert(0, str(CACHE_DIR))
from elibrary_cache import ElibraryCache

# ── OpenAI client ──────────────────────────────────────────────────────────
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
MODEL  = 'gpt-4o-mini'

# ── Load PKL ───────────────────────────────────────────────────────────────
cache = ElibraryCache.load(str(PKL_PATH))


def get_summary_text(ch) -> str:
    """Return the first non-empty chapter summary text, or ''."""
    for s in ch.chapter_summaries:
        t = (s.chapter_summary_text or '').strip()
        if t:
            return t
    return ''


def generate_keywords(book_title: str, ch_title: str, summary: str) -> list[str]:
    """Call the API and return a list of 5 keyword strings."""
    if summary:
        user_content = (
            f"Book: {book_title}\n"
            f"Chapter: {ch_title}\n\n"
            f"Chapter summary:\n{summary}\n\n"
            "Return exactly 5 concise keyword phrases (2–4 words each) that a "
            "researcher would type into a search engine to find this chapter. "
            "Output ONLY a Python list literal on one line, e.g.: "
            '["keyword one", "keyword two", "keyword three", "keyword four", "keyword five"]'
        )
    else:
        user_content = (
            f"Book: {book_title}\n"
            f"Chapter title: {ch_title}\n\n"
            "Return exactly 5 concise keyword phrases (2–4 words each) that a "
            "researcher would type into a search engine to find this chapter. "
            "Output ONLY a Python list literal on one line, e.g.: "
            '["keyword one", "keyword two", "keyword three", "keyword four", "keyword five"]'
        )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research librarian generating search keywords "
                    "for academic chapters. Always respond with exactly one "
                    "Python list literal and nothing else."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
        max_tokens=120,
    )
    raw = response.choices[0].message.content.strip()

    # Parse the list literal safely
    try:
        kws = eval(raw)  # noqa: S307 – controlled environment, known format
        if isinstance(kws, list):
            return [str(k).strip() for k in kws[:5]]
    except Exception:
        pass

    # Fallback: split on commas if eval fails
    cleaned = raw.strip('[]').replace('"', '').replace("'", '')
    return [k.strip() for k in cleaned.split(',') if k.strip()][:5]


# ── Main loop ─────────────────────────────────────────────────────────────
filled    = []
skipped   = []
errors    = []

for book in sorted(cache.books, key=lambda b: b.book_number):
    for ch in sorted(book.book_chapters, key=lambda c: c.chapter_number):
        if ch.chapter_keywords:       # already has keywords
            continue

        summary = get_summary_text(ch)
        label   = f"Book {book.book_number:2d} Ch {ch.chapter_number:2d}  {ch.chapter_title[:55]}"

        try:
            kws = generate_keywords(book.book_title, ch.chapter_title, summary)
            ch.chapter_keywords = kws
            filled.append((book.book_number, ch.chapter_number, ch.chapter_title[:50], kws))
            print(f"  ✓ {label}")
            print(f"      {kws}")
            # Polite rate-limit pause
            time.sleep(0.3)
        except Exception as exc:
            errors.append((book.book_number, ch.chapter_number, str(exc)))
            print(f"  ✗ {label}  ERROR: {exc}")

# ── Save PKL ───────────────────────────────────────────────────────────────
pickle.dump(cache, open(PKL_PATH, 'wb'))
print(f"\nPKL saved: {PKL_PATH}")

# ── Regenerate JSON ───────────────────────────────────────────────────────
result = subprocess.run(
    [sys.executable, 'scripts/export_cache_json.py'],
    cwd=str(BASE_DIR),
    capture_output=True, text=True
)
if result.returncode == 0:
    print("JSON exported successfully.")
else:
    print("JSON export FAILED:")
    print(result.stderr)

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  KEYWORD GENERATION REPORT")
print(f"{'='*60}")
print(f"  Keywords generated : {len(filled)}")
print(f"  Errors             : {len(errors)}")
if errors:
    print("\nERRORS:")
    for bnum, chnum, msg in errors:
        print(f"  Book {bnum} Ch {chnum}: {msg}")
