"""
scripts/generate_book_keywords.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate 5–10 keywords for each of the 22 books using the Claude API,
based on each book's existing summary text.

Saves the keywords into book.book_keywords in the PKL, then re-exports JSON.

Usage:
    python scripts/generate_book_keywords.py
"""

from __future__ import annotations
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import anthropic
from cache.elibrary_cache import ElibraryCache
from scripts.export_cache_json import _truncate_text   # reuse truncation helper

PKL_PATH  = ROOT / "cache" / "elibrary_cache.pkl"
JSON_PATH = ROOT / "cache" / "elibrary_cache.json"

MODEL  = "claude-haiku-4-5-20251001"   # fast + cheap for a keyword task
PROMPT = """\
Below is a summary of an academic book. Generate exactly 7 keywords or short \
keyword phrases (2–4 words each) that best capture the book's core topics, \
methods, and intellectual contributions. Focus on terms that would help a \
researcher decide whether this book is relevant to their query.

Return ONLY a JSON array of strings — no explanation, no markdown fences.
Example: ["group cognition", "computer-supported collaboration", "virtual math teams"]

Book summary:
{summary}"""


def get_keywords(client: anthropic.Anthropic, summary: str) -> list[str]:
    """Call Claude and return a list of keyword strings."""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": PROMPT.format(summary=summary[:3000]),
        }],
    )
    raw = msg.content[0].text.strip()
    # strip any accidental markdown fences
    raw = raw.strip("`").strip()
    if raw.startswith("json"):
        raw = raw[4:].strip()
    return json.loads(raw)


def main():
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print("Loading cache…")
    cache = ElibraryCache.load(str(PKL_PATH))

    print(f"Generating book keywords for {len(cache.books)} books…\n")
    for book in cache.books:
        if not book.book_summaries:
            print(f"  Book {book.book_number:>2}: {book.book_title[:50]}  — no summary, skipped")
            continue

        summary_text = book.book_summaries[0].book_summary_text
        if not summary_text or not summary_text.strip():
            print(f"  Book {book.book_number:>2}: {book.book_title[:50]}  — empty summary, skipped")
            continue

        try:
            keywords = get_keywords(client, summary_text)
            book.book_keywords = keywords
            print(f"  Book {book.book_number:>2}: {book.book_title[:48]}")
            for kw in keywords:
                print(f"           • {kw}")
        except Exception as e:
            print(f"  Book {book.book_number:>2}: ERROR — {e}")

        time.sleep(0.3)   # gentle rate limiting

    print("\nSaving PKL…")
    cache.save(str(PKL_PATH))

    print("Re-exporting JSON…")
    from dataclasses import asdict
    data = _truncate_text(asdict(cache))
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2)
                .replace('\u2028', '\\u2028').replace('\u2029', '\\u2029'))
    print(f"Done → {JSON_PATH}")

    # Print summary table
    print("\n" + "=" * 60)
    print("BOOK KEYWORDS SUMMARY")
    print("=" * 60)
    for book in cache.books:
        print(f"\nBook {book.book_number:>2}: {book.book_title[:52]}")
        for kw in book.book_keywords:
            print(f"   • {kw}")


if __name__ == "__main__":
    main()
