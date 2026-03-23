"""
generate_book_summaries_2016_2025.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate `summary_book` for the 40 ijCSCL editorials from 2016–2025
using the Anthropic (Claude) API with a specific ~100-word prompt.

Skips any editorial that already has a non-empty summary_book.
Saves PKL after each editorial. Re-exports JSON when done.

Run from workspace root:
    python scripts/generate_book_summaries_2016_2025.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

import anthropic

ROOT      = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
PKL_PATH  = CACHE_DIR / "editorials_cache.pkl"
JSON_PATH = CACHE_DIR / "editorials_cache.json"

sys.path.insert(0, str(ROOT))
from cache.editorials_cache import EditorialsCache, EditorialSummary

# ── Anthropic client ───────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL  = "claude-opus-4-5"

# ── Prompt ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "These are editorial introductions to each of the issues of the International "
    "Journal of CSCL during its second decade of publication (2016 through 2025). "
    "You are a teacher summarizing the arguments of this collection for advanced "
    "graduate students studying learning science, social science and computer science. "
    "Write an approximately 100-word-long summary of each editorial, analyzing its "
    "main concerns, arguments, methods and findings. Do not use names or terminology "
    "that do not appear in the editorial."
)

TODAY = date.today().isoformat()


# ── Helpers ────────────────────────────────────────────────────────────────

def _has_summary_book(ed) -> bool:
    s = ed.summary_book
    return s is not None and not isinstance(s, str) and bool(s.summary_text.strip())


def _truncate_text(obj: object, n: int = 10) -> object:
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if (k == "text" or k.endswith("_text") or k.endswith("_prompt")) and isinstance(v, str) and v:
                result[k] = " ".join(v.split()[:n])
            else:
                result[k] = _truncate_text(v, n)
        return result
    if isinstance(obj, list):
        return [_truncate_text(item, n) for item in obj]
    return obj


def save_json(cache: EditorialsCache) -> None:
    data = _truncate_text(asdict(cache))
    with open(JSON_PATH, "w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(data, ensure_ascii=False, indent=2)
            .replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029")
        )


def generate_summary(ed) -> str:
    user_msg = (
        f"Vol {ed.volume} ({ed.year}) Issue {ed.issue}: {ed.title}\n\n"
        f"{ed.text.strip()}"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text.strip()


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    cache = EditorialsCache.load(str(PKL_PATH))

    to_do = [
        ed for ed in cache.editorials
        if 2016 <= ed.year <= 2025 and ed.has_text() and not _has_summary_book(ed)
    ]
    already = sum(1 for ed in cache.editorials if 2016 <= ed.year <= 2025 and _has_summary_book(ed))

    print(f"2016–2025 editorials total : 40")
    print(f"Already have summary_book  : {already}")
    print(f"To generate                : {len(to_do)}")

    for idx, ed in enumerate(to_do, 1):
        print(f"\n[{idx}/{len(to_do)}] {ed.label} — {ed.title[:65]}")
        try:
            text = generate_summary(ed)
            word_count = len(text.split())
            ed.summary_book = EditorialSummary(
                summary_author          = "Claude Sonnet 4.6",
                summary_date            = TODAY,
                summary_prompt          = SYSTEM_PROMPT,
                summary_number_of_words = word_count,
                summary_text            = text,
            )
            cache.save(str(PKL_PATH))
            print(f"   ✓ {word_count} words — saved PKL")
            time.sleep(0.3)
        except Exception as exc:
            print(f"   ✗ ERROR: {exc}")

    save_json(cache)
    print(f"\nJSON updated → {JSON_PATH}")

    done_early = sum(1 for e in cache.editorials if 2006 <= e.year <= 2015 and _has_summary_book(e))
    done_late  = sum(1 for e in cache.editorials if 2016 <= e.year <= 2025 and _has_summary_book(e))
    print(f"summary_book totals: 2006–2015 = {done_early}/39,  2016–2025 = {done_late}/40")


if __name__ == "__main__":
    main()
