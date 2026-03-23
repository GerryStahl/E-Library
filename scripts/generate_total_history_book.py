"""
generate_total_history_book.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate a third total_history summary using the same data-assembly approach
as generate_annual_summaries.py — i.e., feed the summary_book text of all 79
editorials (grouped by year) directly into the API — rather than the already-
aggregated annual summaries.

Stores result in EditorialsCache.total_history.summary_book and writes a report.

Run from workspace root:
    python scripts/generate_total_history_book.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

from openai import OpenAI

ROOT      = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
PKL_PATH  = CACHE_DIR / "editorials_cache.pkl"
JSON_PATH = CACHE_DIR / "editorials_cache.json"
REPORT    = ROOT / "reports" / "total_history_book.txt"

sys.path.insert(0, str(ROOT))
from cache.editorials_cache import EditorialsCache, EditorialSummary, TotalHistory

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL  = "gpt-4o"

# Same system prompt as the other two total_history generators
SYSTEM_PROMPT = (
    "You are a learning-science researcher writing for other leading researchers "
    "and advanced graduate students. You have read all 79 editorial introductions "
    "to the 20 annual volumes of the International Journal of CSCL (2006–2025), "
    "and have already written annual summaries for each year. "
    "Now write a single approximately 500-word synthesis covering the full 20-year "
    "arc of the CSCL field as seen through those editorials. "
    "Focus on how the field changed over time: shifts in theory (e.g., of learning, "
    "cognition, interaction), analytic methodology, software technology, pedagogy, "
    "and curriculum. Identify major turning points, recurring themes, and the "
    "direction the field was heading by 2025. "
    "Do not use names or technical terms that do not appear in the annual summaries "
    "provided. Write this as a resource for constructing a history of CSCL."
)

TODAY = date.today().isoformat()


def _text(s) -> str:
    """Extract summary_text from an EditorialSummary, or '' if absent."""
    if s is None or isinstance(s, str):
        return ""
    return (s.summary_text or "").strip()


def _truncate(obj, n=10):
    if isinstance(obj, dict):
        return {k: (" ".join(v.split()[:n]) if (k == "text" or k.endswith("_text") or k.endswith("_prompt")) and isinstance(v, str) and v else _truncate(v, n)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate(i, n) for i in obj]
    return obj


def build_user_message(cache: EditorialsCache) -> str:
    """
    Build user message using the same per-issue, per-year format as
    generate_annual_summaries.py but using summary_book texts only,
    covering all 20 years in one message.
    """
    lines = [
        "Book-level summaries of all 79 ijCSCL editorials (2006–2025),",
        "grouped by year and issue:\n",
    ]
    for year in range(2006, 2026):
        eds = sorted(
            [e for e in cache.editorials if e.year == year],
            key=lambda e: e.issue,
        )
        if not eds:
            continue
        vol = eds[0].volume
        lines.append(f"=== {year}  (Vol {vol}, {len(eds)} issues) ===")
        for ed in eds:
            book_t = _text(ed.summary_book)
            if book_t:
                lines.append(f"  Issue {ed.issue}: {ed.title}")
                lines.append(f"  {book_t}")
                lines.append("")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    cache = EditorialsCache.load(str(PKL_PATH))

    if not hasattr(cache, 'total_history') or cache.total_history is None:
        cache.total_history = TotalHistory()

    if cache.total_history.has_summary_book():
        print("total_history already has a book summary — regenerating.")

    # Count editorials with summary_book
    filled = sum(1 for e in cache.editorials if _text(e.summary_book))
    print(f"Editorials with summary_book: {filled} / {len(cache.editorials)}")

    user_msg = build_user_message(cache)
    print(f"Sending {len(user_msg.split())} words to API...")

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=900,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
    )
    text = response.choices[0].message.content.strip()
    wc   = len(text.split())
    print(f"Received {wc} words.")

    cache.total_history.summary_book = EditorialSummary(
        summary_author          = "GPT-4o",
        summary_date            = TODAY,
        summary_prompt          = SYSTEM_PROMPT,
        summary_number_of_words = wc,
        summary_text            = text,
    )
    cache.save(str(PKL_PATH))
    print(f"PKL saved → {PKL_PATH}")

    # ── Write report ───────────────────────────────────────────────────────
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ijCSCL — 20-Year History of CSCL (2006–2025) [Book-level summaries, GPT-4o]",
        "=" * 70,
        f"Generated : {TODAY}",
        f"Model     : {MODEL}",
        f"Words     : {wc}",
        f"Input     : summary_book texts from all {filled} editorials, grouped by year",
        "",
        SYSTEM_PROMPT,
        "",
        "=" * 70,
        "",
        text,
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written → {REPORT}")

    # ── Re-export JSON ─────────────────────────────────────────────────────
    data = _truncate(asdict(cache))
    JSON_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2)
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029"),
        encoding="utf-8",
    )
    print(f"JSON updated   → {JSON_PATH}")
    print(f"\n--- Book Total History Preview (first 200 words) ---")
    print(" ".join(text.split()[:200]))


if __name__ == "__main__":
    main()
