"""
generate_total_history_report.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Write a combined total_history.txt report with all 3 agent versions
(Claude, GPT-4o, GitHub Copilot) from the cache PKL.

Run from workspace root:
    python scripts/generate_total_history_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
PKL_PATH  = CACHE_DIR / "editorials_cache.pkl"
REPORT    = ROOT / "reports" / "total_history.txt"

sys.path.insert(0, str(ROOT))
from cache.editorials_cache import EditorialsCache

SEP = "=" * 70


def section(label: str, model: str, es) -> list[str]:
    lines = [
        "",
        SEP,
        f"  {label}",
        SEP,
        f"Generated : {es.summary_date}",
        f"Model     : {model}",
        f"Words     : {es.summary_number_of_words}",
        "",
        es.summary_text.strip(),
        "",
    ]
    return lines


def main() -> None:
    cache = EditorialsCache.load(str(PKL_PATH))
    th = cache.total_history

    lines = [
        "ijCSCL — 20-Year History of CSCL (2006–2025)",
        "All three Level-3 agent summaries",
        SEP,
    ]

    if th.has_summary():
        lines += section("Version 1 — Claude (Anthropic)", th.summary.summary_author, th.summary)
    else:
        lines += ["", SEP, "  Version 1 — Claude: NOT YET GENERATED", SEP]

    if th.has_summary_gpt4o():
        lines += section("Version 2 — GPT-4o (OpenAI)", th.summary_gpt4o.summary_author, th.summary_gpt4o)
    else:
        lines += ["", SEP, "  Version 2 — GPT-4o: NOT YET GENERATED", SEP]

    if th.has_summary_book():
        lines += section("Version 3 — GitHub Copilot", th.summary_book.summary_author, th.summary_book)
    else:
        lines += ["", SEP, "  Version 3 — GitHub Copilot: NOT YET GENERATED", SEP]

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written → {REPORT}")

    # Word count summary
    print()
    if th.has_summary():
        print(f"  Claude  : {th.summary.summary_number_of_words} words")
    if th.has_summary_gpt4o():
        print(f"  GPT-4o  : {th.summary_gpt4o.summary_number_of_words} words")
    if th.has_summary_book():
        print(f"  Copilot : {th.summary_book.summary_number_of_words} words")


if __name__ == "__main__":
    main()
