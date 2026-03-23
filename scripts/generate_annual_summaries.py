"""
generate_annual_summaries.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate a ~300-word annual summary for each of the 20 years (2006-2025)
by merging all Claude, GPT-4o, and Book summaries for that year's issues
and calling the Anthropic API.

Stores results in EditorialsCache.annual_summaries and writes a txt report.

Run from workspace root:
    python scripts/generate_annual_summaries.py
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
REPORT    = ROOT / "reports" / "annual_summaries.txt"

sys.path.insert(0, str(ROOT))
from cache.editorials_cache import EditorialsCache, EditorialSummary

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL  = "claude-opus-4-5"

SYSTEM_PROMPT = (
    "You are a learning-science researcher writing for other leading researchers "
    "and advanced graduate students. You are reviewing the history of the CSCL "
    "(computer-supported collaborative learning) research field from 2006 through "
    "2025 exclusively on the basis of the 79 editorial introductions to the volumes "
    "of the \"international journal of CSCL\". "
    "Write an approximately 300-word summary of each year's volume. "
    "Focus on how the editorials described the evolving trends in the field of CSCL "
    "in the year the articles were written, especially developments concerning "
    "innovations in theory (e.g., of learning, cognition, interaction, etc.), "
    "analytic methodology (e.g., for comparing experimental cases or analyzing "
    "discourse), software technology, pedagogy, curriculum. "
    "Do not use names or technical terms that are not mentioned in this corpus. "
    "Write these summaries of the individual articles as resources for later "
    "constructing a history of the CSCL field during the 20 years covered by the journal."
)

TODAY = date.today().isoformat()


def _text(s) -> str:
    if s is None or isinstance(s, str):
        return ""
    return (s.summary_text or "").strip()


def _truncate(obj, n=10):
    if isinstance(obj, dict):
        return {k: (" ".join(v.split()[:n]) if (k == "text" or k.endswith("_text") or k.endswith("_prompt")) and isinstance(v, str) and v else _truncate(v, n)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate(i, n) for i in obj]
    return obj


def build_user_message(year: int, eds: list) -> str:
    """Compose the user message from all issue summaries for the year."""
    vol = eds[0].volume if eds else year - 2005
    lines = [f"Year {year}  (Vol {vol}) — summaries of {len(eds)} issues\n"]
    for ed in sorted(eds, key=lambda e: e.issue):
        lines.append(f"--- Issue {ed.issue}: {ed.title} ---")
        lines.append(f"Authors: {ed.author}\n")

        claude_t = _text(ed.summary)
        gpt_t    = _text(ed.summary_gpt4o)
        book_t   = _text(ed.summary_book)

        if claude_t:
            lines.append(f"[Claude summary]\n{claude_t}\n")
        if gpt_t:
            lines.append(f"[GPT-4o summary]\n{gpt_t}\n")
        if book_t:
            lines.append(f"[Book summary]\n{book_t}\n")

    return "\n".join(lines)


def generate_annual(year: int, eds: list) -> str:
    msg = build_user_message(year, eds)
    response = client.messages.create(
        model=MODEL,
        max_tokens=700,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": msg}],
    )
    return response.content[0].text.strip()


def main() -> None:
    cache = EditorialsCache.load(str(PKL_PATH))

    # Ensure stubs exist for all 20 years
    for year in range(2006, 2026):
        cache.ensure_annual(year)

    to_do = [
        a for a in cache.annual_summaries
        if not a.has_summary()
    ]
    already = len(cache.annual_summaries) - len(to_do)

    print(f"Annual summaries total   : {len(cache.annual_summaries)}")
    print(f"Already done             : {already}")
    print(f"To generate              : {len(to_do)}")

    for idx, ann in enumerate(to_do, 1):
        year = ann.year
        eds  = sorted([e for e in cache.editorials if e.year == year], key=lambda e: e.issue)
        print(f"\n[{idx}/{len(to_do)}] {year}  ({len(eds)} issues)")
        try:
            text = generate_annual(year, eds)
            wc   = len(text.split())
            ann.summary = EditorialSummary(
                summary_author          = "Claude Sonnet 4.6",
                summary_date            = TODAY,
                summary_prompt          = SYSTEM_PROMPT,
                summary_number_of_words = wc,
                summary_text            = text,
            )
            cache.save(str(PKL_PATH))
            print(f"   ✓ {wc} words — saved PKL")
            time.sleep(0.3)
        except Exception as exc:
            print(f"   ✗ ERROR: {exc}")

    # ── Write txt report ───────────────────────────────────────────────────
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    filled = [a for a in cache.annual_summaries if a.has_summary()]

    lines = [
        "ijCSCL Annual Volume Summaries",
        "=" * 70,
        f"Generated : {TODAY}",
        f"Model     : {MODEL}",
        f"Years     : {len(filled)} / {len(cache.annual_summaries)}",
        "",
        SYSTEM_PROMPT,
        "",
        "=" * 70,
        "",
    ]

    for ann in sorted(cache.annual_summaries, key=lambda a: a.year):
        eds = sorted([e for e in cache.editorials if e.year == ann.year], key=lambda e: e.issue)
        vol = eds[0].volume if eds else ann.year - 2005
        lines.append(f"Year {ann.year}  (Vol {vol})")
        lines.append(f"Issues  : {len(eds)}")
        lines.append(f"Titles  : " + " | ".join(e.title[:40] for e in eds))
        lines.append("")
        if ann.has_summary():
            lines.append(ann.summary.summary_text)
        else:
            lines.append("[No summary generated]")
        lines.append("")
        lines.append("-" * 70)
        lines.append("")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written → {REPORT}")

    # ── Re-export JSON ─────────────────────────────────────────────────────
    data = _truncate(asdict(cache))
    JSON_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2)
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029"),
        encoding="utf-8",
    )
    print(f"JSON updated   → {JSON_PATH}")
    print(f"\nAnnual summaries filled: {sum(1 for a in cache.annual_summaries if a.has_summary())} / 20")


if __name__ == "__main__":
    main()
