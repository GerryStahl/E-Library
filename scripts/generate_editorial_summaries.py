"""
generate_editorial_summaries.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate ~300-word summaries for all 79 ijCSCL editorials stored in
editorials_cache.pkl using the OpenAI API (gpt-4o).

• Skips editorials that already have a non-empty summary.
• Saves the PKL after every editorial so progress is never lost.
• Writes a human-readable TXT report to reports/editorial_summaries.txt.
• Re-exports the JSON mirror when done.

Run from the workspace root:
    python scripts/generate_editorial_summaries.py

Author : Gerry Stahl
Created: March 2026
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

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = Path("/Users/GStahl2/AI/elibrary")
CACHE_DIR = BASE_DIR / "cache"
PKL_PATH  = CACHE_DIR / "editorials_cache.pkl"
JSON_PATH = CACHE_DIR / "editorials_cache.json"
REPORT    = BASE_DIR / "reports" / "editorial_summaries.txt"

sys.path.insert(0, str(BASE_DIR))
from cache.editorials_cache import EditorialsCache, EditorialSummary

# ── Anthropic client ─────────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL  = "claude-opus-4-5"

# ── Prompt ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a learning-science researcher writing for other leading researchers. "
    "You are reviewing the history of the CSCL (computer-supported collaborative learning) "
    "research field from 2006 through 2025 exclusively on the basis of the 79 editorial "
    "introductions to the volumes of the 'international journal of CSCL'. "
    "Write an approximately 300-word summary of each of these articles. "
    "Focus on how it described the evolving trends in the field of CSCL at the time "
    "the article was written, especially developments concerning innovations in theory "
    "(e.g., of learning, cognition, interaction, etc.), analytic methodology "
    "(e.g., for comparing experimental cases or analyzing discourse), software technology, "
    "pedagogy, curriculum. "
    "Do not use names or technical terms that are not mentioned in this corpus. "
    "Write these summaries of the individual articles as resources for later constructing "
    "a history of the CSCL field during the 20 years covered by the journal."
)

TODAY = date.today().isoformat()          # e.g. "2026-03-15"

# ── Helpers ────────────────────────────────────────────────────────────────

def _truncate_text(obj: object, n: int = 10) -> object:
    """Recursively truncate 'text' / '*_text' string fields to n words."""
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


def _has_summary(ed) -> bool:
    """Return True if ed already has a real EditorialSummary with text."""
    s = ed.summary
    if s is None or isinstance(s, str):
        return False
    return bool(s.summary_text.strip())


def generate_summary(ed) -> str:
    """Call the Anthropic API and return the raw summary text."""
    user_msg = (
        f"Vol {ed.volume} ({ed.year}) Issue {ed.issue}: {ed.title}\n\n"
        f"{ed.text.strip()}"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_msg},
        ],
    )
    return response.content[0].text.strip()


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    cache = EditorialsCache.load(str(PKL_PATH))

    # Determine which editorials need summaries
    to_summarize = [ed for ed in cache.editorials if ed.has_text()]
    already_done: list = []

    print(f"Editorials total   : {cache.total}")
    print(f"Already summarized : {len(already_done)}")
    print(f"To summarize       : {len(to_summarize)}")

    if not to_summarize:
        print("Nothing to do — all editorials already have summaries.")
    else:
        for idx, ed in enumerate(to_summarize, 1):
            print(f"\n[{idx}/{len(to_summarize)}] {ed.label} — {ed.title[:60]}")
            try:
                text = generate_summary(ed)
                word_count = len(text.split())
                es = EditorialSummary(
                    summary_author          = "Claude Sonnet 4.6",
                    summary_date            = TODAY,
                    summary_prompt          = SYSTEM_PROMPT,
                    summary_number_of_words = word_count,
                    summary_text            = text,
                )
                ed.set_summary(es)
                cache.save(str(PKL_PATH))
                print(f"   ✓ {word_count} words — saved PKL")
                # Small pause to respect rate limits
                time.sleep(0.5)
            except Exception as exc:
                print(f"   ✗ ERROR: {exc}")

    # ── Write TXT report ───────────────────────────────────────────────────
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    with_summary = [ed for ed in cache.editorials if _has_summary(ed)]
    no_summary   = [ed for ed in cache.editorials if not _has_summary(ed)]

    lines: list[str] = [
        "Editorial Introductions to ijCSCL — Summaries",
        "=" * 60,
        f"Generated : {TODAY}",
        f"Model     : {MODEL}",
        f"Summaries : {len(with_summary)} / {cache.total}",
        "",
        SYSTEM_PROMPT,
        "",
        "=" * 60,
        "",
    ]

    for ed in cache.editorials:
        lines.append(f"Vol {ed.volume} ({ed.year}) Issue {ed.issue}")
        lines.append(f"Title  : {ed.title}")
        lines.append(f"Author : {ed.author}")
        lines.append(f"Words  : {ed.words:,}")
        lines.append("")
        if _has_summary(ed):
            lines.append(ed.summary.summary_text)
        else:
            lines.append("[No summary — editorial has no text]")
        lines.append("")
        lines.append("-" * 60)
        lines.append("")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written → {REPORT}")

    # ── Re-export JSON ─────────────────────────────────────────────────────
    save_json(cache)
    print(f"JSON updated   → {JSON_PATH}")

    # ── Final stats ────────────────────────────────────────────────────────
    print(f"\n{cache.summary_stats()}")
    if no_summary:
        print(f"\nWARNING: {len(no_summary)} editorials still have no summary:")
        for ed in no_summary:
            print(f"  {ed.label}: {ed.title[:60]!r}  (words={ed.words})")


if __name__ == "__main__":
    main()
