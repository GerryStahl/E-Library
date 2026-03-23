"""
generate_editorial_summaries_gpt4o.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate ~300-word summaries for all 79 ijCSCL editorials using the OpenAI
API (gpt-4o) and store them in the new Editorial.summary_gpt4o field.

The existing Editorial.summary field (Claude) is left untouched.

Run from the workspace root:
    python scripts/generate_editorial_summaries_gpt4o.py

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

from openai import OpenAI

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = Path("/Users/GStahl2/AI/elibrary")
CACHE_DIR = BASE_DIR / "cache"
PKL_PATH  = CACHE_DIR / "editorials_cache.pkl"
JSON_PATH = CACHE_DIR / "editorials_cache.json"
REPORT    = BASE_DIR / "reports" / "editorial_summaries_gpt4o.txt"

sys.path.insert(0, str(BASE_DIR))
from cache.editorials_cache import EditorialsCache, EditorialSummary

# ── OpenAI client ──────────────────────────────────────────────────────────
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL  = "gpt-4o"

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

TODAY = date.today().isoformat()


# ── Helpers ────────────────────────────────────────────────────────────────

def _has_gpt4o_summary(ed) -> bool:
    s = ed.summary_gpt4o
    if s is None or isinstance(s, str):
        return False
    return bool(s.summary_text.strip())


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
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=600,
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    cache = EditorialsCache.load(str(PKL_PATH))

    to_summarize = [ed for ed in cache.editorials if ed.has_text() and not _has_gpt4o_summary(ed)]
    already_done = [ed for ed in cache.editorials if _has_gpt4o_summary(ed)]

    print(f"Editorials total         : {cache.total}")
    print(f"Already have gpt4o summ  : {len(already_done)}")
    print(f"To summarize             : {len(to_summarize)}")

    if not to_summarize:
        print("Nothing to do — all editorials already have GPT-4o summaries.")
    else:
        for idx, ed in enumerate(to_summarize, 1):
            print(f"\n[{idx}/{len(to_summarize)}] {ed.label} — {ed.title[:60]}")
            try:
                text = generate_summary(ed)
                word_count = len(text.split())
                es = EditorialSummary(
                    summary_author          = "GPT-4o",
                    summary_date            = TODAY,
                    summary_prompt          = SYSTEM_PROMPT,
                    summary_number_of_words = word_count,
                    summary_text            = text,
                )
                ed.summary_gpt4o = es
                cache.save(str(PKL_PATH))
                print(f"   ✓ {word_count} words — saved PKL")
                time.sleep(0.5)
            except Exception as exc:
                print(f"   ✗ ERROR: {exc}")

    # ── Write TXT report ───────────────────────────────────────────────────
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "Editorial Introductions to ijCSCL — GPT-4o Summaries",
        "=" * 60,
        f"Generated : {TODAY}",
        f"Model     : {MODEL}",
        f"Summaries : {sum(1 for ed in cache.editorials if _has_gpt4o_summary(ed))} / {cache.total}",
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
        if _has_gpt4o_summary(ed):
            lines.append(ed.summary_gpt4o.summary_text)
        else:
            lines.append("[No summary — editorial has no text]")
        lines.append("")
        lines.append("-" * 60)
        lines.append("")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written → {REPORT}")

    save_json(cache)
    print(f"JSON updated   → {JSON_PATH}")

    print(f"\nGPT-4o summaries: {sum(1 for ed in cache.editorials if _has_gpt4o_summary(ed))} / {cache.total}")
    print(f"Claude summaries: {cache.with_summary} / {cache.total}")


if __name__ == "__main__":
    main()
