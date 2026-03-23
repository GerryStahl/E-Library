"""
generate_total_history_gpt4o.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate a single ~500-word total_history summary using GPT-4o, merging all
20 annual summaries into a narrative of the CSCL field's evolution 2006–2025.

Stores result in EditorialsCache.total_history.summary_gpt4o and writes a report.

Run from workspace root:
    python scripts/generate_total_history_gpt4o.py
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
REPORT    = ROOT / "reports" / "total_history_gpt4o.txt"

sys.path.insert(0, str(ROOT))
from cache.editorials_cache import EditorialsCache, EditorialSummary, TotalHistory

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL  = "gpt-4o-mini"

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


def _truncate(obj, n=10):
    if isinstance(obj, dict):
        return {k: (" ".join(v.split()[:n]) if (k == "text" or k.endswith("_text") or k.endswith("_prompt")) and isinstance(v, str) and v else _truncate(v, n)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate(i, n) for i in obj]
    return obj


def build_user_message(cache: EditorialsCache) -> str:
    lines = ["All three Level-2 annual summaries for all 20 volumes of ijCSCL (2006–2025):\n"]
    for ann in sorted(cache.annual_summaries, key=lambda a: a.year):
        lines.append(f"--- {ann.year} ---")
        if ann.has_summary():
            lines.append(f"[Claude]\n{ann.summary.summary_text.strip()}\n")
        if ann.has_summary_gpt4o():
            lines.append(f"[GPT-4o]\n{ann.summary_gpt4o.summary_text.strip()}\n")
        if ann.has_summary_book():
            lines.append(f"[Copilot]\n{ann.summary_book.summary_text.strip()}\n")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    cache = EditorialsCache.load(str(PKL_PATH))

    if not hasattr(cache, 'total_history') or cache.total_history is None:
        cache.total_history = TotalHistory()

    if cache.total_history.has_summary_gpt4o():
        print("total_history already has a GPT-4o summary — regenerating.")

    filled_annual = sum(1 for a in cache.annual_summaries if a.has_summary())
    print(f"Annual summaries available: {filled_annual} / {len(cache.annual_summaries)}")

    user_msg = build_user_message(cache)
    print(f"Sending {len(user_msg.split())} words to GPT-4o API...")

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

    cache.total_history.summary_gpt4o = EditorialSummary(
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
        "ijCSCL — 20-Year History of CSCL (2006–2025) [GPT-4o]",
        "=" * 70,
        f"Generated : {TODAY}",
        f"Model     : {MODEL}",
        f"Words     : {wc}",
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
    print(f"\n--- GPT-4o Total History Preview (first 200 words) ---")
    print(" ".join(text.split()[:200]))


if __name__ == "__main__":
    main()
