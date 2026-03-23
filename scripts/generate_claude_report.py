"""
generate_claude_report.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Write reports/claude_editorial_summaries.txt from the Claude summaries
(Editorial.summary) already stored in editorials_cache.pkl.

Run from the workspace root:
    python scripts/generate_claude_report.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

BASE_DIR  = Path("/Users/GStahl2/AI/elibrary")
CACHE_DIR = BASE_DIR / "cache"
PKL_PATH  = CACHE_DIR / "editorials_cache.pkl"
REPORT    = BASE_DIR / "reports" / "claude_editorial_summaries.txt"

sys.path.insert(0, str(BASE_DIR))
from cache.editorials_cache import EditorialsCache

cache = EditorialsCache.load(str(PKL_PATH))

def _has_claude(ed) -> bool:
    s = ed.summary
    return s is not None and not isinstance(s, str) and bool(s.summary_text.strip())

with_summary = [ed for ed in cache.editorials if _has_claude(ed)]
model = with_summary[0].summary.summary_author if with_summary else "Claude"

lines: list[str] = [
    "Editorial Introductions to ijCSCL — Claude Summaries",
    "=" * 60,
    f"Generated : {date.today().isoformat()}",
    f"Model     : {model}",
    f"Summaries : {len(with_summary)} / {cache.total}",
    "",
]

if with_summary:
    lines.append(with_summary[0].summary.summary_prompt)
    lines.append("")

lines += ["=" * 60, ""]

for ed in cache.editorials:
    lines.append(f"Vol {ed.volume} ({ed.year}) Issue {ed.issue}")
    lines.append(f"Title  : {ed.title}")
    lines.append(f"Author : {ed.author}")
    lines.append(f"Words  : {ed.words:,}")
    lines.append("")
    if _has_claude(ed):
        lines.append(ed.summary.summary_text)
    else:
        lines.append("[No Claude summary]")
    lines.append("")
    lines.append("-" * 60)
    lines.append("")

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(lines), encoding="utf-8")
print(f"Report written → {REPORT}  ({len(with_summary)} summaries)")
