"""
generate_comparison_report.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate reports/editorial_summaries_comparison.txt — a side-by-side
three-column comparison of the Claude, GPT-4o, and Book-16 summaries for
all 79 ijCSCL editorials, ordered by volume and issue.

Run from workspace root:
    python scripts/generate_comparison_report.py
"""

from __future__ import annotations

import sys
import textwrap
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cache.editorials_cache import EditorialsCache, EditorialSummary

CACHE_PKL = ROOT / "cache" / "editorials_cache.pkl"
REPORT    = ROOT / "reports" / "editorial_summaries_comparison.txt"

# ── Layout constants ───────────────────────────────────────────────────────
COL_W    = 52          # chars per column of summary text
COL_SEP  = " │ "      # 3-char separator between columns
PAGE_SEP = "═" * 120  # between editorials


def wrap_col(text: str, width: int = COL_W) -> list[str]:
    """Wrap text to width, preserving paragraph breaks."""
    if not text.strip():
        return ["(none)"]
    paragraphs = text.strip().split("\n")
    lines: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            lines.append("")
        else:
            lines.extend(textwrap.wrap(para, width=width) or [""])
    return lines


def side_by_side(columns: list[tuple[str, str]], col_width: int = COL_W) -> list[str]:
    """
    Render a list of (header, body_text) tuples as side-by-side columns.
    Returns a list of lines ready to join with newlines.
    """
    n = len(columns)
    headers = [h for h, _ in columns]
    bodies  = [wrap_col(b, col_width) for _, b in columns]

    # Pad to equal length
    max_lines = max(len(b) for b in bodies)
    bodies = [b + [""] * (max_lines - len(b)) for b in bodies]

    sep = COL_SEP
    lines: list[str] = []

    # Column header row
    header_line = sep.join(f"{h:<{col_width}}" for h in headers)
    lines.append(header_line)
    lines.append(sep.join("─" * col_width for _ in range(n)))

    for row in zip(*bodies):
        lines.append(sep.join(f"{cell:<{col_width}}" for cell in row))

    return lines


def summary_text(s: object) -> str:
    if s is None or isinstance(s, str):
        return ""
    es: EditorialSummary = s  # type: ignore
    return (es.summary_text or "").strip()


def main() -> None:
    cache = EditorialsCache.load(str(CACHE_PKL))

    # Sort by (volume, issue)
    editorials = sorted(cache.editorials, key=lambda e: (e.volume, e.issue))

    output: list[str] = [
        "ijCSCL Editorial Summaries — Side-by-Side Comparison",
        "=" * 120,
        f"Generated : {date.today().isoformat()}",
        f"Editorials: {len(editorials)}",
        "Columns   : Claude (claude-opus-4-5)  │  GPT-4o  │  Book summary",
        "",
        PAGE_SEP,
        "",
    ]

    for ed in editorials:
        claude_text = summary_text(ed.summary)
        gpt4o_text  = summary_text(ed.summary_gpt4o)
        book_text   = summary_text(ed.summary_book)

        # ── Editorial header ──
        output.append(f"Vol {ed.volume} ({ed.year})  Issue {ed.issue}")
        output.append(f"Title  : {ed.title}")
        output.append(f"Authors: {ed.author}")
        output.append("")

        cols = [
            ("CLAUDE", claude_text),
            ("GPT-4O", gpt4o_text),
            ("BOOK", book_text),
        ]

        output.extend(side_by_side(cols, col_width=COL_W))
        output.append("")
        output.append(PAGE_SEP)
        output.append("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(output), encoding="utf-8")
    print(f"Report written → {REPORT}")
    print(f"  {len(editorials)} editorials, {sum(1 for e in editorials if e.summary_book)} with Book column")


if __name__ == "__main__":
    main()
