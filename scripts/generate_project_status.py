#!/usr/bin/env python3
"""
scripts/generate_project_status.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Auto-generates documents/project_status.md — a compact, always-current
snapshot of the project for use as Copilot Chat context at session start.

No API calls, no model loading. Reads existing output files and synthesises:
  - Pipeline step completion status (based on file existence + timestamps)
  - Cluster inventory (labels, sizes, notes) from cluster_summaries.json
  - Citation statistics from self_citations.csv
  - Top cited years from self_citation_report.txt
  - Report file inventory with modification dates and sizes
  - Open/pending work items (kept as a manually edited section)

Run at the end of each work session, or on demand:
    python scripts/generate_project_status.py

The output file documents/project_status.md is designed to be attached
in Copilot Chat via #file when starting an analysis session.

Author: Gerry Stahl
Created: March 2026
"""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
REPORTS  = ROOT / "reports"
DOCS     = ROOT / "documents"
SCRIPTS  = ROOT / "scripts"
OUT      = DOCS / "project_status.md"

# ── Helpers ───────────────────────────────────────────────────────────────────

def file_stamp(path: Path) -> str:
    """Return 'YYYY-MM-DD  size' string for a file, or 'missing'."""
    if not path.exists():
        return "**missing**"
    mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
    size  = path.stat().st_size
    if size >= 1_000_000:
        sz = f"{size/1_000_000:.1f} MB"
    elif size >= 1_000:
        sz = f"{size/1_000:.0f} KB"
    else:
        sz = f"{size} B"
    return f"{mtime}  {sz}"


def exists_mark(path: Path) -> str:
    return "✅" if path.exists() else "❌"


def csv_row_count(path: Path) -> str:
    if not path.exists():
        return "—"
    with open(path) as f:
        return f"{sum(1 for _ in f) - 1:,}"   # subtract header


def load_cluster_summaries() -> dict:
    p = REPORTS / "cluster_summaries.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def top_cited_years(n: int = 10) -> list[tuple[str, int]]:
    """Parse top cited years from self_citation_report.txt."""
    p = REPORTS / "self_citation_report.txt"
    if not p.exists():
        return []
    text = p.read_text()
    section = re.search(
        r"── TOP CITED YEARS.*?\n(.*?)(?:\n──|\n={6})",
        text, re.DOTALL
    )
    if not section:
        return []
    results = []
    for line in section.group(1).splitlines():
        m = re.match(r"\s+(\d{4}[a-z]?):\s+(\d+)", line)
        if m:
            results.append((m.group(1), int(m.group(2))))
        if len(results) >= n:
            break
    return results


def cluster_citation_counts() -> dict[int, int]:
    """Return {cluster_id: citation_count} from self_citations.csv."""
    p = REPORTS / "self_citations.csv"
    if not p.exists():
        return {}
    counts: dict[int, int] = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            cid = int(row.get("cluster_id", -1))
            counts[cid] = counts.get(cid, 0) + 1
    return counts


# ── Section builders ──────────────────────────────────────────────────────────

def section_pipeline() -> str:
    steps = [
        ("Step 1: Chunk clustering",
         REPORTS / "chunk_clusters.csv",
         "`scripts/cluster_chunks.py`"),
        ("Step 1b: Expanded cluster summaries",
         REPORTS / "cluster_summaries.json",
         "`scripts/expand_cluster_summaries.py`"),
        ("Step 2: Self-citation extraction",
         REPORTS / "self_citations.csv",
         "`scripts/extract_self_citations.py`"),
        ("Step 3: Visualisation",
         REPORTS / "cluster_citation_scatter.png",
         "`scripts/visualize_self_citations.py`"),
        ("Step 4: Cross-cluster citation matrix",
         None,
         "`scripts/cross_cluster_citations.py` — **not yet written**"),
    ]
    lines = ["## Analysis Pipeline Status\n"]
    for label, output_file, script in steps:
        if output_file is None:
            mark = "❌"
            stamp = "pending"
        else:
            mark = exists_mark(output_file)
            stamp = file_stamp(output_file) if output_file.exists() else "output missing"
        lines.append(f"- {mark} **{label}** — {script}  ")
        lines.append(f"  output: `{output_file.name if output_file else '—'}` ({stamp})")
    return "\n".join(lines)


def section_clusters() -> str:
    summaries = load_cluster_summaries()
    cit_counts = cluster_citation_counts()
    if not summaries:
        return "## Clusters\n\n*cluster_summaries.json not found — run `cluster_chunks.py` and `expand_cluster_summaries.py`*"

    NOISE   = {6, 12}
    NON_CSCL = {8, 9, 15}
    PARTIAL = {18}

    # Sort by chunk size descending (we don't have sizes here, so sort by cit count as proxy,
    # with a fallback ordering that matches known cluster ordering)
    KNOWN_ORDER = [11, 1, 5, 3, 7, 4, 19, 0, 10, 6, 12, 2, 17, 13, 18, 14, 16, 9, 8, 15]

    lines = ["## Cluster Inventory (k=20)\n"]
    lines.append("| ID | Label | Citations | Notes |")
    lines.append("|---|---|---|---|")
    for cid_str in [str(c) for c in KNOWN_ORDER]:
        entry = summaries.get(cid_str, {})
        cid   = int(cid_str)
        label = entry.get("label", f"Cluster {cid}")
        cits  = cit_counts.get(cid, 0)
        notes = ""
        if cid in NOISE:
            notes = "⚠ bibliography noise"
        elif cid in NON_CSCL:
            notes = "non-CSCL"
        elif cid in PARTIAL:
            notes = "philosophy (peripheral)"
        lines.append(f"| {cid} | {label} | {cits:,} | {notes} |")
    return "\n".join(lines)


def section_citations() -> str:
    csv_path = REPORTS / "self_citations.csv"
    if not csv_path.exists():
        return "## Citation Statistics\n\n*self_citations.csv not found*"

    total = int(csv_row_count(csv_path).replace(",", ""))
    top   = top_cited_years(10)

    lines = [f"## Citation Statistics\n"]
    lines.append(f"- **Total (chunk, cited_year) pairs:** {total:,}")
    lines.append(f"- **Source file:** `{file_stamp(csv_path)}`\n")
    if top:
        lines.append("**Top 10 cited years:**\n")
        lines.append("| Year | Count |")
        lines.append("|---|---|")
        for yr, ct in top:
            lines.append(f"| {yr} | {ct:,} |")
    return "\n".join(lines)


def section_reports() -> str:
    files = [
        REPORTS / "chunk_clusters.csv",
        REPORTS / "cluster_report.txt",
        REPORTS / "cluster_summaries.json",
        REPORTS / "self_citations.csv",
        REPORTS / "self_citation_report.txt",
        REPORTS / "cluster_citation_scatter.png",
        REPORTS / "global_citation_heatmap.png",
        ROOT / "vector_store" / "query_history.json",
    ]
    lines = ["## Report Files\n"]
    lines.append("| File | Status |")
    lines.append("|---|---|")
    for p in files:
        rel = p.relative_to(ROOT)
        lines.append(f"| `{rel}` | {file_stamp(p)} |")
    return "\n".join(lines)


def section_pending() -> str:
    """
    This section is intentionally static text — edit it manually
    as work items are completed or added.
    """
    # Try to preserve any existing pending section from a prior run
    if OUT.exists():
        existing = OUT.read_text()
        m = re.search(r"(## Open Work Items.*?)(?:\n## |\Z)", existing, re.DOTALL)
        if m:
            return m.group(1).rstrip()

    # Default first-run content
    return """## Open Work Items

### Pending ❌
- **Step 4: Cross-cluster citation matrix** — match `cited_year` back to the cluster
  of that year's chapters; requires building a year → cluster(s) lookup

### Future enhancements
- `chunk_token_count` field — currently 0 for all chunks; needed for tighter context packing
- Batch query script — `query_library_batch.py` for running many questions from a CSV

### Known issues / watch points
- Cluster 12 (bibliography noise) still produces 150 citation pairs — some bleed-through
  from mixed chunks; exclude from content analysis but note the anomaly"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    sections = [
        f"# Elibrary Project — Session Status\n\n"
        f"**Generated:** {now}  \n"
        f"**Workspace:** `/Users/GStahl2/AI/elibrary/`  \n"
        f"**Library:** 22 books · 337 chapters · 36,089 chunks  \n\n"
        f"*Auto-generated by `scripts/generate_project_status.py`. "
        f"Edit the Open Work Items section manually. "
        f"Attach this file in Copilot Chat with `#file` to restore full session context.*",

        section_pipeline(),
        section_clusters(),
        section_citations(),
        section_reports(),
        section_pending(),
    ]

    output = "\n\n---\n\n".join(sections) + "\n"
    OUT.write_text(output)
    print(f"Saved: {OUT}")
    print(f"  {OUT.stat().st_size // 1024} KB  —  attach with #file in Copilot Chat")


if __name__ == "__main__":
    main()
