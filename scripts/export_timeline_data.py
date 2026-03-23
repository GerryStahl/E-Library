#!/usr/bin/env python3
"""
scripts/export_timeline_data.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Export the data underlying cluster_timelines.png and chunk_timelines.png
to a single CSV for inspection in a spreadsheet.

Columns:
  year
  For each cluster (chapters): ch_<id>_count, ch_<id>_pct
  For each cluster (chunks):   ck_<id>_count, ck_<id>_pct
  total_chapters, total_chunks

Output:
  reports/timeline_data.csv

Usage:
    .venv/bin/python scripts/export_timeline_data.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

EXCLUDE_CLUSTERS = {6, 12}

SHORT_LABELS = {
    0: "Heidegger/Philosophy", 1: "Group Processes/KB", 2: "Group Cognition/DG",
    3: "Threading/Chat", 4: "VMT Project", 5: "Perspectives/KCS",
    7: "Group Discourse/Math", 8: "Electronic Music", 9: "Salt Marsh",
    10: "Tacit Knowledge/Design", 11: "Group Cognition/CSCL", 13: "DG Construction",
    14: "HERMES Language", 15: "Sculpture", 16: "Collab Problem-Solving",
    17: "CSCL Foundations", 18: "Marx/Heidegger", 19: "Math Meaning-Making",
}


def load_chapter_counts() -> tuple[dict[int, dict[int, int]], dict[int, int]]:
    """Primary-cluster chapter counts: {cluster: {year: n}}, year totals."""
    votes: dict[tuple[int,int], dict[int,int]] = defaultdict(lambda: defaultdict(int))
    years: dict[tuple[int,int], list[int]] = defaultdict(list)
    with open(REPORTS / "chunk_clusters.csv") as f:
        for row in csv.DictReader(f):
            bk = int(row["book_number"]); ch = int(row["chapter_number"])
            cid = int(row["cluster_id"])
            votes[(bk, ch)][cid] += 1
            try:
                years[(bk, ch)].append(int(row["pub_year"]))
            except (ValueError, KeyError):
                pass
    counts: dict[int, dict[int,int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[int,int] = defaultdict(int)
    for bc, vote_map in votes.items():
        primary = max(vote_map, key=lambda c: vote_map[c])
        if primary in EXCLUDE_CLUSTERS:
            continue
        yl = years[bc]
        year = max(set(yl), key=yl.count) if yl else 0
        if year < 1970 or year > 2030:
            continue
        counts[primary][year] += 1
        totals[year] += 1
    return counts, totals


def load_chunk_counts() -> tuple[dict[int, dict[int, int]], dict[int, int]]:
    """Direct chunk counts: {cluster: {year: n}}, year totals."""
    counts: dict[int, dict[int,int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[int,int] = defaultdict(int)
    with open(REPORTS / "chunk_clusters.csv") as f:
        for row in csv.DictReader(f):
            cid = int(row["cluster_id"])
            if cid in EXCLUDE_CLUSTERS:
                continue
            try:
                year = int(row["pub_year"])
            except (ValueError, KeyError):
                continue
            if year < 1970 or year > 2030:
                continue
            counts[cid][year] += 1
            totals[year] += 1
    return counts, totals


def main() -> None:
    ch_counts, ch_totals = load_chapter_counts()
    ck_counts, ck_totals = load_chunk_counts()

    all_years = sorted(set(ch_totals) | set(ck_totals))
    active = sorted(SHORT_LABELS.keys())

    out = REPORTS / "timeline_data.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)

        # Header row 1: group labels
        header1 = ["year"]
        for cid in active:
            lbl = SHORT_LABELS[cid]
            header1 += [f"CHAPTERS — cluster {cid}: {lbl}", ""]
        header1 += ["", ""]
        for cid in active:
            lbl = SHORT_LABELS[cid]
            header1 += [f"CHUNKS — cluster {cid}: {lbl}", ""]
        header1 += ["", ""]
        writer.writerow(header1)

        # Header row 2: field names
        header2 = ["year"]
        for cid in active:
            header2 += [f"ch_{cid}_count", f"ch_{cid}_pct"]
        header2 += ["total_chapters", ""]
        for cid in active:
            header2 += [f"ck_{cid}_count", f"ck_{cid}_pct"]
        header2 += ["total_chunks", ""]
        writer.writerow(header2)

        # Data rows
        for year in all_years:
            row = [year]
            ch_tot = ch_totals.get(year, 0)
            ck_tot = ck_totals.get(year, 0)
            for cid in active:
                n = ch_counts[cid].get(year, 0)
                pct = round(100 * n / ch_tot, 1) if ch_tot else 0.0
                row += [n, pct]
            row += [ch_tot, ""]
            for cid in active:
                n = ck_counts[cid].get(year, 0)
                pct = round(100 * n / ck_tot, 1) if ck_tot else 0.0
                row += [n, pct]
            row += [ck_tot, ""]
            writer.writerow(row)

    print(f"Saved: {out}")
    print(f"  {len(all_years)} year rows × {2 + 2*len(active)*2} columns")


if __name__ == "__main__":
    main()
