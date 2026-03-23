#!/usr/bin/env python3
"""
scripts/visualize_cluster_timelines.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Plot publication timelines for each semantic cluster:
  - Panel 1: raw chapter count per cluster per year
  - Panel 2: each cluster as share (%) of that year's total chapter output

Unit of analysis: one chapter = whichever cluster holds the plurality
of its level-0 chunks (primary cluster assignment).

Excludes: clusters 6, 12 (bibliography noise) and 8, 9, 15 (non-CSCL).

Outputs:
  reports/cluster_timelines.png   — 2-panel figure, saved at 150 dpi

Usage:
    .venv/bin/python scripts/visualize_cluster_timelines.py

Author: Gerry Stahl / Copilot
Created: March 2026
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT    = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

EXCLUDE_CLUSTERS = {6, 12}  # bibliography noise only; non-CSCL clusters included

# Colour palette — 15 distinct colours, roughly perceptually spaced
PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#aec7e8",
    "#ffbb78", "#98df8a", "#c5b0d5", "#c49c94", "#f7b6d2",
]

# Short labels for the legend (keep under ~35 chars)
SHORT_LABELS = {
    0:  "Heidegger / Philosophy",
    1:  "Group Processes / KB",
    2:  "Group Cognition / DG",
    3:  "Threading / Chat",
    4:  "VMT Project",
    5:  "Perspectives / KCS",
    6:  "Joint Problem Space",     # excluded
    7:  "Group Discourse / Math",
    8:  "Electronic Music",        # excluded
    9:  "Salt Marsh",              # excluded
    10: "Tacit Knowledge / Design",
    11: "Group Cognition / CSCL",
    12: "GC / KB (noise)",         # excluded
    13: "DG Construction",
    14: "HERMES Language",
    15: "Sculpture",               # excluded
    16: "Collab Problem-Solving",
    17: "CSCL Foundations",
    18: "Marx / Heidegger",
    19: "Math Meaning-Making",
}


def load_primary_clusters() -> dict[tuple[int, int], tuple[int, int]]:
    """
    Returns {(book, chapter): (primary_cluster_id, pub_year)}
    Primary cluster = plurality of level-0 chunks for that chapter.
    """
    # cluster vote counts per chapter
    votes: dict[tuple[int, int], dict[int, int]] = defaultdict(lambda: defaultdict(int))
    years: dict[tuple[int, int], list[int]] = defaultdict(list)

    with open(REPORTS / "chunk_clusters.csv") as f:
        for row in csv.DictReader(f):
            bk  = int(row["book_number"])
            ch  = int(row["chapter_number"])
            cid = int(row["cluster_id"])
            votes[(bk, ch)][cid] += 1
            try:
                years[(bk, ch)].append(int(row["pub_year"]))
            except (ValueError, KeyError):
                pass

    result = {}
    for bc, vote_map in votes.items():
        primary = max(vote_map, key=lambda c: vote_map[c])
        year_list = years[bc]
        pub_year = max(set(year_list), key=year_list.count) if year_list else 0
        result[bc] = (primary, pub_year)
    return result


def main() -> None:
    chapter_assignments = load_primary_clusters()

    # Aggregate: cluster_id -> {year -> chapter_count}
    counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    year_totals: dict[int, int] = defaultdict(int)

    for (bk, ch), (cid, year) in chapter_assignments.items():
        if year < 1970 or year > 2030:
            continue
        counts[cid][year] += 1
        year_totals[year] += 1

    active_clusters = sorted(c for c in counts if c not in EXCLUDE_CLUSTERS)
    all_years = sorted(y for y in year_totals if year_totals[y] > 0)

    print(f"Active clusters (excl. noise/non-CSCL): {active_clusters}")
    print(f"Year range: {all_years[0]}–{all_years[-1]}")
    print(f"Total chapters assigned: {sum(year_totals.values())}")

    years_arr = np.array(all_years)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 12), sharex=True,
        gridspec_kw={"hspace": 0.12},
    )

    for i, cid in enumerate(active_clusters):
        colour = PALETTE[i % len(PALETTE)]
        label  = SHORT_LABELS.get(cid, f"Cluster {cid}")
        raw    = np.array([counts[cid].get(y, 0) for y in all_years], dtype=float)
        share  = np.array(
            [100 * counts[cid].get(y, 0) / year_totals[y] if year_totals[y] else 0
             for y in all_years],
            dtype=float,
        )

        ax1.plot(years_arr, raw,   color=colour, linewidth=1.6,
                 marker="o", markersize=3, alpha=0.85, label=label)
        ax2.plot(years_arr, share, color=colour, linewidth=1.6,
                 marker="o", markersize=3, alpha=0.85, label=label)

    # ── Styling ───────────────────────────────────────────────────────────────
    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
        ax.grid(axis="x", color="#eeeeee", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(all_years[0] - 1, all_years[-1] + 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax1.set_ylabel("Chapters per year", fontsize=11)
    ax1.set_title(
        "Cluster publication timelines — raw chapter counts per year\n"
        "(primary cluster = plurality cluster; bibliography-noise clusters 6 & 12 excluded)",
        fontsize=11, pad=8,
    )
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))

    ax2.set_xlabel("Publication year", fontsize=11)
    ax2.set_ylabel("% of that year's chapters", fontsize=11)
    ax2.set_title(
        "Cluster share of annual output — relative emphasis over time",
        fontsize=11, pad=8,
    )
    ax2.set_ylim(0, None)

    # Single shared legend to the right of both panels
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        title="Cluster",
        title_fontsize=9,
    )

    plt.tight_layout(rect=[0, 0, 0.82, 1])

    out = REPORTS / "cluster_timelines.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
