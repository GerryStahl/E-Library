#!/usr/bin/env python3
"""
scripts/visualize_chunk_timelines.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Plot publication timelines for each semantic cluster using level-0 chunks
as the unit of analysis (rather than whole chapters).

Because each chunk has its own cluster assignment, this reveals the
distribution of *topics within* chapters — e.g. how many Heidegger-
themed chunks appear in a given year's output, even if those chunks are
embedded inside primarily CSCL chapters.

Excludes: clusters 6, 12 (bibliography/reference-list noise).
Includes: all other clusters including non-CSCL (8, 9, 15).

Outputs:
  reports/chunk_timelines.png   — 2-panel figure saved at 150 dpi

Usage:
    .venv/bin/python scripts/visualize_chunk_timelines.py

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

EXCLUDE_CLUSTERS = {6, 12}  # bibliography noise only

PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#aec7e8",
    "#ffbb78", "#98df8a", "#c5b0d5", "#c49c94", "#f7b6d2",
    "#7f7f7f", "#dbdb8d", "#9edae5",
]

SHORT_LABELS = {
    0:  "Heidegger / Philosophy",
    1:  "Group Processes / KB",
    2:  "Group Cognition / DG",
    3:  "Threading / Chat",
    4:  "VMT Project",
    5:  "Perspectives / KCS",
    7:  "Group Discourse / Math",
    8:  "Electronic Music",
    9:  "Salt Marsh",
    10: "Tacit Knowledge / Design",
    11: "Group Cognition / CSCL",
    13: "DG Construction",
    14: "HERMES Language",
    15: "Sculpture",
    16: "Collab Problem-Solving",
    17: "CSCL Foundations",
    18: "Marx / Heidegger",
    19: "Math Meaning-Making",
}


def main() -> None:
    # Count chunks per cluster per year directly — no aggregation needed
    counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    year_totals: dict[int, int] = defaultdict(int)

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
            year_totals[year] += 1

    active_clusters = sorted(counts.keys())
    all_years = sorted(y for y in year_totals if year_totals[y] > 0)
    years_arr = np.array(all_years)

    print(f"Active clusters: {active_clusters}")
    print(f"Year range: {all_years[0]}–{all_years[-1]}")
    print(f"Total level-0 chunks: {sum(year_totals.values()):,}")

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

    ax1.set_ylabel("Level-0 chunks per year", fontsize=11)
    ax1.set_title(
        "Cluster timelines — level-0 chunk counts per year\n"
        "(each chunk assigned to its own cluster; bibliography-noise clusters 6 & 12 excluded)",
        fontsize=11, pad=8,
    )
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))

    ax2.set_xlabel("Publication year", fontsize=11)
    ax2.set_ylabel("% of that year's chunks", fontsize=11)
    ax2.set_title(
        "Cluster share of annual chunk output — thematic emphasis over time",
        fontsize=11, pad=8,
    )
    ax2.set_ylim(0, None)

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

    out = REPORTS / "chunk_timelines.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
