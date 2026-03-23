#!/usr/bin/env python3
"""
Visualize Stahl self-citation patterns.

Outputs
-------
reports/cluster_citation_scatter.png  – 20-panel scatter grid
                                        x = citing year (pub_year of chapter)
                                        y = cited year
                                        bubble size ∝ frequency
reports/global_citation_heatmap.png   – full heatmap
                                        rows = cited year, cols = citing year
                                        colour = log(1 + count)
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent.parent
REPORTS   = WORKSPACE / "reports"
CSV       = REPORTS / "self_citations.csv"

# ── Cluster metadata ────────────────────────────────────────────────────────
NOISE_CLUSTERS  = {6, 12}   # bibliography / reference-list artefacts
NON_CSCL        = {8, 9, 15, 18}  # off-topic clusters

CLUSTER_COLORS = {
    "noise":    "#e07b7b",   # muted red
    "non_cscl": "#a8c8e8",  # light blue
    "cscl":     "#2a6496",  # deep blue
}

# ── Data loading ────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["pub_year"]   = pd.to_numeric(df["pub_year"],   errors="coerce")
    df["cited_year"] = pd.to_numeric(df["cited_year"], errors="coerce")
    df = df.dropna(subset=["pub_year", "cited_year"])
    df["pub_year"]   = df["pub_year"].astype(int)
    df["cited_year"] = df["cited_year"].astype(int)
    return df


# ── Plot 1: per-cluster scatter grid ───────────────────────────────────────
def plot_cluster_scatter(df: pd.DataFrame) -> None:
    # Order clusters by total citation count descending
    order = (
        df.groupby(["cluster_id", "cluster_label"])
          .size()
          .reset_index(name="n")
          .sort_values("n", ascending=False)
    )

    NCOLS, NROWS = 5, 4            # 20 panels for 20 clusters
    fig, axes = plt.subplots(
        NROWS, NCOLS,
        figsize=(NCOLS * 3.8, NROWS * 3.4),
        constrained_layout=True,
    )
    axes = axes.flatten()

    for idx, row in enumerate(order.itertuples()):
        ax  = axes[idx]
        cid = row.cluster_id

        sub    = df[df["cluster_id"] == cid]
        counts = (
            sub.groupby(["pub_year", "cited_year"])
               .size()
               .reset_index(name="count")
        )

        # Bubble size: scale so max bubble = 320 pts²
        max_count  = counts["count"].max() if len(counts) else 1
        sizes      = (counts["count"] / max_count * 320).clip(lower=14)

        # Colour by cluster type
        if cid in NOISE_CLUSTERS:
            color = CLUSTER_COLORS["noise"]
            marker = "D"
        elif cid in NON_CSCL:
            color = CLUSTER_COLORS["non_cscl"]
            marker = "s"
        else:
            color = CLUSTER_COLORS["cscl"]
            marker = "o"

        ax.scatter(
            counts["pub_year"], counts["cited_year"],
            s=sizes, c=color, alpha=0.75,
            edgecolors="white", linewidths=0.4,
            marker=marker, zorder=3,
        )

        # Diagonal reference line  y = x  (cited in same year as published)
        if len(counts):
            lo = min(counts["pub_year"].min(), counts["cited_year"].min()) - 2
            hi = max(counts["pub_year"].max(), counts["cited_year"].max()) + 2
            ax.plot([lo, hi], [lo, hi], color="#999", lw=0.7, ls="--", zorder=1)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

        # Labels
        label = row.cluster_label
        short = label[:32] + ("…" if len(label) > 32 else "")
        note  = " ⚠" if cid in NOISE_CLUSTERS else ""
        ax.set_title(
            f"C{cid}: {short}{note}",
            fontsize=6.5, fontweight="bold", pad=3,
        )
        ax.set_xlabel("Citing year", fontsize=6.5, labelpad=2)
        ax.set_ylabel("Cited year",  fontsize=6.5, labelpad=2)
        ax.tick_params(labelsize=6)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=4))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=4))
        ax.grid(True, alpha=0.25, linewidth=0.5, zorder=0)

        # Citation count badge in corner
        ax.text(
            0.97, 0.03, f"n={row.n}",
            transform=ax.transAxes, fontsize=5.5,
            ha="right", va="bottom", color="#555",
        )

    # Hide any spare panels
    for idx in range(len(order), len(axes)):
        axes[idx].set_visible(False)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLUSTER_COLORS["cscl"],
               markersize=7, label="CSCL cluster"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=CLUSTER_COLORS["non_cscl"],
               markersize=7, label="Non-CSCL cluster"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=CLUSTER_COLORS["noise"],
               markersize=7, label="Bibliography noise ⚠"),
        Line2D([0], [0], color="#999", lw=0.8, ls="--", label="y = x (same year)"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=4, fontsize=7, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.015),
    )

    fig.suptitle(
        "Stahl Self-Citation Patterns by Cluster\n"
        "x = citing year  ·  y = cited year  ·  bubble size ∝ frequency  ·  sorted by citation count",
        fontsize=10, fontweight="bold",
    )

    out = REPORTS / "cluster_citation_scatter.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Plot 2: global heatmap ──────────────────────────────────────────────────
def plot_global_heatmap(df: pd.DataFrame) -> None:
    # Aggregate
    agg = (
        df.groupby(["pub_year", "cited_year"])
          .size()
          .reset_index(name="count")
    )

    # Build pivot: rows = cited_year (y), cols = pub_year (x)
    matrix = agg.pivot_table(
        index="cited_year", columns="pub_year",
        values="count", fill_value=0,
    )
    matrix = matrix.sort_index(ascending=True)       # cited_year ascending top→bottom
    matrix = matrix[sorted(matrix.columns)]           # pub_year ascending left→right

    raw   = matrix.values.astype(float)
    log_v = np.log1p(raw)

    # ── Figure layout: main heatmap + marginal bar plots ──────────────────
    fig = plt.figure(figsize=(16, 11))
    gs  = fig.add_gridspec(
        2, 2,
        width_ratios=[15, 2],
        height_ratios=[2, 12],
        hspace=0.05, wspace=0.05,
    )
    ax_heat  = fig.add_subplot(gs[1, 0])   # main heatmap
    ax_top   = fig.add_subplot(gs[0, 0])   # marginal: citations per citing year
    ax_right = fig.add_subplot(gs[1, 1])   # marginal: citations per cited year

    # ── Heatmap ────────────────────────────────────────────────────────────
    im = ax_heat.imshow(
        log_v,
        aspect="auto",
        cmap="YlOrRd",
        origin="lower",           # cited_year ascending bottom→top
        interpolation="nearest",
    )

    # Tick every year but only label some to avoid crowding
    x_years = list(matrix.columns)
    y_years = list(matrix.index)

    ax_heat.set_xticks(range(len(x_years)))
    ax_heat.set_yticks(range(len(y_years)))

    # Label every 5 years on x, every 5 on y
    x_labels = [str(y) if y % 5 == 0 else "" for y in x_years]
    y_labels = [str(y) if y % 5 == 0 else "" for y in y_years]
    ax_heat.set_xticklabels(x_labels, rotation=90, fontsize=7)
    ax_heat.set_yticklabels(y_labels, fontsize=7)

    ax_heat.set_xlabel("Citing year  (pub_year of chapter)", fontsize=9, labelpad=6)
    ax_heat.set_ylabel("Cited year", fontsize=9, labelpad=6)

    # Annotate cells with raw count ≥ 20
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            val = int(raw[i, j])
            if val >= 20:
                ax_heat.text(
                    j, i, str(val),
                    ha="center", va="center",
                    fontsize=4.5, color="black", fontweight="bold",
                )

    # Diagonal reference (cited_year == pub_year)
    if x_years and y_years:
        year_min = max(min(x_years), min(y_years))
        year_max = min(max(x_years), max(y_years))
        diag_x = [x_years.index(y) for y in x_years if year_min <= y <= year_max]
        diag_y = [y_years.index(y) for y in x_years if year_min <= y <= year_max]
        ax_heat.plot(diag_x, diag_y, color="white", lw=0.8, ls="--", alpha=0.6, label="y = x")

    # Colourbar
    cbar = fig.colorbar(im, ax=ax_heat, orientation="vertical", shrink=0.8, pad=0.01)
    cbar.set_label("log(1 + count)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ── Top marginal: total citations per citing year ──────────────────────
    col_sums = raw.sum(axis=0)   # sum over cited_years → per citing year
    ax_top.bar(range(len(x_years)), col_sums, color="#2a6496", alpha=0.8)
    ax_top.set_xlim(-0.5, len(x_years) - 0.5)
    ax_top.set_xticks([])
    ax_top.set_ylabel("Total\ncitations", fontsize=7, labelpad=4)
    ax_top.tick_params(labelsize=6)
    ax_top.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax_top.set_title(
        "Global Stahl Self-Citation Heatmap  ·  rows = cited year · cols = citing year · log colour scale",
        fontsize=10, fontweight="bold", pad=6,
    )

    # ── Right marginal: total citations per cited year ─────────────────────
    row_sums = raw.sum(axis=1)   # sum over citing_years → per cited year
    ax_right.barh(range(len(y_years)), row_sums, color="#c0392b", alpha=0.8)
    ax_right.set_ylim(-0.5, len(y_years) - 0.5)
    ax_right.set_yticks([])
    ax_right.set_xlabel("Total\ncitations", fontsize=7, labelpad=4)
    ax_right.tick_params(labelsize=6)
    ax_right.grid(axis="x", alpha=0.3, linewidth=0.5)

    out = REPORTS / "global_citation_heatmap.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Entry point ─────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading self_citations.csv…")
    df = load_data()
    print(f"  {len(df):,} citation pairs")

    print("Plotting per-cluster scatter grid…")
    plot_cluster_scatter(df)

    print("Plotting global heatmap…")
    plot_global_heatmap(df)

    print("Done.")


if __name__ == "__main__":
    main()
