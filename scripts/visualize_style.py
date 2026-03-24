"""Visualise style features over time and by cluster.

Outputs
-------
reports/style_timeseries.png  – 6-panel scatter + rolling-mean time series
reports/style_by_cluster.png  – 6-panel boxplots grouped by cluster
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV  = ROOT / "reports" / "style_features.csv"

# bibliography-noise clusters: exclude from cluster comparison
EXCLUDE_CLUSTERS: set[int] = {6, 12}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_df() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df = df[df["pub_year"].notna()].copy()
    df["pub_year"] = df["pub_year"].astype(int)
    # cap extreme hedge/assert ratios (near-zero denominator artefacts)
    df["hedge_assert_ratio"] = df["hedge_assert_ratio"].clip(upper=10.0)
    return df


def year_rolling(df: pd.DataFrame, col: str, window: int = 5) -> pd.Series:
    """Compute per-year mean, then a centred rolling mean over *window* years."""
    ym = df.groupby("pub_year")[col].mean().sort_index()
    return ym.rolling(window, center=True, min_periods=2).mean()


def short_label(label: str, max_len: int = 24) -> str:
    return label[:max_len] + "…" if len(label) > max_len else label


# ---------------------------------------------------------------------------
# figure 1 – time series
# ---------------------------------------------------------------------------

TS_PANELS: list[tuple[str, str]] = [
    ("sent_mean_words",    "Mean sentence length (words)"),
    ("ttr",                "Type-token ratio"),
    ("hedge_assert_ratio", "Hedge / Assert ratio"),
    ("i_per_1k",           "1st-person singular  (per 1 k words)"),
    ("we_per_1k",          "1st-person plural  (per 1 k words)"),
    ("citation_per_1k",    "Citation density  (per 1 k words)"),
]


def plot_timeseries(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle("Style features over time  (1967 – 2026)", fontsize=14)

    for ax, (col, label) in zip(axes.flat, TS_PANELS):
        ax.scatter(df["pub_year"], df[col],
                   alpha=0.22, s=14, color="#4878d0", zorder=2, linewidths=0)

        rm = year_rolling(df, col)
        ax.plot(rm.index, rm.values, color="#e8645b", lw=2.2, zorder=3, label="5-yr rolling mean")

        ax.set_xlabel("Publication year", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=10, pad=5)
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# figure 2 – boxplots by cluster
# ---------------------------------------------------------------------------

CL_PANELS: list[tuple[str, str]] = [
    ("sent_mean_words",    "Mean sentence length (words)"),
    ("ttr",                "Type-token ratio"),
    ("hedge_assert_ratio", "Hedge / Assert ratio"),
    ("i_per_1k",           "I / me / my  (per 1 k)"),
    ("we_per_1k",          "We / our / us  (per 1 k)"),
    ("citation_per_1k",    "Citations  (per 1 k)"),
]


def plot_by_cluster(df: pd.DataFrame, out: Path) -> None:
    sub = df[~df["cluster_id"].isin(EXCLUDE_CLUSTERS)].copy()

    # order clusters by median pub_year (loosely chronological)
    order: list[int] = (
        sub.groupby("cluster_id")["pub_year"]
        .median()
        .sort_values()
        .index.tolist()
    )

    label_map: dict[int, str] = (
        sub[["cluster_id", "cluster_label"]]
        .drop_duplicates()
        .set_index("cluster_id")["cluster_label"]
        .to_dict()
    )
    tick_labels = [
        f"C{c}\n{short_label(label_map.get(c, str(c)), 20)}"
        for c in order
    ]

    fig, axes = plt.subplots(3, 2, figsize=(15, 14))
    fig.suptitle(
        "Style features by cluster  (clusters 6 & 12 excluded — bibliography noise)",
        fontsize=12,
    )

    for ax, (col, label) in zip(axes.flat, CL_PANELS):
        data = [sub.loc[sub["cluster_id"] == c, col].dropna().values for c in order]
        ax.boxplot(
            data,
            patch_artist=True,
            medianprops=dict(color="#e8645b", lw=2),
            boxprops=dict(facecolor="#cfe2f3", alpha=0.85),
            whiskerprops=dict(color="#555"),
            capprops=dict(color="#555"),
            flierprops=dict(marker=".", markersize=3, alpha=0.35, color="#888"),
        )
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(tick_labels, rotation=50, ha="right", fontsize=6.5)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=10, pad=5)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    df = load_df()
    print(f"Loaded {len(df)} chapters, {df['pub_year'].min()}–{df['pub_year'].max()}")

    plot_timeseries(df, ROOT / "reports" / "style_timeseries.png")
    plot_by_cluster(df, ROOT / "reports" / "style_by_cluster.png")
    print("Done.")


if __name__ == "__main__":
    main()
