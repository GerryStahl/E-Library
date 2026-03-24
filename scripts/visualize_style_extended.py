"""Reports 3 & 4 — extended style visualisations.

Report 3 (sentence rhythm & complexity):
  reports/style_rhythm_timeseries.png
  reports/style_rhythm_by_cluster.png

Report 4 (argumentative moves):
  reports/style_moves_timeseries.png
  reports/style_moves_by_cluster.png
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
EXCLUDE_CLUSTERS: set[int] = {6, 12}


# ---------------------------------------------------------------------------
# helpers (shared with visualize_style.py)
# ---------------------------------------------------------------------------

def load_df() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df = df[df["pub_year"].notna()].copy()
    df["pub_year"] = df["pub_year"].astype(int)
    df["hedge_assert_ratio"] = df["hedge_assert_ratio"].clip(upper=10.0)
    return df


def year_rolling(df: pd.DataFrame, col: str, window: int = 5) -> pd.Series:
    ym = df.groupby("pub_year")[col].mean().sort_index()
    return ym.rolling(window, center=True, min_periods=2).mean()


def short_label(label: str, max_len: int = 20) -> str:
    return label[:max_len] + "…" if len(label) > max_len else label


def cluster_order(df: pd.DataFrame) -> list[int]:
    sub = df[~df["cluster_id"].isin(EXCLUDE_CLUSTERS)]
    return (
        sub.groupby("cluster_id")["pub_year"]
        .median()
        .sort_values()
        .index.tolist()
    )


def cluster_tick_labels(df: pd.DataFrame, order: list[int]) -> list[str]:
    label_map: dict[int, str] = (
        df[["cluster_id", "cluster_label"]]
        .drop_duplicates()
        .set_index("cluster_id")["cluster_label"]
        .to_dict()
    )
    return [f"C{c}\n{short_label(label_map.get(c, str(c)))}" for c in order]


def make_timeseries_fig(
    df: pd.DataFrame,
    panels: list[tuple[str, str]],
    title: str,
    out: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle(title, fontsize=13)

    for ax, (col, label) in zip(axes.flat, panels):
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


def make_cluster_fig(
    df: pd.DataFrame,
    panels: list[tuple[str, str]],
    title: str,
    out: Path,
) -> None:
    sub = df[~df["cluster_id"].isin(EXCLUDE_CLUSTERS)].copy()
    order = cluster_order(df)
    tick_labels = cluster_tick_labels(df, order)

    fig, axes = plt.subplots(3, 2, figsize=(15, 14))
    fig.suptitle(title, fontsize=12)

    for ax, (col, label) in zip(axes.flat, panels):
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
# Report 3 — sentence rhythm & complexity
# ---------------------------------------------------------------------------

RHYTHM_PANELS: list[tuple[str, str]] = [
    ("sent_mean_words",  "Mean sentence length (words)"),
    ("sent_std_words",   "Sentence-length std dev (words)"),
    ("sent_pct_long",    "% sentences > 30 words"),
    ("sent_pct_short",   "% sentences < 10 words"),
    ("pct_questions",    "% interrogative sentences"),
    ("passive_per_1k",   "Passive constructions  (per 1 k)"),
]

# ---------------------------------------------------------------------------
# Report 4 — argumentative moves
# ---------------------------------------------------------------------------

MOVES_PANELS: list[tuple[str, str]] = [
    ("causal_per_1k",    "Causal connectors  (per 1 k)\ntherefore / thus / hence / consequently"),
    ("contrast_per_1k",  "Contrast connectors  (per 1 k)\nhowever / nevertheless / although / whereas"),
    ("additive_per_1k",  "Additive connectors  (per 1 k)\nfurthermore / moreover / additionally"),
    ("exemplify_per_1k", "Exemplification  (per 1 k)\nfor example / such as / e.g."),
    ("hedge_per_1k",     "Hedging  (per 1 k)\nperhaps / might / suggests / arguably"),
    ("assert_per_1k",    "Assertion  (per 1 k)\nclearly / must / demonstrates / shows"),
]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    df = load_df()
    print(f"Loaded {len(df)} chapters, {df['pub_year'].min()}–{df['pub_year'].max()}")

    print("\nReport 3 — sentence rhythm & complexity")
    make_timeseries_fig(
        df, RHYTHM_PANELS,
        "Sentence rhythm & complexity  (1967 – 2026)",
        ROOT / "reports" / "style_rhythm_timeseries.png",
    )
    make_cluster_fig(
        df, RHYTHM_PANELS,
        "Sentence rhythm & complexity by cluster  (C6 & C12 excluded)",
        ROOT / "reports" / "style_rhythm_by_cluster.png",
    )

    print("\nReport 4 — argumentative moves")
    make_timeseries_fig(
        df, MOVES_PANELS,
        "Argumentative moves  (1967 – 2026)",
        ROOT / "reports" / "style_moves_timeseries.png",
    )
    make_cluster_fig(
        df, MOVES_PANELS,
        "Argumentative moves by cluster  (C6 & C12 excluded)",
        ROOT / "reports" / "style_moves_by_cluster.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
