"""Reports 5 & 6 — new style features (readability, punctuation, vocabulary depth, register).

Report 5 — Readability & punctuation (6 panels):
  reports/style_readability_timeseries.png
  reports/style_readability_by_cluster.png

Report 6 — Vocabulary depth & academic register (6 panels):
  reports/style_register_timeseries.png
  reports/style_register_by_cluster.png
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV  = ROOT / "reports" / "style_features.csv"
EXCLUDE_CLUSTERS: set[int] = {6, 12}


# ---------------------------------------------------------------------------
# helpers (same as in other visualize_style scripts)
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
    return sub.groupby("cluster_id")["pub_year"].median().sort_values().index.tolist()


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
    sub   = df[~df["cluster_id"].isin(EXCLUDE_CLUSTERS)].copy()
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
# Report 5 — Readability & punctuation
# ---------------------------------------------------------------------------

READABILITY_PANELS: list[tuple[str, str]] = [
    ("flesch_ease",      "Flesch Reading Ease\n(higher = easier; academic prose ≈ 20–50)"),
    ("fk_grade",         "Flesch-Kincaid Grade Level\n(US school grade equivalent)"),
    ("fog_index",        "Gunning Fog Index\n(grade equiv.; academic ≈ 12–17)"),
    ("semicolon_per_1k", "Semicolons  (per 1 k words)\ncompound/complex sentence style"),
    ("emdash_per_1k",    "Em-dashes  (per 1 k words)\nparenthetical / digressive style"),
    ("colon_per_1k",     "Colons  (per 1 k words)\nlist-intro & definitional moves"),
]

# ---------------------------------------------------------------------------
# Report 6 — Vocabulary depth & academic register
# ---------------------------------------------------------------------------

REGISTER_PANELS: list[tuple[str, str]] = [
    ("hapax_ratio",            "Hapax ratio\n(fraction of tokens appearing exactly once)"),
    ("abstract_ratio",         "Abstract word ratio\n(-tion / -ity / -ness / -ment / -ism …)"),
    ("mean_word_len",          "Mean word length (characters)"),
    ("metalinguistic_per_1k",  "Metalinguistic announcements  (per 1 k)\n'this chapter argues', 'I demonstrate'"),
    ("definitional_per_1k",    "Definitional phrases  (per 1 k)\n'is defined as', 'refers to', i.e."),
    ("subordinate_per_1k",     "Subordinate conjunctions  (per 1 k)\nbecause / although / since / unless …"),
]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    df = load_df()
    print(f"Loaded {len(df)} chapters, {df['pub_year'].min()}–{df['pub_year'].max()}")

    print("\nReport 5 — Readability & punctuation")
    make_timeseries_fig(
        df, READABILITY_PANELS,
        "Readability & punctuation style  (1967 – 2026)",
        ROOT / "reports" / "style_readability_timeseries.png",
    )
    make_cluster_fig(
        df, READABILITY_PANELS,
        "Readability & punctuation by cluster  (C6 & C12 excluded)",
        ROOT / "reports" / "style_readability_by_cluster.png",
    )

    print("\nReport 6 — Vocabulary depth & academic register")
    make_timeseries_fig(
        df, REGISTER_PANELS,
        "Vocabulary depth & academic register  (1967 – 2026)",
        ROOT / "reports" / "style_register_timeseries.png",
    )
    make_cluster_fig(
        df, REGISTER_PANELS,
        "Vocabulary depth & academic register by cluster  (C6 & C12 excluded)",
        ROOT / "reports" / "style_register_by_cluster.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
