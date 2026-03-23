"""
scripts/extract_self_citations.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Step 2 of the self-citation timeline pipeline.

For every level-0 chunk in chunk_clusters.csv, extract inline self-citations
(i.e., references to Stahl in APA body-text citation format) and record:
  - the citing chunk's cluster and publication year
  - the cited year

Inline citation patterns matched (APA format):
  (Stahl, 2006)            parenthetical solo
  (Stahl, 2006a)           with letter suffix
  (Stahl & X, 2006)        co-authored, Stahl first
  (X & Stahl, 2006)        co-authored, Stahl second  ← via broad scan
  (Stahl et al., 2006)     multi-author
  Stahl (2006)             narrative citation
  See Stahl (2006)         narrative with prefix

Reference list entries are naturally excluded: they take the form
"Stahl, G. (2006). Title…" — the initials appear between the name and
the year, so none of the patterns below match.

Output files:
  reports/self_citations.csv          one row per (chunk, cited_year) pair
  reports/self_citation_report.txt    summary + per-cluster timelines

Usage:
    python scripts/extract_self_citations.py

Author  : Gerry Stahl
Created : March 2026
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Config ─────────────────────────────────────────────────────────────────────
YEAR_MIN      = 1970    # earliest plausible Stahl publication
YEAR_MAX      = 2030    # latest plausible

CLUSTERS_CSV  = ROOT / "reports" / "chunk_clusters.csv"
CHUNK_SIDECAR = ROOT / "vector_store" / "chunks_bm25" / "sidecar.json"
OUT_CSV       = ROOT / "reports" / "self_citations.csv"
OUT_REPORT    = ROOT / "reports" / "self_citation_report.txt"

ROOT.joinpath("reports").mkdir(exist_ok=True)

# ── Citation patterns ──────────────────────────────────────────────────────────
# Matches: Stahl, YYYY  /  Stahl & X, YYYY  /  Stahl et al., YYYY
# Does NOT match: Stahl, G. (YYYY) — initials appear before the year in ref lists
_PAT_STAHL_FIRST = re.compile(
    r'\bStahl(?:\s+et\s+al\.?|\s+&\s+[^,()]{1,40})?,\s*(\d{4}[a-z]?)\b'
)

# Matches: narrative "Stahl (YYYY)" including page variants "Stahl (2006, p. 5)"
_PAT_NARRATIVE = re.compile(r'\bStahl\s+\((\d{4}[a-z]?)\b')

# Broad scan: any parenthetical containing "Stahl" — captures ALL years inside,
# then we attribute only the year immediately following a comma+Stahl pattern.
# Used to catch "X & Stahl, YYYY" where Stahl appears after another author.
_PAT_PAREN_BROAD = re.compile(r'\(([^()]*\bStahl\b[^()]*)\)')


def extract_stahl_years(text: str) -> list[tuple[int, str]]:
    """
    Return a list of (cited_year, snippet) for every inline Stahl citation
    found in *text*. Reference list entries are not matched.
    """
    results: list[tuple[int, str]] = []
    seen: set[tuple[int, int]] = set()   # (match_start, year) dedup

    def _add(m_start: int, yr_str: str, text: str) -> None:
        try:
            yr = int(yr_str[:4])
        except ValueError:
            return
        if not (YEAR_MIN <= yr <= YEAR_MAX):
            return
        key = (m_start, yr)
        if key in seen:
            return
        seen.add(key)
        s = max(0, m_start - 80)
        e = min(len(text), m_start + 80)
        snippet = text[s:e].replace('\n', ' ').strip()
        results.append((yr, snippet))

    # Pattern A: Stahl first / Stahl et al. / Stahl & X
    for m in _PAT_STAHL_FIRST.finditer(text):
        _add(m.start(), m.group(1), text)

    # Pattern B: narrative Stahl (YYYY)
    for m in _PAT_NARRATIVE.finditer(text):
        _add(m.start(), m.group(1), text)

    # Pattern C: X & Stahl, YYYY  (Stahl as second or later author)
    # Find all parentheticals containing Stahl; within each, locate "Stahl, YYYY"
    for m in _PAT_PAREN_BROAD.finditer(text):
        content = m.group(1)
        # Find ", YYYY" that immediately follows "Stahl" (possibly via & chain)
        for inner in re.finditer(r'\bStahl[^,()]{0,40},\s*(\d{4}[a-z]?)\b', content):
            _add(m.start() + inner.start(), inner.group(1), text)

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:

    # Load cluster assignments
    print("Loading chunk cluster assignments…", flush=True)
    clusters: dict[int, dict] = {}
    with open(CLUSTERS_CSV) as f:
        for row in csv.DictReader(f):
            clusters[int(row["vector_id"])] = row
    print(f"  {len(clusters):,} level-0 chunks")

    # Load chunk sidecar for text
    print("Loading chunk sidecar…", flush=True)
    with open(CHUNK_SIDECAR) as f:
        sidecar: dict[str, dict] = json.load(f)
    vid_to_text: dict[int, str] = {
        int(e["vector_id"]): e.get("chunk_text", "")
        for e in sidecar.values()
        if e.get("vector_id") is not None
    }

    # Extract self-citations
    print("Extracting inline Stahl self-citations…", flush=True)
    citation_rows: list[dict] = []
    chunks_with_citations = 0

    for vid, meta in clusters.items():
        text = vid_to_text.get(vid, "")
        if not text:
            continue

        hits = extract_stahl_years(text)
        if not hits:
            continue

        chunks_with_citations += 1
        pub_year_raw = meta.get("pub_year", "")
        pub_year = int(pub_year_raw) if pub_year_raw and pub_year_raw.isdigit() else None

        for cited_year, snippet in hits:
            citation_rows.append({
                "vector_id":      vid,
                "book_number":    meta["book_number"],
                "chapter_number": meta["chapter_number"],
                "book_title":     meta["book_title"][:50],
                "chapter_title":  meta["chapter_title"][:60],
                "cluster_id":     meta["cluster_id"],
                "cluster_label":  meta["cluster_label"],
                "pub_year":       pub_year,
                "cited_year":     cited_year,
                "snippet":        snippet[:220],
            })

    print(f"  Chunks with ≥1 self-citation: {chunks_with_citations:,}")
    print(f"  Total (chunk, cited_year) pairs: {len(citation_rows):,}")

    # Save CSV
    if citation_rows:
        fieldnames = list(citation_rows[0].keys())
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(citation_rows)
        print(f"  Saved: {OUT_CSV}")

    # ── Report ─────────────────────────────────────────────────────────────────
    print("\nWriting report…", flush=True)
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("STAHL SELF-CITATION EXTRACTION REPORT")
    lines.append("=" * 70)
    lines.append(f"\nLevel-0 chunks processed:         {len(clusters):>6,}")
    lines.append(f"Chunks with ≥1 self-citation:     {chunks_with_citations:>6,}")
    lines.append(f"Total (chunk, cited_year) pairs:  {len(citation_rows):>6,}")

    if not citation_rows:
        lines.append("\nNo self-citations found.")
        _write_report(lines, citation_rows)
        return

    # Overall top cited years
    all_cited = Counter(r["cited_year"] for r in citation_rows)
    lines.append("\n")
    lines.append("── TOP CITED YEARS (all clusters) ─────────────────────────────")
    max_count = max(all_cited.values())
    for yr, cnt in sorted(all_cited.items(), key=lambda x: x[1], reverse=True)[:25]:
        bar_len = max(1, round(cnt / max_count * 35))
        bar = "█" * bar_len
        lines.append(f"  {yr}: {cnt:>4}  {bar}")

    # Per-cluster summary table
    by_cluster: dict[str, list[dict]] = defaultdict(list)
    for r in citation_rows:
        by_cluster[r["cluster_id"]].append(r)

    lines.append("\n")
    lines.append("── PER-CLUSTER SELF-CITATION SUMMARY ──────────────────────────")
    header = f"  {'ID':>3}  {'Cluster label':<38}  {'Cits':>5}  {'Citing':>9}  {'Cited':>9}"
    lines.append(header)
    lines.append("  " + "-" * 68)

    cluster_order = sorted(by_cluster.keys(),
                           key=lambda c: len(by_cluster[c]), reverse=True)
    for cid in cluster_order:
        rows = by_cluster[cid]
        label = rows[0]["cluster_label"][:37]
        n = len(rows)
        pub_yrs  = [r["pub_year"]   for r in rows if r["pub_year"]]
        cite_yrs = [r["cited_year"] for r in rows]
        py_rng = f"{min(pub_yrs)}–{max(pub_yrs)}" if pub_yrs else "?"
        cy_rng = f"{min(cite_yrs)}–{max(cite_yrs)}" if cite_yrs else "?"
        lines.append(f"  {cid:>3}  {label:<38}  {n:>5}  {py_rng:>9}  {cy_rng:>9}")

    # Per-cluster timelines
    lines.append("\n")
    lines.append("=" * 70)
    lines.append("SELF-CITATION TIMELINES BY CLUSTER")
    lines.append("(citing year → cited years, count in parens, sorted by frequency)")
    lines.append("=" * 70)

    for cid in cluster_order:
        rows = by_cluster[cid]
        label = rows[0]["cluster_label"]
        lines.append(f"\n── Cluster {cid}: {label}")

        # Group by pub_year → Counter of cited years
        by_pub: dict[int, Counter] = defaultdict(Counter)
        no_year = Counter()
        for r in rows:
            if r["pub_year"]:
                by_pub[r["pub_year"]][r["cited_year"]] += 1
            else:
                no_year[r["cited_year"]] += 1

        for pub_yr in sorted(by_pub.keys()):
            cited_counts = by_pub[pub_yr]
            top = ", ".join(
                f"{yr}({cnt})" for yr, cnt in
                sorted(cited_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            )
            total = sum(cited_counts.values())
            lines.append(f"   {pub_yr} [{total:>3} cits]: {top}")

        if no_year:
            top = ", ".join(
                f"{yr}({cnt})" for yr, cnt in
                sorted(no_year.items(), key=lambda x: x[1], reverse=True)[:8]
            )
            lines.append(f"   ???? [no pub year]: {top}")

    lines.append("\n" + "=" * 70)

    _write_report(lines, citation_rows)


def _write_report(lines: list[str], citation_rows: list[dict]) -> None:
    report_text = "\n".join(lines)
    print("\n" + report_text)
    with open(OUT_REPORT, "w") as f:
        f.write(report_text)
    print(f"\nSaved: {OUT_REPORT}")
    if citation_rows:
        print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
