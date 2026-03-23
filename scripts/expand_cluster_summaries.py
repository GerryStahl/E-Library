#!/usr/bin/env python3
"""
scripts/expand_cluster_summaries.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Re-generate the cluster report with full paragraph summaries (3-5 sentences)
instead of the original single-sentence explanations.

Re-uses the existing chunk_clusters.csv (no re-clustering needed).
For each cluster:
  1. Reconstructs member vectors from FAISS to find the 10 closest to centroid
  2. Sends those 10 exemplar passages to Claude, asking for a full paragraph
  3. Rewrites reports/cluster_report.txt with expanded summaries

Output files (both overwritten):
  reports/cluster_report.txt    — same format, fuller summaries
  reports/cluster_summaries.json — {cluster_id: {"label": ..., "summary": ...}}

Usage:
    python scripts/expand_cluster_summaries.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import faiss
import anthropic
from sklearn.preprocessing import normalize

# ── Config ──────────────────────────────────────────────────────────────────
EXEMPLARS_N    = 10          # passages per cluster sent to Claude
EXEMPLAR_CHARS = 500         # max chars per exemplar
SUMMARY_MODEL  = "claude-haiku-4-5-20251001"
SUMMARY_TOKENS = 400         # room for 3-5 sentences

VS_DIR        = ROOT / "vector_store"
CHUNK_FAISS   = VS_DIR / "elibrary.faiss"
CHUNK_META    = VS_DIR / "elibrary_meta.json"
CHUNK_SIDECAR = VS_DIR / "chunks_bm25" / "sidecar.json"
REPORTS_DIR   = ROOT / "reports"

IN_CSV    = REPORTS_DIR / "chunk_clusters.csv"
OUT_REPORT  = REPORTS_DIR / "cluster_report.txt"
OUT_JSON    = REPORTS_DIR / "cluster_summaries.json"

GLOBAL_MEAN_DIST = 0.0972   # from original run
RANDOM_STATE     = 42
K                = 20


# ── Data loading ─────────────────────────────────────────────────────────────
def load_cluster_csv() -> dict[int, dict]:
    """Load chunk_clusters.csv → {vector_id: row_dict}."""
    rows: dict[int, dict] = {}
    with open(IN_CSV) as f:
        for r in csv.DictReader(f):
            vid = int(r["vector_id"])
            rows[vid] = {
                "book_number":    int(r["book_number"]),
                "chapter_number": int(r["chapter_number"]),
                "book_title":     r["book_title"],
                "chapter_title":  r["chapter_title"],
                "chunk_index":    int(r["chunk_index"]),
                "chunk_page":     int(r["chunk_page"]),
                "pub_year":       r["pub_year"],
                "cluster_id":     int(r["cluster_id"]),
                "cluster_label":  r["cluster_label"],
            }
    return rows


def load_sidecar() -> dict[int, dict]:
    with open(CHUNK_SIDECAR) as f:
        raw = json.load(f)
    return {
        int(e["vector_id"]): e
        for e in raw.values()
        if e.get("vector_id") is not None
    }


# ── Find exemplars ────────────────────────────────────────────────────────────
def compute_exemplars(
    cluster_vids: list[int],
    vid_to_vec: dict[int, np.ndarray],
    vid_to_sc: dict[int, dict],
    n: int = EXEMPLARS_N,
) -> list[str]:
    """Return up to n exemplar texts nearest to the cluster centroid."""
    vecs = np.array([vid_to_vec[v] for v in cluster_vids if v in vid_to_vec],
                    dtype=np.float32)
    if len(vecs) == 0:
        return []
    centroid = normalize(vecs.mean(axis=0, keepdims=True), norm="l2")[0]
    sims = vecs @ centroid                             # cosine sim (already normed)
    top_idx = np.argsort(sims)[::-1][:n]
    texts = []
    for i in top_idx:
        vid = cluster_vids[i] if i < len(cluster_vids) else None
        if vid is None:
            continue
        sc = vid_to_sc.get(vid, {})
        text = sc.get("chunk_text", "").strip()
        if text:
            texts.append(text[:EXEMPLAR_CHARS])
    return texts


# ── Claude summary ────────────────────────────────────────────────────────────
def summarise_cluster(
    client: anthropic.Anthropic,
    cluster_id: int,
    label: str,
    exemplars: list[str],
) -> str:
    """Ask Claude for a 3-5 sentence paragraph summary of the cluster."""
    passages = "\n\n---\n\n".join(
        f"Passage {i+1}:\n{e}" for i, e in enumerate(exemplars)
    )
    prompt = (
        f"The following {len(exemplars)} passages are drawn from the semantic cluster "
        f'labelled "{label}" in a 22-volume academic library by Gerry Stahl on '
        "computer-supported collaborative learning (CSCL) and related topics. "
        "These are the passages whose embedding vectors lie closest to the cluster centroid.\n\n"
        f"{passages}\n\n"
        "Write a paragraph of 3–5 sentences that:\n"
        "  1. States the core theme of this cluster.\n"
        "  2. Identifies the key concepts, methods, or artefacts that appear across the passages.\n"
        "  3. Notes any distinctive characteristics—such as a particular theoretical tradition, "
        "     a specific research project, a time period, or a type of analysis.\n"
        "  4. Explains what role this theme plays within Stahl's broader body of work.\n\n"
        "Write only the paragraph — no bullet points, no heading, no preamble."
    )
    msg = client.messages.create(
        model=SUMMARY_MODEL,
        max_tokens=SUMMARY_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


# ── Report writer ─────────────────────────────────────────────────────────────
def write_report(
    sorted_stats: list[dict],
    cluster_labels: dict[int, str],
    cluster_summaries: dict[int, str],
    exemplar_chunks: dict[int, list[str]],
    cluster_rows_by_id: dict[int, list[dict]],
    total_chunks: int,
) -> None:

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("CHUNK CLUSTER REPORT")
    lines.append(f"k={K}   Level-0 chunks: {total_chunks:,}   Random seed: {RANDOM_STATE}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(
        "Breadth = mean cosine distance of member chunks from cluster centroid.")
    lines.append(
        "Relative breadth = breadth / global mean breadth (>1 = broader than avg).")
    lines.append(f"Global mean distance to centroid: {GLOBAL_MEAN_DIST:.4f}")
    lines.append("")
    lines.append(
        f"{'#':>3}  {'Cluster label':<36}  {'Chunks':>6}  {'%':>5}  "
        f"{'Breadth':>7}  {'Rel.':>5}"
    )
    lines.append("-" * 70)

    for st in sorted_stats:
        c = st["cluster_id"]
        lbl = cluster_labels[c][:35]
        pct = 100 * st["n_chunks"] / total_chunks
        lines.append(
            f"{c:>3}  {lbl:<36}  {st['n_chunks']:>6,}  {pct:>5.1f}%  "
            f"{st['mean_dist']:>7.4f}  {st['rel_breadth']:>5.2f}x"
        )

    lines.append("")
    lines.append("=" * 70)
    lines.append("CLUSTER DETAILS (sorted by size)")
    lines.append("=" * 70)

    for st in sorted_stats:
        c = st["cluster_id"]
        lines.append("")
        lines.append(f"── Cluster {c}: {cluster_labels[c]}")
        lines.append(f"   Chunks:           {st['n_chunks']:,}")
        lines.append(f"   Breadth:          {st['mean_dist']:.4f}")
        lines.append(f"   Relative breadth: {st['rel_breadth']:.2f}x")
        lines.append("")
        # Wrap summary at ~70 chars for readability
        summary = cluster_summaries.get(c, "")
        if summary:
            lines.append("   Summary:")
            words = summary.split()
            line_buf, indent = "     ", "     "
            for w in words:
                if len(line_buf) + len(w) + 1 > 72:
                    lines.append(line_buf)
                    line_buf = indent + w
                else:
                    line_buf = (line_buf + " " + w).lstrip() if line_buf == indent else line_buf + " " + w
            if line_buf.strip():
                lines.append(line_buf)
            lines.append("")

        lines.append("   Top exemplar passages (nearest to centroid):")
        for i, ex in enumerate(exemplar_chunks.get(c, [])[:3], 1):
            snippet = ex.replace("\n", " ").strip()[:200]
            lines.append(f"     {i}. {snippet}…")
        lines.append("")

        # Top books
        book_counts: dict[int, int] = {}
        for r in cluster_rows_by_id.get(c, []):
            book_counts[r["book_number"]] = book_counts.get(r["book_number"], 0) + 1
        top_books = sorted(book_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        bk_strs = [f"Book {bn}({n})" for bn, n in top_books]
        lines.append(f"   Top books: {', '.join(bk_strs)}")

    lines.append("")
    lines.append("=" * 70)

    with open(OUT_REPORT, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {OUT_REPORT}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:

    # 1. Load cluster CSV
    print("Loading chunk_clusters.csv…")
    csv_rows = load_cluster_csv()
    total_chunks = len(csv_rows)
    print(f"  {total_chunks:,} level-0 chunks")

    # 2. Sidecar for chunk text
    print("Loading sidecar…")
    vid_to_sc = load_sidecar()

    # 3. FAISS index + metadata to get vectors
    print("Loading FAISS index…")
    index = faiss.read_index(str(CHUNK_FAISS))
    with open(CHUNK_META) as f:
        meta: dict = json.load(f)

    # Level-0 vids in order (same as original clustering)
    level0_vids = sorted(
        int(vid) for vid, m in meta.items() if m["chunk_level"] == 0
    )

    print(f"Reconstructing {len(level0_vids):,} level-0 vectors from FAISS…")
    dim = index.d
    vecs = np.zeros((len(level0_vids), dim), dtype=np.float32)
    for i, vid in enumerate(level0_vids):
        index.reconstruct(vid, vecs[i])
    vid_to_vec: dict[int, np.ndarray] = {
        vid: vecs[i] for i, vid in enumerate(level0_vids)
    }
    print("  Done.")

    # 4. Group vids and rows by cluster
    cluster_vids: dict[int, list[int]] = {c: [] for c in range(K)}
    cluster_rows_by_id: dict[int, list[dict]] = {c: [] for c in range(K)}
    for vid, row in csv_rows.items():
        c = row["cluster_id"]
        cluster_vids[c].append(vid)
        cluster_rows_by_id[c].append(row)

    # Cluster labels (from CSV)
    cluster_labels: dict[int, str] = {
        c: (cluster_rows_by_id[c][0]["cluster_label"] if cluster_rows_by_id[c] else f"Cluster {c}")
        for c in range(K)
    }

    # Cluster stats (breadth from centroid, same computation as original)
    sorted_stats: list[dict] = []
    for c in range(K):
        vids = cluster_vids[c]
        member_vecs = np.array([vid_to_vec[v] for v in vids if v in vid_to_vec],
                               dtype=np.float32)
        if len(member_vecs) == 0:
            sorted_stats.append({"cluster_id": c, "n_chunks": 0,
                                  "mean_dist": 0.0, "rel_breadth": 0.0})
            continue
        centroid = normalize(member_vecs.mean(axis=0, keepdims=True), norm="l2")[0]
        cos_sims = member_vecs @ centroid
        mean_dist = float(np.mean(1.0 - cos_sims))
        rel = mean_dist / GLOBAL_MEAN_DIST
        sorted_stats.append({
            "cluster_id": c,
            "n_chunks": len(vids),
            "mean_dist": mean_dist,
            "rel_breadth": rel,
        })
    sorted_stats.sort(key=lambda x: x["n_chunks"], reverse=True)

    # 5. Find exemplar passages per cluster
    print("Finding exemplar passages per cluster…")
    exemplar_chunks: dict[int, list[str]] = {}
    for c in range(K):
        exemplar_chunks[c] = compute_exemplars(
            cluster_vids[c], vid_to_vec, vid_to_sc, n=EXEMPLARS_N
        )

    # 6. Generate full paragraph summaries with Claude
    print(f"\nGenerating paragraph summaries for {K} clusters with Claude…")
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    cluster_summaries: dict[int, str] = {}

    for st in sorted_stats:
        c = st["cluster_id"]
        label = cluster_labels[c]
        print(f"  [{c:2d}] {label[:55]}…", end=" ", flush=True)
        exemplars = exemplar_chunks[c]
        if exemplars:
            summary = summarise_cluster(client, c, label, exemplars)
        else:
            summary = ""
        cluster_summaries[c] = summary
        print("✓")

    # Save summaries JSON for later use
    with open(OUT_JSON, "w") as f:
        json.dump(
            {str(c): {"label": cluster_labels[c], "summary": cluster_summaries[c]}
             for c in range(K)},
            f, indent=2, ensure_ascii=False,
        )
    print(f"Saved: {OUT_JSON}")

    # 7. Write updated report
    print("Writing updated cluster_report.txt…")
    write_report(
        sorted_stats, cluster_labels, cluster_summaries,
        exemplar_chunks, cluster_rows_by_id, total_chunks,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
