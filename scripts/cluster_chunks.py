"""
scripts/cluster_chunks.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Cluster level-0 chunk vectors (k=20) using k-means on the FAISS embeddings,
label each cluster with Claude, and produce a human-readable report.

Output files:
  reports/chunk_clusters.csv     — chunk-level cluster assignments
  reports/cluster_report.txt     — labelled clusters, sizes, breadth stats

Breadth measures (both reported per cluster):
  mean_dist_to_centroid  — average cosine distance from member vectors to the
                           cluster centroid; range [0,2] but typically [0,0.5].
                           Lower = tighter / more focused theme.
  relative_breadth       — mean_dist / global_mean_dist; >1 = broader than
                           average, <1 = tighter than average.

Usage:
    python scripts/cluster_chunks.py [--k 20] [--no-claude]

Author  : Gerry Stahl
Created : March 2026
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import faiss
import anthropic
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# ── Config ─────────────────────────────────────────────────────────────────────
K                = 20       # number of clusters
EXEMPLARS_N      = 5        # exemplar passages per cluster sent to Claude
EXEMPLAR_CHARS   = 400      # max chars per exemplar passage
LABEL_MODEL      = "claude-haiku-4-5-20251001"
RANDOM_STATE     = 42

VS_DIR           = ROOT / "vector_store"
CHUNK_FAISS      = VS_DIR / "elibrary.faiss"
CHUNK_META_FILE  = VS_DIR / "elibrary_meta.json"
CHUNK_SIDECAR    = VS_DIR / "chunks_bm25" / "sidecar.json"
PKL_PATH         = ROOT / "cache" / "elibrary_cache.pkl"
REPORTS_DIR      = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

OUT_CSV          = REPORTS_DIR / "chunk_clusters.csv"
OUT_REPORT       = REPORTS_DIR / "cluster_report.txt"


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_year(reference: str) -> int | None:
    """Extract 4-digit year from an APA reference string."""
    m = re.search(r'\((\d{4}[a-z]?)\)', reference or '')
    if m:
        try:
            return int(m.group(1)[:4])
        except ValueError:
            pass
    return None


def build_year_lookup(cache) -> dict[tuple[int,int], int | None]:
    """Return {(book_number, chapter_number): publication_year} from cache."""
    lookup: dict[tuple[int,int], int | None] = {}
    for book in cache.books:
        book_year = extract_year(book.book_reference)
        for ch in book.book_chapters:
            ch_year = extract_year(ch.chapter_reference) or book_year
            lookup[(book.book_number, ch.chapter_number)] = ch_year
    return lookup


def label_cluster_claude(
    client: anthropic.Anthropic,
    exemplars: list[str],
    cluster_id: int,
) -> str:
    """Ask Claude Haiku to label a cluster from its exemplar passages."""
    passages = "\n\n---\n\n".join(
        f"Passage {i+1}:\n{e}" for i, e in enumerate(exemplars)
    )
    prompt = (
        f"The following {len(exemplars)} passages are from the same semantic cluster "
        "in a 22-volume academic library by Gerry Stahl on computer-supported "
        "collaborative learning (CSCL) and related topics.\n\n"
        f"{passages}\n\n"
        "Give this cluster a concise thematic label of 3–6 words that captures "
        "the shared topic. Then write one sentence explaining what unifies these passages.\n"
        "Format your response as:\n"
        "LABEL: <label>\n"
        "EXPLANATION: <one sentence>"
    )
    msg = client.messages.create(
        model=LABEL_MODEL,
        max_tokens=120,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text.strip()
    label, explanation = f"Cluster {cluster_id}", ""
    for line in text.splitlines():
        if line.startswith("LABEL:"):
            label = line.replace("LABEL:", "").strip()
        elif line.startswith("EXPLANATION:"):
            explanation = line.replace("EXPLANATION:", "").strip()
    return label, explanation


# ── Main ───────────────────────────────────────────────────────────────────────

def main(k: int = K, use_claude: bool = True) -> None:

    # ── 1. Load cache for year lookup ──────────────────────────────────────────
    print("Loading cache for publication years…", flush=True)
    with open(PKL_PATH, "rb") as f:
        cache = pickle.load(f)
    year_lookup = build_year_lookup(cache)
    del cache   # free RAM

    # ── 2. Load chunk FAISS and metadata ───────────────────────────────────────
    print("Loading chunk FAISS index and metadata…", flush=True)
    index = faiss.read_index(str(CHUNK_FAISS))
    with open(CHUNK_META_FILE) as f:
        chunk_meta: dict[str, dict] = json.load(f)

    # ── 3. Load chunk sidecar for text ────────────────────────────────────────
    print("Loading chunk sidecar…", flush=True)
    with open(CHUNK_SIDECAR) as f:
        sidecar: dict[str, dict] = json.load(f)
    # Build vid → sidecar entry for fast lookup
    vid_to_sc: dict[int, dict] = {
        int(e["vector_id"]): e
        for e in sidecar.values()
        if e.get("vector_id") is not None
    }

    # ── 4. Extract level-0 vector IDs and their vectors ───────────────────────
    print("Extracting level-0 vectors…", flush=True)
    level0_vids: list[int] = [
        int(vid) for vid, m in chunk_meta.items()
        if m["chunk_level"] == 0
    ]
    level0_vids.sort()

    # Reconstruct matrix from FAISS index
    # faiss.IndexFlatIP stores vectors; we reconstruct via index.reconstruct()
    dim = index.d
    vecs = np.zeros((len(level0_vids), dim), dtype=np.float32)
    for i, vid in enumerate(level0_vids):
        index.reconstruct(vid, vecs[i])

    print(f"  Level-0 chunks: {len(level0_vids):,}")

    # ── 5. K-means clustering ─────────────────────────────────────────────────
    print(f"\nRunning k-means (k={k})… this may take ~30s", flush=True)
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
    labels = km.fit_predict(vecs)   # (N,) cluster assignments
    centroids = km.cluster_centers_ # (k, 768) — not L2-normalised yet

    print("  K-means converged.")

    # ── 6. Compute breadth per cluster ────────────────────────────────────────
    # Cosine distance = 1 - cosine_similarity
    # Vectors are already L2-normalised; centroids may not be — normalise them
    centroids_norm = normalize(centroids, norm="l2")

    # Global mean distance (reference point for relative breadth)
    all_cos_sim = np.einsum("nd,nd->n", vecs, centroids_norm[labels])  # dot per member
    global_mean_dist = float(np.mean(1.0 - all_cos_sim))

    cluster_stats: list[dict] = []
    for c in range(k):
        mask = labels == c
        members = vecs[mask]       # (nc, 768)
        nc = int(mask.sum())
        centroid = centroids_norm[c]   # (768,)
        cos_sim = members @ centroid   # (nc,) — inner product = cosine (normalised)
        mean_dist = float(np.mean(1.0 - cos_sim))
        rel_breadth = mean_dist / global_mean_dist if global_mean_dist > 0 else 1.0
        cluster_stats.append({
            "cluster_id":          c,
            "n_chunks":            nc,
            "mean_dist_centroid":  mean_dist,
            "relative_breadth":    rel_breadth,
        })

    # ── 7. Find exemplar passages per cluster ─────────────────────────────────
    # Exemplars = chunks whose vector is closest to the centroid
    print("Finding exemplar passages…", flush=True)
    exemplar_chunks: dict[int, list[str]] = {c: [] for c in range(k)}
    for c in range(k):
        centroid = centroids_norm[c]
        mask = np.where(labels == c)[0]   # indices into level0_vids
        sims = vecs[mask] @ centroid
        top_idx = mask[np.argsort(sims)[::-1][:EXEMPLARS_N]]
        for idx in top_idx:
            vid = level0_vids[idx]
            sc_entry = vid_to_sc.get(vid)
            if sc_entry:
                text = sc_entry.get("chunk_text", "")[:EXEMPLAR_CHARS]
                exemplar_chunks[c].append(text)

    # ── 8. Label clusters with Claude ─────────────────────────────────────────
    cluster_labels: dict[int, str] = {}
    cluster_explanations: dict[int, str] = {}

    if use_claude:
        print(f"\nLabelling {k} clusters with Claude Haiku…", flush=True)
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        for c in range(k):
            exemplars = exemplar_chunks[c]
            if exemplars:
                label, explanation = label_cluster_claude(client, exemplars, c)
            else:
                label, explanation = f"Cluster {c}", ""
            cluster_labels[c] = label
            cluster_explanations[c] = explanation
            print(f"  [{c:2d}] {label}")
    else:
        for c in range(k):
            cluster_labels[c] = f"Cluster {c}"
            cluster_explanations[c] = ""
        print("  (Claude labelling skipped — use --no-claude to enable)")

    # ── 9. Build chunk-level assignment table ─────────────────────────────────
    print("\nBuilding cluster assignment table…", flush=True)
    rows: list[dict] = []
    for idx, vid in enumerate(level0_vids):
        m = chunk_meta.get(str(vid), {})
        sc = vid_to_sc.get(vid, {})
        bn = int(m.get("book_number", 0))
        cn = int(m.get("chapter_number", 0))
        pub_year = year_lookup.get((bn, cn))
        c = int(labels[idx])
        rows.append({
            "vector_id":       vid,
            "book_number":     bn,
            "chapter_number":  cn,
            "book_title":      m.get("book_title", ""),
            "chapter_title":   m.get("chapter_title", ""),
            "chunk_index":     int(m.get("chunk_index", 0)),
            "chunk_page":      int(m.get("chunk_page", 0)),
            "pub_year":        pub_year,
            "cluster_id":      c,
            "cluster_label":   cluster_labels[c],
        })

    # Save CSV
    import csv
    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {OUT_CSV}")

    # ── 10. Write report ───────────────────────────────────────────────────────
    print("Writing report…", flush=True)

    # Sort clusters by size descending for the report
    sorted_stats = sorted(cluster_stats, key=lambda x: x["n_chunks"], reverse=True)

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("CHUNK CLUSTER REPORT")
    lines.append(f"k={k}   Level-0 chunks: {len(level0_vids):,}   Random seed: {RANDOM_STATE}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(
        "Breadth = mean cosine distance of member chunks from cluster centroid.")
    lines.append(
        "Relative breadth = breadth / global mean breadth (>1 = broader than avg).")
    lines.append(
        f"Global mean distance to centroid: {global_mean_dist:.4f}")
    lines.append("")
    lines.append(
        f"{'#':>3}  {'Cluster label':<36}  {'Chunks':>6}  {'%':>5}  "
        f"{'Breadth':>7}  {'Rel.':>5}")
    lines.append("-" * 70)

    total = len(level0_vids)
    for st in sorted_stats:
        c = st["cluster_id"]
        lbl = cluster_labels[c][:35]
        pct = 100 * st["n_chunks"] / total
        lines.append(
            f"{c:>3}  {lbl:<36}  {st['n_chunks']:>6,}  {pct:>5.1f}%  "
            f"{st['mean_dist_centroid']:>7.4f}  {st['relative_breadth']:>5.2f}x"
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
        lines.append(f"   Breadth:          {st['mean_dist_centroid']:.4f}")
        lines.append(f"   Relative breadth: {st['relative_breadth']:.2f}x")
        if cluster_explanations[c]:
            lines.append(f"   Summary:          {cluster_explanations[c]}")

        lines.append(f"   Top exemplar passages (nearest to centroid):")
        for i, ex in enumerate(exemplar_chunks[c][:3], 1):
            snippet = ex.replace("\n", " ").strip()[:200]
            lines.append(f"     {i}. {snippet}…")

        # Which books are represented in this cluster?
        cluster_rows = [r for r in rows if r["cluster_id"] == c]
        book_counts: dict[int, int] = {}
        for r in cluster_rows:
            book_counts[r["book_number"]] = book_counts.get(r["book_number"], 0) + 1
        top_books = sorted(book_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        bk_strs = [f"Book {bn}({n})" for bn, n in top_books]
        lines.append(f"   Top books: {', '.join(bk_strs)}")

    lines.append("")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    with open(OUT_REPORT, "w") as f:
        f.write(report_text)
    print(f"  Saved: {OUT_REPORT}")

    print()
    print(report_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster level-0 chunks.")
    parser.add_argument("--k",         type=int, default=K,
                        help=f"Number of clusters (default: {K})")
    parser.add_argument("--no-claude", action="store_true",
                        help="Skip Claude labelling (faster; uses generic labels)")
    args = parser.parse_args()
    main(k=args.k, use_claude=not args.no_claude)
