#!/usr/bin/env python3
"""
scripts/build_citation_pairs.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Build the resolved self-citation pair dataset for vector-based analysis.

For each of the 2,852 inline Stahl self-citations in self_citations.csv:
  1. Search ALL chunks from the same (book, chapter) for a Stahl APA
     reference-list entry matching the cited year.
  2. Parse the APA entry to extract the cited work title and type
     (chapter-in-book, monograph, article/other).
  3. Look up the cited title in the library's chapter index.
  4. If resolved: load the FAISS vectors for both the citing chunk and
     the cited chapter (centroid of its level-0 chunks).
  5. Save a JSONL record for every citation (resolved OR unresolved)
     and an NPZ of vectors for the resolved subset.

Outputs
-------
  reports/citation_pairs.jsonl       — one JSON record per citation
  reports/citation_vectors.npz       — from_vecs, to_vecs (resolved only)
  reports/citation_pairs_report.txt  — resolution summary

JSONL record schema
-------------------
  citation_id          int
  from_vector_id       int
  from_snippet         str      — citing sentence(s) from self_citations.csv
  from_pub_year        int
  from_book_number     int
  from_chapter_number  int
  from_chapter_title   str
  from_cluster_id      int
  from_cluster_label   str
  cited_year           str      — e.g. "2006" or "2006a"
  resolved             bool
  to_book_number       int|null
  to_chapter_number    int|null
  to_chapter_title     str|null
  to_pub_year          int|null
  to_apa_entry         str|null — the APA line we matched
  to_type              str|null — "chapter", "monograph", "article_or_other"
  ref_source           str|null — "chapter" (per-chapter refs) or "book" (consolidated)
  unresolved_reason    str|null — one of:
      "no_ref_list_in_book"         — book has no scannable Stahl reference list anywhere
      "year_not_in_ref_list"        — ref list found but cited year not present
      "cited_whole_book"            — APA entry is a monograph, not a chapter
      "cited_article_not_in_library"— article/conference paper, not in library
      "ambiguous_year"              — multiple Stahl entries for same year
      "chapter_title_not_in_library"— chapter APA entry but title not found
      "apa_parse_failed"            — APA entry found but couldn't parse title

NPZ arrays (N_resolved rows, aligned by citation_id)
-----------------------------------------------------
  citation_ids  shape (N,)      int32  — maps rows back to JSONL citation_id
  from_vids     shape (N,)      int32  — FAISS vector_id of citing chunk
  to_vids       shape (N,)      int32  — representative FAISS vid of cited chapter
  from_vecs     shape (N, 768)  float32
  to_vecs       shape (N, 768)  float32  — centroid of cited chapter's level-0 vecs
  sum_vecs      shape (N, 768)  float32  — from_vec + to_vec

Usage
-----
  .venv/bin/python scripts/build_citation_pairs.py [--verbose]

Author: Gerry Stahl / Copilot
Created: March 2026
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np

ROOT    = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

# ── Regex: find a Stahl APA entry line for a specific cited year ──────────────
# Matches lines like:
#   Stahl, G. (2006). Group cognition...
#   Stahl, G. (2006a). Analyzing...
#   Fischer, G., ..., Stahl, G., ... (1993). Embedding...  ← multi-author; skip
#
# We require "Stahl" to appear near the START of the line (within 10 chars)
# to exclude lines where Stahl is a co-author listed mid-line.

def _stahl_entry_pattern(cited_year: str) -> re.Pattern:
    year_esc = re.escape(cited_year)
    return re.compile(
        r'^.{0,15}Stahl,\s*G\..*?\(' + year_esc + r'\)\.\s*(.+)',
        re.IGNORECASE | re.DOTALL,
    )


# ── APA title/type parser ─────────────────────────────────────────────────────

_MONOGRAPH_PUBLISHERS = (
    "mit press", "springer", "cambridge", "oxford", "routledge",
    "erlbaum", "wiley", "elsevier", "kluwer", "lulu",
)

def _normalize(s: str) -> str:
    return re.sub(r'\W+', ' ', s.lower()).strip()


def parse_apa_entry(entry_text: str, cited_year: str) -> tuple[str, str]:
    """
    Parse a Stahl APA entry to (title, type).

    type is one of: "chapter", "monograph", "article_or_other", "unknown"
    Returns ("", "unknown") on failure.
    """
    # Find where the year token ends — title starts after "(YEAR). "
    year_m = re.search(r'\(' + re.escape(cited_year) + r'\)\.\s*', entry_text, re.IGNORECASE)
    if not year_m:
        return "", "unknown"

    rest = entry_text[year_m.end():]

    # Does the entry reference a chapter "In [editor(s)]..."?
    in_m = re.search(r'\.\s+[Ii]n\b', rest)
    if in_m:
        title = rest[: in_m.start()].strip().rstrip('.')
        return title, "chapter"

    # No "In" — take text up to the first ". " (title ends there)
    period_m = re.search(r'\.\s+', rest)
    if period_m:
        title = rest[: period_m.start()].strip()
        tail = rest[period_m.end():].lower()
        for pub in _MONOGRAPH_PUBLISHERS:
            if pub in tail:
                return title, "monograph"
        return title, "article_or_other"

    # Fallback: use the whole rest as title
    title = rest.strip().rstrip('.')
    return title, "unknown"


# ── Title-to-chapter index ────────────────────────────────────────────────────

def build_chapter_index(
    chunk_clusters_path: Path,
) -> dict[str, tuple[int, int, str, int]]:
    """
    Returns {normalized_title: (book_number, chapter_number, chapter_title, pub_year)}
    for every unique (book, chapter) in chunk_clusters.csv.
    """
    seen: dict[str, tuple[int, int, str, int]] = {}
    with open(chunk_clusters_path) as f:
        for row in csv.DictReader(f):
            bk  = int(row["book_number"])
            ch  = int(row["chapter_number"])
            ct  = row["chapter_title"].strip()
            try:
                py = int(row["pub_year"])
            except (ValueError, KeyError):
                py = 0
            key = _normalize(ct)
            if key and key not in seen:
                seen[key] = (bk, ch, ct, py)
    return seen


def lookup_chapter(
    apa_title: str,
    apa_year_str: str,
    chapter_index: dict[str, tuple[int, int, str, int]],
    pub_year_map: dict[tuple[int, int], int],
) -> tuple[int, int, str, int] | None:
    """
    Match APA title against the library's chapter index.
    Uses (1) exact normalised match, (2) prefix match, (3) substring match.
    Returns (book_number, chapter_number, chapter_title, pub_year) or None.
    """
    norm = _normalize(apa_title)
    if not norm:
        return None

    # Exact match
    if norm in chapter_index:
        return chapter_index[norm]

    # Prefix match (APA titles are sometimes truncated with "...")
    for key, val in chapter_index.items():
        if key.startswith(norm) or norm.startswith(key):
            if len(norm) > 10:   # avoid matching on tiny fragments
                return val

    # Substring match (≥ 8 words of the APA title appear in the key)
    norm_words = set(norm.split())
    if len(norm_words) >= 4:
        best_overlap = 0
        best_val = None
        for key, val in chapter_index.items():
            key_words = set(key.split())
            overlap = len(norm_words & key_words)
            if overlap >= min(6, len(norm_words) - 1) and overlap > best_overlap:
                best_overlap = overlap
                best_val = val
        if best_val is not None:
            return best_val

    return None


# ── FAISS helpers ─────────────────────────────────────────────────────────────

def chapter_centroid(
    book: int,
    chapter: int,
    chapter_vids: dict[tuple[int, int], list[int]],
    index: faiss.Index,
) -> np.ndarray | None:
    """Mean of all level-0 FAISS vectors for a (book, chapter). Returns None if empty."""
    vids = chapter_vids.get((book, chapter), [])
    if not vids:
        return None
    vecs = np.array([index.reconstruct(v) for v in vids], dtype=np.float32)
    return vecs.mean(axis=0)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(verbose: bool = False) -> None:
    print("Loading indexes and data…")

    index = faiss.read_index(str(ROOT / "vector_store/elibrary.faiss"))
    sidecar: dict = json.loads((ROOT / "vector_store/chunks_bm25/sidecar.json").read_text())

    # cluster_id and cluster_label per vector_id
    cluster_id_map: dict[int, int] = {}
    cluster_label_map: dict[int, str] = {}
    chapter_vids: dict[tuple[int, int], list[int]] = defaultdict(list)
    with open(REPORTS / "chunk_clusters.csv") as f:
        for row in csv.DictReader(f):
            vid = int(row["vector_id"])
            cluster_id_map[vid] = int(row["cluster_id"])
            cluster_label_map[vid] = row["cluster_label"]
            chapter_vids[(int(row["book_number"]), int(row["chapter_number"]))].append(vid)

    # pub_year per (book, chapter) — use most common value in chunk_clusters
    pub_year_map: dict[tuple[int, int], int] = {}
    year_votes: dict[tuple[int, int], dict[int, int]] = defaultdict(lambda: defaultdict(int))
    with open(REPORTS / "chunk_clusters.csv") as f:
        for row in csv.DictReader(f):
            try:
                py = int(row["pub_year"])
            except (ValueError, KeyError):
                continue
            year_votes[(int(row["book_number"]), int(row["chapter_number"]))][py] += 1
    for bc, votes in year_votes.items():
        pub_year_map[bc] = max(votes, key=lambda y: votes[y])

    # All chunks grouped by (book, chapter) and by book
    chapter_chunks: dict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)
    book_chunks: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for doc_id, meta in sidecar.items():
        bk = meta.get("book_number")
        ch = meta.get("chapter_number")
        vid = meta.get("vector_id")
        if bk and ch and vid is not None:
            chapter_chunks[(int(bk), int(ch))].append((int(vid), meta.get("chunk_text", "")))
            book_chunks[int(bk)].append((int(vid), meta.get("chunk_text", "")))

    # Chapter title index: normalized title → (book, chapter, title, pub_year)
    chapter_index = build_chapter_index(REPORTS / "chunk_clusters.csv")

    # Self-citations
    with open(REPORTS / "self_citations.csv") as f:
        cit_rows = list(csv.DictReader(f))
    print(f"  {len(cit_rows):,} self-citation rows to process")

    # ── Process ──────────────────────────────────────────────────────────────
    records: list[dict] = []
    resolved_ids: list[int] = []
    resolved_from_vecs: list[np.ndarray] = []
    resolved_to_vecs: list[np.ndarray] = []
    resolved_from_vids: list[int] = []
    resolved_to_vids_rep: list[int] = []   # representative vid of cited chapter

    # Track reason counts for summary
    reason_counts: dict[str, int] = defaultdict(int)

    for i, row in enumerate(cit_rows):
        cid       = i
        from_vid  = int(row["vector_id"])
        snippet   = row.get("snippet", "")
        from_year = row.get("pub_year", "")
        from_bk   = int(row["book_number"])
        from_ch   = int(row["chapter_number"])
        from_ct   = row.get("chapter_title", "")
        from_clu  = cluster_id_map.get(from_vid, -1)
        from_lbl  = cluster_label_map.get(from_vid, "")
        cited_yr  = row.get("cited_year", "").strip()

        base_record = {
            "citation_id":         cid,
            "from_vector_id":      from_vid,
            "from_snippet":        snippet,
            "from_pub_year":       from_year,
            "from_book_number":    from_bk,
            "from_chapter_number": from_ch,
            "from_chapter_title":  from_ct,
            "from_cluster_id":     from_clu,
            "from_cluster_label":  from_lbl,
            "cited_year":          cited_yr,
        }

        def record(resolved: bool, reason: str | None = None, **kw) -> dict:
            r = {**base_record, "resolved": resolved,
                 "ref_source": None,
                 "to_book_number": None, "to_chapter_number": None,
                 "to_chapter_title": None, "to_pub_year": None,
                 "to_apa_entry": None, "to_type": None,
                 "unresolved_reason": reason}
            r.update(kw)
            return r

        # ── Step 1: find APA entries — chapter-level first, book-level fallback ──
        pat = _stahl_entry_pattern(cited_yr)

        def _search_chunks(chunk_list: list[tuple[int, str]]) -> list[str]:
            found = []
            for _vid, text in chunk_list:
                for line in text.splitlines():
                    line = line.strip()
                    if pat.match(line):
                        found.append(line)
            return found

        ref_src: str | None = None
        matched_entries: list[str] = []
        has_any_ref_list = False  # did we find a ref list for this book at all?

        chapter_chunk_list = chapter_chunks.get((from_bk, from_ch), [])
        if chapter_chunk_list:
            matched_entries = _search_chunks(chapter_chunk_list)
            if matched_entries:
                ref_src = "chapter"
            else:
                # Check whether ANY Stahl year entry exists in this chapter
                any_stahl = re.compile(r'Stahl,\s*G\.?\s*\(\d{4}[a-z]?\)', re.IGNORECASE)
                has_any_ref_list = any(any_stahl.search(t) for _, t in chapter_chunk_list)

        # Fallback: search all chunks in the book (consolidated bibliography)
        if not matched_entries:
            book_chunk_list = book_chunks.get(from_bk, [])
            if book_chunk_list:
                matched_entries = _search_chunks(book_chunk_list)
                if matched_entries:
                    ref_src = "book"
                else:
                    any_stahl = re.compile(r'Stahl,\s*G\.?\s*\(\d{4}[a-z]?\)', re.IGNORECASE)
                    has_any_ref_list = has_any_ref_list or any(
                        any_stahl.search(t) for _, t in book_chunk_list
                    )

        if not matched_entries:
            if has_any_ref_list:
                # Reference list exists but this year is absent
                reason_counts["year_not_in_ref_list"] += 1
                records.append(record(False, "year_not_in_ref_list"))
            else:
                # No reference list found anywhere in the book
                reason_counts["no_ref_list_in_book"] += 1
                records.append(record(False, "no_ref_list_in_book"))
            continue

        # Deduplicate
        matched_entries = list(dict.fromkeys(matched_entries))

        # Deduplicate
        matched_entries = list(dict.fromkeys(matched_entries))

        if len(matched_entries) > 1:
            reason_counts["ambiguous_year"] += 1
            records.append(record(False, "ambiguous_year",
                                  ref_source=ref_src,
                                  to_apa_entry=" | ".join(matched_entries[:3])))
            continue

        apa_line = matched_entries[0]
        apa_title, apa_type = parse_apa_entry(apa_line, cited_yr)

        if verbose:
            print(f"  [{i}] cited_year={cited_yr} src={ref_src} type={apa_type} title={apa_title[:60]!r}")

        if apa_type == "monograph":
            reason_counts["cited_whole_book"] += 1
            records.append(record(False, "cited_whole_book",
                                  ref_source=ref_src,
                                  to_apa_entry=apa_line, to_type=apa_type))
            continue

        if apa_type in ("article_or_other", "unknown") and not apa_title:
            reason_counts["apa_parse_failed"] += 1
            records.append(record(False, "apa_parse_failed",
                                  ref_source=ref_src,
                                  to_apa_entry=apa_line, to_type=apa_type))
            continue

        if apa_type == "article_or_other":
            match = lookup_chapter(apa_title, cited_yr, chapter_index, pub_year_map)
            if match is None:
                reason_counts["cited_article_not_in_library"] += 1
                records.append(record(False, "cited_article_not_in_library",
                                      ref_source=ref_src,
                                      to_apa_entry=apa_line, to_type=apa_type))
                continue

        else:  # "chapter" or "unknown" with title
            match = lookup_chapter(apa_title, cited_yr, chapter_index, pub_year_map)
            if match is None:
                reason_counts["chapter_title_not_in_library"] += 1
                records.append(record(False, "chapter_title_not_in_library",
                                      ref_source=ref_src,
                                      to_apa_entry=apa_line, to_type=apa_type))
                continue

        to_bk, to_ch, to_ct, to_py = match

        # ── Step 2: compute vectors ─────────────────────────────────────────
        from_vec = index.reconstruct(from_vid).astype(np.float32)
        to_vec   = chapter_centroid(to_bk, to_ch, chapter_vids, index)
        if to_vec is None:
            reason_counts["chapter_title_not_in_library"] += 1
            records.append(record(False, "chapter_title_not_in_library",
                                  to_apa_entry=apa_line, to_type=apa_type))
            continue

        # Representative vid = closest chunk vid to centroid for the to-chapter
        to_vids_list = chapter_vids[(to_bk, to_ch)]
        rep_vid = to_vids_list[0]  # good enough for indexing

        reason_counts["resolved"] += 1
        rec = record(
            True, None,
            ref_source=ref_src,
            to_book_number=to_bk, to_chapter_number=to_ch,
            to_chapter_title=to_ct, to_pub_year=to_py,
            to_apa_entry=apa_line, to_type=apa_type,
        )
        records.append(rec)
        resolved_ids.append(cid)
        resolved_from_vecs.append(from_vec)
        resolved_to_vecs.append(to_vec)
        resolved_from_vids.append(from_vid)
        resolved_to_vids_rep.append(rep_vid)

    # ── Save JSONL ────────────────────────────────────────────────────────────
    jsonl_path = REPORTS / "citation_pairs.jsonl"
    with open(jsonl_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved: {jsonl_path}  ({len(records):,} records)")

    # ── Save NPZ ─────────────────────────────────────────────────────────────
    n = len(resolved_ids)
    if n > 0:
        from_arr = np.array(resolved_from_vecs, dtype=np.float32)
        to_arr   = np.array(resolved_to_vecs,   dtype=np.float32)
        sum_arr  = from_arr + to_arr
        npz_path = REPORTS / "citation_vectors.npz"
        np.savez_compressed(
            npz_path,
            citation_ids = np.array(resolved_ids,        dtype=np.int32),
            from_vids    = np.array(resolved_from_vids,  dtype=np.int32),
            to_vids      = np.array(resolved_to_vids_rep, dtype=np.int32),
            from_vecs    = from_arr,
            to_vecs      = to_arr,
            sum_vecs     = sum_arr,
        )
        print(f"Saved: {npz_path}  ({n} resolved pairs, {from_arr.nbytes // 1024 // 3:.0f} KB per array)")
    else:
        print("No resolved pairs — NPZ not written.")

    # ── Report ────────────────────────────────────────────────────────────────
    total = len(records)
    report_lines = [
        "Citation Pair Resolution Report",
        "=" * 50,
        f"Total self-citations: {total:,}",
        f"  Resolved (with vectors): {reason_counts['resolved']:,}  "
        f"({100*reason_counts['resolved']/total:.1f}%)",
        "",
        "Unresolved breakdown:",
    ]
    unresolved_reasons = [
        "no_ref_list_in_book",
        "year_not_in_ref_list",
        "cited_whole_book",
        "cited_article_not_in_library",
        "ambiguous_year",
        "chapter_title_not_in_library",
        "apa_parse_failed",
    ]
    for reason in unresolved_reasons:
        n_r = reason_counts[reason]
        if n_r:
            report_lines.append(f"  {reason:<40s} {n_r:>5,}  ({100*n_r/total:.1f}%)")

    report_text = "\n".join(report_lines) + "\n"
    rpt_path = REPORTS / "citation_pairs_report.txt"
    rpt_path.write_text(report_text)
    print(f"Saved: {rpt_path}")
    print()
    print(report_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build resolved self-citation pair dataset.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print each resolved entry as it is processed")
    args = parser.parse_args()
    main(verbose=args.verbose)
