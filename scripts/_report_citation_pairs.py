#!/usr/bin/env python3
"""
scripts/_report_citation_pairs.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Produce a human-readable diagnostic report of citation_pairs.jsonl,
showing examples of every resolution outcome so problem cases can be
inspected and the pipeline improved.

Usage:
    .venv/bin/python scripts/_report_citation_pairs.py [--examples N]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def main(n_examples: int = 10) -> None:
    records = []
    with open(REPORTS / "citation_pairs.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    by_reason: dict[str, list[dict]] = defaultdict(list)
    resolved = []
    for r in records:
        if r["resolved"]:
            resolved.append(r)
        else:
            by_reason[r.get("unresolved_reason", "unknown")].append(r)

    lines = []
    def w(*args):
        lines.append(" ".join(str(a) for a in args))

    w("=" * 70)
    w("CITATION PAIR RESOLUTION — DIAGNOSTIC REPORT")
    w("=" * 70)
    w(f"Total citations : {total:,}")
    w(f"Resolved        : {len(resolved):,}  ({100*len(resolved)/total:.1f}%)")
    w(f"Unresolved      : {total - len(resolved):,}  ({100*(total-len(resolved))/total:.1f}%)")
    w()

    # ── Resolved examples ────────────────────────────────────────────────────
    w("─" * 70)
    w(f"RESOLVED ({len(resolved):,})  — sample of {min(n_examples, len(resolved))}")
    w("─" * 70)
    for r in resolved[:n_examples]:
        w(f"  from: Book {r['from_book_number']} Ch {r['from_chapter_number']}"
          f"  ({r['from_pub_year']})  cluster {r['from_cluster_id']}")
        w(f"  snippet : {r['from_snippet'][:120]}")
        w(f"  cited_year: {r['cited_year']}  ref_src: {r['ref_source']}"
          f"  type: {r['to_type']}")
        w(f"  APA     : {(r['to_apa_entry'] or '')[:120]}")
        w(f"  → Book {r['to_book_number']} Ch {r['to_chapter_number']}"
          f"  \"{r['to_chapter_title']}\"  ({r['to_pub_year']})")
        w()

    # ── Each unresolved category ─────────────────────────────────────────────
    ordered_reasons = [
        "ambiguous_year",
        "chapter_title_not_in_library",
        "year_not_in_ref_list",
        "cited_whole_book",
        "cited_article_not_in_library",
        "no_ref_list_in_book",
        "apa_parse_failed",
    ]

    for reason in ordered_reasons:
        group = by_reason.get(reason, [])
        if not group:
            continue
        w("─" * 70)
        w(f"{reason.upper()}  ({len(group):,} = {100*len(group)/total:.1f}%)")
        w("─" * 70)

        # Unique APA entries seen (for title-matching failures, very informative)
        if reason in ("ambiguous_year", "chapter_title_not_in_library"):
            apa_set: dict[str, int] = defaultdict(int)
            for r in group:
                apa = (r.get("to_apa_entry") or "").strip()
                if apa:
                    apa_set[apa[:120]] += 1
            w(f"  Distinct APA entries seen (top 30):")
            for apa, cnt in sorted(apa_set.items(), key=lambda x: -x[1])[:30]:
                w(f"    [{cnt:4d}×]  {apa}")
            w()

        w(f"  Examples (first {min(n_examples, len(group))}):")
        for r in group[:n_examples]:
            w(f"    from: Book {r['from_book_number']} Ch {r['from_chapter_number']}"
              f"  ({r['from_pub_year']})  cluster {r['from_cluster_id']}")
            w(f"    snippet   : {r['from_snippet'][:120]}")
            w(f"    cited_year: {r['cited_year']}"
              f"  ref_src: {r.get('ref_source', '—')}")
            apa = (r.get("to_apa_entry") or "").strip()
            if apa:
                w(f"    APA match : {apa[:120]}")
            w()

    # ── Any unexpected reasons ───────────────────────────────────────────────
    for reason, group in by_reason.items():
        if reason not in ordered_reasons:
            w("─" * 70)
            w(f"OTHER: {reason}  ({len(group):,})")
            for r in group[:5]:
                w(f"  {r}")
            w()

    report = "\n".join(lines)
    out = REPORTS / "citation_pairs_diagnostic.txt"
    out.write_text(report)
    print(f"Saved: {out}")
    print(f"  {len(lines)} lines")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", "-n", type=int, default=10,
                        help="Number of examples per category (default 10)")
    args = parser.parse_args()
    main(n_examples=args.examples)
