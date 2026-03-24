"""
chapter_inventory.py
Produce a chronological report: pub_year, word_count, book_num, chapter_num, title
for every chapter in the elibrary cache.

Year resolution:
  1. chapter_reference (APA string) — used when present
  2. book_reference fallback — used when chapter_reference is empty
  3. '????' — no year found anywhere

Output: reports/chapter_inventory.txt  +  console print
"""

import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CACHE_PATH = ROOT / "cache" / "elibrary_cache.pkl"
REPORT_PATH = ROOT / "reports" / "chapter_inventory.txt"

YEAR_PAT = re.compile(r'\((\d{4})\)')


def extract_year(ref: str | None) -> int | None:
    m = YEAR_PAT.search(ref or "")
    return int(m.group(1)) if m else None


def main() -> None:
    cache = pickle.load(open(CACHE_PATH, "rb"))

    # Build book-level year fallback
    book_year: dict[int, int | None] = {}
    book_title: dict[int, str] = {}
    for book in cache.books:
        book_year[book.book_number] = extract_year(book.book_reference)
        book_title[book.book_number] = book.book_title or book.book_name or ""

    rows: list[tuple] = []
    src_counts = {"chapter_ref": 0, "book_ref": 0, "unknown": 0}

    for book in cache.books:
        by = book_year[book.book_number]
        for ch in book.book_chapters:
            cy = extract_year(ch.chapter_reference)
            if cy:
                year = cy
                src = "chapter_ref"
            elif by:
                year = by
                src = "book_ref"
            else:
                year = 0
                src = "unknown"
            src_counts[src] += 1
            words = ch.chapter_number_of_words or 0
            rows.append((
                year,
                book.book_number,
                ch.chapter_number,
                words,
                ch.chapter_title,
                book_title[book.book_number],
                src,
            ))

    rows.sort(key=lambda r: (r[0], r[1], r[2]))

    # ── format ──────────────────────────────────────────────────────────────
    lines: list[str] = []
    header = (
        f"{'Year':<6}  {'Bk':>3}  {'Ch':>3}  {'Words':>7}  "
        f"{'Title':<60}  Book"
    )
    sep = "-" * 120

    lines.append("ELIBRARY CHAPTER INVENTORY — chronological")
    lines.append(f"Generated: March 24, 2026   |   Total chapters: {len(rows)}")
    lines.append(f"Year source: {src_counts['chapter_ref']} from chapter_ref, "
                 f"{src_counts['book_ref']} from book_ref (marked *), "
                 f"{src_counts['unknown']} unknown")
    lines.append("")
    lines.append(header)
    lines.append(sep)

    current_year = None
    total_words = 0

    for year, bk, ch, words, title, btitle, src in rows:
        yr_str = str(year) if year else "????"
        flag = "*" if src == "book_ref" else (" " if src == "chapter_ref" else "?")
        if yr_str != current_year:
            if current_year is not None:
                lines.append("")
            current_year = yr_str

        line = (
            f"{yr_str:<6}{flag} {bk:>3}  {ch:>3}  {words:>7}  "
            f"{title[:60]:<60}  {btitle[:40]}"
        )
        lines.append(line)
        total_words += words

    lines.append(sep)
    lines.append(f"TOTAL WORDS: {total_words:,}")

    output = "\n".join(lines)
    print(output)

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(output, encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
