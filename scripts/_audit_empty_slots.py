"""
Audit elibrary_cache.pkl for every empty slot across all books and chapters.
Saves a detailed report to reports/empty_slots_report.txt
"""
import sys
from pathlib import Path
from datetime import date

CACHE_DIR = Path('/Users/GStahl2/AI/elibrary/cache')
REPORTS   = Path('/Users/GStahl2/AI/elibrary/reports')
sys.path.insert(0, str(CACHE_DIR))
from elibrary_cache import ElibraryCache

cache = ElibraryCache.load(str(CACHE_DIR / 'elibrary_cache.pkl'))

lines = []
def w(s=""): lines.append(s)

def is_empty(v) -> bool:
    if v is None:           return True
    if isinstance(v, str):  return v.strip() == ""
    if isinstance(v, int):  return v == 0
    if isinstance(v, list): return len(v) == 0
    return False

# ─── Header ────────────────────────────────────────────────────────────────
w("=" * 72)
w("ELIBRARY CACHE — EMPTY SLOT AUDIT")
w(f"Generated: {date.today().isoformat()}  |  PKL: elibrary_cache.pkl")
w("=" * 72)
w()

# ─── Book-level fields to check ────────────────────────────────────────────
BOOK_FIELDS = [
    "book_title", "book_author", "book_keywords", "book_reference",
    "book_text", "book_number_of_pages", "book_kind",
]
BOOK_SUMMARY_FIELDS = [
    "book_summary_author", "book_summary_date", "book_summary_prompt",
    "book_summary_number_of_words", "book_summary_text",
]

# ─── Chapter-level fields to check ─────────────────────────────────────────
CH_FIELDS = [
    "chapter_title", "chapter_author", "chapter_keywords", "chapter_reference",
    "chapter_text", "chapter_number_of_pages", "chapter_number_of_words",
    "chapter_number_of_tokens", "chapter_number_of_symbols",
]
CH_SUMMARY_FIELDS = [
    "chapter_summary_author", "chapter_summary_date", "chapter_summary_prompt",
    "chapter_summary_number_of_words", "chapter_summary_text",
]

total_book_empty = 0
total_ch_empty   = 0
total_books      = len(cache.books)
total_chapters   = sum(len(b.book_chapters) for b in cache.books)

for b in cache.books:
    bnum   = b.book_number
    btitle = b.book_title or b.book_name
    book_empties = []

    # Book-level scalar fields
    for f in BOOK_FIELDS:
        v = getattr(b, f, None)
        if is_empty(v):
            book_empties.append(f"  book.{f}")

    # Book summaries
    bsums = getattr(b, 'book_summaries', [])
    if not bsums:
        book_empties.append("  book_summaries  [NO ENTRIES]")
    elif len(bsums) != 1:
        book_empties.append(f"  book_summaries  [{len(bsums)} entries — expected 1]")
    else:
        bs = bsums[0]
        for f in BOOK_SUMMARY_FIELDS:
            v = getattr(bs, f, None)
            if is_empty(v):
                book_empties.append(f"  book_summaries[0].{f}")

    # Chapter-level
    ch_section = []
    for ch in b.book_chapters:
        cnum   = ch.chapter_number
        ctitle = ch.chapter_title or f"(ch {cnum})"
        ch_empties = []

        for f in CH_FIELDS:
            v = getattr(ch, f, None)
            if is_empty(v):
                ch_empties.append(f"    ch.{f}")

        # Chapter summaries
        csums = getattr(ch, 'chapter_summaries', [])
        if not csums:
            ch_empties.append("    chapter_summaries  [NO ENTRIES]")
        elif len(csums) != 1:
            ch_empties.append(f"    chapter_summaries  [{len(csums)} entries — expected 1]")
        else:
            cs = csums[0]
            for f in CH_SUMMARY_FIELDS:
                v = getattr(cs, f, None)
                if is_empty(v):
                    ch_empties.append(f"    chapter_summaries[0].{f}")

        if ch_empties:
            total_ch_empty += len(ch_empties)
            ch_section.append(f"  ── Ch {cnum:>2}: {ctitle[:55]}")
            ch_section.extend(ch_empties)

    if book_empties or ch_section:
        w(f"┌─ Book {bnum:>2}: {btitle[:58]}")
        if book_empties:
            total_book_empty += len(book_empties)
            w("  [BOOK LEVEL]")
            for e in book_empties:
                w(e)
        if ch_section:
            w("  [CHAPTERS]")
            for line in ch_section:
                w(line)
        w()

# ─── Summary counts ─────────────────────────────────────────────────────────
w("=" * 72)
w("SUMMARY")
w("=" * 72)
w(f"  Books scanned   : {total_books}")
w(f"  Chapters scanned: {total_chapters}")
w(f"  Empty book-level slots   : {total_book_empty}")
w(f"  Empty chapter-level slots: {total_ch_empty}")
w(f"  Total empty slots        : {total_book_empty + total_ch_empty}")
w()

report_text = "\n".join(lines)
print(report_text)

out_path = REPORTS / "empty_slots_report.txt"
out_path.write_text(report_text, encoding="utf-8")
print(f"\nReport saved → {out_path}")
