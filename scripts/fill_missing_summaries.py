"""
fill_missing_summaries.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Reads each report TXT file in /reports/, parses chapter summaries and the book
overview, and fills any missing entries in the PKL (and then regenerates JSON).

Missing data as identified by the diagnostic:
  Book summaries  : books 4, 5, 6, 10, 11, 15, 16, 17, 18, 20, 21, 22
  Chapter summaries:
    Book  2 – Ch3, Ch10, Ch15  (part headers / appendix → no report section)
    Book  3 – Ch11             (part header → no report section)
    Book 10 – Ch2, Ch4, Ch6, Ch7  (real chapters → are in the report)

Author  : Gerry Stahl
Created : March 4, 2026
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = Path('/Users/GStahl2/AI/elibrary')
CACHE_DIR = ROOT / 'cache'
PKL_PATH  = CACHE_DIR / 'elibrary_cache.pkl'
JSON_PATH = CACHE_DIR / 'elibrary_cache.json'
REPORTS   = ROOT / 'reports'

SUMMARY_AUTHOR = "Claude agent"
SUMMARY_DATE   = "March 4, 2026"

# Titles that indicate a structural PKL entry (part intro, appendix, etc.)
# — these don't have matching report sections so we leave them empty.
_STRUCTURAL = re.compile(
    r'^(?:Part\s+[IVX\d]+|Appendix|Introduction\s+to\s+Part)',
    re.IGNORECASE,
)

# Map book number → report filename
REPORT_FILES = {
    int(p.name.split('.')[0]): p
    for p in sorted(REPORTS.glob('*.txt'))
}

# ── Load PKL ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(CACHE_DIR))
pkl = pickle.load(PKL_PATH.open('rb'))


# ── Parser ─────────────────────────────────────────────────────────────────
# Header patterns in approximate priority order:
#  1. "Chapter N:" or "Chapter N." (digit, then colon or period)
#  2. "CHAPTER N." or "CHAPTER I:" etc (uppercase, digit or Roman numeral)
#  3. "N. Title" for numbered essays (book 10)

CHAPTER_HEADING = re.compile(
    r'^(?:'
    r'Chapter\s+(\d+)\s*[.:]'          # "Chapter N:" / "Chapter N."
    r'|CHAPTER\s+(\d+)\s*[.:]'         # "CHAPTER N:" / "CHAPTER N."
    r'|CHAPTER\s+([IVXLC]+)\s*[.:]'    # "CHAPTER IV:" Roman numeral
    r'|(\d+)\.\s+\S'                   # "4. Title" — numbered essay
    r')',
    re.IGNORECASE
)

# Patterns that signal the start of the book-level overview / summary
OVERVIEW_HEADING = re.compile(
    r'^(?:'
    r'OVERVIEW(?:\s+OF\s+THE\s+COLLECTION)?'  # OVERVIEW / OVERVIEW OF THE COLLECTION
    r'|Overview(?:\s*:\s*\S)?'                # Overview: Title / Overview\n
    r'|.+—\s*(?:SUMMARY\s+OF|Collection\s+Overview)'  # Title — SUMMARY OF / — Collection Overview
    r')',
    re.IGNORECASE
)

ROMAN = {
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
    'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
    'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15,
    'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20,
    'XXI': 21, 'XXII': 22, 'XXIII': 23, 'XXIV': 24, 'XXV': 25,
    'XXVI': 26, 'XXVII': 27, 'XXVIII': 28, 'XXIX': 29, 'XXX': 30,
}


def _heading_chapter_num(line: str) -> int | None:
    """Return chapter number from a heading line, or None if not a chapter heading."""
    m = CHAPTER_HEADING.match(line)
    if not m:
        return None
    # group 1: "Chapter N"
    if m.group(1):
        return int(m.group(1))
    # group 2: "CHAPTER N" (digit)
    if m.group(2):
        return int(m.group(2))
    # group 3: "CHAPTER IV" Roman numeral
    if m.group(3):
        return ROMAN.get(m.group(3).upper())
    # group 4: "4. Title" numbered essay
    if m.group(4):
        return int(m.group(4))
    return None


def _is_separator(line: str) -> bool:
    """Lines that are pure separator decoration (===, ----, ════, ━━━, etc.)."""
    stripped = line.strip()
    if not stripped:
        return False
    return all(c in '=─━─—━═-_~' for c in stripped) and len(stripped) >= 4


def _is_keyword_line(line: str) -> bool:
    """Lines like 'Keywords: foo, bar' that we skip."""
    return line.startswith('Keywords:') or line.startswith('---')


def parse_report(path: Path) -> tuple[dict[int, str], str]:
    """
    Parse a summary report file.

    Returns:
        chapter_texts  : {chapter_number: text_string}
        book_overview  : str  (may be empty)
    """
    raw = path.read_text(encoding='utf-8', errors='replace')
    lines = raw.splitlines()

    # ── Locate section boundaries ──────────────────────────────────────────
    # Each boundary is (line_index, section_type, chapter_num_or_None)
    # section_type is 'chapter' or 'overview'
    boundaries: list[tuple[int, str, int | None]] = []

    for i, line in enumerate(lines):
        ch_num = _heading_chapter_num(line)
        if ch_num is not None:
            boundaries.append((i, 'chapter', ch_num))
            continue
        if OVERVIEW_HEADING.match(line):
            # Make sure it's not a stray word inside running text by checking
            # that the previous line is empty or a separator.
            prev = lines[i - 1].strip() if i > 0 else ''
            if not prev or _is_separator(prev):
                boundaries.append((i, 'overview', None))

    # ── Extract text for each section ─────────────────────────────────────
    chapter_texts: dict[int, str] = {}
    book_overview: str = ''

    for idx, (line_i, stype, ch_num) in enumerate(boundaries):
        # Text runs from the line after the heading to the start of the next section
        end_i = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(lines)
        body_lines = lines[line_i + 1: end_i]

        # Strip leading/trailing separators and blanks
        while body_lines and (_is_separator(body_lines[0]) or not body_lines[0].strip()):
            body_lines.pop(0)
        while body_lines and (_is_separator(body_lines[-1]) or not body_lines[-1].strip()):
            body_lines.pop()

        # Remove keyword lines
        body_lines = [l for l in body_lines if not _is_keyword_line(l)]

        text = '\n'.join(body_lines).strip()

        if stype == 'chapter' and ch_num is not None:
            # If we already have text for this chapter number, append
            if ch_num in chapter_texts:
                chapter_texts[ch_num] += '\n\n' + text
            else:
                chapter_texts[ch_num] = text
        elif stype == 'overview':
            book_overview += ('\n\n' + text) if book_overview else text

    return chapter_texts, book_overview.strip()


# ── Fill PKL ───────────────────────────────────────────────────────────────
book_summaries_filled   = 0
chapter_summaries_filled = 0

for book in sorted(pkl.books, key=lambda b: b.book_number):
    bnum  = book.book_number
    rfile = REPORT_FILES.get(bnum)
    if not rfile:
        print(f"Book {bnum:2d}: no report file found — skipping")
        continue

    chapter_texts, book_overview = parse_report(rfile)

    # ── Book summary ───────────────────────────────────────────────────────
    bsums = book.book_summaries
    btext = bsums[0].book_summary_text.strip() if bsums else ''
    if not btext:
        if book_overview:
            if bsums:
                bsums[0].book_summary_text            = book_overview
                bsums[0].book_summary_number_of_words = len(book_overview.split())
                bsums[0].book_summary_author          = SUMMARY_AUTHOR
                bsums[0].book_summary_date            = SUMMARY_DATE
            else:
                from elibrary_cache import BookSummary
                book.book_summaries.append(BookSummary(
                    book_summary_text            = book_overview,
                    book_summary_number_of_words = len(book_overview.split()),
                    book_summary_author          = SUMMARY_AUTHOR,
                    book_summary_date            = SUMMARY_DATE,
                ))
            book_summaries_filled += 1
            print(f"Book {bnum:2d}: book summary filled ({len(book_overview.split())} words)")
        else:
            print(f"Book {bnum:2d}: book summary MISSING in report — nothing to fill")

    # ── Chapter summaries ──────────────────────────────────────────────────
    for ch in book.book_chapters:
        # Skip structural entries (part introductions, appendices, etc.)
        if _STRUCTURAL.match(ch.chapter_title):
            print(f"  Book {bnum:2d} Ch{ch.chapter_number:2d} ({ch.chapter_title[:40]}): structural entry — skipping")
            continue
        csums = ch.chapter_summaries
        ctext = csums[0].chapter_summary_text.strip() if csums else ''
        if not ctext:
            # Try to find by chapter number in the parsed report
            rtext = chapter_texts.get(ch.chapter_number, '').strip()
            if rtext:
                if csums:
                    csums[0].chapter_summary_text            = rtext
                    csums[0].chapter_summary_number_of_words = len(rtext.split())
                    csums[0].chapter_summary_author          = SUMMARY_AUTHOR
                    csums[0].chapter_summary_date            = SUMMARY_DATE
                else:
                    from elibrary_cache import ChapterSummary
                    ch.chapter_summaries.append(ChapterSummary(
                        chapter_summary_text            = rtext,
                        chapter_summary_number_of_words = len(rtext.split()),
                        chapter_summary_author          = SUMMARY_AUTHOR,
                        chapter_summary_date            = SUMMARY_DATE,
                    ))
                chapter_summaries_filled += 1
                print(f"  Book {bnum:2d} Ch{ch.chapter_number:2d}: summary filled ({len(rtext.split())} words)")
            else:
                print(f"  Book {bnum:2d} Ch{ch.chapter_number:2d} ({ch.chapter_title[:40]}): no report section — leaving empty")

print(f"\nBook summaries filled   : {book_summaries_filled}")
print(f"Chapter summaries filled: {chapter_summaries_filled}")

# ── Save PKL ───────────────────────────────────────────────────────────────
print("\nSaving PKL …")
with PKL_PATH.open('wb') as f:
    pickle.dump(pkl, f)
print("PKL saved.")

# ── Regenerate JSON ────────────────────────────────────────────────────────
print("Regenerating JSON …")

def _obj_to_dict(obj) -> object:
    """Recursively convert dataclass/list objects to plain dicts for JSON.
    Text fields (keys ending in '_text') are truncated to 10 words so the
    JSON stays human-readable; full text lives in the PKL.
    """
    if hasattr(obj, '__dict__'):
        result = {}
        for k, v in obj.__dict__.items():
            key = k.replace('_', ' ')
            if k.endswith('_text') and isinstance(v, str):
                result[key] = ' '.join(v.split()[:10])
            else:
                result[key] = _obj_to_dict(v)
        return result
    if isinstance(obj, list):
        return [_obj_to_dict(i) for i in obj]
    return obj

data = _obj_to_dict(pkl)
with JSON_PATH.open('w', encoding='utf-8') as f:
    f.write(json.dumps(data, indent=2, ensure_ascii=False)
            .replace('\u2028', '\\u2028').replace('\u2029', '\\u2029'))
print(f"JSON saved → {JSON_PATH}")
print("Done.")
