"""
main.py — Parse the 22 source PDFs and populate elibrary_cache with chapter text and metrics.

Workflow
--------
1. Load elibrary_cache.pkl.
2. For each book in the cache:
   a. Open the matching PDF from sourcepdfs/.
   b. Parse it with PDFParser (using per-PDF chapter-level and font-size settings
      ported from the tested configuration in main.old.py).
   c. Slice the full document text into per-chapter segments using heading positions.
   d. Normalize each parsed chapter title and compare against the book's cache entries.
   e. On a title match  → populate chapter.text, .number_of_pages, .number_of_words,
                           .number_of_tokens, .number_of_symbols.
   f. On no match       → record in the mismatch list.
3. Save the updated cache (pickle + JSON re-export).
4. Write a mismatch report to reports/parsing_mismatches.txt.

No LLM calls are made by this script.

Configuration knobs (all near the top of the file):
  CHAPTER_LEVEL_OVERRIDES   – PDFs whose chapters live at heading level H2 instead of H1
  CHAPTER_FONT_SIZE_FILTERS – minimum font-size gate per PDF
  EXCLUDED_CHAPTER_TITLES   – front/back-matter titles to skip
  SKIP_PAGES                – pages to skip at the start of each PDF (front matter)
  MIN_CHAPTER_WORDS         – discard parsed chapters shorter than this

Author : Gerry Stahl
Created: March 2026
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ── ensure cache dir is on sys.path ─────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent   # elibrary/
CACHE_DIR = ROOT / "cache"                           # elibrary/cache/
if str(CACHE_DIR) not in sys.path:
    sys.path.insert(0, str(CACHE_DIR))

from elibrary_cache import ElibraryCache, Book, Chapter
from parsers.pdf_parser import PDFParser

# ── optional tiktoken for accurate GPT-style token counts ───────────────────
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text))

    TOKEN_METHOD = "tiktoken/cl100k_base"
except ImportError:
    def _count_tokens(text: str) -> int:
        """Approximate: ~1.3 tokens per word (academic English)."""
        return round(len(text.split()) * 1.3)

    TOKEN_METHOD = "approx (words × 1.3)"


# ═══════════════════════════════════════════════════════════════════════════════
# ── Configuration (ported from main.old.py – tested values) ─────────────────
# ═══════════════════════════════════════════════════════════════════════════════

PDF_DIR    = ROOT / "sourcepdfs"
CACHE_PKL  = CACHE_DIR / "elibrary_cache.pkl"
CACHE_JSON = CACHE_DIR / "elibrary_cache.json"
REPORT_DIR = ROOT / "reports"

# Number of front-matter pages to skip at the start of each PDF.
# These are before any real chapter headings (title page, copyright, TOC, etc.).
SKIP_PAGES: int = 3

# PDFs whose content chapters are detected as H2 headings rather than H1.
CHAPTER_LEVEL_OVERRIDES: Dict[str, int] = {
    "9.cscl":       2,   # essays use H2 as chapter level
    "15.global":    2,   # language-section chapters are H2
    "17.proposals": 2,   # individual proposals are H2
}

# Minimum font-size gate for chapter headings (filters out smaller subsections).
CHAPTER_FONT_SIZE_FILTERS: Dict[str, float] = {
    "15.global":    18.0,  # 8 language sections at 18pt; ignore 111 subsections
    "17.proposals": 16.0,  # main proposals at ≥16.1pt; ignore 13.9pt subsections
}

# Chapter titles to exclude from parsing.
#
# EXCLUDED_EXACT  — only match the exact (normalised) title.
#   e.g. "introduction" excludes a bare "Introduction" heading but NOT
#        "Introduction to CSCL" or "Introduction to the Analysis".
#
# EXCLUDED_PREFIX — match exact title AND any title that starts with the
#   term followed by a space (catches plurals, elaborated forms, etc.).
#   e.g. "notes" also excludes "Notes on the Investigations", "Notes & …".

EXCLUDED_EXACT: Set[str] = {
    "introduction",   # standalone only; "Introduction to X" passes through
    "preface",
    "abstract",
    "notice",
    "vita",
}

EXCLUDED_PREFIX: Set[str] = {
    "contents", "notes", "references", "bibliography",
    "tables", "logs", "figures", "acknowledgements", "acknowledgment",
    "authors and collaborators", "notes ….",
    "notes & comments", "notes on the investigations", "note",
    "index of names", "index of terms", "author index", "author's biography",
    "authors biography",
}

# Parsed chapters with fewer words than this are discarded (dividers / graphics).
MIN_CHAPTER_WORDS: int = 100

# Per-PDF titles that are normally excluded by EXCLUDED_EXACT but should be
# allowed through for a specific book.  Keyed by pdf_stem (e.g. "2.tacit").
PER_PDF_ALLOWED_TITLES: Dict[str, Set[str]] = {
    "2.tacit": {"introduction"},  # "Introduction" is a real content chapter here
}


# ═══════════════════════════════════════════════════════════════════════════════
# ── Chapter-exclusion logic (ported verbatim from main.old.py) ───────────────
# ═══════════════════════════════════════════════════════════════════════════════

def is_excluded_chapter(title: str) -> bool:
    """
    Return True for headings that represent front/back matter rather than
    substantive content chapters.

    Checks (in order):
      1. Exact match against EXCLUDED_EXACT  (e.g. bare "Introduction").
      2. Exact match against EXCLUDED_PREFIX.
      3. Plural or 'starts-with' variant of EXCLUDED_PREFIX terms
         (e.g. "Acknowledgments", "Notes on…").
      4. Table-of-contents entries: three or more consecutive dots.

    NOTE: EXCLUDED_EXACT entries are NOT prefix-matched, so
    "Introduction to CSCL" passes through even though "introduction" is
    in EXCLUDED_EXACT.
    """
    normalized = title.lower().strip()
    normalized = re.sub(r"[\s\.…]+$", "", normalized)
    # Normalise apostrophes: curly → straight → remove
    normalized_no_apos = normalized.replace("\u2019", "'").replace("'", "")

    # Exact-only set: only exclude if the entire normalised title matches.
    if normalized in EXCLUDED_EXACT or normalized_no_apos in EXCLUDED_EXACT:
        return True

    # Prefix set: exact match OR starts-with variants.
    if normalized in EXCLUDED_PREFIX or normalized_no_apos in EXCLUDED_PREFIX:
        return True
    for excl in EXCLUDED_PREFIX:
        if (
            normalized == excl + "s"
            or normalized.startswith(excl + " ")
            or normalized_no_apos == excl + "s"
            or normalized_no_apos.startswith(excl + " ")
        ):
            return True

    # Table-of-contents entries (e.g. "Chapter 1 ........ 42")
    if re.search(r"\.{3,}", title):
        return True

    return False


# ═══════════════════════════════════════════════════════════════════════════════
# ── Title normalisation for matching ────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# Matches leading labels like "Chapter I.", "Part 3:", "Section IV –"
_RE_CHAPTER_LABEL = re.compile(
    r"^(?:chapter|part|section)\s+(?:[ivxlcdmIVXLCDM]+|\d+)[\.\:\s\–\-]+",
    re.IGNORECASE,
)
# Matches bare leading digits like "1. " or "10: "
_RE_LEADING_DIGITS = re.compile(r"^\d+[\.\:\s]+")
# Matches bare leading Roman numerals like "IV. "
_RE_LEADING_ROMAN  = re.compile(r"^[ivxlcdmIVXLCDM]+[\.\:\s]+")
# Strips "Introduction to Part I: " / "Introduction to Part 3: " prefix.
# Allows cache titles like "Introduction to Part I: Introducing Group Cognition…"
# to normalise to the same form as the parsed heading "Introducing Group Cognition…".
_RE_INTRO_TO_PART = re.compile(
    r"^introduction\s+to\s+part\s+(?:[ivxlcdmIVXLCDM]+|\d+)[\s\.\:\–\-]+",
    re.IGNORECASE,
)
# Strips trailing " by [Author Name]" from cache titles such as
# "Spanish Translation by Cesar Alberto & Collazos Ordoñez" so they normalise
# to the same form as the shorter parsed heading "Spanish Translation".
# Requires the word after "by" to start with an uppercase/accented letter
# (i.e. a proper name) to avoid false positives on mid-title "by" usage.
_RE_BY_AUTHOR = re.compile(r"\s+by\s+[A-Z\u00C0-\u024F].+$")


def _normalize_title(title: str) -> str:
    """
    Produce a canonical form used for title matching.

    Steps:
      1. Strip surrounding whitespace.
      2. Remove leading "Chapter/Part/Section [Roman/Arabic]." labels.
      3. Remove remaining bare leading "[digits]." labels.
      4. Remove remaining bare leading "[Roman]." labels.
      5. Remove leading "Introduction to Part X:" prefix.
      6. Remove trailing " by [Author Name]" suffix.
      7. Lowercase.
      8. Normalise apostrophes (curly → straight → remove).
      9. Collapse multiple spaces.
    """
    t = title.strip()
    t = _RE_CHAPTER_LABEL.sub("", t).strip()
    t = _RE_LEADING_DIGITS.sub("", t).strip()
    t = _RE_LEADING_ROMAN.sub("", t).strip()
    t = _RE_INTRO_TO_PART.sub("", t).strip()   # e.g. "Introduction to Part I: Foo" → "Foo"
    t = _RE_BY_AUTHOR.sub("", t).strip()        # e.g. "Spanish Translation by X" → "Spanish Translation"
    t = t.lower()
    t = t.replace("\u2019", "'").replace("\u2018", "'")   # curly → straight
    t = t.replace("'", "")                                # remove apostrophes
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# ── PDF parsing helpers ──────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# Matches a bare "Introduction to Part I" / "Introduction to Part 3" heading
# with nothing after the Roman/Arabic numeral — used in the 4.svmt merge.
_RE_BARE_INTRO_PART = re.compile(
    r"^Introduction\s+to\s+Part\s+(?:[IVXivx]+|\d+)\s*$"
)


def _get_chapter_headings(
    heading_structure: List[Dict],
    chapter_level: int,
    min_font_size: Optional[float],
    pdf_stem: str = "",
) -> List[Dict]:
    """
    Filter the full heading list to only the headings that qualify as
    chapter-level headings for this PDF.

    pdf_stem is used for two special cases:
      • 2.tacit  — "Introduction" is a real chapter (not front-matter).
      • 4.svmt   — bare "Introduction to Part X" is immediately followed by
                   its subtitle as a separate heading at a different font size;
                   the two are merged into "Introduction to Part X: Subtitle".
    """
    allowed = PER_PDF_ALLOWED_TITLES.get(pdf_stem, set())

    result = []
    for h in heading_structure:
        if h["level"] != chapter_level:
            continue
        if min_font_size is not None and h.get("font_size", 0) < min_font_size:
            continue
        # Skip headings whose text is clearly a full sentence rather than a
        # chapter title (e.g. a decorative drop-cap opening sentence where only
        # the first letter has a large font, or a garbled OCR fragment ending
        # with a period).  50-char threshold avoids clipping short abbreviations.
        stripped_text = h["text"].strip()
        if stripped_text.endswith(".") and len(stripped_text) > 50:
            continue
        norm = h["text"].lower().strip()
        if norm not in allowed and is_excluded_chapter(h["text"]):
            continue
        result.append(h)

    # ── Merge cross-page continuation headings ───────────────────────────────
    # The PDF parser combines split headings only within the same page.
    # Sometimes a long heading wraps onto the next page; the second fragment
    # starts with a lowercase letter (e.g. "and Reseeding-on-demand").
    # Merge such consecutive pairs so the full title is available for matching.
    merged: List[Dict] = []
    i = 0
    while i < len(result):
        h = result[i]
        if i + 1 < len(result):
            nxt = result[i + 1]
            same_or_next_page = nxt["page"] <= h["page"] + 1
            starts_lower      = nxt["text"][:1].islower()
            same_font         = abs(nxt.get("font_size", 0) - h.get("font_size", 0)) < 2.0
            if same_or_next_page and starts_lower and same_font:
                sep = "" if h["text"].endswith("-") else " "
                combined = dict(h)
                combined["text"] = h["text"] + sep + nxt["text"]
                merged.append(combined)
                i += 2
                continue
        merged.append(h)
        i += 1

    # ── 4.svmt: merge "Part X" / subtitle / "Introduction to Part X" triples ─
    # In this PDF each part opens with three consecutive H1 headings:
    #   1. "Part I"  (36pt)  — decorative title, no cache entry
    #   2. "Introducing Group Cognition in Virtual Math Teams"  (36pt) — subtitle
    #   3. "Introduction to Part I"  (22pt) — the actual section label
    # We collapse these into a single heading:
    #   "Introduction to Part I: Introducing Group Cognition in Virtual Math Teams"
    # using the position of "Part I" so the body text spans from the title page
    # all the way to the first chapter, giving enough words to pass MIN_CHAPTER_WORDS.
    if pdf_stem == "4.svmt":
        _re_part_title = re.compile(r"^Part\s+(?:[IVXivx]+|\d+)\s*$")
        final: List[Dict] = []
        j = 0
        while j < len(merged):
            h = merged[j]
            if (
                _re_part_title.match(h["text"].strip())
                and j + 2 < len(merged)
                and _RE_BARE_INTRO_PART.match(merged[j + 2]["text"].strip())
            ):
                subtitle = merged[j + 1]   # "Introducing Group Cognition…"
                intro    = merged[j + 2]   # "Introduction to Part I"
                combined = dict(h)         # keep "Part I"'s position (earliest → more body text)
                combined["text"] = intro["text"].strip() + ": " + subtitle["text"].strip()
                final.append(combined)
                j += 3
            else:
                final.append(h)
                j += 1
        merged = final

    return merged


def _count_pages_in_range(
    start_pos: int, end_pos: int, page_chunks: List[Dict]
) -> int:
    """Return the number of distinct PDF pages overlapping [start_pos, end_pos)."""
    pages: Set[int] = set()
    for pc in page_chunks:
        if pc["start_pos"] < end_pos and pc["end_pos"] > start_pos:
            pages.add(pc["page_number"])
    return max(1, len(pages))


def _extract_chapter_texts(
    full_text: str,
    chapter_headings: List[Dict],
    page_chunks: List[Dict],
) -> List[Dict]:
    """
    Slice full_text into per-chapter segments using heading character positions.

    Each segment begins immediately after the heading line and ends at the
    start of the next chapter heading (or end-of-document).

    Returns a list of dicts:
      { title, text, start_page, num_pages }
    """
    parsed: List[Dict] = []
    text_len = len(full_text)

    for i, heading in enumerate(chapter_headings):
        seg_start = heading["position"]
        seg_end   = (
            chapter_headings[i + 1]["position"]
            if i + 1 < len(chapter_headings)
            else text_len
        )

        # Advance past the heading line itself
        newline_pos = full_text.find("\n", seg_start)
        body_start  = newline_pos + 1 if newline_pos != -1 and newline_pos < seg_end else seg_start

        chapter_text = full_text[body_start:seg_end].strip()
        word_count   = len(chapter_text.split())

        if word_count < MIN_CHAPTER_WORDS:
            continue   # divider, graphic-heavy page, etc.

        parsed.append({
            "title":      heading["text"],
            "text":       chapter_text,
            "start_page": heading["page"],
            "num_pages":  _count_pages_in_range(seg_start, seg_end, page_chunks),
        })

    return parsed


# ═══════════════════════════════════════════════════════════════════════════════
# ── Per-book processing ──────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

ParsedChapter  = Dict   # one item from _extract_chapter_texts
MismatchRecord = Tuple[int, str, str, str]   # (book_num, book_name, parsed_title, reason)


def process_book(
    book: Book,
    parser: PDFParser,
    verbose: bool = True,
) -> Tuple[int, int, List[MismatchRecord]]:
    """
    Parse one book's PDF and populate matching cache chapters.

    Returns:
        (matched_count, unmatched_parsed_count, list_of_mismatch_records)
    """
    pdf_stem  = book.book_name.replace(".pdf", "")          # e.g. "3.gc"
    pdf_path  = PDF_DIR / book.book_name

    if not pdf_path.exists():
        print(f"  \u2717 PDF not found: {pdf_path}")
        return 0, 0, [(book.book_number, book.book_name, "\u2014", "PDF file not found")]

    chapter_level = CHAPTER_LEVEL_OVERRIDES.get(pdf_stem, 1)
    min_font_size = CHAPTER_FONT_SIZE_FILTERS.get(pdf_stem, None)

    if verbose:
        print(f"\n{'\u2500'*70}")
        print(f"  Book {book.book_number:>2}: {book.book_name}")
        print(f"           heading level = H{chapter_level}"
              + (f", min font = {min_font_size}pt" if min_font_size else ""))

    # ── Parse PDF ────────────────────────────────────────────────────────────
    try:
        doc_data = parser.parse_with_headings(pdf_path, skip_pages=SKIP_PAGES)
    except Exception as exc:
        print(f"  ✗ Parser error: {exc}")
        return 0, 0, [(book.book_number, book.book_name, "—", f"Parser error: {exc}")]

    # ── Extract chapter-level headings & slice text ──────────────────────────
    ch_headings   = _get_chapter_headings(
        doc_data["heading_structure"], chapter_level, min_font_size, pdf_stem=pdf_stem
    )
    parsed_chapters = _extract_chapter_texts(
        doc_data["text"], ch_headings, doc_data["page_chunks"]
    )

    if verbose:
        print(f"           {len(ch_headings)} headings found  →  "
              f"{len(parsed_chapters)} chapter segments (≥{MIN_CHAPTER_WORDS} words)")

    # ── Build lookup: normalised-title → cache Chapter ───────────────────────
    # Use only chapters with non-empty titles; chapter 0 is the book-level entry.
    cache_lookup: Dict[str, Chapter] = {}
    for ch in book.content_chapters:
        if ch.chapter_title:
            norm = _normalize_title(ch.chapter_title)
            cache_lookup[norm] = ch

    # ── Match and populate ───────────────────────────────────────────────────
    mismatches: List[MismatchRecord] = []
    matched = 0
    matched_norm_keys: Set[str] = set()

    for pc in parsed_chapters:
        # Tier 1: exact match (stripped, not lowercased)
        exact_key = pc["title"].strip()
        norm_key  = _normalize_title(pc["title"])

        cache_ch: Optional[Chapter] = None

        # Check exact stripped match first
        for ch in book.content_chapters:
            if ch.chapter_title and ch.chapter_title.strip() == exact_key:
                cache_ch = ch
                break

        # Fall back to normalised match
        if cache_ch is None:
            cache_ch = cache_lookup.get(norm_key)

        if cache_ch is not None:
            # Populate metrics
            cache_ch.chapter_text              = pc["text"]
            cache_ch.chapter_number_of_pages   = pc["num_pages"]
            cache_ch.chapter_number_of_words   = len(pc["text"].split())
            cache_ch.chapter_number_of_tokens  = _count_tokens(pc["text"])
            cache_ch.chapter_number_of_symbols = len(pc["text"])
            matched += 1
            matched_norm_keys.add(_normalize_title(cache_ch.chapter_title))

            if verbose:
                print(f"    ✓ Ch{cache_ch.chapter_number:>2}  {cache_ch.chapter_title[:60]}")
        else:
            mismatches.append((
                book.book_number, book.book_name,
                pc["title"],
                "no matching cache entry",
            ))
            if verbose:
                print(f"    ✗ NO MATCH  »{pc['title'][:60]}«")

    # ── Report cache chapters that received no text ──────────────────────────
    for ch in book.content_chapters:
        if ch.chapter_title and _normalize_title(ch.chapter_title) not in matched_norm_keys:
            mismatches.append((
                book.book_number, book.book_name,
                f"[cache] {ch.chapter_title}",
                "no parsed chapter matched this cache entry",
            ))
            if verbose:
                print(f"    \u26a0  UNMATCHED CACHE  \u00bb{ch.chapter_title[:60]}\u00ab")

    unmatched_parsed = len([m for m in mismatches if not m[2].startswith("[cache]")])
    return matched, unmatched_parsed, mismatches


# ═══════════════════════════════════════════════════════════════════════════════
# ── Report writer ────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def write_mismatch_report(
    all_mismatches: List[MismatchRecord],
    total_matched: int,
    total_cache_chapters: int,
) -> Path:
    """Write parsing_mismatches.txt to REPORT_DIR."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "parsing_mismatches.txt"

    parsed_no_match = [m for m in all_mismatches if not m[2].startswith("[cache]")]
    cache_no_match  = [m for m in all_mismatches if     m[2].startswith("[cache]")]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("ELIBRARY PARSE — CHAPTER MISMATCH REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"  Cache chapters        : {total_cache_chapters}\n")
        f.write(f"  Successfully matched  : {total_matched}\n")
        f.write(f"  Unmatched (parsed)    : {len(parsed_no_match)}\n")
        f.write(f"  Unmatched (cache)     : {len(cache_no_match)}\n\n")

        if parsed_no_match:
            f.write("─" * 80 + "\n")
            f.write("PARSED CHAPTERS WITH NO CACHE MATCH\n")
            f.write("(PDF chapter headings that don't correspond to any cache entry)\n")
            f.write("─" * 80 + "\n")
            last_book = None
            for book_num, book_name, title, reason in parsed_no_match:
                if book_name != last_book:
                    f.write(f"\n  Book {book_num:>2}: {book_name}\n")
                    last_book = book_name
                f.write(f"    • {title}\n")
            f.write("\n")

        if cache_no_match:
            f.write("─" * 80 + "\n")
            f.write("CACHE CHAPTERS WITH NO PARSED MATCH\n")
            f.write("(Cache entries whose text was not populated from the PDF)\n")
            f.write("─" * 80 + "\n")
            last_book = None
            for book_num, book_name, title, reason in cache_no_match:
                display = title.replace("[cache] ", "")
                if book_name != last_book:
                    f.write(f"\n  Book {book_num:>2}: {book_name}\n")
                    last_book = book_name
                f.write(f"    • {display}\n")

    return report_path


# ═══════════════════════════════════════════════════════════════════════════════
# ── JSON re-export ───────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _truncate_text(obj: object, n: int = 10) -> object:
    """Recursively walk dicts/lists and truncate every '*_text' value to *n* words."""
    if isinstance(obj, dict):
        return {
            k: (" ".join(v.split()[:n]) if k.endswith("_text") and isinstance(v, str) else _truncate_text(v, n))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_truncate_text(item, n) for item in obj]
    return obj


def export_json(cache: ElibraryCache) -> None:
    """Serialise cache to elibrary_cache.json (human-readable, text fields truncated to 10 words)."""
    # asdict() already produces prefixed field names after the field rename
    data = _truncate_text(asdict(cache))
    with open(CACHE_JSON, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(data, ensure_ascii=False, indent=2)
                 .replace('\u2028', '\\u2028').replace('\u2029', '\\u2029'))
    print(f"\nJSON exported  → {CACHE_JSON}")


# ═══════════════════════════════════════════════════════════════════════════════
# ── Main entry point ─────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def main(verbose: bool = True) -> None:
    print("=" * 70)
    print("ELIBRARY PDF PARSING PIPELINE")
    print("=" * 70)
    print(f"  Cache    : {CACHE_PKL}")
    print(f"  PDF dir  : {PDF_DIR}")
    print(f"  Skip pages / book : {SKIP_PAGES}")
    print(f"  Token method      : {TOKEN_METHOD}")

    # ── Load cache ───────────────────────────────────────────────────────────
    cache = ElibraryCache.load(str(CACHE_PKL))
    print(f"\nLoaded cache: {cache.total_books} books, {cache.total_chapters} chapters\n")

    # ── Initialise parser ────────────────────────────────────────────────────
    parser = PDFParser()

    # ── Process every book ───────────────────────────────────────────────────
    total_matched    = 0
    total_unmatched  = 0
    all_mismatches: List[MismatchRecord] = []

    for book in cache.books:
        matched, unmatched, mismatches = process_book(book, parser, verbose=verbose)
        total_matched   += matched
        total_unmatched += unmatched
        all_mismatches.extend(mismatches)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PARSING COMPLETE")
    print("=" * 70)
    print(f"  Cache chapters total    : {cache.total_chapters}")
    print(f"  Chapters matched        : {total_matched}")
    print(f"  Parsed, no cache match  : {total_unmatched}")
    print(f"  Cache, no parsed match  : {len(all_mismatches) - total_unmatched}")
    print(f"  Total words in cache    : {cache.total_words:,}")

    # ── Save updated cache ───────────────────────────────────────────────────
    saved = cache.save(str(CACHE_PKL))
    print(f"\nCache saved    → {saved}")

    # ── Re-export JSON ───────────────────────────────────────────────────────
    export_json(cache)

    # ── Write mismatch report ────────────────────────────────────────────────
    report_path = write_mismatch_report(
        all_mismatches, total_matched, cache.total_chapters
    )
    print(f"Mismatch report→ {report_path}")

    if all_mismatches:
        print(f"\n  ⚠  {len(all_mismatches)} mismatch(es) — see report for details.")
    else:
        print("\n  ✓ All chapters matched perfectly.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Parse elibrary PDFs into the cache chapters."
    )
    ap.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-chapter output (show summary only)",
    )
    args = ap.parse_args()

    main(verbose=not args.quiet)
