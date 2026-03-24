"""
build_narrative_chunks.py
─────────────────────────
One-pass cleaning of every level-0 sidecar chunk, producing a new JSON of
narrative-only text suitable for stylistic analysis.

Cleaning pipeline (applied to each chunk in order):
  1. Strip running page headers  — BookTitle + whitespace-only line + PageNum
  2. Strip reference/bibliography sections — from header keyword to end of chunk
  3. Strip labeled chat-log blocks — from "Log N." label through numbered turns
  4. Strip unlabeled runs of chat turns — 3+ consecutive numbered turn lines
  5. Strip figure/table caption lines
  6. Strip residual standalone page numbers
  7. Strip inline footnote clusters — dense numbered footnote blocks (early work)
  8. Normalise whitespace

Output: reports/narrative_chunks.json
  { doc_id: {
      vector_id, book_number, chapter_number, book_title, chapter_title,
      pub_year, cluster_id, cluster_label,
      original_word_count, narrative_word_count,
      narrative_text   } }

Chunks with narrative_word_count < 50 are excluded.
All clusters are retained (no cluster-based filtering here).
"""

import csv
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SIDECAR_PATH  = ROOT / "vector_store" / "chunks_bm25" / "sidecar.json"
CLUSTERS_PATH = ROOT / "reports" / "chunk_clusters.csv"
CACHE_PATH    = ROOT / "cache" / "elibrary_cache.pkl"
OUTPUT_PATH   = ROOT / "reports" / "narrative_chunks.json"

MIN_NARRATIVE_WORDS = 50
MIN_APA_RATIO_TO_EXCLUDE = 0.55   # chunks > 55% APA reference lines → excluded

# ── Compiled regex patterns ──────────────────────────────────────────────────

# 1. Running page headers
#    BookTitle line, then a whitespace-ONLY line, then a page number line.
#    The whitespace-only line is the key discriminator.
RUNNING_HEADER_PAT = re.compile(
    r'\n'
    r'[A-Z][^\n]{3,79}'    # title line (starts with capital, ≤80 chars)
    r'\n'
    r'[ \t]+\n'             # a line containing ONLY spaces/tabs
    r'[ \t]*\d{1,4}[ \t]*' # the page number (possibly with surrounding spaces)
    r'\n',
    re.MULTILINE,
)

# 2. Reference / bibliography sections (strip from header to end of chunk)
REF_SECTION_PAT = re.compile(
    r'\n[ \t]*(References|Bibliography|Works Cited|REFERENCES|BIBLIOGRAPHY'
    r'|Works\s+Cited|Reference List)'
    r'[ \t]*\n'
    r'.*',           # everything after the header
    re.DOTALL | re.IGNORECASE,
)

# 3a. Labeled chat-log blocks
#     Introduced by "Log N." or "Log N-N." on its own line.
LOG_LABEL_PAT = re.compile(
    r'\nLog\s+[\d]+[-.\d]*\.\s*\n',
    re.IGNORECASE,
)

# 3b-pre. Normalise multi-line VMT turn format to single-line before detection.
#   Format B: "\n10  \nImH: \nmessage"  →  "\n10  ImH: message"
#   Format C: "\n350 \n4:31:55 Mic \nmessage" → "\n350 4:31:55 Mic message"
FORMAT_B_NORMALISE = re.compile(
    r'(\n\d{1,3}[ \t]+)\n'          # number line ending in newline
    r'([A-Za-z][a-zA-Z0-9_]{0,20}[ \t]*:[ \t]*)\n'  # Name: on next line
    r'([^\n]*)',                       # message on third line
)
# Also handle 2-line format where Name: and message share a line
FORMAT_B2_NORMALISE = re.compile(
    r'(\n\d{1,3}[ \t]*)\n'              # number line
    r'([A-Za-z][a-zA-Z0-9_]{0,20}[ \t]*:[^\n]+)',   # Name: message on second line
)

# 3b. Individual numbered turn lines (all VMT formats):
#   Format B simple:   "1  \nImH: \n"   (number, then newline, then name:)
#   Format D dense:    "144  mathis: message"
#   Format C table:    "350  4:31:55 Mic  message"
#   Format A timestamp: "17.  \nAvr  \n(8:23:27 PM):"
CHAT_TURN_LINE = re.compile(
    r'^\d{1,3}\.?[ \t]+'                # line or fragment starting with 1-3 digits
    r'(?:'
    r'[A-Za-z][a-zA-Z0-9_]{0,20}[ \t]*[:(]'  # name followed by colon or open-paren
    r'|\d{1,2}:\d{2}'                         # or timestamp (HH:MM)
    r')',
    re.MULTILINE,
)

# 4. Figure / table caption lines
FIG_TABLE_PAT = re.compile(
    r'^[ \t]*(Fig(?:ure)?\.?|Table)[ \t]+[\d\-]+[.\s][^\n]*\n?',
    re.MULTILINE | re.IGNORECASE,
)

# 5. Standalone page numbers (a line containing only digits and whitespace)
LONE_PAGE_NUM = re.compile(
    r'^\s*\d{1,4}\s*$',
    re.MULTILINE,
)

# 6. Inline footnote clusters (Book 1, 2, 3 — old dissertation footnote style)
#    A "footnote line" starts with 1-2 digits then a space, then typical
#    citation content (author name, "Ibid.", "Cf.", or quoted title).
#    We strip runs of 2+ consecutive such lines.
FOOTNOTE_LINE_PAT = re.compile(
    r'^(\d{1,2})[ \t]+'
    r'(?:'
    r'"[^"]{5,}'           # quoted title
    r'|[A-Z][a-z]+,[ \t]'  # Lastname,
    r'|Ibid\.'
    r'|ibid\.'
    r'|Cf\.'
    r'|[A-Z][A-Z]\s'       # abbreviation (e.g. "TW Adorno")
    r')',
    re.MULTILINE,
)


# Detect APA-style reference lines — matches citation entries containing:
# year-in-parens (including YYYY/YYYY translation years), pp., Vol., doi:, Press
_APA_ENTRY = re.compile(
    r'(?:'
    r'\([12]\d{3}[/,\-a-z\d]*\)'   # (YYYY) or (1950/1967) or (2003, rev. ed.)
    r'|pp\.\s*\d+'                  # pp. 123
    r'|Vol\.\s*\d+'                 # Vol. 3
    r'|\bdoi:'                      # DOI
    r'|\bPress\b'                   # publisher word
    r')',
    re.IGNORECASE,
)


def is_bibliography_chunk(text: str) -> bool:
    """Return True if >50% of non-empty lines contain APA citation markers."""
    lines = [l for l in text.split('\n') if l.strip()]
    if len(lines) < 4:
        return False
    matching = sum(1 for l in lines if _APA_ENTRY.search(l))
    return (matching / len(lines)) > 0.50


# ── Cleaning functions ────────────────────────────────────────────────────────

def strip_running_headers(text: str) -> str:
    return RUNNING_HEADER_PAT.sub('\n', text)


def strip_reference_sections(text: str) -> str:
    return REF_SECTION_PAT.sub('', text)


def normalise_multiline_turns(text: str) -> str:
    """Collapse format-B VMT turns into single lines (two passes)."""
    # Pass 1: number\nName:\nmessage  →  number Name: message
    text = FORMAT_B_NORMALISE.sub(r'\1\2\3', text)
    # Pass 2: number\nName: message  →  number Name: message
    text = FORMAT_B2_NORMALISE.sub(r'\1\2', text)
    return text


def strip_chat_logs(text: str) -> str:
    """Remove labeled Log blocks, then any remaining runs of 3+ chat-turn lines."""
    # Pre-step: normalise multiline turn format so CHAT_TURN_LINE can match them
    text = normalise_multiline_turns(text)

    # Phase 1: find "Log N." labels and remove the block of turns that follows.
    parts = LOG_LABEL_PAT.split(text)
    if len(parts) > 1:
        cleaned_parts = [parts[0]]
        # Every odd index is the content AFTER a Log label — strip its turn lines.
        for block in parts[1:]:
            cleaned_block = _strip_turn_run(block)
            cleaned_parts.append(cleaned_block)
        text = '\n'.join(cleaned_parts)

    # Phase 2: strip any remaining runs of 3+ consecutive turn lines
    text = _strip_turn_run_global(text)
    return text


# A "prose line" starts with a capital letter, is long, and is NOT a Name: message
_PROSE_LINE   = re.compile(r'^[A-Z].{30,}')
_NAME_COLON   = re.compile(r'^[A-Za-z][a-zA-Z0-9_]{0,20}[: \t]')


def _strip_turn_run(block: str) -> str:
    """
    Given a block of text that STARTS with a chat excerpt (after a Log label),
    find where narrative prose resumes and return only that part.
    Prose = a line that starts with a capital letter, is >30 chars, and is
    NOT a Name: message line.
    """
    lines = block.split('\n')
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (_PROSE_LINE.match(stripped)
                and not _NAME_COLON.match(stripped)
                and len(stripped) > 30):
            return '\n'.join(lines[i:])
    return ''  # whole block was chat


def _strip_turn_run_global(text: str) -> str:
    """Remove contiguous runs of ≥3 chat-turn lines anywhere in the text."""
    lines = text.split('\n')
    result = []
    run = []
    for line in lines:
        if CHAT_TURN_LINE.match(line):
            run.append(line)
        else:
            if len(run) >= 3:
                pass  # discard the run
            else:
                result.extend(run)
            run = []
            result.append(line)
    if len(run) < 3:
        result.extend(run)
    return '\n'.join(result)


def strip_figure_captions(text: str) -> str:
    return FIG_TABLE_PAT.sub('\n', text)


def strip_lone_page_numbers(text: str) -> str:
    return LONE_PAGE_NUM.sub('', text)


# Improved footnote block detection: blank-line-bounded blocks that start
# with a small number and contain citation markers.
_CITA_MARKER = re.compile(
    r'p\.\s*\d|pp\.\s*\d|\(\d{4}\)|Ibid\.|Cf\.\s|'
    r'\bPress\b|\bUniversity\b|\bJournal\b|\bTrans\.?\b',
    re.IGNORECASE,
)
# Match a blank-ish line, then one or more footnote entries, up to the next blank line.
_FOOTNOTE_BLOCK_PAT = re.compile(
    r'\n[ \t]*\n'          # blank line before block
    r'(\d{1,2})[ \t]+'     # footnote number (1-99)
    r'[A-Z"\'\'«\d]'       # starts with capital, quote, or digit
    r'(?:[^\n]+\n?)*?'     # content (non-greedy, stops at blank line)
    r'(?=\n[ \t]*\n|\Z)',  # lookahead: next blank line or end of string
    re.MULTILINE,
)


def strip_footnote_clusters(text: str) -> str:
    """
    Remove footnote blocks (old dissertation style, Books 1-2).
    A footnote block: blank line → small-number-prefixed lines containing
    citation markers (year, p., Press, etc.) → blank line or end of chunk.
    Only strips when the block contains ≥2 citation markers.
    """
    def maybe_strip(m: re.Match) -> str:
        content = m.group(0)
        if len(_CITA_MARKER.findall(content)) >= 2:
            return '\n\n'  # replace block with a single blank line
        return content

    return _FOOTNOTE_BLOCK_PAT.sub(maybe_strip, text)


def normalize_whitespace(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)           # collapse inline spaces
    text = re.sub(r' *\n *', '\n', text)          # trim spaces around newlines
    text = re.sub(r'\n{3,}', '\n\n', text)        # max two consecutive newlines
    return text.strip()


def clean_chunk(text: str) -> str:
    text = strip_running_headers(text)
    text = strip_reference_sections(text)
    text = strip_chat_logs(text)
    text = strip_figure_captions(text)
    text = strip_lone_page_numbers(text)
    text = strip_footnote_clusters(text)
    text = normalize_whitespace(text)
    return text


def word_count(text: str) -> int:
    return len(text.split())


# ── Data loading ─────────────────────────────────────────────────────────────

def load_pub_years(cache_path: Path) -> dict[tuple[int, int], int]:
    """Return {(book_number, chapter_number): pub_year}."""
    cache = pickle.load(open(cache_path, "rb"))
    YEAR_PAT = re.compile(r'\((\d{4})\)')

    def extract_year(ref: str | None) -> int | None:
        m = YEAR_PAT.search(ref or "")
        return int(m.group(1)) if m else None

    book_year: dict[int, int | None] = {}
    for book in cache.books:
        book_year[book.book_number] = extract_year(book.book_reference)

    result: dict[tuple[int, int], int] = {}
    for book in cache.books:
        by = book_year[book.book_number]
        for ch in book.book_chapters:
            cy = extract_year(ch.chapter_reference)
            year = cy or by or 0
            result[(book.book_number, ch.chapter_number)] = year
    return result


def load_clusters(clusters_path: Path) -> dict[str, tuple[int, str]]:
    """Return {vector_id: (cluster_id, cluster_label)}."""
    result: dict[str, tuple[int, str]] = {}
    with open(clusters_path) as f:
        for row in csv.DictReader(f):
            result[row["vector_id"]] = (int(row["cluster_id"]), row["cluster_label"])
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data...")
    sidecar   = json.loads(SIDECAR_PATH.read_text())
    pub_years = load_pub_years(CACHE_PATH)
    clusters  = load_clusters(CLUSTERS_PATH)

    # Only keep level-0 chunks (base chunks; chunk_level == 0)
    level0_ids = [
        doc_id for doc_id, meta in sidecar.items()
        if meta.get("chunk_level", 0) == 0
    ]
    print(f"Level-0 chunks: {len(level0_ids):,}")

    output: dict[str, dict] = {}
    stats = {
        "total":           0,
        "kept":            0,
        "excluded_short":  0,
        "excluded_biblio": 0,
        "words_original":  0,
        "words_narrative": 0,
    }

    for doc_id in sorted(level0_ids, key=lambda x: int(x)):
        meta  = sidecar[doc_id]
        text  = meta.get("chunk_text", "")
        bk    = meta.get("book_number", 0)
        ch    = meta.get("chapter_number", 0)
        vid   = str(meta.get("vector_id", ""))

        orig_wc = word_count(text)
        stats["total"]          += 1
        stats["words_original"] += orig_wc

        # Exclude raw bibliography chunks BEFORE cleaning (no header present)
        if is_bibliography_chunk(text):
            stats["excluded_biblio"] += 1
            continue

        narrative = clean_chunk(text)
        narr_wc   = word_count(narrative)
        stats["words_narrative"] += narr_wc

        if narr_wc < MIN_NARRATIVE_WORDS:
            stats["excluded_short"] += 1
            continue

        stats["kept"] += 1
        cluster_id, cluster_label = clusters.get(vid, (-1, "unknown"))
        pub_year = pub_years.get((bk, ch), 0)

        output[doc_id] = {
            "vector_id":          vid,
            "book_number":        bk,
            "chapter_number":     ch,
            "book_title":         meta.get("book_title", ""),
            "chapter_title":      meta.get("chapter_title", ""),
            "pub_year":           pub_year,
            "cluster_id":         cluster_id,
            "cluster_label":      cluster_label,
            "original_word_count": orig_wc,
            "narrative_word_count": narr_wc,
            "narrative_text":     narrative,
        }

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2))

    # ── Summary ──
    pct_kept = 100 * stats["kept"] / stats["total"] if stats["total"] else 0
    pct_narr = 100 * stats["words_narrative"] / stats["words_original"] if stats["words_original"] else 0
    print(f"\n{'─'*55}")
    print(f"Total level-0 chunks:       {stats['total']:>8,}")
    print(f"  Kept (≥{MIN_NARRATIVE_WORDS} words):        {stats['kept']:>8,}  ({pct_kept:.1f}%)")
    print(f"  Excluded (too short):     {stats['excluded_short']:>8,}")
    print(f"  Excluded (bibliography):  {stats['excluded_biblio']:>8,}")
    print(f"Original words:             {stats['words_original']:>8,}")
    print(f"Narrative words (kept):     {stats['words_narrative']:>8,}  ({pct_narr:.1f}% of original)")
    print(f"{'─'*55}")
    print(f"\nOutput → {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / 1_000_000:.1f} MB")


if __name__ == "__main__":
    main()
