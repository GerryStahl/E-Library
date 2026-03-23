"""
fill_editorial_metadata.py — Parse title, author, and word count from the
extracted text of each editorial and store the results in editorials_cache.pkl.

Title/author are extracted by detecting the header layout variant in use:

  Layout A (2016–2020, simple):
      {Title line(s)}
      {Author(s): digit-superscripts + & separators}
      Published online: ... / # Copyright ...

  Layout B (2021+, header block):
      Vol.:(0123456789) / https://doi.org/... / 1 3
      {Title line(s)}
      {Author(s): digit-superscripts + · separators}
       (blank)
      © ...

  Layout C (2023–2024, email-first):
      Published online: ... / © ... / (blank)
      (contact block: name, email, name, email, affiliation...)
      {Title line(s)}
      {Author(s): · separators}

Strategy: find the first author line (detected by separator and digit-superscript
patterns), then collect the non-preamble lines immediately before it as the title.
Falls back to the PDF filename if no title can be extracted.

Run:
    python scripts/fill_editorial_metadata.py
"""

import json
import re
import sys
import fitz  # PyMuPDF — used to re-read raw text for header parsing
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cache.editorials_cache import EditorialsCache

CACHE_PKL      = ROOT / "cache" / "editorials_cache.pkl"
CACHE_JSON     = ROOT / "cache" / "editorials_cache.json"
EDITORIALS_DIR = ROOT / "editorials"

def _raw_text_from_pdf(pdf_path: Path) -> str:
    """Extract raw text from PDF without any superscript stripping — used only for header parsing."""
    doc = fitz.open(str(pdf_path))
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# Preamble detection
# ---------------------------------------------------------------------------

# Affiliation keywords: present in institution/location lines but never in titles.
_AFF_KW = (
    'University', 'Institute', 'Centre', 'Center',
    'Carnegie Mellon', 'Télécom',
    ', USA', ', France', ', Finland', ', Australia', ', Germany',
    ', Netherlands', ', Sweden', ', Norway', ', Denmark', ', Spain',
    ', Italy', ', Japan', ', China', ', Korea', ', Israel', ', Switzerland',
    ', UK', ', England', ', Scotland', ', Ireland',
)


def _is_preamble(line: str) -> bool:
    """True if *line* is metadata/boilerplate rather than title or body text."""
    # Leading tab → affiliation continuation (e.g. '\tCarnegie Mellon...')
    if line.startswith('\t'):
        return True
    s = line.strip()
    if not s:
        return True                           # blank / whitespace-only
    if s.startswith('Vol.:('):               # Vol.:(0123456789)
        return True
    if s.startswith('https://') or s.startswith('http://'):
        return True
    if re.match(r'^1\s+3\s*$', s):          # page-number artefact "1 3"
        return True
    if s.startswith('Published online:'):
        return True
    if s.startswith('#') or s.startswith('©'):
        return True
    if 'Intern. J. Comput.' in s:           # journal abbreviation line
        return True
    if re.match(r'^\(\d{4}\)\s+\d+', s):   # (2020) 15:249–255
        return True
    if '@' in s:                             # e-mail address
        return True
    if re.match(r'^\d+\s*$', s):            # standalone affiliation number "1", "2"
        return True
    if re.match(r'^[A-Z]{2},\s*USA$', s):   # "PA, USA"
        return True
    # Institution / country affiliation lines
    if any(kw in s for kw in _AFF_KW) and len(s) > 15:
        # Guard: affiliation lines never have · or & (those are author separators)
        if '·' not in s and ' & ' not in s:
            return True
    return False


# ---------------------------------------------------------------------------
# Author-line detection
# ---------------------------------------------------------------------------

def _is_author_line(line: str) -> bool:
    """True if *line* looks like an editorial author list."""
    # Normalise non-breaking space before testing
    s = line.strip().replace('\xa0', ' ')
    if not s:
        return False
    # New format: middle-dot separator AND digit superscript
    if '·' in s and re.search(r'[A-Za-z\u00C0-\u024F]\d+', s):
        return True
    # Old format: ampersand separator AND digit superscript
    if (' & ' in s or s.rstrip().endswith('&')) and re.search(r'[A-Za-z\u00C0-\u024F]\d+', s):
        return True
    # Single author: "CapName ... <digit>" — short line, no institution keywords
    if (re.match(r'^[A-Z][A-Za-z\.\s\u00C0-\u024F\-]+\d+\s*$', s)
            and len(s) < 70
            and not _is_preamble(line)):
        return True
    return False


# ---------------------------------------------------------------------------
# Header parser
# ---------------------------------------------------------------------------

def _title_from_filename(pdf: str) -> str:
    """Derive a best-effort title from the PDF filename (used as fallback)."""
    stem = Path(pdf).stem                                    # strip .pdf
    stem = re.sub(r'^ijcscl_\d{4}_v\d+_i\d+_', '', stem)  # strip prefix
    return stem.replace('_', ' ')


def parse_header(text: str, pdf: str = '') -> tuple[str, str]:
    """
    Return (title, authors) extracted from the editorial's full text.

    Both values are lightly cleaned:
      - digit superscripts removed from author names
      - separator characters (·, &) normalised to ", "
      - non-breaking hyphens/spaces normalised
    """
    # Normalise non-breaking characters throughout
    text = (text
            .replace('\xa0', ' ')
            .replace('\u202f', ' ')
            .replace('\u2011', '-')   # non-breaking hyphen → regular hyphen
            .replace('\u2010', '-'))

    lines = text.split('\n')

    # ── Find the first author line ──────────────────────────────────────────
    author_idx = -1
    for i, line in enumerate(lines):
        if _is_author_line(line):
            author_idx = i
            break

    if author_idx == -1:
        # No author line found — fall back to filename for title
        return _title_from_filename(pdf), ''

    # ── Collect author (old format may wrap: line ends with &) ──────────────
    raw_author_parts = [lines[author_idx].strip().replace('\xa0', ' ')]
    j = author_idx + 1
    while raw_author_parts[-1].rstrip().endswith('&') and j < len(lines):
        nxt = lines[j].strip().replace('\xa0', ' ')
        if nxt and not _is_preamble(lines[j]) and re.search(r'[A-Za-z\u00C0-\u024F]\d+', nxt):
            raw_author_parts.append(nxt)
            j += 1
        else:
            break

    raw_author = ' '.join(raw_author_parts)
    # Remove digit superscripts (e.g. "Ludvigsen1" → "Ludvigsen")
    authors = re.sub(r'(?<=[A-Za-z\u00C0-\u024F])\d+', '', raw_author)
    # Normalise separators → comma
    authors = re.sub(r'\s*[·&]\s*', ', ', authors)
    authors = re.sub(r',\s*,', ',', authors).strip().strip(',').strip()

    # ── Collect title: non-preamble lines immediately before author_idx ─────
    title_parts = []
    i = author_idx - 1
    while i >= 0:
        if _is_preamble(lines[i]):
            break
        part = lines[i].strip()
        if part:
            title_parts.insert(0, part)
        i -= 1

    title = ' '.join(title_parts).strip()
    if not title:
        title = _title_from_filename(pdf)

    return title, authors


# ---------------------------------------------------------------------------
# JSON truncation helper
# ---------------------------------------------------------------------------

def _truncate_text(obj: object, n: int = 10) -> object:
    """Recursively truncate 'text' and '*_text' string values to *n* words."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if (k == 'text' or k.endswith('_text')) and isinstance(v, str) and v:
                result[k] = ' '.join(v.split()[:n])
            else:
                result[k] = _truncate_text(v, n)
        return result
    if isinstance(obj, list):
        return [_truncate_text(item, n) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cache = EditorialsCache.load(str(CACHE_PKL))
    print(f"Processing {cache.total} editorials...\n")

    errors = []
    for ed in cache.editorials:
        # Parse header from the RAW pdf text (superscript digits intact),
        # so the author-line detector can find its digit markers.
        pdf_path = EDITORIALS_DIR / ed.pdf
        if pdf_path.exists():
            raw = _raw_text_from_pdf(pdf_path)
        else:
            raw = ed.text  # fallback: use stored clean text if PDF missing

        title, authors = parse_header(raw, ed.pdf)
        ed.title  = title
        ed.author = authors
        ed.words  = len(ed.text.replace('\xa0', ' ').split()) if ed.text else 0

        status = '✓' if title and authors else '⚠'
        print(f"  {status} {ed.label}  ({ed.words:,} words)")
        print(f"      Title : {ed.title}")
        print(f"      Author: {ed.author}")

        if not title or not authors:
            errors.append(ed.label)

    # ── Save PKL ──────────────────────────────────────────────────────────
    cache.save(str(CACHE_PKL))
    print(f"\n  PKL saved → {CACHE_PKL}")

    # ── Re-export JSON ────────────────────────────────────────────────────
    data = _truncate_text(asdict(cache))
    with open(CACHE_JSON, 'w', encoding='utf-8') as fh:
        fh.write(
            json.dumps(data, ensure_ascii=False, indent=2)
            .replace('\u2028', '\\u2028')
            .replace('\u2029', '\\u2029')
        )
    print(f"  JSON saved → {CACHE_JSON}")

    if errors:
        print(f"\n  ⚠  Missing title or author ({len(errors)}):")
        for e in errors:
            print(f"     {e}")
    else:
        print(f"\n  All {cache.total} editorials parsed successfully.")


if __name__ == '__main__':
    main()
