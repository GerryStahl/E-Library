"""
add_early_editorials_to_cache.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Scans the editorials/ folder for any ijcscl_*.pdf files not yet in
editorials_cache.pkl, adds skeleton Editorial entries for them, then fills:
  • text   — full extracted text (superscript digits stripped)
  • title  — extracted from raw PDF header
  • author — extracted from raw PDF header
  • words  — word count of the cleaned text

Already-populated entries are skipped for text extraction but always have
their title/author/words refreshed (idempotent).

Saves the updated PKL and re-exports editorials_cache.json.

Run AFTER download_editorials_2006_2015.py:
    python scripts/add_early_editorials_to_cache.py

Author  : Gerry Stahl
Created : March 5, 2026
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cache.editorials_cache import Editorial, EditorialsCache, EditorialSummary

EDITORIALS_DIR = ROOT / "editorials"
CACHE_PKL      = ROOT / "cache" / "editorials_cache.pkl"
CACHE_JSON     = ROOT / "cache" / "editorials_cache.json"

FNAME_RE = re.compile(r'^ijcscl_(\d{4})_v(\d+)_i(\d+)_(.+)\.pdf$')


# ── Text extraction ──────────────────────────────────────────────────────────

def extract_clean_text(pdf_path: Path) -> str:
    """Extract all text from a PDF, stripping affiliation-superscript digits."""
    doc = fitz.open(str(pdf_path))
    pages = [page.get_text("text") for page in doc]
    doc.close()
    raw = "\n".join(pages)
    # Remove digit superscripts immediately following a Unicode letter
    cleaned = re.sub(r'(?<=[A-Za-z\u00C0-\u024F\u1E00-\u1EFF])\d+', '', raw)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned


def extract_raw_text(pdf_path: Path) -> str:
    """Extract raw text (superscripts intact) — used only for header parsing."""
    doc = fitz.open(str(pdf_path))
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages)


# ── Header parser (copy of logic in fill_editorial_metadata.py) ─────────────

_AFF_KW = (
    'University', 'Institute', 'Centre', 'Center',
    'Carnegie Mellon', 'Télécom',
    ', USA', ', France', ', Finland', ', Australia', ', Germany',
    ', Netherlands', ', Sweden', ', Norway', ', Denmark', ', Spain',
    ', Italy', ', Japan', ', China', ', Korea', ', Israel', ', Switzerland',
    ', UK', ', England', ', Scotland', ', Ireland',
)


def _is_preamble(line: str) -> bool:
    if line.startswith('\t'):
        return True
    s = line.strip()
    if not s:
        return True
    if s.startswith('Vol.:('):
        return True
    if s.startswith('https://') or s.startswith('http://'):
        return True
    if re.match(r'^1\s+3\s*$', s):
        return True
    if s.startswith('Published online:'):
        return True
    if s.startswith('#') or s.startswith('©'):
        return True
    if 'Intern. J. Comput.' in s:
        return True
    if re.match(r'^\(\d{4}\)\s+\d+', s):
        return True
    if '@' in s:
        return True
    if re.match(r'^\d+\s*$', s):
        return True
    if re.match(r'^[A-Z]{2},\s*USA$', s):
        return True
    if any(kw in s for kw in _AFF_KW) and len(s) > 15:
        if '·' not in s and ' & ' not in s:
            return True
    return False


def _is_author_line(line: str) -> bool:
    s = line.strip().replace('\xa0', ' ')
    if not s:
        return False
    if '·' in s and re.search(r'[A-Za-z\u00C0-\u024F]\d+', s):
        return True
    if (' & ' in s or s.rstrip().endswith('&')) and re.search(r'[A-Za-z\u00C0-\u024F]\d+', s):
        return True
    if (re.match(r'^[A-Z][A-Za-z\.\s\u00C0-\u024F\-]+\d+\s*$', s)
            and len(s) < 70
            and not _is_preamble(line)):
        return True
    return False


def _title_from_filename(pdf: str) -> str:
    stem = Path(pdf).stem
    stem = re.sub(r'^ijcscl_\d{4}_v\d+_i\d+_', '', stem)
    return stem.replace('_', ' ')


def parse_header(text: str, pdf: str = '') -> tuple[str, str]:
    """Return (title, authors) from raw PDF text."""
    text = (text
            .replace('\xa0', ' ').replace('\u202f', ' ')
            .replace('\u2011', '-').replace('\u2010', '-'))
    lines = text.split('\n')

    author_idx = -1
    for i, line in enumerate(lines):
        if _is_author_line(line):
            author_idx = i
            break

    if author_idx == -1:
        return _title_from_filename(pdf), ''

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
    authors = re.sub(r'(?<=[A-Za-z\u00C0-\u024F])\d+', '', raw_author)
    authors = re.sub(r'\s*[·&]\s*', ', ', authors)
    authors = re.sub(r',\s*,', ',', authors).strip().strip(',').strip()

    title_parts = []
    i = author_idx - 1
    while i >= 0:
        if _is_preamble(lines[i]):
            break
        part = lines[i].strip()
        if part:
            title_parts.insert(0, part)
        i -= 1

    title = ' '.join(title_parts).strip() or _title_from_filename(pdf)
    return title, authors


# ── JSON truncation ──────────────────────────────────────────────────────────

def _truncate_text(obj: object, n: int = 10) -> object:
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cache = EditorialsCache.load(str(CACHE_PKL))

    # Build set of PDF filenames already in cache
    known_pdfs = {ed.pdf for ed in cache.editorials}

    # ── Step 1: find new PDFs and add skeleton entries ────────────────────
    new_pdfs = []
    for pdf in sorted(EDITORIALS_DIR.glob('ijcscl_*.pdf')):
        if pdf.name not in known_pdfs:
            m = FNAME_RE.match(pdf.name)
            if not m:
                print(f'  ⚠  Skipping unrecognised filename: {pdf.name}')
                continue
            year  = int(m.group(1))
            vol   = int(m.group(2))
            issue = int(m.group(3))
            ed = Editorial(year=year, volume=vol, issue=issue, pdf=pdf.name)
            cache.add(ed)
            new_pdfs.append(pdf.name)
            print(f'  + Added to cache: {pdf.name}')

    if not new_pdfs:
        print('  No new PDFs found — cache is already up to date.')
    else:
        print(f'\n  {len(new_pdfs)} new editorial(s) added.\n')

    # ── Step 2: fill text for any entry that lacks it ─────────────────────
    print('Filling text fields...')
    text_filled = 0
    for ed in cache.editorials:
        if ed.text.strip():
            continue
        pdf_path = EDITORIALS_DIR / ed.pdf
        if not pdf_path.exists():
            print(f'  ⚠  PDF not found: {ed.pdf}')
            continue
        ed.text = extract_clean_text(pdf_path)
        words = len(ed.text.split())
        print(f'  ✓ {ed.label}  ({words:,} words)')
        text_filled += 1

    print(f'  Text filled: {text_filled}  |  already had text: {cache.total - text_filled}\n')

    # ── Step 3: fill title / author / words for all entries ───────────────
    print('Filling title / author / words...')
    errors = []
    for ed in cache.editorials:
        pdf_path = EDITORIALS_DIR / ed.pdf
        raw = extract_raw_text(pdf_path) if pdf_path.exists() else ed.text
        title, authors = parse_header(raw, ed.pdf)
        ed.title  = title
        ed.author = authors
        ed.words  = len(ed.text.replace('\xa0', ' ').split()) if ed.text else 0

        status = '✓' if title and authors else '⚠'
        print(f'  {status} {ed.label}')
        print(f'      Title : {ed.title}')
        print(f'      Author: {ed.author}')
        if not title or not authors:
            errors.append(ed.label)

    # ── Step 4: save PKL + export JSON ───────────────────────────────────
    cache.save(str(CACHE_PKL))
    print(f'\n  PKL saved → {CACHE_PKL}')

    data = _truncate_text(asdict(cache))
    with open(CACHE_JSON, 'w', encoding='utf-8') as fh:
        fh.write(
            json.dumps(data, ensure_ascii=False, indent=2)
            .replace('\u2028', '\\u2028')
            .replace('\u2029', '\\u2029')
        )
    print(f'  JSON saved → {CACHE_JSON}')

    if errors:
        print(f'\n  ⚠  Missing title or author ({len(errors)}):')
        for e in errors:
            print(f'     {e}')
    else:
        print(f'\n  All {cache.total} editorials complete. ✓')


if __name__ == '__main__':
    main()
