"""
fill_editorial_texts.py — Parse the 40 ijCSCL editorial PDFs and store
extracted text in editorials_cache.pkl (and re-export JSON).

For each editorial in the cache whose `text` field is empty, the script:
  1. Locates the PDF in editorials/
  2. Extracts all text with PyMuPDF (no pages skipped — editorials are short)
  3. Cleans up whitespace
  4. Stores the result in editorial.text

After all PDFs are processed, the updated pkl is saved and the JSON
is re-exported.

Run:
    python scripts/fill_editorial_texts.py
"""

import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fitz  # PyMuPDF

from cache.editorials_cache import EditorialsCache

EDITORIALS_DIR = ROOT / "editorials"
CACHE_PKL      = ROOT / "cache" / "editorials_cache.pkl"
CACHE_JSON     = ROOT / "cache" / "editorials_cache.json"


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def extract_text(pdf_path: Path) -> str:
    """Extract and clean all text from a PDF using PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    raw = "\n".join(pages)
    # Strip inline superscript digits used as affiliation markers
    # e.g. "Ludvigsen1", "Cress2 &", "Rosé1 ·"  →  "Ludvigsen", "Cress &", "Rosé ·"
    # Only strip digits that immediately follow a Unicode letter (not standalone numbers)
    cleaned = re.sub(r'(?<=[A-Za-z\u00C0-\u024F\u1E00-\u1EFF])\d+', '', raw)
    # Collapse runs of blank lines to a single blank line
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned


def truncate_text(obj: object, n: int = 10) -> object:
    """Recursively truncate 'text' and '*_text' fields to *n* words (for JSON)."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if (k == "text" or k.endswith("_text")) and isinstance(v, str) and v:
                result[k] = " ".join(v.split()[:n])
            else:
                result[k] = truncate_text(v, n)
        return result
    if isinstance(obj, list):
        return [truncate_text(item, n) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cache = EditorialsCache.load(str(CACHE_PKL))

    filled  = 0
    skipped = 0
    missing = []

    for ed in cache.editorials:
        if ed.text.strip():
            skipped += 1
            continue

        pdf_path = EDITORIALS_DIR / ed.pdf
        if not pdf_path.exists():
            print(f"  ⚠  PDF not found: {pdf_path.name}")
            missing.append(ed.label)
            continue

        text = extract_text(pdf_path)
        ed.text = text
        words = len(text.split())
        print(f"  ✓ {ed.label}  ({words:,} words)  {ed.pdf}")
        filled += 1

    print(f"\n{'─'*60}")
    print(f"  Filled : {filled}")
    print(f"  Already had text: {skipped}")
    if missing:
        print(f"  Missing PDFs ({len(missing)}): {missing}")

    # Save pkl
    cache.save(str(CACHE_PKL))
    print(f"\n  PKL saved → {CACHE_PKL}")

    # Re-export JSON (text truncated to 10 words for readability)
    data = truncate_text(asdict(cache))
    with open(CACHE_JSON, "w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(data, ensure_ascii=False, indent=2)
            .replace('\u2028', '\\u2028')
            .replace('\u2029', '\\u2029')
        )
    print(f"  JSON saved → {CACHE_JSON}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
