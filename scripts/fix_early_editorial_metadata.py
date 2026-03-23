"""
Fix title and author fields for early (2006–2015) editorials using the simple
block structure: block[0]=title, block[1]=author (" & " separated).

Overwrites title/author for all editorials that have a PDF in the 2007-2015
range, so any previously-wrong extractions get corrected.
"""

import sys, re, pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from cache.editorials_cache import EditorialsCache

import fitz  # PyMuPDF

CACHE_PKL  = Path(__file__).parent.parent / "cache" / "editorials_cache.pkl"
CACHE_JSON = CACHE_PKL.with_suffix(".json")
EDT_DIR    = Path(__file__).parent.parent / "editorials"


def clean(s: str) -> str:
    """Normalize whitespace and strip."""
    return re.sub(r"\s+", " ", s.replace("\n", " ")).strip()


def author_from_block(raw: str) -> str:
    """Turn 'A & B & C' or 'A, B and C' into 'A, B, C'."""
    # Replace ' & ' with ', '
    s = re.sub(r"\s*&\s*", ", ", raw)
    return clean(s)


def looks_like_author(s: str) -> bool:
    """Heuristic: short, no URL, no 'Published', no DOI."""
    if len(s) > 200:
        return False
    bad = ("published online", "doi ", "springer", "computer-supported",
           "intern. j.", "©", "#", "http", "e-mail", "@")
    low = s.lower()
    return not any(b in low for b in bad)


def extract_title_author(pdf_path: Path):
    """Return (title, author) from the first page blocks."""
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    raw_blocks = [b[4].strip() for b in page.get_text("blocks") if b[4].strip()]
    doc.close()

    if not raw_blocks:
        return "", ""

    # Block 0 is always the title
    title = clean(raw_blocks[0])

    # Block 1 should be the author — verify heuristic
    author = ""
    if len(raw_blocks) > 1:
        cand = clean(raw_blocks[1])
        if looks_like_author(cand):
            author = author_from_block(cand)
        else:
            # Try block 2 as fallback
            if len(raw_blocks) > 2:
                cand2 = clean(raw_blocks[2])
                if looks_like_author(cand2):
                    author = author_from_block(cand2)

    return title, author


def main():
    cache: EditorialsCache = pickle.loads(CACHE_PKL.read_bytes())

    fixed = 0
    for ed in cache.editorials:
        # Only process early editorials (2007–2015) that have a PDF
        if ed.year > 2015 or ed.year < 2007:
            continue
        if not ed.pdf:
            continue

        pdf_path = EDT_DIR / ed.pdf
        if not pdf_path.exists():
            print(f"  ⚠ PDF not found: {ed.pdf}")
            continue

        title, author = extract_title_author(pdf_path)

        old_title  = ed.title
        old_author = ed.author

        ed.title  = title
        ed.author = author

        t_changed = old_title  != title
        a_changed = old_author != author
        tag = "✓" if author else "⚠"
        print(f"  {tag} Vol {ed.volume} ({ed.year}) Issue {ed.issue}")
        if t_changed:
            print(f"      Title : {title!r}")
        if a_changed:
            print(f"      Author: {author!r}")
        fixed += 1

    # Save
    CACHE_PKL.write_bytes(pickle.dumps(cache))
    print(f"\n  Processed {fixed} early editorial(s).")

    # Re-export JSON using the cache's own serialiser
    import json

    def _trunc(s, n=10):
        words = s.split()
        return " ".join(words[:n]) + ("…" if len(words) > n else "")

    rows = []
    for ed in sorted(cache.editorials, key=lambda e: (e.year, e.volume, e.issue)):
        rows.append({
            "year":   ed.year,
            "volume": ed.volume,
            "issue":  ed.issue,
            "author": ed.author,
            "title":  ed.title,
            "text":   _trunc(ed.text) if ed.text else "",
            "words":  ed.words,
            "summary": _trunc(ed.summary) if ed.summary else "",
            "pdf":    ed.pdf,
        })
    CACHE_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"  PKL saved → {CACHE_PKL}")
    print(f"  JSON saved → {CACHE_JSON}")


if __name__ == "__main__":
    main()
