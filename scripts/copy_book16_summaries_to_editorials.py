"""
copy_book16_summaries_to_editorials.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Add `summary_book` to early ijCSCL editorials (2006–2015) in editorials_cache.pkl
by copying chapter summaries from `16.ijcscl.pdf` in elibrary_cache.pkl.

For all copied summaries, metadata (`summary_author`, `summary_date`,
`summary_prompt`) comes from book 16's book_summaries[0].

Run from workspace root:
    python scripts/copy_book16_summaries_to_editorials.py
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"

# Keep compatibility with old pickle module paths
sys.path.insert(0, str(CACHE_DIR))
sys.path.insert(0, str(ROOT))

from elibrary_cache import ElibraryCache  # type: ignore
from cache.editorials_cache import EditorialsCache, EditorialSummary

ELIB_PKL = CACHE_DIR / "elibrary_cache.pkl"
EDT_PKL = CACHE_DIR / "editorials_cache.pkl"
EDT_JSON = CACHE_DIR / "editorials_cache.json"

# chapter title prefix: "V(I): ..." or combined issue "V(I&J): ..."
_PREFIX = re.compile(r"^(\d+)\((\d+)(?:&\d+)?\)\s*:\s*(.*)", re.DOTALL)


def parse_chapter_prefix(title: str) -> tuple[int, int] | None:
    m = _PREFIX.match((title or "").strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _truncate_text(obj: object, n: int = 10) -> object:
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if (k == "text" or k.endswith("_text") or k.endswith("_prompt")) and isinstance(v, str) and v:
                result[k] = " ".join(v.split()[:n])
            else:
                result[k] = _truncate_text(v, n)
        return result
    if isinstance(obj, list):
        return [_truncate_text(item, n) for item in obj]
    return obj


def main() -> None:
    elib = ElibraryCache.load(str(ELIB_PKL))
    edt = EditorialsCache.load(str(EDT_PKL))

    book16 = next((b for b in elib.books if b.book_name == "16.ijcscl.pdf"), None)
    if book16 is None:
        raise RuntimeError("Book 16 (16.ijcscl.pdf) not found in elibrary_cache.pkl")
    if not book16.book_summaries:
        raise RuntimeError("Book 16 has no book_summaries entry")

    book_meta = book16.book_summaries[0]

    # map (volume, issue) -> chapter summary text
    chapter_summary_map: dict[tuple[int, int], str] = {}
    missing_chapter_summaries = 0

    for ch in book16.book_chapters:
        key = parse_chapter_prefix(ch.chapter_title)
        if key is None:
            continue
        if not ch.chapter_summaries or not ch.chapter_summaries[0].chapter_summary_text.strip():
            missing_chapter_summaries += 1
            continue
        chapter_summary_map[key] = ch.chapter_summaries[0].chapter_summary_text.strip()

    print(f"Mapped chapter summaries: {len(chapter_summary_map)}")
    if missing_chapter_summaries:
        print(f"Chapters missing chapter_summaries: {missing_chapter_summaries}")

    updated = 0
    missing = 0

    for ed in edt.editorials:
        if not (2006 <= ed.year <= 2015):
            continue

        key = (ed.volume, ed.issue)
        text = chapter_summary_map.get(key)
        if not text:
            print(f"  ⚠ Missing map for {ed.label}: {ed.title}")
            missing += 1
            continue

        ed.summary_book = EditorialSummary(
            summary_author=book_meta.book_summary_author,
            summary_date=book_meta.book_summary_date,
            summary_prompt=book_meta.book_summary_prompt,
            summary_number_of_words=len(text.split()),
            summary_text=text,
        )
        print(f"  ✓ {ed.label} → summary_book ({len(text.split())} words)")
        updated += 1

    edt.save(str(EDT_PKL))

    data = _truncate_text(asdict(edt))
    with open(EDT_JSON, "w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(data, ensure_ascii=False, indent=2)
            .replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029")
        )

    print(f"\nUpdated summary_book for {updated} early editorials")
    print(f"Missing mappings: {missing}")
    print(f"Saved: {EDT_PKL}")
    print(f"Saved: {EDT_JSON}")


if __name__ == "__main__":
    main()
