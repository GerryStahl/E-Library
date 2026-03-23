"""
export_cache_json.py — Export elibrary_cache.pkl to elibrary_cache.json.

Run from the workspace root:

    python3 scripts/export_cache_json.py

Produces a human-readable JSON file that mirrors the full cache structure.
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
sys.path.insert(0, str(ROOT))

from cache.elibrary_cache import ElibraryCache

INPUT  = CACHE_DIR / "elibrary_cache.pkl"
OUTPUT = CACHE_DIR / "elibrary_cache.json"

cache = ElibraryCache.load(str(INPUT))


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


# After the field rename, asdict(cache) already produces the correct prefixed
# field names (book_summaries, book_chapters, chapter_summaries, etc.).
# We only need to truncate text fields before writing.

# dataclasses → plain dict, then truncate *_text fields
data = _truncate_text(asdict(cache))

with open(OUTPUT, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(data, ensure_ascii=False, indent=2)
                 .replace('\u2028', '\\u2028').replace('\u2029', '\\u2029'))
print(f"Exported {cache.total_books} books / {cache.total_chapters} chapters → {OUTPUT}")
