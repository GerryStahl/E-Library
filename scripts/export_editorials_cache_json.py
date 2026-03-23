"""
export_editorials_cache_json.py — Export editorials_cache.pkl → editorials_cache.json.

Run from the workspace root:

    python scripts/export_editorials_cache_json.py

Produces a human-readable JSON file that mirrors the full editorials cache
structure.  The `text` field is truncated to 10 words (like *_text fields in
the main cache) so the file stays readable; full text lives in the pkl.
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
sys.path.insert(0, str(ROOT))

from cache.editorials_cache import EditorialsCache

INPUT  = CACHE_DIR / "editorials_cache.pkl"
OUTPUT = CACHE_DIR / "editorials_cache.json"

cache = EditorialsCache.load(str(INPUT))


def _truncate_text(obj: object, n: int = 10) -> object:
    """Recursively walk dicts/lists and truncate text fields to *n* words.

    Catches both fields named exactly 'text' and fields ending in '_text'
    (e.g. summary_text).
    """
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


data = _truncate_text(asdict(cache))

with open(OUTPUT, "w", encoding="utf-8") as fh:
    fh.write(
        json.dumps(data, ensure_ascii=False, indent=2)
        .replace('\u2028', '\\u2028')
        .replace('\u2029', '\\u2029')
    )

print(f"Exported {cache.total} editorials → {OUTPUT}")
