"""Strip affiliation superscript digits from all author fields in the cache."""
import sys, pickle, re, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

CACHE_PKL  = Path(__file__).parent.parent / "cache" / "editorials_cache.pkl"
CACHE_JSON = CACHE_PKL.with_suffix(".json")

def strip_super(s):
    return re.sub(r"(?<=[A-Za-z\u00C0-\u024F\u1E00-\u1EFF])\d+", "", s).strip()

cache = pickle.loads(CACHE_PKL.read_bytes())
changed = 0
for ed in cache.editorials:
    cleaned = strip_super(ed.author)
    if cleaned != ed.author:
        print(f"  Vol {ed.volume} ({ed.year}) Issue {ed.issue}: {ed.author!r} → {cleaned!r}")
        ed.author = cleaned
        changed += 1

print(f"  Fixed {changed} author(s).")
CACHE_PKL.write_bytes(pickle.dumps(cache))

def _trunc(s, n=10):
    words = s.split()
    return " ".join(words[:n]) + ("…" if len(words) > n else "")

rows = []
for ed in sorted(cache.editorials, key=lambda e: (e.year, e.volume, e.issue)):
    rows.append({
        "year": ed.year, "volume": ed.volume, "issue": ed.issue,
        "author": ed.author, "title": ed.title,
        "text": _trunc(ed.text) if ed.text else "",
        "words": ed.words, "summary": _trunc(ed.summary) if ed.summary else "",
        "pdf": ed.pdf,
    })
CACHE_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
print(f"  PKL saved → {CACHE_PKL}")
print(f"  JSON saved → {CACHE_JSON}")
