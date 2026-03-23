"""
Add stub cache entries for the 11 editorials whose PDFs couldn't be downloaded
(paywall before login for 2006 & 2008; intermittent failure for 2007/2 and 2014/9/2).

Titles are known from the Springer article page scraped during the download run.
text / author / words / pdf are left empty — to be filled once PDFs are obtained.
"""
import sys, pickle, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cache.editorials_cache import EditorialsCache, Editorial

CACHE_PKL  = Path(__file__).parent.parent / "cache" / "editorials_cache.pkl"
CACHE_JSON = CACHE_PKL.with_suffix(".json")

# Known missing entries: (year, volume, issue, title)
STUBS = [
    # 2006 — all 4 issues (download ran before login)
    (2006, 1, 1, "ijCSCL—a journal for research in CSCL"),
    (2006, 1, 2, "Building knowledge in the classroom, building knowledge in the CSCL community"),
    (2006, 1, 3, "Focusing on participation in group meaning making"),
    (2006, 1, 4, "Social practices of computer-supported collaborative learning"),
    # 2007 — issues 2 and 4 (issue 3 is the combined-issue duplicate — no separate entry)
    (2007, 2, 2, "A double issue for CSCL 2007"),
    (2007, 2, 4, "CSCL and its flash themes"),
    # 2008 — all 4 issues
    (2008, 3, 1, "The many levels of CSCL"),
    (2008, 3, 2, "The strength of the lone wolf"),
    (2008, 3, 3, "Explorations of participation in discourse"),
    (2008, 3, 4, "CSCL practices"),
    # 2014 — issue 2 only
    (2014, 9, 2, ""),
]

cache: EditorialsCache = pickle.loads(CACHE_PKL.read_bytes())

existing = {(e.year, e.volume, e.issue) for e in cache.editorials}

added = 0
for year, vol, iss, title in STUBS:
    key = (year, vol, iss)
    if key in existing:
        print(f"  skip  {year} v{vol:02d} i{iss}  (already present)")
        continue
    stub = Editorial(
        year=year, volume=vol, issue=iss,
        title=title, author="", text="", words=0, summary="", pdf="",
    )
    cache.editorials.append(stub)
    existing.add(key)
    added += 1
    print(f"  + stub {year} v{vol:02d} i{iss}  {title!r}")

print(f"\n  {added} stub(s) added.  Total in cache: {len(cache.editorials)}")

CACHE_PKL.write_bytes(pickle.dumps(cache))

def _trunc(s, n=10):
    words = s.split()
    return " ".join(words[:n]) + ("…" if len(words) > n else "")

rows = []
for ed in sorted(cache.editorials, key=lambda e: (e.year, e.volume, e.issue)):
    rows.append({
        "year":    ed.year,
        "volume":  ed.volume,
        "issue":   ed.issue,
        "author":  ed.author,
        "title":   ed.title,
        "text":    _trunc(ed.text) if ed.text else "",
        "words":   ed.words,
        "summary": _trunc(ed.summary) if ed.summary else "",
        "pdf":     ed.pdf,
    })
CACHE_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
print(f"  PKL saved → {CACHE_PKL}")
print(f"  JSON saved → {CACHE_JSON}")
