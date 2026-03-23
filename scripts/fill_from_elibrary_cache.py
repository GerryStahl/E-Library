"""
Copy text, author, and words from elibrary_cache.pkl (book 16) into
editorials_cache.pkl for any 2006-2015 entry that is missing text.

Does NOT copy summaries.
"""
import sys, re, pickle, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "cache"))

ELIB_PKL = Path(__file__).parent.parent / "cache" / "elibrary_cache.pkl"
EDT_PKL  = Path(__file__).parent.parent / "cache" / "editorials_cache.pkl"
EDT_JSON = EDT_PKL.with_suffix(".json")

# ── Load both caches ──────────────────────────────────────────────────────────
elib  = pickle.loads(ELIB_PKL.read_bytes())
edt   = pickle.loads(EDT_PKL.read_bytes())

# ── Find book 16 ──────────────────────────────────────────────────────────────
book16 = next((b for b in elib.books if b.book_number == 16), None)
if book16 is None:
    sys.exit("ERROR: book 16 not found in elibrary_cache.pkl")

# ── Parse chapter_title prefix  "V(I):" or "V(I1&I2):" ──────────────────────
_PREFIX = re.compile(r'^(\d+)\((\d+)(?:&\d+)?\)\s*:\s*(.*)', re.DOTALL)

def parse_chapter(ch):
    """Return (volume, issue, bare_title) or None."""
    m = _PREFIX.match(ch.chapter_title or "")
    if not m:
        return None
    vol  = int(m.group(1))
    iss  = int(m.group(2))          # first issue for combined (e.g. 2&3 → 2)
    title = m.group(3).strip()
    return vol, iss, title

# ── Build lookup: (vol, iss) → Chapter ───────────────────────────────────────
ch_map: dict[tuple[int,int], object] = {}
for ch in book16.book_chapters:
    parsed = parse_chapter(ch)
    if parsed:
        vol, iss, _ = parsed
        ch_map[(vol, iss)] = ch
        print(f"  mapped {vol}({iss}) → {ch.chapter_title[:60]}")

print(f"\n  {len(ch_map)} chapter(s) mapped from book 16.\n")

# Vol → year for 2006-2015  (vol = year - 2005)
def vol_to_year(vol):
    return vol + 2005

# ── Copy into editorials_cache ────────────────────────────────────────────────
updated = 0
skipped = 0
for ed in edt.editorials:
    if ed.year < 2006 or ed.year > 2015:
        continue
    if ed.text:                  # already has text — skip
        skipped += 1
        continue
    key = (ed.volume, ed.issue)
    ch  = ch_map.get(key)
    if ch is None:
        print(f"  ⚠ No chapter found for Vol {ed.volume} ({ed.year}) Issue {ed.issue}")
        continue
    # Copy fields (no summaries)
    ed.text   = ch.chapter_text or ""
    ed.words  = ch.chapter_number_of_words or 0
    # Clean up author: replace " & " with ", "
    raw_author = ch.chapter_author or ""
    ed.author = re.sub(r"\s*&\s*", ", ", raw_author).strip()
    # Use bare title from chapter if our stub title is empty
    if not ed.title:
        _, _, bare = parse_chapter(ch)
        ed.title = bare
    print(f"  ✓ Vol {ed.volume} ({ed.year}) Issue {ed.issue}  [{ed.words} words]  {ed.author}")
    updated += 1

print(f"\n  Updated: {updated}  |  Already had text (skipped): {skipped}")

# ── Save ──────────────────────────────────────────────────────────────────────
EDT_PKL.write_bytes(pickle.dumps(edt))

def _trunc(s, n=10):
    words = s.split()
    return " ".join(words[:n]) + ("…" if len(words) > n else "")

rows = []
for ed in sorted(edt.editorials, key=lambda e: (e.year, e.volume, e.issue)):
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
EDT_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
print(f"\n  PKL saved → {EDT_PKL}")
print(f"  JSON saved → {EDT_JSON}")
