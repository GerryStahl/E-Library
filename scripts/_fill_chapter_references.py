"""
Fill missing chapter_reference fields in the PKL from documents/chapters.csv.
Only updates chapters where chapter_reference is currently empty.
Skips book-level rows (chapter_number == 0) and rows with empty references.
"""
import sys, csv, pickle, subprocess
from pathlib import Path

BASE_DIR  = Path('/Users/GStahl2/AI/elibrary')
CACHE_DIR = BASE_DIR / 'cache'
CSV_PATH  = BASE_DIR / 'documents' / 'chapters.csv'
PKL_PATH  = CACHE_DIR / 'elibrary_cache.pkl'

sys.path.insert(0, str(CACHE_DIR))
from elibrary_cache import ElibraryCache

cache = ElibraryCache.load(str(PKL_PATH))

# Build lookup: {book_number: {chapter_number: chapter_obj}}
lookup = {}
for b in cache.books:
    lookup[b.book_number] = {ch.chapter_number: ch for ch in b.book_chapters}

filled       = []
skipped_empty = []
already_had  = []
not_found    = []

with open(CSV_PATH, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # skip header row
    for row in reader:
        if len(row) < 8:
            continue
        try:
            bnum  = int(row[0])
            chnum = int(row[4])
        except ValueError:
            continue
        ref = row[7].strip()

        if chnum == 0:       # book-level row — skip
            continue
        if not ref:          # no reference in CSV
            skipped_empty.append((bnum, chnum, row[6][:50]))
            continue

        ch = lookup.get(bnum, {}).get(chnum)
        if ch is None:
            not_found.append((bnum, chnum, row[6][:50]))
            continue

        if ch.chapter_reference.strip():
            already_had.append((bnum, chnum, ch.chapter_title[:50]))
            continue

        ch.chapter_reference = ref
        filled.append((bnum, chnum, ch.chapter_title[:50]))

# ── Save PKL ──────────────────────────────────────────────────────────────
pickle.dump(cache, open(PKL_PATH, 'wb'))
print(f"PKL saved: {PKL_PATH}")

# ── Regenerate JSON ───────────────────────────────────────────────────────
result = subprocess.run(
    [sys.executable, 'scripts/export_cache_json.py'],
    cwd=str(BASE_DIR),
    capture_output=True, text=True
)
if result.returncode == 0:
    print("JSON exported successfully.")
else:
    print("JSON export FAILED:")
    print(result.stderr)

# ── Report ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  CHAPTER REFERENCE FILL REPORT")
print(f"{'='*60}")
print(f"  Filled (newly added):      {len(filled)}")
print(f"  Already had a reference:   {len(already_had)}")
print(f"  No ref in CSV (skipped):   {len(skipped_empty)}")
print(f"  Chapter not in PKL:        {len(not_found)}")
print()

if filled:
    print("FILLED:")
    for bnum, chnum, title in sorted(filled):
        print(f"  Book {bnum:2d} Ch {chnum:2d}  {title}")

if not_found:
    print("\nNOT FOUND IN PKL (CSV has entry but PKL doesn't):")
    for bnum, chnum, title in sorted(not_found):
        print(f"  Book {bnum:2d} Ch {chnum:2d}  {title}")

if skipped_empty:
    print(f"\nSKIPPED (no reference in CSV) — {len(skipped_empty)} chapters")
