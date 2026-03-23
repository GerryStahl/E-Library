import sys, pickle
from pathlib import Path
CACHE_DIR = Path('/Users/GStahl2/AI/elibrary/cache')
sys.path.insert(0, str(CACHE_DIR))
pkl = pickle.load(open(CACHE_DIR / 'elibrary_cache.pkl', 'rb'))

print("=== BOOK SUMMARIES ===")
for b in sorted(pkl.books, key=lambda x: x.book_number):
    bsums = b.book_summaries
    btext = bsums[0].book_summary_text.strip() if bsums else ''
    bwc = len(btext.split()) if btext else 0
    status = "OK" if bwc > 0 else "MISSING"
    print(f"  Book {b.book_number:2d}: {bwc:4d} words  [{status}]  {b.book_title[:45]}")

print("\n=== CHAPTER SUMMARIES (missing only) ===")
missing = 0
for b in sorted(pkl.books, key=lambda x: x.book_number):
    for ch in b.book_chapters:
        csums = ch.chapter_summaries
        ctext = csums[0].chapter_summary_text.strip() if csums else ''
        cwc = len(ctext.split()) if ctext else 0
        if cwc == 0:
            missing += 1
            print(f"  Book {b.book_number:2d} Ch{ch.chapter_number:2d}: '{ch.chapter_title[:55]}'")
print(f"\nTotal missing chapter summaries: {missing}")
