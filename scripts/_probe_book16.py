"""Probe elibrary_cache.pkl — show book 16 (ijcscl) structure."""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "cache"))

ELIB_PKL = Path(__file__).parent.parent / "cache" / "elibrary_cache.pkl"
cache = pickle.loads(ELIB_PKL.read_bytes())

# Find book 16
book = None
for b in cache.books:
    if b.book_number == 16:
        book = b
        break

if book is None:
    print("Book 16 not found. Available books:")
    for b in cache.books:
        print(f"  {b.book_number}: {b.book_title}")
else:
    print(f"Book: {book.book_title}")
    print(f"Fields: {list(book.__dict__.keys())}")
    print()
    # Find whatever holds the chapters
    for attr, val in book.__dict__.items():
        if isinstance(val, list) and val and attr == 'book_chapters':
            print(f"  List attr '{attr}' has {len(val)} items.")
            item = val[0]
            print(f"    First item type: {type(item).__name__}")
            print(f"    Fields: {list(item.__dict__.keys())}")
            print()
            # Show all chapters
            print(f"  {'#':<5} {'chapter_title'[:55]:<57} {'words':<7} {'author'}")
            print("  " + "-"*110)
            for ch in val:
                t = getattr(ch, 'chapter_title', '')
                a = getattr(ch, 'chapter_author', '')
                w = getattr(ch, 'chapter_number_of_words', '')
                n = getattr(ch, 'chapter_number', '')
                t_disp = (t[:54] + '…') if len(t) > 55 else t
                print(f"  {n:<5} {t_disp:<57} {str(w):<7} {a}")
            break
