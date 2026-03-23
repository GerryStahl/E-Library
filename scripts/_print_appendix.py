import sys, pickle
from pathlib import Path
CACHE_DIR = Path('/Users/GStahl2/AI/elibrary/cache')
sys.path.insert(0, str(CACHE_DIR))
pkl = pickle.load(open(CACHE_DIR / 'elibrary_cache.pkl', 'rb'))
b2 = next(b for b in pkl.books if b.book_number == 2)
ch15 = next(ch for ch in b2.book_chapters if ch.chapter_number == 15)
print(ch15.chapter_text)
