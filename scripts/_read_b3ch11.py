"""Print Book 3 Ch 11 text and the book summary prompt."""
import sys, pickle
from pathlib import Path

CACHE_DIR = Path('/Users/GStahl2/AI/elibrary/cache')
sys.path.insert(0, str(CACHE_DIR))
from elibrary_cache import ElibraryCache

cache = ElibraryCache.load(str(CACHE_DIR / 'elibrary_cache.pkl'))
b3 = next(b for b in cache.books if b.book_number == 3)

print("=== BOOK PROMPT ===")
print(b3.book_summaries[0].book_summary_prompt)
print()

ch11 = next(ch for ch in b3.book_chapters if ch.chapter_number == 11)
print(f"=== Ch {ch11.chapter_number}: {ch11.chapter_title} ===")
print(f"Words: {ch11.chapter_number_of_words}")
print(f"Summaries: {len(ch11.chapter_summaries)}")
print()
print("--- FULL TEXT ---")
print(ch11.chapter_text)
