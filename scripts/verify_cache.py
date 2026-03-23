"""Quick verification of elibrary_cache.pkl after build."""
import sys
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
sys.path.insert(0, str(_CACHE_DIR))

from elibrary_cache import ElibraryCache, ChapterSummary, BookSummary

cache = ElibraryCache.load()
print(repr(cache))
print()

# Book 3
b3 = cache.get_book(3)
print("=== Book 3 ===")
print(f"  name      : {b3.book_name}")
print(f"  title     : {b3.book_title}")
print(f"  author    : {b3.book_author}")
print(f"  kind      : {b3.book_kind}")
print(f"  reference : {b3.book_reference[:80]}...")
print(f"  chapters  : {len(b3.content_chapters)}")
for ch in b3.book_chapters[:4]:
    print(f"    Ch {ch.chapter_number:>2}: {ch.chapter_title!r}  author={ch.chapter_author!r}")
print()

# Book 11, chapter 7
b11 = cache.get_book(11)
print("=== Book 11 ===")
ch7 = b11.get_chapter(7)
print(f"  Ch 7 title     : {ch7.chapter_title}")
print(f"  Ch 7 author    : {ch7.chapter_author}")
print(f"  Ch 7 reference : {ch7.chapter_reference[:80]}...")
print()

# Test summary slots
print("=== Summary slot test ===")
ch7.add_summary(ChapterSummary(
    chapter_summary_author="test", chapter_summary_date="2026-03-01",
    chapter_summary_prompt="Summarise in 300 words", chapter_summary_number_of_words=5,
    chapter_summary_text="This is a test summary.",
))
print(f"  Chapter summary : {ch7.latest_summary()}")

b11.add_summary(BookSummary(
    book_summary_author="test", book_summary_date="2026-03-01",
    book_summary_prompt="Summarise book in 500 words", book_summary_number_of_words=5,
    book_summary_text="Book-level test summary.",
))
print(f"  Book summary    : {b11.latest_summary()}")
print()

# Lookup by name
found = cache.get_book_by_name("19.investigations.pdf")
print(f"Lookup by name '19.investigations.pdf': Book {found.book_number} — {found.book_title[:60]}")
print()

# Full stats table
print(cache.summary_stats())
