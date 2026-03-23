"""Check that every book in the PKL has exactly one book_summaries entry with a book_summary_prompt."""
import sys
from pathlib import Path

CACHE_DIR = Path('/Users/GStahl2/AI/elibrary/cache')
sys.path.insert(0, str(CACHE_DIR))

from elibrary_cache import ElibraryCache  # noqa: E402

cache = ElibraryCache.load(str(CACHE_DIR / 'elibrary_cache.pkl'))

print(f"Total books in PKL: {len(cache.books)}\n")

issues = []
for b in cache.books:
    bnum = b.book_number
    bname = b.book_name
    summaries = getattr(b, 'book_summaries', None)

    if summaries is None:
        issues.append(f"Book {bnum:>2} ({bname}): book_summaries attribute MISSING")
        continue

    count = len(summaries)
    if count != 1:
        issues.append(f"Book {bnum:>2} ({bname}): has {count} summaries (expected 1)")
        # still report details if count > 0
        for i, s in enumerate(summaries):
            prompt = getattr(s, 'book_summary_prompt', '')
            pw = len(prompt.split()) if prompt else 0
            print(f"  entry[{i}]  prompt_words={pw:>4}  text_words={len(getattr(s,'book_summary_text','').split()):>4}")
        continue

    s = summaries[0]
    prompt = getattr(s, 'book_summary_prompt', '')
    text   = getattr(s, 'book_summary_text', '')
    has_prompt = bool(prompt and prompt.strip())
    prompt_words = len(prompt.split()) if prompt else 0
    text_words   = len(text.split())   if text   else 0

    status = "OK" if has_prompt else "NO PROMPT"
    print(f"Book {bnum:>2}  [{status:9s}]  prompt={prompt_words:>4}w  text={text_words:>4}w  {bname}")
    if not has_prompt:
        issues.append(f"Book {bnum:>2} ({bname}): book_summary_prompt is empty/missing")

print()
if issues:
    print(f"ISSUES ({len(issues)}):")
    for i in issues:
        print(f"  {i}")
else:
    print("All 22 books: exactly 1 summary entry, each with a book_summary_prompt. ✓")
