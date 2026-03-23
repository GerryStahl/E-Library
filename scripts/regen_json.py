"""Regenerate elibrary_cache.json from elibrary_cache.pkl,
truncating all text/content fields to the first 10 words."""
import sys
import json
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
sys.path.insert(0, str(CACHE_DIR))
from elibrary_cache import ElibraryCache


def trunc(text, n=10):
    """Return first n words of text, or empty string if None/empty."""
    if not text:
        return ""
    words = str(text).split()
    return ' '.join(words[:n])


cache = ElibraryCache.load()

data = {
    "DEFAULT_PATH": getattr(cache, 'DEFAULT_PATH',
                            str(CACHE_DIR / "elibrary_cache.pkl")),
    "books": []
}

for b in cache.books:
    book_summaries = []
    for bs in (b.book_summaries or []):
        book_summaries.append({
            "book summary author": bs.book_summary_author,
            "book summary date": bs.book_summary_date,
            "book summary prompt": trunc(bs.book_summary_prompt),
            "book summary number of words": bs.book_summary_number_of_words,
            "book summary text": trunc(bs.book_summary_text)
        })

    chapters = []
    for ch in b.book_chapters:
        ch_summaries = []
        for cs in (ch.chapter_summaries or []):
            ch_summaries.append({
                "chapter summary author": cs.chapter_summary_author,
                "chapter summary date": cs.chapter_summary_date,
                "chapter summary prompt": trunc(cs.chapter_summary_prompt),
                "chapter summary number of words": cs.chapter_summary_number_of_words,
                "chapter summary text": trunc(cs.chapter_summary_text)
            })
        chapters.append({
            "chapter number": ch.chapter_number,
            "chapter title": ch.chapter_title,
            "chapter author": ch.chapter_author,
            "chapter keywords": ch.chapter_keywords,
            "chapter reference": trunc(ch.chapter_reference),
            "chapter text": trunc(ch.chapter_text),
            "chapter number of pages": ch.chapter_number_of_pages,
            "chapter number of words": ch.chapter_number_of_words,
            "chapter number of tokens": ch.chapter_number_of_tokens,
            "chapter number of symbols": ch.chapter_number_of_symbols,
            "chapter summaries": ch_summaries
        })

    data["books"].append({
        "book number": b.book_number,
        "book name": b.book_name,
        "book title": b.book_title,
        "book author": b.book_author,
        "book keywords": b.book_keywords,
        "book reference": trunc(b.book_reference),
        "book text": trunc(b.book_text),
        "book number of pages": b.book_number_of_pages,
        "book kind": b.book_kind,
        "book summaries": book_summaries,
        "book chapters": chapters
    })

out_path = str(CACHE_DIR / "elibrary_cache.json")
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(data, indent=2, ensure_ascii=False)
            .replace('\u2028', '\\u2028').replace('\u2029', '\\u2029'))

print(f"Written: {out_path}")
print(f"Total books: {len(data['books'])}")
for b in data['books']:
    print(f"  Book {b['book number']:>2}: {b['book title']:<50} "
          f"{len(b['book chapters'])} ch, {len(b['book summaries'])} summaries")
