"""fill_cache_metadata.py
Fill in missing 'book summary author', 'book summary date',
'chapter summary author', 'chapter summary date' fields in
elibrary_cache.json — only where the field is currently empty or absent.
"""

import json
from pathlib import Path

CACHE_PATH = Path(__file__).resolve().parent.parent / 'cache' / 'elibrary_cache.json'

AUTHOR = 'Claude agent'
DATE   = 'March 2, 2026'


def fill(d: dict, key: str, value: str) -> bool:
    """Set d[key]=value if missing or empty. Returns True if changed."""
    if not d.get(key, ''):
        d[key] = value
        return True
    return False


def main():
    cache = json.loads(CACHE_PATH.read_text('utf-8'))

    book_author_filled = book_date_filled = 0
    ch_author_filled   = ch_date_filled   = 0

    for book in cache['books']:
        for bs in book.get('book summaries', []):
            book_author_filled += fill(bs, 'book summary author', AUTHOR)
            book_date_filled   += fill(bs, 'book summary date',   DATE)

        for chapter in book.get('book chapters', []):
            for cs in chapter.get('chapter summaries', []):
                ch_author_filled += fill(cs, 'chapter summary author', AUTHOR)
                ch_date_filled   += fill(cs, 'chapter summary date',   DATE)

    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2)
            .replace('\u2028', '\\u2028').replace('\u2029', '\\u2029'),
        encoding='utf-8'
    )

    print(f'book summary author   filled: {book_author_filled}')
    print(f'book summary date     filled: {book_date_filled}')
    print(f'chapter summary author filled: {ch_author_filled}')
    print(f'chapter summary date  filled: {ch_date_filled}')
    print('Done.')


if __name__ == '__main__':
    main()
