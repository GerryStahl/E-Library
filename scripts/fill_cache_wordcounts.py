"""fill_cache_wordcounts.py
Compute and fill missing 'book summary number of words' and
'chapter summary number of words' from their respective text fields.
Only fills where the field is currently absent or empty.
"""

import json
from pathlib import Path

CACHE_PATH = Path(__file__).resolve().parent.parent / 'cache' / 'elibrary_cache.json'


def word_count(text: str) -> str:
    return str(len(text.split()))


def fill(d: dict, count_key: str, text_key: str) -> bool:
    if not d.get(count_key, ''):
        text = d.get(text_key, '')
        if text:
            d[count_key] = word_count(text)
            return True
    return False


def main():
    cache = json.loads(CACHE_PATH.read_text('utf-8'))

    book_filled = ch_filled = 0

    for book in cache['books']:
        for bs in book.get('book summaries', []):
            book_filled += fill(bs, 'book summary number of words', 'book summary text')

        for chapter in book.get('book chapters', []):
            for cs in chapter.get('chapter summaries', []):
                ch_filled += fill(cs, 'chapter summary number of words', 'chapter summary text')

    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2)
            .replace('\u2028', '\\u2028').replace('\u2029', '\\u2029'),
        encoding='utf-8'
    )

    print(f'book summary number of words    filled: {book_filled}')
    print(f'chapter summary number of words filled: {ch_filled}')
    print('Done.')


if __name__ == '__main__':
    main()
