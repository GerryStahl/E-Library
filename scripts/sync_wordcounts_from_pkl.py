"""sync_wordcounts_from_pkl.py
Update 'book summary number of words' and 'chapter summary number of words'
in elibrary_cache.json using the word counts computed from the full text
stored in elibrary_cache.pkl.

The PKL has the real (non-truncated) text; the JSON texts are truncated.
We match entries by book number and chapter number, then overwrite the
word count in the JSON with len(pkl_text.split()).
"""

import json
import pickle
import sys
from pathlib import Path

CACHE_DIR  = Path(__file__).resolve().parent.parent / 'cache'
PKL_PATH   = CACHE_DIR / 'elibrary_cache.pkl'
JSON_PATH  = CACHE_DIR / 'elibrary_cache.json'

# The ElibraryCache class must be importable
sys.path.insert(0, str(CACHE_DIR))


def wc(text: str) -> str:
    return str(len(text.split())) if text else '0'


def main():
    # Load PKL
    pkl = pickle.load(PKL_PATH.open('rb'))
    # pkl is an ElibraryCache object with a .books list of Book objects

    # Build lookup: book_number -> Book object
    pkl_books = {b.book_number: b for b in pkl.books}

    # Load JSON
    cache = json.loads(JSON_PATH.read_text('utf-8'))

    book_updated = ch_updated = 0

    for jbook in cache['books']:
        bn = jbook['book number']
        pbook = pkl_books.get(bn)
        if pbook is None:
            print(f'Book {bn}: not found in PKL — skipping')
            continue

        # Book summaries
        pkl_bsums = pbook.book_summaries
        for i, jbs in enumerate(jbook.get('book summaries', [])):
            pbs = pkl_bsums[i] if i < len(pkl_bsums) else None
            text = pbs.book_summary_text if pbs else ''
            new_wc = wc(text)
            if jbs.get('book summary number of words', '') != new_wc:
                jbs['book summary number of words'] = new_wc
                book_updated += 1

        # Chapter summaries
        # Build PKL chapter lookup: chapter_number -> Chapter object
        pkl_chapters = {ch.chapter_number: ch for ch in pbook.book_chapters}
        for jch in jbook.get('book chapters', []):
            cn = jch['chapter number']
            pch = pkl_chapters.get(cn)
            if pch is None:
                continue
            pkl_csums = pch.chapter_summaries
            for i, jcs in enumerate(jch.get('chapter summaries', [])):
                pcs = pkl_csums[i] if i < len(pkl_csums) else None
                text = pcs.chapter_summary_text if pcs else ''
                new_wc = wc(text)
                if jcs.get('chapter summary number of words', '') != new_wc:
                    jcs['chapter summary number of words'] = new_wc
                    ch_updated += 1

    JSON_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2)
            .replace('\u2028', '\\u2028').replace('\u2029', '\\u2029'),
        encoding='utf-8'
    )
    print(f'book summary word counts   updated: {book_updated}')
    print(f'chapter summary word counts updated: {ch_updated}')
    print('Done.')


if __name__ == '__main__':
    main()
