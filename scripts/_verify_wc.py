import json, pickle, sys
sys.path.insert(0, '/Users/GStahl2/AI/elibrary/cache')
pkl = pickle.load(open('/Users/GStahl2/AI/elibrary/cache/elibrary_cache.pkl','rb'))
cache = json.loads(open('/Users/GStahl2/AI/elibrary/cache/elibrary_cache.json').read())

pkl_b1 = pkl.books[0]
json_b1 = cache['books'][0]

print('=== Book 1 summary ===')
print('  PKL text words:', len(pkl_b1.book_summaries[0].book_summary_text.split()))
print('  JSON stored:   ', json_b1['book summaries'][0]['book summary number of words'])

print()
print('=== Book 1 chapters 1-3 ===')
for pch, jch in zip(pkl_b1.book_chapters[:3], json_b1['book chapters'][:3]):
    pkl_wc = len(pch.chapter_summaries[0].chapter_summary_text.split()) if pch.chapter_summaries else 0
    json_wc = jch['chapter summaries'][0]['chapter summary number of words'] if jch.get('chapter summaries') else 'n/a'
    print(f'  Ch{pch.chapter_number}: PKL={pkl_wc}  JSON={json_wc}')

print()
print('=== All book summary word counts ===')
for pbook, jbook in zip(pkl.books, cache['books']):
    pkl_wc = len(pbook.book_summaries[0].book_summary_text.split()) if pbook.book_summaries else 0
    json_wc = jbook['book summaries'][0]['book summary number of words'] if jbook.get('book summaries') else 'n/a'
    match = 'OK' if str(pkl_wc) == str(json_wc) else 'MISMATCH'
    print(f'  Book {jbook["book number"]:>2}: PKL={pkl_wc:>4}  JSON={json_wc:>4}  {match}')
