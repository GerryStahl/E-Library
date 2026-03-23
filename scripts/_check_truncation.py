"""Check that all *_text fields in elibrary_cache.json are truncated to ≤10 words."""
import json

with open('cache/elibrary_cache.json') as f:
    data = json.load(f)

violations = []
samples = []

for b in data['books']:
    bnum = b['book_number']

    # book-level summaries
    for s in (b.get('book_summaries') or []):
        for k, v in s.items():
            if isinstance(v, str) and k.endswith('_text') and v:
                wc = len(v.split())
                if wc > 11:
                    violations.append(f'Book {bnum} book_summaries[{k}]: {wc}w: {repr(v[:80])}')
                elif len(samples) < 12:
                    samples.append(f'Book {bnum} {k} ({wc}w): {repr(v[:70])}')

    # chapter-level
    for ch in (b.get('book_chapters') or []):
        cnum = ch.get('chapter_number', '?')

        ct = ch.get('chapter_text', '')
        if ct:
            wc = len(ct.split())
            if wc > 11:
                violations.append(f'  Book {bnum} Ch {cnum} chapter_text: {wc}w: {repr(ct[:80])}')

        for s in (ch.get('chapter_summaries') or []):
            for k, v in s.items():
                if isinstance(v, str) and k.endswith('_text') and v:
                    wc = len(v.split())
                    if wc > 11:
                        violations.append(f'  Book {bnum} Ch {cnum} {k}: {wc}w: {repr(v[:80])}')
                    elif len(samples) < 12:
                        samples.append(f'  Book {bnum} Ch {cnum} {k} ({wc}w): {repr(v[:70])}')

print(f'VIOLATIONS ({len(violations)}):')
for v in violations:
    print(' ', v)

print()
print(f'GOOD SAMPLES (first {min(len(samples), 12)}):')
for s in samples[:12]:
    print(' ', s)
