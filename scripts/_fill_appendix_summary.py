import sys, pickle, json
from pathlib import Path

CACHE_DIR = Path('/Users/GStahl2/AI/elibrary/cache')
sys.path.insert(0, str(CACHE_DIR))

pkl = pickle.load(open(CACHE_DIR / 'elibrary_cache.pkl', 'rb'))
b2  = next(b for b in pkl.books if b.book_number == 2)
ch15 = next(ch for ch in b2.book_chapters if ch.chapter_number == 15)

SUMMARY = """\
The Appendix to Interpretation in Design contains three technical supplements to
the language design work developed in Chapter 10.

Section A, Programming Walkthrough of the HERMES Language, documents two formal
programming walkthroughs (April and August 1992) in which computer-science
experts attempted to write a HERMES query — "list people with four or more
grandchildren" — using the language's syntax. The walkthroughs exposed systematic
writability problems: the query template presented too many syntactic terms
simultaneously; the concept of a query's "Subject" was counterintuitive (the
search must begin from grandparents, not grandchildren, contrary to English
reading); set operations conflicted with ordinary English conjunctions; and the
English-like surface structure obscured the underlying hypertext-navigation
computations. Each problem is traced through four steps of query formulation,
producing concrete redesign decisions — eliminating glue words, renaming terms,
and restructuring the syntax to reflect the system's computational model rather
than English word order.

Section B illustrates tacit, direct-manipulation usage of the resulting language
through representative queries. Section C provides the complete annotated BNF
syntax for the HERMES end-user language, organized by category: DataLists (base,
stored, and computed), Associations (simple, predicates, InputAssociations, and
computed), and Filters (simple, multimedia, and computed). Representative examples
drawn from the lunar-habitat design domain accompany each syntactic category.
Together the three sections serve as both an empirical record of the iterative
design process and a formal reference for the language as implemented.\
"""

WC = len(SUMMARY.split())
print(f"Summary word count: {WC}")

csums = ch15.chapter_summaries
if csums:
    csums[0].chapter_summary_text            = SUMMARY
    csums[0].chapter_summary_number_of_words = WC
    csums[0].chapter_summary_author          = "Claude agent"
    csums[0].chapter_summary_date            = "March 4, 2026"
else:
    from elibrary_cache import ChapterSummary
    ch15.chapter_summaries.append(ChapterSummary(
        chapter_summary_text            = SUMMARY,
        chapter_summary_number_of_words = WC,
        chapter_summary_author          = "Claude agent",
        chapter_summary_date            = "March 4, 2026",
    ))

# Save PKL
with open(CACHE_DIR / 'elibrary_cache.pkl', 'wb') as f:
    pickle.dump(pkl, f)
print("PKL saved.")

# Regenerate JSON
def _obj_to_dict(obj):
    """Recursively convert dataclass/list objects to plain dicts for JSON.
    Text fields (keys ending in '_text') are truncated to 10 words so the
    JSON stays human-readable; full text lives in the PKL.
    """
    if hasattr(obj, '__dict__'):
        result = {}
        for k, v in obj.__dict__.items():
            key = k.replace('_', ' ')
            if k.endswith('_text') and isinstance(v, str):
                result[key] = ' '.join(v.split()[:10])
            else:
                result[key] = _obj_to_dict(v)
        return result
    if isinstance(obj, list):
        return [_obj_to_dict(i) for i in obj]
    return obj

data = _obj_to_dict(pkl)
json_path = CACHE_DIR / 'elibrary_cache.json'
with json_path.open('w', encoding='utf-8') as f:
    f.write(json.dumps(data, indent=2, ensure_ascii=False)
            .replace('\u2028', '\\u2028').replace('\u2029', '\\u2029'))
print(f"JSON saved → {json_path}")
print("Done.")
