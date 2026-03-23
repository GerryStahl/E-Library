"""
Generate and store a summary for Book 3, Ch 11 (Introduction to Part II).
"""
import sys, pickle
from pathlib import Path
import subprocess

CACHE_DIR = Path('/Users/GStahl2/AI/elibrary/cache')
sys.path.insert(0, str(CACHE_DIR))
from elibrary_cache import ElibraryCache, ChapterSummary

PKL_PATH = CACHE_DIR / 'elibrary_cache.pkl'
cache = ElibraryCache.load(str(PKL_PATH))

b3   = next(b for b in cache.books if b.book_number == 3)
ch11 = next(ch for ch in b3.book_chapters if ch.chapter_number == 11)

SUMMARY = """\
This transitional introduction serves a triple function: it looks back critically \
on the eight groupware design studies of Part I, introduces the theoretical \
frameworks that inform Part II, and previews the five analytical essays that follow.

The retrospective section reassesses each Part I study in light of subsequent \
developments. The Teacher Curriculum Archive (TCA) anticipated digital library \
movements that later materialized through major NSF initiatives, but its \
community-sharing vision remains unrealized. The Essence system used latent \
semantic analysis effectively only through extensive hand-tuning, cautioning \
against treating LSA as a general-purpose model of human understanding. The CREW \
astronaut-grouping software illustrated the data-quality problem that undermines \
AI approaches when adequate contextual data cannot be collected. The Hermes \
domain-oriented design environment proved too costly to configure for practical \
educational deployment. CIE, WebGuide, Synergeia, and BSCL each confronted the \
central obstacle shared by all collaboration systems: the adoption problem. \
Technically sophisticated groupware repeatedly failed because users — even \
motivated students — defaulted to email, messaging, and meetings. The section \
concludes that effective groupware requires designing whole socio-technical \
activity systems, not just adding features, and that this requires a far deeper \
understanding of how groups actually collaborate.

The theoretical background section introduces the two analytic perspectives that \
drive Part II. Socio-cultural psychology, drawing on Vygotsky, focuses on how \
cognition is mediated by artifacts and socially constructed through interaction. \
Communication analysis — particularly ethnomethodology and conversation analysis \
— provides the empirical methods for examining how meaning is actually negotiated \
in small-group discourse.

The preview section maps the five chapters ahead: a graphical model of \
collaborative knowledge building (Ch 9); a methodological critique of CSCL \
research for ignoring group-level phenomena (Ch 10); a theoretical synthesis \
proposing four foundational notions for CSCL (Ch 11, i.e., "Contributions"); \
and two micro-analyses of a 30-second SimRocket transcript that anchor the \
book's central concept of group cognition at the small-group unit of analysis \
(Chs 12–13).\
"""

WC = len(SUMMARY.split())
print(f"Summary word count: {WC}")

# Book 3 Ch 11 currently has no summaries — add one
ch11.chapter_summaries.append(ChapterSummary(
    chapter_summary_author          = "Claude agent",
    chapter_summary_date            = "March 4, 2026",
    chapter_summary_prompt          = "Same prompt as for the book",
    chapter_summary_number_of_words = WC,
    chapter_summary_text            = SUMMARY,
))

with open(PKL_PATH, 'wb') as f:
    pickle.dump(cache, f)
print(f"PKL saved.")

# Regenerate JSON
result = subprocess.run(
    [sys.executable, 'scripts/export_cache_json.py'],
    cwd='/Users/GStahl2/AI/elibrary',
    capture_output=True, text=True
)
print(result.stdout.strip())
if result.returncode != 0:
    print("JSON export error:", result.stderr.strip())

# Verify
ch11_check = next(ch for ch in b3.book_chapters if ch.chapter_number == 11)
print(f"\nVerify: Ch 11 now has {len(ch11_check.chapter_summaries)} summary entry/entries.")
s = ch11_check.chapter_summaries[0]
print(f"  author : {s.chapter_summary_author}")
print(f"  date   : {s.chapter_summary_date}")
print(f"  prompt : {s.chapter_summary_prompt}")
print(f"  words  : {s.chapter_summary_number_of_words}")
print(f"  text   : {s.chapter_summary_text[:80]}...")
