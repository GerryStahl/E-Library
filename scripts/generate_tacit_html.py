"""
Create /documents/tacit.html and update the cache for book 2 (2.tacit.pdf)
with book summary text and chapter summary text from /reports/2.tacit_summary.txt.
"""
import sys, re, html
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
sys.path.insert(0, str(CACHE_DIR))
from elibrary_cache import ElibraryCache, BookSummary, ChapterSummary

WORKSPACE = Path(__file__).resolve().parent.parent
REPORT_PATH  = WORKSPACE / "reports" / "2.tacit_summary.txt"
HTML_OUT     = WORKSPACE / "documents" / "tacit.html"
AUTHOR       = "Claude agent"
DATE         = "March 2, 2026"
BOOK_NUMBER  = 2
BOOK_NAME    = "tacit"
BOOK_TITLE   = "Tacit and Explicit Understanding in Computer Support"

# ── 1. Parse the report ────────────────────────────────────────────────────
raw = REPORT_PATH.read_text(encoding="utf-8")

# Separator: a single long row of ━ characters (followed by optional spaces/newline)
SEP = re.compile(r'━{5,}[^\n]*\n')

parts = SEP.split(raw)
# parts[0] = book header (title/subtitle before first separator)
# parts[1..n] = each section: blank line + heading line + body text

sections = {}  # normalised heading -> body text
for part in parts[1:]:       # skip part[0] (book header block)
    part = part.strip()
    if not part:
        continue
    lines = part.splitlines()
    # first non-empty line is the heading
    heading = ""
    body_start = 0
    for idx_l, l in enumerate(lines):
        if l.strip():
            heading = l.strip()
            body_start = idx_l + 1
            break
    body = "\n".join(lines[body_start:]).strip()
    if heading:
        sections[heading] = body

print("Parsed sections:", list(sections.keys()))

# ── 2. Helper: text → list of HTML paragraphs ─────────────────────────────
def to_html_paras(text):
    """Split on blank lines, HTML-escape each paragraph, wrap in <p class='indent'>."""
    paras = re.split(r'\n{2,}', text.strip())
    out = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        escaped = html.escape(p, quote=False)
        # rejoin within-paragraph line breaks as spaces
        escaped = re.sub(r'\n+', ' ', escaped)
        out.append(f'<p class="indent">{escaped}</p>')
    return "\n".join(out)

# ── 3. Map cache chapter titles → section headings in report ──────────────
# Cache chapter titles (from inspect output):
CHAPTER_TO_SECTION = {
    "INTRODUCTION":                              "INTRODUCTION",
    "CHAPTER 1. OVERVIEW":                       "CHAPTER 1. OVERVIEW",
    "CHAPTER 2. THREE METHODOLOGIES OF DESIGN":  "CHAPTER 2. THREE METHODOLOGIES OF DESIGN",
    "CHAPTER 3. INTERPRETATION IN LUNAR HABITAT DESIGN":
                                                 "CHAPTER 3. INTERPRETATION IN LUNAR HABITAT DESIGN",
    "CHAPTER 4. HEIDEGGER\u2019S PHILOSOPHY OF INTERPRETATION":
                                                 "CHAPTER 4. HEIDEGGER'S PHILOSOPHY OF INTERPRETATION",
    "CHAPTER 5. GROUNDING EXPLICIT DESIGN KNOWLEDGE":
                                                 "CHAPTER 5. GROUNDING EXPLICIT DESIGN KNOWLEDGE",
    "CHAPTER 6. A THEORY OF COMPUTER SUPPORT":   "CHAPTER 6. A THEORY OF COMPUTER SUPPORT",
    "CHAPTER 7. RELATED COMPUTER SYSTEMS FOR DESIGN":
                                                 "CHAPTER 7. RELATED COMPUTER SYSTEMS FOR DESIGN",
    "CHAPTER 8. REPRESENTING THE DESIGN SITUATION":
                                                 "CHAPTER 8. REPRESENTING THE DESIGN SITUATION",
    "CHAPTER 9. INTERPRETIVE PERSPECTIVES FOR COLLABORATION":
                                                 "CHAPTER 9. INTERPRETIVE PERSPECTIVES FOR COLLABORATION",
    "CHAPTER 10. A LANGUAGE FOR SUPPORTING INTERPRETATION":
                                                 "CHAPTER 10. A LANGUAGE FOR SUPPORTING INTERPRETATION",
    "CHAPTER 11. CONTRIBUTIONS":                 "CHAPTER 11. CONTRIBUTIONS",
}

# Book summary = OVERVIEW section
BOOK_SUMMARY_HEADING = "OVERVIEW"

# ── 4. Build the HTML ─────────────────────────────────────────────────────
# Book summary paragraphs
book_summary_body = sections.get(BOOK_SUMMARY_HEADING, "")
book_summary_html = to_html_paras(book_summary_body)

# Collect chapter titles+summaries that have report content
chapter_entries = []   # list of (title, body_text) for chapters with summaries
for cache_title, section_heading in CHAPTER_TO_SECTION.items():
    body = sections.get(section_heading, "")
    if body:
        chapter_entries.append((cache_title, body))

# TOC lines
toc_lines = "\n".join(f"{title}<br>" for title, _ in chapter_entries)

# Chapter summary blocks
ch_summary_blocks = []
for title, body in chapter_entries:
    display_title = html.escape(title, quote=False)
    ch_summary_blocks.append(
        f'<p class="indent"><strong>{display_title}</strong></p>\n'
        f'</p>\n'
        f'{to_html_paras(body)}'
    )
ch_summary_html = "\n\n</p>\n".join(ch_summary_blocks)

html_content = f"""\
<!--
Substitute for every term in brackets []
For example, for 1.marx.pdf
[book number] -> 1
[book] -> marx
[book title] -> Marx and Heidegger
[book summary] -> summary of 1.marx.pdf in cache
[chapter 1 title] -> title of first chapter in 1.marx.pdf
[summary of chapter 1] -> summary of first chapter in 1.marx.pdf
-->

<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
<title>Gerry Stahl's e-Library</title>
<!--#include virtual="../../home/header2.html" -->
</head>
<body>

<p class="title"; align="center">Volume {BOOK_NUMBER}. {BOOK_TITLE} </p>
</p>

<p align="center"><img src="{BOOK_NAME}.jpg" height="500"></p>
</p>

{book_summary_html}

<hr>

<h1>download volume</h1>

<p class="indent">
* <strong>Download PDF free for reading online or printing: <a href="{BOOK_NAME}.pdf">{BOOK_NAME}.pdf</a> </strong><BR />
(<em>This is the best version: up-to-date, complete, full color, hi-res.</em>)
<p></p>

<p class="indent">
* Download iBook version free for iPad, Mac, etc.: <a href="{BOOK_NAME}.epub">{BOOK_NAME}.epub</a>  
<p></p>

<p class="indent">
* Download MOBI version free for Kindle: <a href="{BOOK_NAME}.mobi">{BOOK_NAME}.mobi</a>  
<p></p>

<p class="indent">
* Order paperback from Lulu at printing cost: <a href="http://www.lulu.com/spotlight/GerryStahl">Lulu page for Gerry Stahl</a>  
<p></p>

<p class="indent">
* Order paperback from Amazon at printing cost: <a href="https://www.amazon.com/s?k=%22Gerry+Stahl%22">Amazon page for Gerry Stahl</a>  

<hr>

<h1>table of contents</h1>
{toc_lines}

<hr>
<h1>summaries of the chapters</h1>
</p>
{ch_summary_html}

<hr>
<!--#include virtual="../../home/footer.html" -->
</body>
</html>
"""

HTML_OUT.write_text(html_content, encoding="utf-8")
print(f"Written: {HTML_OUT}")

# ── 5. Update the cache ───────────────────────────────────────────────────
cache = ElibraryCache.load()
book  = next(b for b in cache.books if b.book_number == BOOK_NUMBER)

# 5a. Book summary
book_summary_text = f"TACIT AND EXPLICIT UNDERSTANDING — SUMMARY OF THE DISSERTATION\n\n{book_summary_body}"
word_count = len(book_summary_text.split())
if book.book_summaries:
    bs = book.book_summaries[0]
    bs.book_summary_text             = book_summary_text
    bs.book_summary_author           = AUTHOR
    bs.book_summary_date             = DATE
    bs.book_summary_number_of_words  = word_count
else:
    bs = BookSummary(
        book_summary_author=AUTHOR,
        book_summary_date=DATE,
        book_summary_prompt="",
        book_summary_number_of_words=word_count,
        book_summary_text=book_summary_text,
    )
    book.book_summaries = [bs]
print(f"Book summary set ({word_count} words)")

# 5b. Chapter summaries
updated_chapters = 0
for ch in book.book_chapters:
    section_heading = CHAPTER_TO_SECTION.get(ch.chapter_title)
    if section_heading is None:
        continue
    body = sections.get(section_heading, "")
    if not body:
        continue
    wc = len(body.split())
    cs = ChapterSummary(
        chapter_summary_author=AUTHOR,
        chapter_summary_date=DATE,
        chapter_summary_prompt="",
        chapter_summary_number_of_words=wc,
        chapter_summary_text=body,
    )
    ch.chapter_summaries = [cs]
    updated_chapters += 1
    print(f"  Ch {ch.chapter_number}: {ch.chapter_title!r} — {wc} words")

print(f"Updated {updated_chapters} chapter summaries")

cache.save()
print("Cache saved.")
