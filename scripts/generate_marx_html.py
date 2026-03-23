"""Generate documents/marx.html from the cache data for book 1 (1.marx.pdf)."""
import sys
import html
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
if str(CACHE_DIR) not in sys.path:
    sys.path.insert(0, str(CACHE_DIR))

from elibrary_cache import ElibraryCache  # noqa: E402

CACHE_PATH = CACHE_DIR / 'elibrary_cache.pkl'
OUT_PATH   = ROOT / 'documents' / 'marx.html'

cache = ElibraryCache.load(str(CACHE_PATH))

book = cache.get_book(1)
book_number = book.book_number        # 1
book_name   = "marx"
book_title  = book.book_title         # "Marx and Heidegger"

# ── Book summary (plain text → HTML paragraphs) ──────────────────────────────
def text_to_html_paras(text, css_class="indent"):
    """Wrap each blank-line-separated paragraph in a <p class='indent'> tag."""
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    return '\n'.join(
        f'<p class="{css_class}">{html.escape(p)}</p>' for p in paras
    )

bs = book.latest_summary()
book_summary_html = text_to_html_paras(bs.book_summary_text if bs else "")

# ── Chapters ──────────────────────────────────────────────────────────────────
chapters = book.content_chapters   # chapters with number > 0, in order

# Table of contents
toc_lines = [f'{html.escape(ch.chapter_title)}<br>' for ch in chapters]
toc_html  = '\n'.join(toc_lines)

# Chapter summary blocks
ch_blocks = []
for ch in chapters:
    cs = ch.latest_summary()
    summary_text = cs.chapter_summary_text if cs else ""
    title_esc    = html.escape(ch.chapter_title)
    summary_html = text_to_html_paras(summary_text)
    ch_blocks.append(
        f'<p class="indent"><strong>{title_esc}</strong></p>\n'
        f'{summary_html}\n'
    )
ch_html = '\n\n'.join(ch_blocks)

# ── Assemble page ─────────────────────────────────────────────────────────────
page = f"""\
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

<p class="title"; align="center">Volume {book_number}. {html.escape(book_title)} </p>
</p>

<p align="center"><img src="{book_name}.jpg" height="350"></p>
</p>

{book_summary_html}

<hr>

<h1>download volume</h1>

<p class="indent">
* <strong>Download PDF free for reading online or printing: <a href="{book_name}.pdf">{book_name}.pdf</a> </strong><BR />
(<em>This is the best version: up-to-date, complete, full color, hi-res.</em>)
<p></p>

<p class="indent">
* Download iBook version free for iPad, Mac, etc.: <a href="{book_name}.epub">{book_name}.epub</a>  
<p></p>

<p class="indent">
* Download MOBI version free for Kindle: <a href="{book_name}.mobi">{book_name}.mobi</a>  
<p></p>

<p class="indent">
* Order paperback from Lulu at printing cost: <a href="http://www.lulu.com/spotlight/GerryStahl">Lulu page for Gerry Stahl</a>  
<p></p>

<p class="indent">
* Order paperback from Amazon at printing cost: <a href="https://www.amazon.com/s?k=%22Gerry+Stahl%22">Amazon page for Gerry Stahl</a>  

<hr>

<h1>table of contents</h1>
{toc_html}

<hr>
<h1>summaries of the chapters</h1>

{ch_html}

<hr>

<!--#include virtual="../../home/footer.html" -->
</body>
</html>
"""

with open(OUT_PATH, 'w', encoding='utf-8') as f:
    f.write(page)

print(f"Written: {OUT_PATH}  ({len(page):,} bytes)")
print(f"Chapters in TOC: {len(chapters)}")
