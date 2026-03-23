"""
generate_all_html.py

For each elibrary book (3-22), parse the report in /reports/,
create the corresponding /documents/<stem>.html, and update the cache
with book summary text, chapter summary text, and keywords.

Handles all report format variations:
  - Separator styles: ━  ═  =  ─  -  (dashes: 3+ chars; others: 5+ chars)
  - Block patterns:  title-only blocks merged with next body block
  - No-separator formats: secondary split on "Chapter N:" heading patterns
  - Book overview headings: OVERVIEW, BOOK OVERVIEW
  - Keywords: extracted from 'Keywords: ...' lines
"""
import sys, re, html as htmlmod
from pathlib import Path
from difflib import SequenceMatcher

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
sys.path.insert(0, str(CACHE_DIR))
from elibrary_cache import ElibraryCache, BookSummary, ChapterSummary

WORKSPACE = Path(__file__).resolve().parent.parent
AUTHOR = "Claude agent"
DATE   = "March 2, 2026"

# ── Separator / line helpers ───────────────────────────────────────────────

def is_sep_line(line):
    """True if line consists entirely of repeated separator chars.
    Dashes require 3+; all other chars require 5+."""
    s = line.strip()
    if not s:
        return False
    c = s[0]
    if c not in '━═=─-':
        return False
    if len(set(s)) != 1:
        return False
    return len(s) >= (3 if c == '-' else 5)

def is_frag_line(line):
    """True if line is a short separator fragment like '==' or '--' (1-6 chars)."""
    s = line.strip()
    return 1 <= len(s) <= 6 and len(set(s)) == 1 and s[0] in '━═=─-'

# ── Keywords helpers ──────────────────────────────────────────────────────

def extract_keywords(block_text):
    """Extract comma-separated keywords from any 'Keywords: ...' line(s)."""
    lines = block_text.splitlines()
    collecting = False
    kw_raw = ''
    for l in lines:
        if re.match(r'Keywords?:', l, re.I):
            kw_raw = re.sub(r'Keywords?:\s*', '', l, flags=re.I)
            collecting = True
        elif collecting:
            s = l.strip()
            if s and not kw_raw.rstrip().endswith('.'):
                kw_raw += ' ' + s
            else:
                collecting = False
    if not kw_raw.strip():
        return []
    return [k.strip() for k in re.split(r',', kw_raw) if k.strip()]

def remove_keywords_lines(text):
    """Remove Keywords: ... line and its continuation."""
    lines = text.splitlines()
    result = []
    skip = False
    for l in lines:
        if re.match(r'Keywords?:', l, re.I):
            skip = True
        elif skip:
            if not l.strip():
                skip = False
        if not skip:
            result.append(l)
    return '\n'.join(result).strip()

# ── Secondary chapter-level splitting ────────────────────────────────────

# Patterns that start a new chapter-level section
_CH_PAT = re.compile(
    r'^((?:Chapter|Investigation)\s+\d+[\.:][^\n]{0,250}|'
    r'BOOK\s+OVERVIEW|OVERVIEW)$',
    re.MULTILINE | re.IGNORECASE
)

def secondary_split(full_block_text):
    """
    If a block contains multiple 'Chapter N:' / 'Chapter N.' headings,
    split it into individual (title, body, keywords) sections.
    Also extracts BOOK OVERVIEW / OVERVIEW as a special section.
    Returns list of (title, body, kw) or [] if < 2 headings found.
    """
    matches = list(_CH_PAT.finditer(full_block_text))
    if len(matches) < 2:
        return []

    sections = []
    for i, m in enumerate(matches):
        sec_title = m.group(1).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(full_block_text)
        body_raw = full_block_text[body_start:body_end].strip()
        kw = extract_keywords(body_raw)
        body_clean = remove_keywords_lines(body_raw)
        body_clean = '\n'.join(
            l for l in body_clean.splitlines() if not is_frag_line(l)
        ).strip()
        sections.append((sec_title, body_clean, kw))
    return sections

# ── Block-based report parser ─────────────────────────────────────────────

def parse_report(raw):
    """
    Parse a report into a list of (title, body, keywords) sections.

    Steps:
    1. Split on separator lines into raw text blocks.
    2. Each block: first real (non-fragment) line = title; rest = body.
    3. Title-only blocks (body < 30 chars) are merged with the next
       non-title-only block; the last title before the body is used as
       the section title (skips PART headers).
    4. Any section whose body contains multiple 'Chapter N:' headings is
       further split via secondary_split().
    """
    lines = raw.splitlines()

    # Step 1: split into raw blocks
    raw_blocks = []
    current = []
    for line in lines:
        if is_sep_line(line):
            content = '\n'.join(current).strip()
            if content:
                raw_blocks.append(content)
            current = []
        else:
            current.append(line)
    content = '\n'.join(current).strip()
    if content:
        raw_blocks.append(content)

    # Step 2: parse each block → (title, body, full)
    def split_title_body(block):
        blines = block.splitlines()
        title = ''
        title_idx = -1
        for i, l in enumerate(blines):
            s = l.strip()
            if s and not is_frag_line(l):
                title = s
                title_idx = i
                break
        if title_idx < 0:
            return '', block
        body = '\n'.join(blines[title_idx + 1:]).strip()
        return title, body

    parsed = []  # (title, body, full_text)
    for block in raw_blocks:
        title, body = split_title_body(block)
        parsed.append((title, body, block))

    # Step 3: merge title-only blocks with next body block.
    # Key rule: if the current block's title is a structural header
    # (Book N:, CHAPTER SUMMARIES, OVERVIEW, PART X:) then use the NEXT
    # body block's own title as the merged section's title, so structural
    # headers don't steal Chapter 1's content.
    # If the current title IS a valid chapter title, keep it.
    _STRUCTURAL = re.compile(
        r'^(?:PART\s+\w+|CHAPTER\s+SUMMARIES|Book\s+\d+:|OVERVIEW|BOOK\s+OVERVIEW)',
        re.I
    )
    merged = []
    i = 0
    while i < len(parsed):
        title, body, full = parsed[i]

        if len(body.strip()) < 30:
            # Find the next block that has real content
            j = i + 1
            # Seed last_title:
            #  - None if current title is a structural header (we'll look further)
            #  - current title otherwise (it IS the chapter title)
            if _STRUCTURAL.match(title):
                last_title = None
            else:
                last_title = title
            while j < len(parsed) and len(parsed[j][1].strip()) < 30:
                candidate_t = parsed[j][0]
                # Update last_title for non-structural titles
                if not _STRUCTURAL.match(candidate_t):
                    last_title = candidate_t
                j += 1

            if j < len(parsed):
                next_title, next_body, next_full = parsed[j]
                # Use last_title (best chapter title seen so far) if available;
                # fall back to the body block's own title only for structural cases.
                use_title = last_title if last_title else next_title
                kw = extract_keywords(next_full)
                body_clean = remove_keywords_lines(next_full)
                body_clean = '\n'.join(
                    l for l in body_clean.splitlines() if not is_frag_line(l)
                ).strip()
                merged.append((use_title, body_clean, kw, next_full))
                i = j + 1
            else:
                i += 1
        else:
            kw = extract_keywords(full)
            body_clean = remove_keywords_lines(body)
            body_clean = '\n'.join(
                l for l in body_clean.splitlines() if not is_frag_line(l)
            ).strip()
            merged.append((title, body_clean, kw, full))
            i += 1

    # Step 4: secondary split for sections containing multiple chapters
    result = []
    for title, body, kw, full in merged:
        sub = secondary_split(full)
        if sub:
            result.extend(sub)
        else:
            result.append((title, body, kw))

    return result  # list of (title, body, keywords_list)

# ── Book overview extraction ──────────────────────────────────────────────

OVERVIEW_TITLES = {
    'overview', 'book overview', 'book summary',
    'collection overview', 'summary of the collection',
    'summary', 'book overview and summary',
}

def find_overview(sections):
    """Return (body, keywords) for the book-level overview section."""
    for title, body, kw in sections:
        if title.lower().strip() in OVERVIEW_TITLES:
            return body, kw
    return '', []

# ── Fuzzy title matching ──────────────────────────────────────────────────

def normalize(s):
    s = s.lower()
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def similarity(a, b):
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()

SKIP_HEADINGS = {
    'chapter summaries', 'book overview', 'overview', 'summary',
    'introduction', 'keywords', 'appendix', 'chapter summaries and collection overview',
    'book summary', 'chapter and book summaries', 'collection overview',
    'chapter summaries and overview', 'book summaries', 'bibliography',
    'references', 'index', 'table of contents', 'foreword', 'preface',
    'chapter summaries and book overview',
}

def match_to_chapters(sections, cache_chapters, book_title, threshold=0.45):
    """
    Match cache chapters to report sections by fuzzy title similarity.
    Pairs are ranked globally by ratio and assigned greedily (best first),
    preventing cascade shifts where one wrong high-ratio match blocks others.
    Returns: {chapter_number: (matched_title, body, keywords, ratio)}
    """
    norm_book_title = normalize(book_title)

    # Build candidates: skip meta/overview sections and book-header sections
    candidates = []
    for i, (title, body, kw) in enumerate(sections):
        nt = normalize(title)
        # Skip known meta headings
        if nt in SKIP_HEADINGS:
            continue
        # Skip if title looks like a book header (starts with "Book N:" or
        # closely matches the full book title)
        if re.match(r'^book\s+\d+', nt):
            continue
        if similarity(title, book_title) > 0.75:
            continue
        # Must have substantial content
        if len(body) < 30:
            continue
        candidates.append((i, title, body, kw))

    # Build all candidate pairs above threshold, then sort by ratio descending
    all_pairs = []
    for cand_idx, (sec_idx, sec_title, sec_body, sec_kw) in enumerate(candidates):
        for ch in cache_chapters:
            r = similarity(sec_title, ch.chapter_title)
            if r >= threshold:
                all_pairs.append((r, ch.chapter_number, cand_idx,
                                   sec_title, sec_body, sec_kw))
    all_pairs.sort(key=lambda x: -x[0])  # best ratio first

    result = {}
    used_chs = set()
    used_cands = set()
    for r, ch_num, cand_idx, sec_title, sec_body, sec_kw in all_pairs:
        if ch_num in used_chs or cand_idx in used_cands:
            continue
        result[ch_num] = (sec_title, sec_body, sec_kw, r)
        used_chs.add(ch_num)
        used_cands.add(cand_idx)

    return result

# ── HTML helpers ──────────────────────────────────────────────────────────

def to_html_paras(text):
    """Convert plain text to <p class='indent'>...</p> blocks."""
    paras = re.split(r'\n{2,}', text.strip())
    out = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        escaped = htmlmod.escape(p, quote=False)
        escaped = re.sub(r'\n+', ' ', escaped)
        out.append(f'<p class="indent">{escaped}</p>')
    return '\n'.join(out)

def build_html(book_number, book_name_stem, book_title, book_summary_html,
               chapter_entries):
    toc_lines = '\n'.join(f'{t}<br>' for t, _ in chapter_entries)

    ch_blocks = []
    for title, body in chapter_entries:
        disp = htmlmod.escape(title, quote=False)
        ch_blocks.append(
            f'<p class="indent"><strong>{disp}</strong></p>\n</p>\n{to_html_paras(body)}'
        )
    ch_html = '\n\n</p>\n'.join(ch_blocks)

    return f"""\
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

<p class="title"; align="center">Volume {book_number}. {book_title} </p>
</p>

<p align="center"><img src="{book_name_stem}.jpg" height="500"></p>
</p>

{book_summary_html}

<hr>

<h1>download volume</h1>

<p class="indent">
* <strong>Download PDF free for reading online or printing: <a href="{book_name_stem}.pdf">{book_name_stem}.pdf</a> </strong><BR />
(<em>This is the best version: up-to-date, complete, full color, hi-res.</em>)
<p></p>

<p class="indent">
* Download iBook version free for iPad, Mac, etc.: <a href="{book_name_stem}.epub">{book_name_stem}.epub</a>  
<p></p>

<p class="indent">
* Download MOBI version free for Kindle: <a href="{book_name_stem}.mobi">{book_name_stem}.mobi</a>  
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
{ch_html}

<hr>
<!--#include virtual="../../home/footer.html" -->
</body>
</html>
"""

# ── Per-book stem mapping ─────────────────────────────────────────────────

def book_stem(book_name):
    """'3.gc.pdf' -> 'gc'"""
    return book_name.replace('.pdf', '').split('.', 1)[1]

# ── Main processing ───────────────────────────────────────────────────────

def process_book(book, cache):
    bn    = book.book_number
    bname = book.book_name
    stem  = book_stem(bname)
    title = book.book_title

    report_path = WORKSPACE / 'reports' / f'{bn}.{stem}_summary.txt'
    if not report_path.exists():
        print(f'  [SKIP] No report found for book {bn}: {bname}')
        return False

    print(f'\nBook {bn}: {bname} ({stem})')
    raw = report_path.read_text('utf-8')

    sections = parse_report(raw)
    print(f'  Parsed {len(sections)} sections: '
          + str([s[0][:40] for s in sections[:6]]))

    overview_body, overview_kw = find_overview(sections)
    matches = match_to_chapters(sections, book.book_chapters, title)
    print(f'  Matched {len(matches)}/{len(book.book_chapters)} chapters')

    # ── Update cache ──────────────────────────────────────────────────

    if overview_body:
        wc = len(overview_body.split())
        if book.book_summaries:
            bs = book.book_summaries[0]
            bs.book_summary_text            = overview_body
            bs.book_summary_author          = AUTHOR
            bs.book_summary_date            = DATE
            bs.book_summary_number_of_words = wc
        else:
            book.book_summaries = [BookSummary(
                book_summary_author=AUTHOR,
                book_summary_date=DATE,
                book_summary_prompt='',
                book_summary_number_of_words=wc,
                book_summary_text=overview_body,
            )]
        print(f'  Book summary set ({wc} words)')

    if overview_kw:
        book.book_keywords = overview_kw

    for ch in book.book_chapters:
        if ch.chapter_number not in matches:
            continue
        matched_title, body, kw, ratio = matches[ch.chapter_number]
        wc = len(body.split())
        cs = ChapterSummary(
            chapter_summary_author=AUTHOR,
            chapter_summary_date=DATE,
            chapter_summary_prompt='',
            chapter_summary_number_of_words=wc,
            chapter_summary_text=body,
        )
        ch.chapter_summaries = [cs]
        if kw:
            ch.chapter_keywords = kw
        print(f'  Ch {ch.chapter_number:2}: r={ratio:.2f} '
              f'{ch.chapter_title[:40]!r} -> {matched_title[:40]!r} '
              f'({wc}w, {len(kw)}kw)')

    # ── Build HTML ────────────────────────────────────────────────────

    book_summary_html = to_html_paras(overview_body) if overview_body else ''

    chapter_entries = []
    for ch in book.book_chapters:
        if ch.chapter_number in matches:
            _, body, _, _ = matches[ch.chapter_number]
            chapter_entries.append((ch.chapter_title, body))

    html_content = build_html(bn, stem, title, book_summary_html, chapter_entries)
    out_path = WORKSPACE / 'documents' / f'{stem}.html'
    out_path.write_text(html_content, encoding='utf-8')
    print(f'  Written: {out_path}')
    return True


def main():
    cache = ElibraryCache.load()
    skip = {1, 2}  # marx and tacit already done

    for book in cache.books:
        if book.book_number in skip:
            continue
        process_book(book, cache)

    cache.save()
    print('\nCache saved.')


if __name__ == '__main__':
    main()
