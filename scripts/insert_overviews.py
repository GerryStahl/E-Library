"""insert_overviews.py
Extract the overview block from each numbered report and insert it into
the corresponding HTML in websites/, after the TOC and before the chapter
summaries.  Also fixes any already-inserted overviews that are missing
<p></p> spacers between paragraphs.

Heading patterns handled:
  • BOOK OVERVIEW / OVERVIEW / OVERVIEW OF THE COLLECTION  (books 5-9, 11-14, 19)
  • Overview: <Book Title>                                  (books 15-18, 20-22)
  • <Title> — Collection Overview                          (book 10)
"""

import re
import html as htmlmod
from pathlib import Path

WORKSPACE   = Path(__file__).resolve().parent.parent
REPORTS_DIR = WORKSPACE / 'reports'
WEBSITES    = WORKSPACE / 'websites'

# Map book number → html stem.
# Books 1-4 were done manually by the user; we still process them here
# so their paragraph spacing can be fixed if needed.
BOOK_MAP = {
    2:  'tacit',
    3:  'gc',
    4:  'svmt',
    5:  'euclid',
    6:  'analysis',
    7:  'philosophy',
    8:  'software',
    9:  'cscl',
    10: 'science',
    11: 'theory',
    12: 'math',
    13: 'dynamic',
    14: 'topics',
    15: 'global',
    16: 'ijcscl',
    17: 'proposals',
    18: 'overview',
    19: 'investigations',
    20: 'form',
    21: 'game',
    22: 'climate',
}

# Separator: 5+ of = ═ ━ — -
_SEP_RE = re.compile(r'^[\u2550=\u2501\u2014\u2012\-]{5,}$')

# Overview heading: any of the corpus patterns (case-insensitive)
_OV_HEAD_RE = re.compile(
    r'(?i)^(?:'
    r'book\s+overview'
    r'|overview(?:\s+of\s+the\s+collection)?(?::\s*.+)?'
    r'|.+\bCollection\s+Overview\b'
    r')$'
)


def extract_overview(report_text: str) -> str:
    """
    Find the overview section in a report and return its body text.
    The heading must be adjacent to a separator line (above or below).
    """
    lines = report_text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not _OV_HEAD_RE.match(stripped):
            continue
        # Look within 3 lines in each direction (blank lines may intervene)
        prev_sep = any(
            _SEP_RE.match(lines[j].strip())
            for j in range(max(0, i - 3), i)
        )
        next_sep = any(
            _SEP_RE.match(lines[j].strip())
            for j in range(i + 1, min(len(lines), i + 4))
        )
        if not (prev_sep or next_sep):
            continue
        # Skip heading + any following separator + blank lines
        start = i + 1
        while start < len(lines) and (
            _SEP_RE.match(lines[start].strip()) or lines[start].strip() == ''
        ):
            start += 1
        return '\n'.join(lines[start:]).strip()
    return ''


def to_html_paragraphs(text: str) -> str:
    """
    Convert plain-text paragraphs (separated by blank lines) to HTML.
    Each paragraph → <p class="indent">…</p>.
    A <p></p> spacer is placed between consecutive paragraphs.
    *italic* → <em>italic</em>; HTML entities are escaped.
    """
    paras = re.split(r'\n\s*\n', text)
    html_parts = []
    for para in paras:
        para = ' '.join(para.split('\n')).strip()
        if not para:
            continue
        para = htmlmod.escape(para)
        para = re.sub(r'\*(.+?)\*', r'<em>\1</em>', para)
        html_parts.append(f'<p class="indent">{para}</p>')
    return '\n<p></p>\n'.join(html_parts)


INSERTION_MARKER = '\n<hr>\n<h1>summaries of the chapters</h1>'


def insert_overview_block(html_text: str, overview_html: str) -> str | None:
    """Insert the overview block before the chapter-summaries section."""
    if INSERTION_MARKER not in html_text:
        return None
    block = '\n<hr>\n<h1>Overview</h1>\n<p></p>\n' + overview_html + '\n'
    return html_text.replace(INSERTION_MARKER, block + INSERTION_MARKER, 1)


def fix_paragraph_spacing(html_text: str) -> tuple[str, int]:
    """
    In the overview section of an already-processed HTML file, insert
    <p></p> between consecutive <p class="indent"> paragraphs where it
    is missing.  Returns (new_html, number_of_spacers_added).
    """
    ov_start = html_text.find('<h1>Overview</h1>')
    if ov_start == -1:
        return html_text, 0
    hr_after = html_text.find('<hr>', ov_start + 1)
    if hr_after == -1:
        hr_after = len(html_text)

    section = html_text[ov_start:hr_after]
    # Insert <p></p> between </p> and <p class="indent"> only where absent
    fixed, n = re.subn(
        r'(?<!<p>)(</p>)\n(<p class="indent">)',
        r'\1\n<p></p>\n\2',
        section
    )
    return html_text[:ov_start] + fixed + html_text[hr_after:], n


def main():
    for book_num, stem in sorted(BOOK_MAP.items()):
        # Locate the report (prefer exact name, fall back to number prefix)
        report_path = REPORTS_DIR / f'{book_num}.{stem}_summary.txt'
        if not report_path.exists():
            candidates = list(REPORTS_DIR.glob(f'{book_num}.*.txt'))
            if not candidates:
                print(f'Book {book_num}: report not found — skipping')
                continue
            report_path = candidates[0]

        html_path = WEBSITES / f'{stem}.html'
        if not html_path.exists():
            print(f'Book {book_num}: HTML not found ({stem}.html) — skipping')
            continue

        html_text = html_path.read_text('utf-8')

        # Already has an overview → just fix paragraph spacing if needed
        if '<h1>Overview</h1>' in html_text:
            new_html, n = fix_paragraph_spacing(html_text)
            if n > 0:
                html_path.write_text(new_html, 'utf-8')
                print(f'Book {book_num} ({stem}): fixed spacing (+{n} <p></p> spacers)')
            else:
                print(f'Book {book_num} ({stem}): overview present, spacing OK')
            continue

        # Extract overview from report
        report_text = report_path.read_text('utf-8')
        overview_body = extract_overview(report_text)
        if not overview_body:
            print(f'Book {book_num} ({stem}): no overview found in report — skipping')
            continue

        words = len(overview_body.split())
        overview_html = to_html_paragraphs(overview_body)
        paras = overview_html.count('<p class="indent">')

        new_html = insert_overview_block(html_text, overview_html)
        if new_html is None:
            print(f'Book {book_num} ({stem}): insertion marker not found — skipping')
            continue

        html_path.write_text(new_html, 'utf-8')
        print(f'Book {book_num} ({stem}): inserted overview ({words}w, {paras} paras)')


if __name__ == '__main__':
    main()
