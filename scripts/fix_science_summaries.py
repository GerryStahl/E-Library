"""fix_science_summaries.py
Rebuild the chapter summaries section of science.html using
complete summaries from the report file 10.science_summary.txt.
"""

import re
import html as htmlmod
from pathlib import Path

WORKSPACE   = Path(__file__).resolve().parent.parent
REPORT      = WORKSPACE / 'reports' / '10.science_summary.txt'
HTML_PATH   = WORKSPACE / 'websites' / 'science.html'

# ── Parse report ──────────────────────────────────────────────────────────────

def parse_report(text: str) -> list[dict]:
    """
    Return list of {title, paragraphs} for each numbered chapter section.
    Handles single- and multi-line titles (e.g. a wrapped title whose
    continuation line is indented).  Stops at the first ==== separator
    so the overview section is not included.
    """
    # The report has a header separator (====) on line 3, then chapters,
    # then another ==== before the overview section.  Skip the first one.
    stop_re = re.compile(r'^={5,}', re.MULTILINE)
    stops = list(stop_re.finditer(text))
    if len(stops) >= 2:
        text = text[stops[0].end():stops[1].start()]
    elif len(stops) == 1:
        text = text[stops[0].end():]

    lines = text.splitlines()
    # Find the start line of each numbered chapter: "N. Title..."
    chapter_start_re = re.compile(r'^\d+\.\s+\S')
    dash_re = re.compile(r'^[-─]{5,}$')

    # Collect (line_index, title) for each chapter heading
    headings = []
    i = 0
    while i < len(lines):
        if chapter_start_re.match(lines[i]):
            # Collect title: may span multiple lines until a dash line
            title_parts = [lines[i].split('.', 1)[1].strip()]
            j = i + 1
            while j < len(lines) and not dash_re.match(lines[j]):
                continuation = lines[j].strip()
                if continuation:
                    title_parts.append(continuation)
                j += 1
            title = ' '.join(title_parts)
            body_start = j + 1  # first line after the dash underline
            headings.append((body_start, title))
            i = j + 1
        else:
            i += 1

    chapters = []
    for idx, (body_start, title) in enumerate(headings):
        body_end_line = headings[idx + 1][0] - 3 if idx + 1 < len(headings) else len(lines)
        body = '\n'.join(lines[body_start:body_end_line]).strip()
        paras = [' '.join(p.split()) for p in re.split(r'\n\s*\n', body) if p.strip()]
        chapters.append({'title': title, 'paragraphs': paras})
    return chapters


# ── Build HTML block ──────────────────────────────────────────────────────────

def chapters_to_html(chapters: list[dict]) -> str:
    parts = []
    for ch in chapters:
        title_esc = htmlmod.escape(ch['title'])
        parts.append(f'</p>\n<p class="indent"><strong>{title_esc}</strong></p>')
        for para in ch['paragraphs']:
            parts.append(f'<p class="indent">{htmlmod.escape(para)}</p>')
    return '\n'.join(parts)


# ── Splice into HTML ──────────────────────────────────────────────────────────

SUMM_OPEN  = '<h1>summaries of the chapters</h1>'
FOOTER     = '\n<hr>\n<!--#include virtual="../../home/footer.html" -->'


def rebuild_summaries(html: str, summaries_html: str) -> str:
    start = html.find(SUMM_OPEN)
    if start == -1:
        raise ValueError('summaries marker not found')
    end = html.find(FOOTER, start)
    if end == -1:
        raise ValueError('footer marker not found')
    new_section = SUMM_OPEN + '\n' + summaries_html + '\n'
    return html[:start] + new_section + html[end:]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    report_text = REPORT.read_text('utf-8')
    chapters = parse_report(report_text)
    print(f'Parsed {len(chapters)} chapters from report')
    for i, ch in enumerate(chapters):
        print(f'  {i+1}. {ch["title"][:70]} ({len(ch["paragraphs"])} paras)')

    summaries_html = chapters_to_html(chapters)

    html = HTML_PATH.read_text('utf-8')
    new_html = rebuild_summaries(html, summaries_html)
    HTML_PATH.write_text(new_html, 'utf-8')
    print(f'\nWrote {HTML_PATH}')


if __name__ == '__main__':
    main()
