"""reposition_overviews.py
Move each book's Overview section from its current position
(before <h1>summaries of the chapters</h1>) to immediately after
the book cover image (before the download-volume <hr>).

Target structure:
  <p align="center"><img src="stem.jpg" height="500"></p>
  </p>
  <hr>
  <h1>Overview</h1>
  <p></p>
  <p class="indent">...</p>
  ...
  <hr>
  <h1>download volume</h1>
"""

import re
from pathlib import Path

WEBSITES = Path(__file__).resolve().parent.parent / 'websites'

STEMS = [
    'euclid', 'analysis', 'philosophy', 'software', 'cscl', 'science',
    'theory', 'math', 'dynamic', 'topics', 'global', 'ijcscl',
    'proposals', 'overview', 'investigations', 'form', 'game', 'climate',
]

SUMM_MARKER = '\n<hr>\n<h1>summaries of the chapters</h1>'


def reposition(path: Path) -> str:
    txt = path.read_text('utf-8')

    # Locate the overview block (starts with \n<hr>\n<h1>Overview</h1>)
    ov_start = txt.find('\n<hr>\n<h1>Overview</h1>')
    if ov_start == -1:
        return 'no overview found'

    # Locate the summaries marker (where the overview block ends)
    summ_pos = txt.find(SUMM_MARKER, ov_start)
    if summ_pos == -1:
        return 'summaries marker not found'

    # Guard: if overview is already before the first <hr>, it's in place
    first_hr = txt.find('<hr>')
    if 0 <= ov_start <= first_hr:
        return 'already in post-image position — skipped'

    # Extract the overview block (\n<hr>\n<h1>Overview</h1>...<last para>\n)
    overview_block = txt[ov_start:summ_pos]

    # Remove overview from its current position
    txt2 = txt[:ov_start] + txt[summ_pos:]

    # Find the image element + its two closing </p> lines
    m = re.search(r'<img[^>]+></p>\n</p>', txt2, re.IGNORECASE)
    if not m:
        return 'img section pattern not found'

    end_of_img = m.end()  # right after the second </p>

    # Find the next <hr> (start of download-volume section)
    next_hr = txt2.find('<hr>', end_of_img)
    if next_hr == -1:
        return 'no <hr> after image'

    # Insert: ...img</p></p> + overview_block(\n<hr>\n<h1>Overview</h1>...) + \n<hr>...
    new_txt = txt2[:end_of_img] + overview_block + '\n' + txt2[next_hr:]
    path.write_text(new_txt, 'utf-8')

    paras = overview_block.count('<p class="indent">')
    return f'moved ({paras} paras)'


def main():
    for stem in STEMS:
        path = WEBSITES / f'{stem}.html'
        if not path.exists():
            print(f'{stem}: file not found')
            continue
        result = reposition(path)
        print(f'{stem}: {result}')


if __name__ == '__main__':
    main()
