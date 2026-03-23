"""fix_overview_format.py
For books 5-22:
  1. Lowercase <h1>Overview</h1> → <h1>overview</h1>
  2. Remove <p></p> spacers between consecutive <p class="indent"> paragraphs
     within the overview section (keep the initial <p></p> after the heading).
"""

import re
from pathlib import Path

WEBSITES = Path(__file__).resolve().parent.parent / 'websites'

STEMS = [
    'euclid', 'analysis', 'philosophy', 'software', 'cscl', 'science',
    'theory', 'math', 'dynamic', 'topics', 'global', 'ijcscl',
    'proposals', 'overview', 'investigations', 'form', 'game', 'climate',
]


def fix_file(path: Path) -> str:
    txt = path.read_text('utf-8')

    # Locate the overview section
    ov_start = txt.find('<h1>Overview</h1>')
    if ov_start == -1:
        ov_start = txt.find('<h1>overview</h1>')
        if ov_start == -1:
            return 'no overview heading found'

    # The overview section ends at the next <hr>
    ov_end = txt.find('<hr>', ov_start + 1)
    if ov_end == -1:
        ov_end = len(txt)

    section = txt[ov_start:ov_end]

    # 1. Lowercase the heading
    section = section.replace('<h1>Overview</h1>', '<h1>overview</h1>', 1)

    # 2. Remove <p></p> between consecutive <p class="indent"> paragraphs
    #    Pattern: </p>\n<p></p>\n<p class="indent">  →  </p>\n<p class="indent">
    cleaned, n = re.subn(
        r'(</p>)\n<p></p>\n(<p class="indent">)',
        r'\1\n\2',
        section
    )

    new_txt = txt[:ov_start] + cleaned + txt[ov_end:]
    path.write_text(new_txt, 'utf-8')
    return f'heading lowercased, {n} <p></p> spacers removed'


def main():
    for stem in STEMS:
        path = WEBSITES / f'{stem}.html'
        if not path.exists():
            print(f'{stem}: file not found')
            continue
        result = fix_file(path)
        print(f'{stem}: {result}')


if __name__ == '__main__':
    main()
