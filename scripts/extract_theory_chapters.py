"""Extract text of each essay in 11.theory.pdf by page ranges."""
import fitz

doc = fitz.open('/Users/GStahl2/AI/elibrary/sourcepdfs/11.theory.pdf')

# Page ranges (1-based), determined from find_chapters_11b.py
chapters = [
    (1,  "Introduction to CSCL",                  14,  36),
    (2,  "Theories of CSCL",                       37,  59),
    (3,  "Analysis of CSCL",                       60,  80),
    (4,  "Curriculum for CSCL",                    81,  99),
    (5,  "Technological Artifacts",               100, 122),
    (6,  "Introducing Theoretical Investigations", 123, 127),
    (7,  "A Vision of Group Cognition",            128, 162),
    (8,  "The Theory of Group Cognition",          163, 211),
    (9,  "A Post-cognitive Theoretical Paradigm",  212, 241),
    (10, "Practices in Group Cognition",           242, 264),
    (11, "Co-experiencing a Virtual World",        265, 283),
    (12, "From Intersubjectivity to Group Cognition", 284, 314),
    (13, "Constituting Group Cognition",           315, 330),
    (14, "Sustaining Group Cognition",             331, 362),
    (15, "Structuring Group Cognition",            363, 396),
]

for num, title, start, end in chapters:
    text_parts = []
    for pg in range(start - 1, min(end, len(doc))):  # 0-based
        text_parts.append(doc[pg].get_text('text'))
    text = '\n'.join(text_parts)
    out = '/Users/GStahl2/AI/elibrary/documents/ch%02d_theory.txt' % num
    with open(out, 'w', encoding='utf-8') as f:
        f.write(text)
    words = len(text.split())
    print("Ch%02d (%d pages, ~%d words): %s -> %s" % (num, end-start+1, words, title, out))
