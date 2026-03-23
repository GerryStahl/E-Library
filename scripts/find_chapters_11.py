"""Find the starting page of each essay in 11.theory.pdf."""
import fitz

doc = fitz.open('/Users/GStahl2/AI/elibrary/sourcepdfs/11.theory.pdf')

chapter_keywords = [
    'Introduction to CSCL',
    'Theories of CSCL',
    'Analysis of CSCL',
    'Curriculum for CSCL',
    'Technological Artifacts',
    'Introducing Theoretical Investigations',
    'A Vision of Group Cognition',
    'The Theory of Group Cognition',
    'Post-cognitive',
    'Practices in Group Cognition',
    'Co-experiencing',
    'From Intersubjectivity',
    'Constituting Group Cognition',
    'Sustaining Group Cognition',
    'Structuring Group Cognition',
]

hits = {}
for i, page in enumerate(doc):
    text = page.get_text('text')
    for kw in chapter_keywords:
        if kw in text and kw not in hits:
            idx = text.find(kw)
            if idx < 400:
                hits[kw] = i + 1
                print("Page %d: %r  (pos %d)" % (i + 1, kw, idx))
                break

print("\nTotal chapters found:", len(hits))
