"""Find essay start pages in 11.theory.pdf using font-size detection."""
import fitz

doc = fitz.open('/Users/GStahl2/AI/elibrary/sourcepdfs/11.theory.pdf')

# Skip the intro section (pages 1-14), look for big-font headings that signal essay starts
print("=== Scanning for large-font headings (possible essay starts) ===")
for i in range(13, len(doc)):
    page = doc[i]
    blocks = page.get_text('dict')['blocks']
    for block in blocks:
        if 'lines' not in block:
            continue
        for line in block['lines']:
            for span in line['spans']:
                if span['size'] >= 16 and len(span['text'].strip()) > 10:
                    text = span['text'].strip()
                    print("Page %d  size=%.1f  %r" % (i + 1, span['size'], text[:80]))
