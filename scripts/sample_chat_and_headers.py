"""
sample_chat_and_headers.py
Targeted diagnostic: actual VMT chat excerpts + page header patterns.
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SIDECAR = ROOT / "vector_store" / "chunks_bm25" / "sidecar.json"
sidecar = json.loads(SIDECAR.read_text())

def show(doc_id):
    meta = sidecar[doc_id]
    print(f"\n--- doc_id={doc_id}  bk={meta.get('book_number')} ch={meta.get('chapter_number')} ---")
    print(repr(meta.get("chunk_text", "")[:1000]))

# 1. Book 4, chapters 8-11 (actual VMT chat analysis chapters)
print("="*70)
print("BOOK 4 ch 8-11: VMT session analysis (likely has chat excerpts)")
bk4_ch8_ids = sorted([d for d in sidecar
                       if sidecar[d].get("book_number") == 4
                       and sidecar[d].get("chapter_number") in (8, 9, 10, 11)],
                      key=lambda d: int(d))
for doc_id in bk4_ch8_ids[5:12]:
    show(doc_id)

# 2. Book 6 (Constructing Dynamic Triangles) session chapters — numbered log lines?
print("\n" + "="*70)
print("BOOK 6 ch 4-10: Session analysis chapters")
bk6_ids = sorted([d for d in sidecar
                  if sidecar[d].get("book_number") == 6
                  and sidecar[d].get("chapter_number") in (4, 5, 6)],
                 key=lambda d: int(d))
for doc_id in bk6_ids[3:9]:
    show(doc_id)

# 3. Book 19, investigation chapters with actual chat
print("\n" + "="*70)
print("BOOK 19 ch 5-9: Investigations with chat data")
bk19_ids = sorted([d for d in sidecar
                   if sidecar[d].get("book_number") == 19
                   and sidecar[d].get("chapter_number") in (5, 6, 7)],
                  key=lambda d: int(d))
for doc_id in bk19_ids[10:17]:
    show(doc_id)

# 4. Show raw page-header/footer pattern analysis
print("\n" + "="*70)
print("PAGE HEADER PATTERN: short lines before/after numbers")
# Collect all line-length distributions across all chunks
header_pat = re.compile(r'\n[A-Z][^\n]{3,60}\n\s*\n?\s*\d{1,4}\s*\n', re.MULTILINE)
page_num_only = re.compile(r'^\d{1,4}\s*$', re.MULTILINE)
count = 0
for doc_id, meta in list(sidecar.items())[:500]:
    text = meta.get("chunk_text", "")
    m = header_pat.search(text)
    if m and count < 5:
        print(f"\n--- doc_id={doc_id}  bk={meta.get('book_number')} ch={meta.get('chapter_number')} ---")
        # Show 100 chars around match
        start = max(0, m.start()-20)
        print(repr(text[start:m.end()+20]))
        count += 1
