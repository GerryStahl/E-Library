"""
sample_vmt_chat_format.py
Find actual student chat turns in sidecar by searching for known VMT pseudonyms.
Also look at numbered log-entry patterns.
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SIDECAR = ROOT / "vector_store" / "chunks_bm25" / "sidecar.json"
sidecar = json.loads(SIDECAR.read_text())

# Known VMT student pseudonyms
vmt_names = r'Qwertyuiop|Bwang|Aznx|Jason|137|QuickSilver|Quicksilver|Cheerio|Cornflake|Frosty|Pin|Sup|Avr|ImH|Jas'
vmt_pat = re.compile(r'(' + vmt_names + r')\s*:', re.IGNORECASE)

print("="*70)
print("SEARCHING FOR VMT PSEUDONYM TURNS")
print("="*70)
found = 0
for doc_id, meta in sidecar.items():
    text = meta.get("chunk_text", "")
    if vmt_pat.search(text) and found < 6:
        print(f"\n--- doc_id={doc_id}  bk={meta.get('book_number')} ch={meta.get('chapter_number')} ---")
        print(repr(text[:1200]))
        found += 1

# Also search for numbered log lines: "^[0-9]{1,4} " at line start
print("\n" + "="*70)
print("SEARCHING FOR NUMBERED LOG LINES (e.g., '42  Qwertyuiop: ...')")
print("="*70)
log_pat = re.compile(r'^\d{1,4}[ \t]+\S.{10,}', re.MULTILINE)
found = 0
for doc_id, meta in sidecar.items():
    text = meta.get("chunk_text", "")
    matches = log_pat.findall(text)
    # Only interested if there are multiple consecutive numbered lines
    if len(matches) >= 3 and found < 5:
        print(f"\n--- doc_id={doc_id}  bk={meta.get('book_number')} ch={meta.get('chapter_number')} ---")
        print(repr(text[:1200]))
        found += 1

# Also look for "Log N." or "Excerpt N" patterns
print("\n" + "="*70)
print("SEARCHING FOR 'Log' or 'Excerpt' LABELS")
print("="*70)
log2_pat = re.compile(r'\b(Log|Excerpt|Protocol|Transcript|Chat log)\s+\d+', re.IGNORECASE)
found = 0
for doc_id, meta in sidecar.items():
    text = meta.get("chunk_text", "")
    if log2_pat.search(text) and found < 5:
        print(f"\n--- doc_id={doc_id}  bk={meta.get('book_number')} ch={meta.get('chapter_number')} ---")
        # Show surrounding context
        m = log2_pat.search(text)
        start = max(0, m.start()-100)
        print(repr(text[start:start+800]))
        found += 1
