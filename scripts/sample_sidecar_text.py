"""
sample_sidecar_text.py
Quick diagnostic: print raw chunk text samples across different content types
so we can see exactly what formatting artifacts survive PDF→text extraction.
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SIDECAR = ROOT / "vector_store" / "chunks_bm25" / "sidecar.json"
CLUSTERS = ROOT / "reports" / "chunk_clusters.csv"

import csv

# Load cluster lookup: doc_id -> cluster_id
cluster_of = {}
with open(CLUSTERS) as f:
    for row in csv.DictReader(f):
        cluster_of[row["vector_id"]] = int(row["cluster_id"])

sidecar = json.loads(SIDECAR.read_text())

# Build reverse: cluster -> list of doc_ids
from collections import defaultdict
by_cluster = defaultdict(list)
for doc_id, meta in sidecar.items():
    vid = str(meta.get("vector_id", ""))
    cid = cluster_of.get(vid, -1)
    by_cluster[cid].append(doc_id)

def show_samples(label, doc_ids, n=3):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print('='*70)
    for doc_id in doc_ids[:n]:
        meta = sidecar[doc_id]
        text = meta.get("chunk_text", "")
        print(f"\n--- doc_id={doc_id}  bk={meta.get('book_number')} ch={meta.get('chapter_number')} ---")
        print(text[:600])
        print("...")

# 1. A VMT chapter (bk 4) — chat excerpts expected
vmt_ids = [d for d in by_cluster.get(4, [])
           if sidecar[d].get("book_number") == 4]
show_samples("CLUSTER 4 / Book 4 (VMT) — expect chat turns", vmt_ids, 4)

# 2. Cluster 6 — bibliography noise
show_samples("CLUSTER 6 — bibliography noise", by_cluster[6], 3)

# 3. Cluster 3 — threading/response structure (VMT methodology)
show_samples("CLUSTER 3 — threading analysis", by_cluster[3][:50], 3)

# 4. Book 7 (social philosophy) early writing
bk7_ids = [d for d in sidecar if sidecar[d].get("book_number") == 7]
show_samples("BOOK 7 (Social Philosophy) — early essays", bk7_ids, 3)

# 5. Book 19 (Theoretical Investigations) — contains reprinted chat data
bk19_ids = [d for d in sidecar if sidecar[d].get("book_number") == 19]
show_samples("BOOK 19 (Theoretical Investigations)", bk19_ids[30:34], 4)

# 6. Show chunks with obvious chat-turn-like patterns
print(f"\n{'='*70}")
print("  CHAT-TURN PATTERN SEARCH")
print('='*70)
chat_pat = re.compile(r'^\s{0,4}[A-Za-z][a-z0-9_]{1,12}\s*:\s+\S', re.MULTILINE)
count = 0
for doc_id, meta in sidecar.items():
    text = meta.get("chunk_text", "")
    if chat_pat.search(text) and count < 5:
        print(f"\n--- doc_id={doc_id}  bk={meta.get('book_number')} ch={meta.get('chapter_number')} ---")
        print(text[:800])
        count += 1

# 7. Chunks with Figure/Table captions
print(f"\n{'='*70}")
print("  FIGURE/TABLE CAPTIONS")
print('='*70)
fig_pat = re.compile(r'^(Fig\.?|Figure|Table)\s+\d+', re.MULTILINE | re.IGNORECASE)
count = 0
for doc_id, meta in sidecar.items():
    text = meta.get("chunk_text", "")
    if fig_pat.search(text) and count < 4:
        print(f"\n--- doc_id={doc_id}  bk={meta.get('book_number')} ch={meta.get('chapter_number')} ---")
        print(text[:600])
        count += 1
