"""
Sample actual Stahl APA reference-list entries to understand formatting variation.
Also check whether the 54 'no cluster-6/12' chapters have refs in other clusters.
"""
import json, csv, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sidecar = json.loads((ROOT / "vector_store/chunks_bm25/sidecar.json").read_text())

cluster_map = {}
chapter_clusters = defaultdict(set)
with open(ROOT / "reports/chunk_clusters.csv") as f:
    for row in csv.DictReader(f):
        vid = int(row["vector_id"])
        cid = int(row["cluster_id"])
        cluster_map[vid] = cid
        chapter_clusters[(int(row["book_number"]), int(row["chapter_number"]))].add(cid)

# Build: (book, chapter) -> list of (vid, text)
chapter_chunks = defaultdict(list)
for doc_id, meta in sidecar.items():
    bk = meta.get("book_number"); ch = meta.get("chapter_number"); vid = meta.get("vector_id")
    if bk and ch:
        chapter_chunks[(int(bk), int(ch))].append((vid, meta.get("chunk_text", "")))

stahl_ref_pat = re.compile(r'Stahl,\s*G\..*?\(\d{4}[a-z]?\)', re.IGNORECASE)

# Sample 20 actual APA entries from clusters 6/12
print("=== SAMPLE APA ENTRIES FROM CLUSTERS 6/12 ===\n")
count = 0
for doc_id, meta in sidecar.items():
    vid = meta.get("vector_id")
    if cluster_map.get(vid) not in (6, 12):
        continue
    text = meta.get("chunk_text", "")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines:
        if stahl_ref_pat.search(line):
            print(repr(line[:180]))
            count += 1
            if count >= 20:
                break
    if count >= 20:
        break

# Now check: for chapters WITHOUT cluster-6/12 chunks, do they have Stahl refs elsewhere?
print("\n\n=== CHAPTERS WITHOUT CLUSTER-6/12: do they have Stahl refs in other chunks? ===\n")
with open(ROOT / "reports/self_citations.csv") as f:
    cit_rows = list(csv.DictReader(f))
citing_chapters = set((int(r["book_number"]), int(r["chapter_number"])) for r in cit_rows)

found_elsewhere = 0
truly_missing = 0
for bc in sorted(citing_chapters):
    if 6 in chapter_clusters[bc] or 12 in chapter_clusters[bc]:
        continue
    # Search all chunks from this chapter
    has_ref = any(stahl_ref_pat.search(text) for _, text in chapter_chunks[bc])
    if has_ref:
        found_elsewhere += 1
    else:
        truly_missing += 1
        print(f"  TRULY MISSING: Book {bc[0]:2d} Ch {bc[1]:2d}  clusters: {sorted(chapter_clusters[bc])}")

print(f"\nOf 54 chapters without cluster-6/12:")
print(f"  Have Stahl refs in other clusters : {found_elsewhere}")
print(f"  Truly no reference list at all     : {truly_missing}")
