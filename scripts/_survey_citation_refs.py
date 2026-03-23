"""Survey: where are Stahl reference-list entries, and do all citing chapters have them?"""
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

stahl_ref_pat = re.compile(r'Stahl,\s*G\.?\s*\((\d{4}[a-z]?)\)', re.IGNORECASE)

noise_with_stahl = 0
other_with_stahl = 0
for doc_id, meta in sidecar.items():
    vid = meta.get("vector_id")
    text = meta.get("chunk_text", "")
    if stahl_ref_pat.search(text):
        cid = cluster_map.get(vid, -1)
        if cid in (6, 12):
            noise_with_stahl += 1
        else:
            other_with_stahl += 1

print(f"Chunks with Stahl ref-list entries in clusters 6/12 : {noise_with_stahl:,}")
print(f"Chunks with Stahl ref-list entries in OTHER clusters : {other_with_stahl:,}")

with open(ROOT / "reports/self_citations.csv") as f:
    cit_rows = list(csv.DictReader(f))
print(f"\nTotal self-citation rows: {len(cit_rows):,}")

citing_chapters = set((int(r["book_number"]), int(r["chapter_number"])) for r in cit_rows)
with_noise  = sum(1 for bc in citing_chapters if 6 in chapter_clusters[bc] or 12 in chapter_clusters[bc])
without_noise = len(citing_chapters) - with_noise
print(f"Citing chapters WITH cluster-6/12 chunks   : {with_noise}")
print(f"Citing chapters WITHOUT cluster-6/12 chunks: {without_noise}")

# Sample a few chapters without noise chunks to understand why
print("\nSample chapters without cluster-6/12 chunks:")
count = 0
for bc in sorted(citing_chapters):
    if 6 not in chapter_clusters[bc] and 12 not in chapter_clusters[bc]:
        clusters_present = sorted(chapter_clusters[bc])
        print(f"  Book {bc[0]:2d} Ch {bc[1]:2d}  clusters: {clusters_present}")
        count += 1
        if count >= 15:
            break
