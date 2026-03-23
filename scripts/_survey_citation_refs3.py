"""
Check if books with 'truly missing' chapters have consolidated reference
sections — i.e. cluster-6/12 chunks in a different chapter of the same book.
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

# For each book, which chapters have cluster-6/12 chunks?
book_noise_chapters = defaultdict(list)
for (bk, ch), clusters in sorted(chapter_clusters.items()):
    if 6 in clusters or 12 in clusters:
        book_noise_chapters[bk].append(ch)

# The truly-missing books from the previous survey
missing_books = {2, 3, 4, 5, 6, 10, 15, 18}

stahl_ref_pat = re.compile(r'Stahl,\s*G\.?\s*\(\d{4}[a-z]?\)', re.IGNORECASE)

# Build (book, chapter) -> list of chunks
chapter_chunks = defaultdict(list)
for doc_id, meta in sidecar.items():
    bk = meta.get("book_number"); ch = meta.get("chapter_number")
    if bk and ch:
        chapter_chunks[(int(bk), int(ch))].append(meta.get("chunk_text", ""))

print("Books with consolidated reference sections (cluster-6/12 in book but not per-chapter):\n")
for bk in sorted(missing_books):
    noise_chs = book_noise_chapters.get(bk, [])
    if noise_chs:
        # Check which of those chapters actually contain Stahl entries
        stahl_chs = []
        for ch in noise_chs:
            texts = chapter_chunks.get((bk, ch), [])
            if any(stahl_ref_pat.search(t) for t in texts):
                stahl_chs.append(ch)
        print(f"  Book {bk:2d}: cluster-6/12 chapters = {noise_chs}")
        print(f"          chapters with Stahl entries = {stahl_chs}")
    else:
        print(f"  Book {bk:2d}: NO cluster-6/12 chapters at all")

# Also show max chapter number per book (to see if refs are truly at the end)
print("\nMax chapter numbers per missing book:")
book_max_ch = defaultdict(int)
for (bk, ch) in chapter_clusters:
    book_max_ch[bk] = max(book_max_ch[bk], ch)
for bk in sorted(missing_books):
    noise_chs = book_noise_chapters.get(bk, [])
    print(f"  Book {bk:2d}: max_chapter={book_max_ch[bk]:3d}  noise_chapters={noise_chs}")
