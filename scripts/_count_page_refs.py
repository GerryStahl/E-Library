"""Count how many of the 2,852 inline self-citations include page numbers."""
import csv, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Page number patterns in inline citations:
# (Stahl, 2006, p. 45)  (Stahl, 2006, pp. 45-47)  (Stahl, 2006, p.45)
page_pat = re.compile(r'\bpp?\.\s*\d+', re.IGNORECASE)

total = 0
with_page = 0
examples = []

with open(ROOT / "reports/self_citations.csv") as f:
    for row in csv.DictReader(f):
        total += 1
        snippet = row.get("snippet", "")
        if page_pat.search(snippet):
            with_page += 1
            if len(examples) < 10:
                examples.append(snippet[:150])

print(f"Total inline self-citations : {total:,}")
print(f"With page number            : {with_page:,}  ({100*with_page/total:.1f}%)")
print(f"Without page number         : {total - with_page:,}  ({100*(total-with_page)/total:.1f}%)")
print("\nExamples with page numbers:")
for ex in examples:
    print(f"  {ex}")
