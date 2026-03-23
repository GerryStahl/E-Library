"""
Check that every chapter in the PKL has exactly one chapter_summaries entry.
For any chapter with exactly one entry and a blank chapter_summary_prompt,
fill it in with "Same prompt as for the book".
Then save the PKL and regenerate JSON.
"""
import sys, pickle
from pathlib import Path

CACHE_DIR = Path('/Users/GStahl2/AI/elibrary/cache')
sys.path.insert(0, str(CACHE_DIR))

from elibrary_cache import ElibraryCache  # noqa: E402

PKL_PATH = CACHE_DIR / 'elibrary_cache.pkl'
cache = ElibraryCache.load(str(PKL_PATH))

issues      = []   # chapters that don't have exactly 1 summary
filled      = []   # chapters where we set the prompt
already_ok  = []   # chapters already with a prompt

for b in cache.books:
    bnum  = b.book_number
    bname = b.book_name

    for ch in b.book_chapters:
        cnum  = ch.chapter_number
        ctitle = ch.chapter_title
        sums  = ch.chapter_summaries

        count = len(sums)
        label = f"Book {bnum:>2} Ch {cnum:>2} ({ctitle[:45]})"

        if count != 1:
            issues.append(f"{label}: has {count} summaries (expected 1)")
            continue

        prompt = sums[0].chapter_summary_prompt
        if not prompt or not prompt.strip():
            sums[0].chapter_summary_prompt = "Same prompt as for the book"
            filled.append(label)
        else:
            already_ok.append(label)

# ── Report ─────────────────────────────────────────────────────────────────
print(f"Chapters with wrong summary count ({len(issues)}):")
for x in issues:
    print(f"  ⚠  {x}")

print()
print(f"Chapters prompt filled → 'Same prompt as for the book' ({len(filled)}):")
for x in filled:
    print(f"  ✎  {x}")

print()
print(f"Chapters already had a prompt ({len(already_ok)}): (not shown)")

# ── Save PKL ───────────────────────────────────────────────────────────────
if filled:
    with open(PKL_PATH, 'wb') as f:
        pickle.dump(cache, f)
    print(f"\nPKL saved → {PKL_PATH}")
else:
    print("\nNo changes — PKL not rewritten.")

# ── Regenerate JSON ─────────────────────────────────────────────────────────
if filled:
    import subprocess, sys as _sys
    result = subprocess.run(
        [_sys.executable, 'scripts/export_cache_json.py'],
        cwd='/Users/GStahl2/AI/elibrary',
        capture_output=True, text=True
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print("JSON export error:", result.stderr.strip())
