"""
build_editorials_cache.py — Build editorials_cache.pkl from the 40 ijCSCL PDFs.

Scans editorials/ for all ijcscl_*.pdf files, parses year / volume / issue
from each filename, and creates an Editorial entry for each.  The text and
summary fields are left empty; they will be filled in by later pipeline steps.

Filename pattern:
    ijcscl_{year}_v{vol:02d}_i{issue}_{safe_title}.pdf

Run:
    python scripts/build_editorials_cache.py
"""

import re
import sys
from pathlib import Path

# ── project root on sys.path so we can import cache/ modules ──────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cache.editorials_cache import Editorial, EditorialsCache  # noqa: E402

EDITORIALS_DIR = ROOT / "editorials"
CACHE_PATH     = ROOT / "cache" / "editorials_cache.pkl"

# Regex: ijcscl_{year}_v{vol}_i{issue}_*.pdf
FNAME_RE = re.compile(r"^ijcscl_(\d{4})_v(\d+)_i(\d+)_(.+)\.pdf$")


def main() -> None:
    pdf_files = sorted(EDITORIALS_DIR.glob("ijcscl_*.pdf"))

    if not pdf_files:
        print(f"No ijcscl_*.pdf files found in {EDITORIALS_DIR}")
        sys.exit(1)

    cache = EditorialsCache()
    skipped = []

    for pdf in pdf_files:
        m = FNAME_RE.match(pdf.name)
        if not m:
            print(f"  ⚠  Skipping unrecognised filename: {pdf.name}")
            skipped.append(pdf.name)
            continue

        year  = int(m.group(1))
        vol   = int(m.group(2))
        issue = int(m.group(3))

        editorial = Editorial(
            year   = year,
            volume = vol,
            issue  = issue,
            pdf    = pdf.name,
            # text and summary left empty — filled later
        )
        cache.add(editorial)

    # ── save ──────────────────────────────────────────────────────────────
    saved = cache.save(str(CACHE_PATH))

    print(f"\n{'='*60}")
    print(f"  editorials_cache.pkl created — {cache.total} editorials")
    print(f"  Saved to: {saved}")
    if skipped:
        print(f"\n  Skipped ({len(skipped)}):")
        for s in skipped:
            print(f"    {s}")
    print(f"{'='*60}\n")
    print(cache.summary_stats())


if __name__ == "__main__":
    main()
