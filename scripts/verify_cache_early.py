"""Show final cache state for early editorials."""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

cache = pickle.loads((Path(__file__).parent.parent / "cache" / "editorials_cache.pkl").read_bytes())

print(f"Total editorials in cache: {len(cache.editorials)}")
print()

early = [e for e in cache.editorials if e.year <= 2015]
early.sort(key=lambda e: (e.year, e.volume, e.issue))
print(f"Early editorials (≤2015) in cache: {len(early)}")
print()
print(f"  {'Year':<6} {'Vol':<4} {'Iss':<4} {'Words':<6} {'Title':<35} {'Author'}")
print("  " + "-"*90)
for e in early:
    t = (e.title[:33] + "…") if len(e.title) > 35 else e.title
    a = e.author or "—"
    w = str(e.words) if e.words else "—"
    print(f"  {e.year:<6} {e.volume:<4} {e.issue:<4} {w:<6} {t:<36} {a}")

missing_auth = [e for e in early if not e.author]
missing_text = [e for e in early if not e.text]
print()
print(f"Missing author : {len(missing_auth)}")
print(f"Missing text   : {len(missing_text)}")
