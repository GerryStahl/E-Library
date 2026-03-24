"""
verify_cleaning.py
Check that the cleaning pipeline correctly handled each content type.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SIDECAR  = ROOT / "vector_store" / "chunks_bm25" / "sidecar.json"
NARRATIVE = ROOT / "reports" / "narrative_chunks.json"

sidecar   = json.loads(SIDECAR.read_text())
narrative = json.loads(NARRATIVE.read_text())

def compare(doc_id: str, label: str):
    orig = sidecar.get(doc_id, {}).get("chunk_text", "[not in sidecar]")
    if doc_id in narrative:
        narr = narrative[doc_id]["narrative_text"]
        orig_wc = narrative[doc_id]["original_word_count"]
        narr_wc = narrative[doc_id]["narrative_word_count"]
        flag = f"KEPT  {orig_wc}→{narr_wc} words"
    else:
        narr = "[EXCLUDED]"
        flag = "EXCLUDED"

    print(f"\n{'='*65}")
    print(f"  {label}  [doc_id={doc_id}]  {flag}")
    print(f"{'='*65}")
    print("ORIGINAL:")
    print(repr(orig[:600]))
    print("\nCLEANED:")
    print(repr(narr[:600]))

# 1. Page header (doc_id=3608 — mid-page header "Studying Virtual Math Teams\n  \n113\n")
compare("3608", "PAGE HEADER in VMT")

# 2. Chat log block (doc_id=3509 — Log 4-2 with ImH, Jas turns)
compare("3509", "CHAT LOG (Log 4-2)")

# 3. Another chat format (doc_id=3502 — Log 4-1 with timestamps)
compare("3502", "CHAT LOG with timestamps (Log 4-1)")

# 4. Dense chat (doc_id=3635 — Log 6-1: mathis, bob1, qw)
compare("3635", "CHAT LOG dense (Log 6-1)")

# 5. Reference section (doc_id=617 — pure bibliography)
compare("617", "BIBLIOGRAPHY chunk")

# 6. Figure caption (doc_id=558 — Figure 1 reference)
compare("558", "FIGURE CAPTION")

# 7. Early footnotes (doc_id=42 — Book 1 ch 1 with footnotes 4-7)
compare("42", "EARLY FOOTNOTES (Book 1)")

# 8. Clean prose that should be unchanged (doc_id=3609 — regular VMT prose)
compare("3609", "CLEAN PROSE (should be ~unchanged)")
