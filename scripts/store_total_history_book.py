"""
store_total_history_book.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Stores the Copilot-generated Level-3 total_history summary_book,
synthesizing all 3 × 20 Level-2 annual summaries.
Run from workspace root:
    python scripts/store_total_history_book.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
PKL_PATH  = CACHE_DIR / "editorials_cache.pkl"
JSON_PATH = CACHE_DIR / "editorials_cache.json"
REPORT    = ROOT / "reports" / "total_history_book.txt"

sys.path.insert(0, str(ROOT))
from cache.editorials_cache import EditorialsCache, EditorialSummary, TotalHistory

TODAY  = date.today().isoformat()
AUTHOR = "GitHub Copilot"

PROMPT = (
    "You are a learning-science researcher writing for other leading researchers "
    "and advanced graduate students. You have read all 79 editorial introductions "
    "to the 20 annual volumes of the International Journal of CSCL (2006–2025), "
    "and have already written annual summaries for each year. "
    "Now write a single approximately 500-word synthesis covering the full 20-year "
    "arc of the CSCL field as seen through those editorials. "
    "Focus on how the field changed over time: shifts in theory (e.g., of learning, "
    "cognition, interaction), analytic methodology, software technology, pedagogy, "
    "and curriculum. Identify major turning points, recurring themes, and the "
    "direction the field was heading by 2025. "
    "Do not use names or technical terms that do not appear in the annual summaries "
    "provided. Write this as a resource for constructing a history of CSCL."
)

TEXT = """\
The twenty-year arc of ijCSCL editorials traces a field that moved from paradigm-declaration to paradigm-consolidation, while repeatedly circling back to a handful of foundational tensions that were never fully resolved—and may have been productively sustained rather than dissolved.

**Founding commitments (2006–2009).** The inaugural volume announced a decisive break from individually-focused educational research: the field's defining object was intersubjective meaning making, a group-level phenomenon irreducible to the sum of individual learners. Social practices, participation in discourse, and the affordances and deficits of technology for mediating interaction were placed at the center. Flash themes—scripting, argumentation—were introduced as a mechanism for sustaining theoretical debate across issues, reflecting a self-conscious commitment to collective knowledge-building within the research community itself. The 2009 volume introduced the practice turn and four paradigms for conceptualizing shared knowledge, establishing that CSCL's theoretical pluralism was a feature rather than a deficiency.

**Institutional consolidation and theoretical sharpening (2010–2012).** Acceptance into the Web of Science marked external validation of the field. Internally, editorials demanded theoretically self-conscious research specifically suited to CSCL—criticizing both atheoretical empiricism and the mechanical import of frameworks from other disciplines. Visual models of group cognition depicted sequential dialogical interaction as the mechanism through which individual voices weave into collective meaning-making. The challenge of traversing planes of analysis—individual learning, small-group cognition, and community knowledge building—was identified as the field's central unresolved problem.

**Multi-level framework and methodological expansion (2013–2015).** The theme of learning across levels deepened into a sustained research program. Transactivity was introduced as a specific analytical lens for discourse research; eye-tracking emerged as a new methodological tool; dialogical theory and artifact-centered analysis provided additional theoretical scaffolding. The 2015 retrospective consolidated a decade of work under the banner of intersubjectivity as the field's defining characteristic and identified the post-cognitive paradigm as the frame within which progress had been made.

**Second decade: regulatory, emotional, and computational turns (2016–2020).** New editorial leadership shifted emphasis toward socially shared regulation of learning, the social-emotional dimensions of collaboration, and computational tools for group formation and awareness. The question of whether CSCL constituted one community or several was posed explicitly, with the observation that constructivist and socio-cultural frameworks dominated while information processing approaches remained a minority. The COVID-19 pandemic forced an abrupt demonstration of CSCL's societal relevance and brought equity and accessibility to the foreground for the first time as central rather than peripheral concerns.

**Learning analytics, extended reality, and maturity claims (2021–2025).** The final years showed three simultaneous movements: methodological maturation (sophisticated discourse analysis, real-time analytics, learning analytics as a partner field), technological expansion (extended reality environments, AI-mediated communication), and explicit reflection on field identity. The 2024 declaration that CSCL had reached "normal science" was immediately qualified by two unresolved foundational questions—whether group processes reduce to individual characteristics, and whether technology transforms or merely supports collaboration. The 2025 proposal of mechanistic explanation as a unifying construct offered the most ambitious theoretical synthesis of the twenty-year period: a meta-theoretical framework intended to bridge cognitive and sociocultural approaches that had coexisted without integration since the field's founding.

Throughout, the editorials returned consistently to the individual-group tension, to the challenge of connecting levels of analysis, and to the dual nature of technology as both enabling and potentially undermining the collaborative conditions the field sought to create.\
"""


def _truncate(obj, n=10):
    if isinstance(obj, dict):
        return {k: (" ".join(v.split()[:n]) if (k == "text" or k.endswith("_text") or k.endswith("_prompt")) and isinstance(v, str) and v else _truncate(v, n)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate(i, n) for i in obj]
    return obj


def main() -> None:
    cache = EditorialsCache.load(str(PKL_PATH))
    if not hasattr(cache, 'total_history') or cache.total_history is None:
        cache.total_history = TotalHistory()

    wc = len(TEXT.split())
    cache.total_history.summary_book = EditorialSummary(
        summary_author          = AUTHOR,
        summary_date            = TODAY,
        summary_prompt          = PROMPT,
        summary_number_of_words = wc,
        summary_text            = TEXT,
    )
    cache.save(str(PKL_PATH))
    print(f"PKL saved → {PKL_PATH}  ({wc} words)")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ijCSCL — 20-Year History of CSCL (2006–2025) [GitHub Copilot / Book-level]",
        "=" * 70,
        f"Generated : {TODAY}",
        f"Model     : {AUTHOR}",
        f"Words     : {wc}",
        f"Input     : All 3 Level-2 annual summaries × 20 years (60 total)",
        "",
        PROMPT,
        "",
        "=" * 70,
        "",
        TEXT,
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written → {REPORT}")

    data = _truncate(asdict(cache))
    JSON_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2)
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029"),
        encoding="utf-8",
    )
    print(f"JSON updated   → {JSON_PATH}")
    print(f"\nTotalHistory now: {cache.total_history}")


if __name__ == "__main__":
    main()
