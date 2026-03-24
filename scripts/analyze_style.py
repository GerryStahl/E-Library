"""
analyze_style.py
────────────────
Compute stylometric features per chapter from narrative_chunks.json.

Strategy: concatenate all narrative chunks belonging to a chapter,
then compute all metrics on the full chapter text. This gives accurate
type-token ratios, sentence distributions, and discourse counts.

Output: reports/style_features.csv
  One row per chapter with the following columns:

  IDENTITY
    book_number, chapter_number, book_title, chapter_title,
    pub_year, cluster_id, cluster_label,
    chunk_count, narrative_words

  SENTENCE RHYTHM
    sent_count          sentences in chapter
    sent_mean_words     mean words/sentence
    sent_std_words      std dev words/sentence
    sent_pct_long       % sentences > 30 words
    sent_pct_short      % sentences < 10 words

  VOCABULARY
    ttr                 type-token ratio (unique_tokens / total_tokens)
    mean_word_len       mean characters per word token
    abstract_ratio      words ending in -tion/-ity/-ness/-ment/-ism / total

  HEDGING vs ASSERTION
    hedge_per_1k        hedging words per 1000 narrative words
    assert_per_1k       assertion words per 1000 narrative words
    hedge_assert_ratio  hedge / (hedge + assert)  [0=all assertion, 1=all hedge]

  DISCOURSE CONNECTORS (per 1000 words)
    causal_per_1k       therefore/thus/hence/consequently
    contrast_per_1k     however/nevertheless/although/whereas/yet
    additive_per_1k     furthermore/moreover/additionally/in addition
    exemplify_per_1k    for example/for instance/such as
    pct_questions       % of sentences ending with ?

  VOICE (per 1000 words)
    i_per_1k            first-person singular (I/me/my/mine/myself)
    we_per_1k           first-person plural (we/our/us/ourselves)
    passive_per_1k      passive constructions (was/were … -ed form)

  CITATION DENSITY (per 1000 words)
    citation_per_1k     inline citations  (Surname, YYYY) or Surname (YYYY)
    self_cite_per_1k    self-citations   Stahl, YYYY / Stahl (YYYY)
"""

import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import nltk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NARRATIVE_PATH = ROOT / "reports" / "narrative_chunks.json"
OUTPUT_PATH    = ROOT / "reports" / "style_features.csv"

# Ensure sentence tokenizer data is available
nltk.download("punkt_tab", quiet=True)

# ── Word / sentence helpers ──────────────────────────────────────────────────

_WORD_PAT  = re.compile(r"\b[a-zA-Z'\-]{2,}\b")   # word tokens for TTR / lengths
_VOWEL_RUN = re.compile(r"[aeiouy]+", re.I)


def _syllables(word: str) -> int:
    """Heuristic syllable count (strip trailing silent-e, count vowel groups)."""
    w = word.lower()
    if len(w) > 2 and w[-1] == "e" and w[-2] not in "aeiouy":
        w = w[:-1]
    return max(1, len(_VOWEL_RUN.findall(w)))


def word_tokens(text: str) -> list[str]:
    return _WORD_PAT.findall(text)


def sentences(text: str) -> list[str]:
    return nltk.sent_tokenize(text)


# ── Vocabulary metrics ───────────────────────────────────────────────────────

_ABSTRACT_SUFFIX = re.compile(
    r"(?:tion|ity|ness|ment|ism|ance|ence|ivity)$",
    re.IGNORECASE,
)

def vocab_metrics(tokens: list[str]) -> dict:
    n = len(tokens)
    if n == 0:
        return {"ttr": 0.0, "mean_word_len": 0.0, "abstract_ratio": 0.0, "hapax_ratio": 0.0}
    lower  = [t.lower() for t in tokens]
    counts = Counter(lower)
    ttr   = len(counts) / n
    mwl   = sum(len(t) for t in lower) / n
    abst  = sum(1 for t in lower if _ABSTRACT_SUFFIX.search(t)) / n
    hapax = sum(1 for v in counts.values() if v == 1) / n
    return {
        "ttr": round(ttr, 4), "mean_word_len": round(mwl, 3),
        "abstract_ratio": round(abst, 4), "hapax_ratio": round(hapax, 4),
    }


# ── Sentence rhythm ──────────────────────────────────────────────────────────

def sentence_metrics(sents: list[str]) -> dict:
    n = len(sents)
    if n == 0:
        return {
            "sent_count": 0, "sent_mean_words": 0.0,
            "sent_std_words": 0.0, "sent_pct_long": 0.0,
            "sent_pct_short": 0.0, "pct_questions": 0.0,
        }
    lens = [len(s.split()) for s in sents]
    mean = sum(lens) / n
    variance = sum((l - mean) ** 2 for l in lens) / n
    std  = math.sqrt(variance)
    long_pct  = 100 * sum(1 for l in lens if l > 30) / n
    short_pct = 100 * sum(1 for l in lens if l < 10) / n
    q_pct     = 100 * sum(1 for s in sents if s.rstrip().endswith("?")) / n
    return {
        "sent_count":     n,
        "sent_mean_words": round(mean, 2),
        "sent_std_words":  round(std, 2),
        "sent_pct_long":   round(long_pct, 2),
        "sent_pct_short":  round(short_pct, 2),
        "pct_questions":   round(q_pct, 2),
    }


# ── Readability ──────────────────────────────────────────────────────────────

def readability_metrics(tokens: list[str], sents: list[str]) -> dict:
    """Flesch Reading Ease, Flesch-Kincaid Grade Level, Gunning Fog Index."""
    n_words = len(tokens)
    n_sents = len(sents)
    if n_words == 0 or n_sents == 0:
        return {"flesch_ease": 0.0, "fk_grade": 0.0, "fog_index": 0.0}
    syl_counts = [_syllables(t) for t in tokens]
    n_syl      = sum(syl_counts)
    n_complex  = sum(1 for s in syl_counts if s >= 3)
    asl = n_words / n_sents          # avg sentence length
    asw = n_syl   / n_words          # avg syllables per word
    flesch = 206.835 - 1.015 * asl - 84.6 * asw
    fk     = 0.39  * asl + 11.8 * asw - 15.59
    fog    = 0.4   * (asl + 100 * n_complex / n_words)
    return {
        "flesch_ease": round(flesch, 2),
        "fk_grade":    round(fk,    2),
        "fog_index":   round(fog,   2),
    }


# ── Hedge / assert ───────────────────────────────────────────────────────────

_HEDGE_PAT = re.compile(
    r"\b(perhaps|maybe|might|could|seems?|appear(?:s|ed)?|arguably|suggest"
    r"s?|suggest(?:ing|ed)?|possibly|likely|probable|probably|tentative"
    r"ly?|uncertain|unclear|it\s+is\s+possible|one\s+might|may\s+be)\b",
    re.IGNORECASE,
)
_ASSERT_PAT = re.compile(
    r"\b(clearly|necessarily|obviously|certainly|must\b|indeed|undoubtedly"
    r"|always|never|inevitably|fundamentally|essentially|demonstrates?"
    r"|demonstrate(?:d|ing)?|shows?\b|show(?:ed|ing)?|proves?\b|"
    r"prove(?:d|n|ing)?|establishes?)\b",
    re.IGNORECASE,
)

def hedge_metrics(text: str, words_1k: float) -> dict:
    if words_1k == 0:
        return {"hedge_per_1k": 0.0, "assert_per_1k": 0.0, "hedge_assert_ratio": 0.5}
    h = len(_HEDGE_PAT.findall(text))
    a = len(_ASSERT_PAT.findall(text))
    ratio = h / (h + a) if (h + a) > 0 else 0.5
    return {
        "hedge_per_1k":     round(h / words_1k, 3),
        "assert_per_1k":    round(a / words_1k, 3),
        "hedge_assert_ratio": round(ratio, 4),
    }


# ── Discourse connectors ─────────────────────────────────────────────────────

_CAUSAL    = re.compile(r"\b(therefore|thus|hence|consequently|as\s+a\s+result)\b", re.I)
_CONTRAST  = re.compile(r"\b(however|nevertheless|nonetheless|although|whereas"
                         r"|yet\b|while\b|on\s+the\s+other\s+hand|in\s+contrast)\b", re.I)
_ADDITIVE  = re.compile(r"\b(furthermore|moreover|additionally|in\s+addition"
                         r"|also\b|as\s+well\s+as)\b", re.I)
_EXEMPLIFY = re.compile(r"\b(for\s+example|for\s+instance|such\s+as|e\.g\.|"
                         r"as\s+illustrated|as\s+shown)\b", re.I)

def connector_metrics(text: str, words_1k: float) -> dict:
    if words_1k == 0:
        return {k: 0.0 for k in
                ("causal_per_1k", "contrast_per_1k", "additive_per_1k", "exemplify_per_1k")}
    return {
        "causal_per_1k":    round(len(_CAUSAL.findall(text))    / words_1k, 3),
        "contrast_per_1k":  round(len(_CONTRAST.findall(text))  / words_1k, 3),
        "additive_per_1k":  round(len(_ADDITIVE.findall(text))  / words_1k, 3),
        "exemplify_per_1k": round(len(_EXEMPLIFY.findall(text)) / words_1k, 3),
    }


# ── Voice ────────────────────────────────────────────────────────────────────

_I_PAT      = re.compile(r"\b(I|me|my|mine|myself)\b")
_WE_PAT     = re.compile(r"\b(we|our|us|ourselves)\b", re.I)
_PASSIVE_PAT = re.compile(r"\b(was|were)\s+(?:\w+ly\s+)?(\w+ed)\b", re.I)

def voice_metrics(text: str, words_1k: float) -> dict:
    if words_1k == 0:
        return {"i_per_1k": 0.0, "we_per_1k": 0.0, "passive_per_1k": 0.0}
    return {
        "i_per_1k":       round(len(_I_PAT.findall(text))       / words_1k, 3),
        "we_per_1k":      round(len(_WE_PAT.findall(text))       / words_1k, 3),
        "passive_per_1k": round(len(_PASSIVE_PAT.findall(text)) / words_1k, 3),
    }


# ── Punctuation density ──────────────────────────────────────────────────────

_EMDASH_PAT = re.compile(r"—|(?<!\s)--(?![\s>])")

def punctuation_metrics(text: str, words_1k: float) -> dict:
    """Semicolons, em-dashes, and colons per 1000 words."""
    if words_1k == 0:
        return {"semicolon_per_1k": 0.0, "emdash_per_1k": 0.0, "colon_per_1k": 0.0}
    semicolons = text.count(";")
    emdashes   = len(_EMDASH_PAT.findall(text))
    colons     = len(re.findall(r"(?<!\d):(?!\d)", text))  # exclude time/ratio
    return {
        "semicolon_per_1k": round(semicolons / words_1k, 3),
        "emdash_per_1k":    round(emdashes   / words_1k, 3),
        "colon_per_1k":     round(colons     / words_1k, 3),
    }


# ── Citation density ─────────────────────────────────────────────────────────

_CITE_PAT = re.compile(
    r"(?:[A-Z][a-z]+(?:\s+et\s+al\.?|\s+&\s+[A-Z][a-z]+)?)"  # Author(s)
    r"(?:,\s*|\s+)\(?\d{4}[a-z]?\)?",                          # , YYYY or (YYYY)
    re.MULTILINE,
)
_SELF_PAT = re.compile(
    r"Stahl(?:\s+et\s+al\.?|\s+&\s+[^,()]{1,40})?(?:,\s*|\s+)\(?\d{4}[a-z]?\)?",
    re.IGNORECASE,
)

def citation_metrics(text: str, words_1k: float) -> dict:
    if words_1k == 0:
        return {"citation_per_1k": 0.0, "self_cite_per_1k": 0.0}
    all_cites  = len(_CITE_PAT.findall(text))
    self_cites = len(_SELF_PAT.findall(text))
    return {
        "citation_per_1k":  round(all_cites  / words_1k, 3),
        "self_cite_per_1k": round(self_cites / words_1k, 3),
    }


# ── Academic register ────────────────────────────────────────────────────────

_METALINGUISTIC = re.compile(
    r"\bthis\s+(?:paper|chapter|article|study|analysis|book|essay|section)"
    r"\s+(?:argues?|shows?|demonstrates?|examines?|explores?|investigates?"
    r"|presents?|proposes?|suggests?|contends?|claims?)"
    r"|\bI\s+(?:argue|show|demonstrate|claim|propose|suggest|examine|explore"
    r"|contend|maintain|assert)\b",
    re.I,
)
_DEFINITIONAL = re.compile(
    r"\b(?:is\s+defined\s+as|refers?\s+to\b|(?<!\w)means?\b|is\s+called"
    r"|is\s+known\s+as|that\s+is(?:\s+to\s+say)?|i\.e\.|in\s+other\s+words"
    r"|to\s+put\s+it\s+(?:another\s+way|simply|differently))\b",
    re.I,
)
_SUBORDINATE = re.compile(
    r"\b(?:because|although|since\b|unless|whereas|despite\b|even\s+though"
    r"|so\s+that|in\s+order\s+(?:to|that)|given\s+that|provided\s+that"
    r"|assuming\s+that)\b",
    re.I,
)

def register_metrics(text: str, words_1k: float) -> dict:
    """Metalinguistic announcements, definitional phrases, subordinate conjunctions."""
    if words_1k == 0:
        return {"metalinguistic_per_1k": 0.0, "definitional_per_1k": 0.0, "subordinate_per_1k": 0.0}
    return {
        "metalinguistic_per_1k": round(len(_METALINGUISTIC.findall(text)) / words_1k, 3),
        "definitional_per_1k":   round(len(_DEFINITIONAL.findall(text))   / words_1k, 3),
        "subordinate_per_1k":    round(len(_SUBORDINATE.findall(text))    / words_1k, 3),
    }


# ── Per-chapter aggregation ──────────────────────────────────────────────────

def compute_chapter_features(
    chapter_text: str,
    chunk_count: int,
    narrative_words: int,
    meta: dict,
) -> dict:
    """Compute all style features for a concatenated chapter text."""
    tokens   = word_tokens(chapter_text)
    sents    = sentences(chapter_text)
    words_1k = narrative_words / 1000.0

    row = {
        # identity
        "book_number":      meta["book_number"],
        "chapter_number":   meta["chapter_number"],
        "book_title":       meta["book_title"],
        "chapter_title":    meta["chapter_title"],
        "pub_year":         meta["pub_year"],
        "cluster_id":       meta["cluster_id"],
        "cluster_label":    meta["cluster_label"],
        "chunk_count":      chunk_count,
        "narrative_words":  narrative_words,
    }
    row.update(sentence_metrics(sents))
    row.update(readability_metrics(tokens, sents))
    row.update(vocab_metrics(tokens))
    row.update(hedge_metrics(chapter_text, words_1k))
    row.update(connector_metrics(chapter_text, words_1k))
    row.update(voice_metrics(chapter_text, words_1k))
    row.update(punctuation_metrics(chapter_text, words_1k))
    row.update(citation_metrics(chapter_text, words_1k))
    row.update(register_metrics(chapter_text, words_1k))
    return row


# ── Main ─────────────────────────────────────────────────────────────────────

FIELDNAMES = [
    # identity
    "book_number", "chapter_number", "book_title", "chapter_title",
    "pub_year", "cluster_id", "cluster_label", "chunk_count", "narrative_words",
    # sentence
    "sent_count", "sent_mean_words", "sent_std_words",
    "sent_pct_long", "sent_pct_short", "pct_questions",
    # readability
    "flesch_ease", "fk_grade", "fog_index",
    # vocabulary
    "ttr", "mean_word_len", "abstract_ratio", "hapax_ratio",
    # hedging
    "hedge_per_1k", "assert_per_1k", "hedge_assert_ratio",
    # connectors
    "causal_per_1k", "contrast_per_1k", "additive_per_1k", "exemplify_per_1k",
    # voice
    "i_per_1k", "we_per_1k", "passive_per_1k",
    # punctuation
    "semicolon_per_1k", "emdash_per_1k", "colon_per_1k",
    # citation
    "citation_per_1k", "self_cite_per_1k",
    # academic register
    "metalinguistic_per_1k", "definitional_per_1k", "subordinate_per_1k",
]


def main() -> None:
    print("Loading narrative chunks…")
    data = json.loads(NARRATIVE_PATH.read_text())

    # Group chunks by (book_number, chapter_number), preserving metadata
    chapters: dict[tuple, dict] = {}    # key → {texts, words, meta}
    for doc_id, rec in data.items():
        key = (rec["book_number"], rec["chapter_number"])
        if key not in chapters:
            chapters[key] = {
                "texts":  [],
                "words":  0,
                "count":  0,
                "meta":   {
                    "book_number":    rec["book_number"],
                    "chapter_number": rec["chapter_number"],
                    "book_title":     rec.get("book_title", ""),
                    "chapter_title":  rec.get("chapter_title", ""),
                    "pub_year":       rec.get("pub_year", 0),
                    "cluster_id":     rec.get("cluster_id", -1),
                    "cluster_label":  rec.get("cluster_label", ""),
                },
            }
        chapters[key]["texts"].append(rec["narrative_text"])
        chapters[key]["words"] += rec["narrative_word_count"]
        chapters[key]["count"] += 1
        # Use plurality cluster: update if this chunk's cluster is the same or
        # inherit the dominant cluster_id from the data (set on first chunk)
        # — for style analysis, the chapter-level cluster is the one from
        # chunk_clusters.csv majority; we'll just carry forward the first seen
        # (a separate aggregation step can improve this later).

    print(f"Chapters found: {len(chapters):,}")
    print("Computing style features…")

    rows = []
    for key in sorted(chapters.keys()):
        ch = chapters[key]
        full_text = "\n\n".join(ch["texts"])
        row = compute_chapter_features(
            full_text,
            chunk_count=ch["count"],
            narrative_words=ch["words"],
            meta=ch["meta"],
        )
        rows.append(row)

    # Sort by pub_year then book/chapter
    rows.sort(key=lambda r: (r["pub_year"] or 9999, r["book_number"], r["chapter_number"]))

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary stats ──────────────────────────────────────────────────────
    years = [r["pub_year"] for r in rows if r["pub_year"]]
    print(f"\n{'─'*55}")
    print(f"Chapters analysed:  {len(rows):>6,}")
    print(f"Year range:         {min(years)}–{max(years)}")
    print(f"Total narr. words:  {sum(r['narrative_words'] for r in rows):>10,}")
    print(f"\nMeans across all chapters:")
    for col in ("sent_mean_words", "flesch_ease", "fk_grade", "fog_index",
                "ttr", "hapax_ratio", "abstract_ratio",
                "hedge_per_1k", "assert_per_1k",
                "semicolon_per_1k", "emdash_per_1k",
                "i_per_1k", "we_per_1k", "citation_per_1k", "self_cite_per_1k",
                "metalinguistic_per_1k", "definitional_per_1k", "subordinate_per_1k"):
        vals = [r[col] for r in rows if r[col] is not None]
        print(f"  {col:<24} {sum(vals)/len(vals):.3f}")
    print(f"{'─'*55}")
    print(f"\nOutput → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
