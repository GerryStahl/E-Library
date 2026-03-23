"""
editorials_cache.py — Data model and persistence for the ijCSCL editorials cache.

The cache (editorials_cache.pkl) stores structured data for 40 ijCSCL editorial
PDFs (2016–2025, Vol 11–20, 4 issues/year), ready for text extraction,
summarization, and RAG querying.

Structure (mirrors cache_structure.txt):

  EditorialsCache
  ├── editorials: List[Editorial]
  ├── annual_summaries: List[AnnualSummary]
  │     ├── year             int
  │     └── summary          EditorialSummary | None
  └── total_history: TotalHistory
        └── summary          EditorialSummary | None

  (Editorial fields — see Editorial dataclass)
        ├── year             int      — calendar year  (2016–2025)
        ├── volume           int      — journal volume (11–20; vol = year − 2005)
        ├── issue            int      — issue number   (1–4)
        ├── title            str      — editorial title (extracted from PDF)
        ├── author           str      — author(s), comma-separated
        ├── words            int      — word count of the extracted text
        ├── text             str      — full extracted text of the editorial PDF
        ├── summary          EditorialSummary | None   — Claude summary
        │     ├── summary_author          str
        │     ├── summary_date            str
        │     ├── summary_prompt          str
        │     ├── summary_number_of_words int
        │     └── summary_text            str
        ├── summary_gpt4o    EditorialSummary | None   — GPT-4o summary
        │     └── (same fields as summary)
        ├── summary_book     EditorialSummary | None   — copied from book 16 chapter summaries
        │     └── (same fields as summary)
        └── pdf              str      — PDF filename (in editorials/)

Author : Gerry Stahl
Created: March 2026
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# EditorialSummary dataclass
# ---------------------------------------------------------------------------

@dataclass
class EditorialSummary:
    """AI-generated summary of a single ijCSCL editorial."""
    summary_author: str = ""            # model or human author of the summary
    summary_date: str = ""              # ISO date string  (e.g. "2026-03-05")
    summary_prompt: str = ""            # prompt template used to generate the summary
    summary_number_of_words: int = 0    # word count of the summary text
    summary_text: str = ""              # summary text


# ---------------------------------------------------------------------------
# AnnualSummary dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnnualSummary:
    """Container for one year's summary of ijCSCL editorials."""
    year: int = 0                                         # calendar year (2006–2025)
    summary: Optional[EditorialSummary] = None            # Claude annual summary (from Level-1 Claude texts)
    summary_gpt4o: Optional[EditorialSummary] = None      # GPT-4o annual summary (from Level-1 GPT-4o texts)
    summary_book: Optional[EditorialSummary] = None       # Copilot annual summary (from Level-1 Book texts)

    def __setstate__(self, state: dict) -> None:
        state.setdefault('year', 0)
        state.setdefault('summary', None)
        state.setdefault('summary_gpt4o', None)
        state.setdefault('summary_book', None)
        self.__dict__.update(state)

    def has_summary(self) -> bool:
        return self.summary is not None and bool(self.summary.summary_text.strip())

    def has_summary_gpt4o(self) -> bool:
        return self.summary_gpt4o is not None and bool(self.summary_gpt4o.summary_text.strip())

    def has_summary_book(self) -> bool:
        return self.summary_book is not None and bool(self.summary_book.summary_text.strip())

    def __repr__(self) -> str:
        c = 'yes' if self.has_summary() else 'no'
        g = 'yes' if self.has_summary_gpt4o() else 'no'
        b = 'yes' if self.has_summary_book() else 'no'
        return f"AnnualSummary(year={self.year}, claude={c}, gpt4o={g}, book={b})"


# ---------------------------------------------------------------------------
# TotalHistory dataclass
# ---------------------------------------------------------------------------

@dataclass
class TotalHistory:
    """Container for the single 20-year history summary of all ijCSCL editorials."""
    summary: Optional[EditorialSummary] = None        # Claude total-history summary
    summary_gpt4o: Optional[EditorialSummary] = None  # GPT-4o total-history summary
    summary_book: Optional[EditorialSummary] = None   # Claude summary from book-level editorial texts

    def __setstate__(self, state: dict) -> None:
        state.setdefault('summary', None)
        state.setdefault('summary_gpt4o', None)
        state.setdefault('summary_book', None)
        self.__dict__.update(state)

    def has_summary(self) -> bool:
        return self.summary is not None and bool(self.summary.summary_text.strip())

    def has_summary_gpt4o(self) -> bool:
        return self.summary_gpt4o is not None and bool(self.summary_gpt4o.summary_text.strip())

    def has_summary_book(self) -> bool:
        return self.summary_book is not None and bool(self.summary_book.summary_text.strip())

    def __repr__(self) -> str:
        c = 'yes' if self.has_summary() else 'no'
        g = 'yes' if self.has_summary_gpt4o() else 'no'
        b = 'yes' if self.has_summary_book() else 'no'
        return f"TotalHistory(claude={c}, gpt4o={g}, book={b})"


# ---------------------------------------------------------------------------
# Editorial dataclass
# ---------------------------------------------------------------------------

@dataclass
class Editorial:
    """
    One ijCSCL editorial (the first article of a journal issue).

    ``text`` is populated when the PDF is parsed.
    ``summary`` is populated by the summarization pipeline.
    ``pdf`` is the filename (basename only) inside the editorials/ folder.
    """
    year: int = 0                               # calendar year   (2016–2025)
    volume: int = 0                             # journal volume  (11–20)
    issue: int = 0                              # issue number    (1–4)
    title: str = ""                             # editorial title (extracted from PDF)
    author: str = ""                            # author(s), comma-separated
    words: int = 0                              # word count of extracted text
    text: str = ""                              # full extracted PDF text
    summary: Optional[EditorialSummary] = None      # AI-generated summary (Claude)
    summary_gpt4o: Optional[EditorialSummary] = None  # AI-generated summary (GPT-4o)
    summary_book: Optional[EditorialSummary] = None   # copied from elibrary book 16 chapter summaries
    pdf: str = ""                               # PDF filename (basename)

    def __setstate__(self, state: dict) -> None:
        """Ensure missing fields get default values when loading older pkl files."""
        defaults = {
            'year': 0, 'volume': 0, 'issue': 0,
            'title': '', 'author': '', 'words': 0,
            'text': '', 'summary': None, 'summary_gpt4o': None, 'summary_book': None, 'pdf': '',
        }
        for k, v in defaults.items():
            state.setdefault(k, v)
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def set_summary(self, summary: EditorialSummary) -> None:
        """Replace the current summary."""
        self.summary = summary

    def clear_summary(self) -> None:
        """Remove the current summary."""
        self.summary = None

    def has_text(self) -> bool:
        return bool(self.text.strip())

    def has_summary(self) -> bool:
        return self.summary is not None and bool(self.summary.summary_text.strip())

    @property
    def label(self) -> str:
        """Human-readable identifier, e.g. 'Vol 11 (2016) Issue 1'."""
        return f"Vol {self.volume} ({self.year}) Issue {self.issue}"

    def __repr__(self) -> str:
        return (
            f"Editorial(year={self.year}, vol={self.volume}, issue={self.issue}, "
            f"text={'yes' if self.has_text() else 'no'}, "
            f"summary={'yes' if self.has_summary() else 'no'}, "
            f"pdf={self.pdf!r})"
        )


# ---------------------------------------------------------------------------
# Top-level cache
# ---------------------------------------------------------------------------

@dataclass
class EditorialsCache:
    """
    Root container for the ijCSCL editorials cache.

    Usage
    -----
    Load existing cache::

        cache = EditorialsCache.load()
        ed    = cache.get(year=2024, issue=2)

    Build from scratch::

        cache = EditorialsCache()
        cache.add(Editorial(year=2016, volume=11, issue=1, pdf="ijcscl_...pdf"))
        cache.save()

    Default pickle path: ``~/AI/elibrary/cache/editorials_cache.pkl``
    """

    DEFAULT_PATH: str = field(
        default="/Users/GStahl2/AI/elibrary/cache/editorials_cache.pkl",
        init=False,
        repr=False,
        compare=False,
    )

    editorials: List[Editorial] = field(default_factory=list)
    annual_summaries: List[AnnualSummary] = field(default_factory=list)
    total_history: TotalHistory = field(default_factory=TotalHistory)

    def __setstate__(self, state: dict) -> None:
        """Ensure missing top-level fields get defaults when loading older pkl."""
        state.setdefault('editorials', [])
        state.setdefault('annual_summaries', [])
        state.setdefault('total_history', TotalHistory())
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Annual summary helpers
    # ------------------------------------------------------------------

    def get_annual(self, year: int) -> Optional[AnnualSummary]:
        """Return the AnnualSummary for *year*, or None."""
        for a in self.annual_summaries:
            if a.year == year:
                return a
        return None

    def ensure_annual(self, year: int) -> AnnualSummary:
        """Return the AnnualSummary for *year*, creating a stub if absent."""
        a = self.get_annual(year)
        if a is None:
            a = AnnualSummary(year=year)
            self.annual_summaries.append(a)
            self.annual_summaries.sort(key=lambda x: x.year)
        return a

    # ------------------------------------------------------------------
    # Editorial helpers
    # ------------------------------------------------------------------

    def add(self, editorial: Editorial) -> None:
        """Add an editorial, keeping the list sorted by (year, issue)."""
        self.editorials.append(editorial)
        self.editorials.sort(key=lambda e: (e.year, e.issue))

    def get(self, year: int, issue: int) -> Optional[Editorial]:
        """Return the editorial for a given year and issue, or None."""
        for ed in self.editorials:
            if ed.year == year and ed.issue == issue:
                return ed
        return None

    def get_by_volume(self, volume: int, issue: int) -> Optional[Editorial]:
        """Return the editorial for a given volume and issue, or None."""
        for ed in self.editorials:
            if ed.volume == volume and ed.issue == issue:
                return ed
        return None

    def by_year(self, year: int) -> List[Editorial]:
        """Return all four issues for a given year."""
        return [ed for ed in self.editorials if ed.year == year]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> Path:
        """Serialise the cache to *path* (defaults to DEFAULT_PATH)."""
        target = Path(path or self.DEFAULT_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return target

    @classmethod
    def load(cls, path: Optional[str] = None) -> "EditorialsCache":
        """Deserialise the cache from *path* (defaults to DEFAULT_PATH)."""
        default = "/Users/GStahl2/AI/elibrary/cache/editorials_cache.pkl"
        target = Path(path or default)
        if not target.exists():
            raise FileNotFoundError(
                f"Cache file not found: {target}\n"
                "Run  scripts/build_editorials_cache.py  to create it."
            )
        with open(target, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Unexpected pickle type: {type(obj)}")
        return obj

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def total(self) -> int:
        return len(self.editorials)

    @property
    def with_text(self) -> int:
        return sum(1 for ed in self.editorials if ed.has_text())

    @property
    def with_summary(self) -> int:
        return sum(1 for ed in self.editorials if ed.has_summary())

    def summary_stats(self) -> str:
        """Return a human-readable statistics string."""
        lines = [
            "=== EditorialsCache statistics ===",
            f"  Editorials  : {self.total}",
            f"  With text   : {self.with_text}",
            f"  With summary: {self.with_summary}",
            "",
            f"  {'Year':>4}  {'Vol':>3}  {'Iss':>3}  {'Words':>6}  {'Summ':>4}  {'Author':<30}  Title",
            f"  {'-'*4}  {'-'*3}  {'-'*3}  {'-'*6}  {'-'*4}  {'-'*30}  {'-'*40}",
        ]
        for ed in self.editorials:
            author_short = (ed.author[:28] + '…') if len(ed.author) > 29 else ed.author
            title_short  = (ed.title[:38] + '…')  if len(ed.title)  > 39 else ed.title
            lines.append(
                f"  {ed.year:>4}  {ed.volume:>3}  {ed.issue:>3}  "
                f"{ed.words:>6,}  "
                f"{'yes' if ed.has_summary() else 'no':>4}  "
                f"{author_short:<30}  {title_short}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"EditorialsCache(total={self.total}, "
            f"with_text={self.with_text}, "
            f"with_summary={self.with_summary})"
        )
