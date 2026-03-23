"""
elibrary_cache.py — Data model and persistence for the Gerry Stahl e-library cache.

The cache (elibrary_cache.pkl) stores structured metadata for 22 academic books
and their 337 chapters, ready for embedding, summarization, and RAG querying.

Structure (mirrors cache_structure.txt):

  ElibraryCache
  └── books: List[Book]
        ├── book_number          int      — 1–22
        ├── book_name            str      — PDF filename  (e.g. "3.gc.pdf")
        ├── book_title           str      — Full book title
        ├── book_author          str      — Primary book author(s)
        ├── book_keywords        List[str]
        ├── book_reference       str      — APA-style book citation
        ├── book_text            str      — Full book text (from PDF parser)
        ├── book_number_of_pages int      — Total pages
        ├── book_kind            str      — sections|essays|chapters|…
        ├── book_summaries       List[BookSummary]
        │     ├── book_summary_author          str
        │     ├── book_summary_date            str
        │     ├── book_summary_prompt          str
        │     ├── book_summary_number_of_words int
        │     └── book_summary_text            str
        └── book_chapters        List[Chapter]
              ├── chapter_number          int
              ├── chapter_title           str
              ├── chapter_author          str
              ├── chapter_keywords        List[str]
              ├── chapter_reference       str
              ├── chapter_text            str
              ├── chapter_number_of_pages  int
              ├── chapter_number_of_words  int
              ├── chapter_number_of_tokens int
              ├── chapter_number_of_symbols int
              └── chapter_summaries       List[ChapterSummary]
                    ├── chapter_summary_author          str
                    ├── chapter_summary_date            str
                    ├── chapter_summary_prompt          str
                    ├── chapter_summary_number_of_words int
                    └── chapter_summary_text            str

Author : Gerry Stahl
Created: March 2026
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Summary dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ChapterSummary:
    """AI-generated summary of a single chapter."""
    chapter_summary_author: str = ""           # model or human author of the summary
    chapter_summary_date: str = ""             # ISO date string  (e.g. "2026-03-01")
    chapter_summary_prompt: str = ""           # prompt template used to generate the summary
    chapter_summary_number_of_words: int = 0   # word count of the summary text
    chapter_summary_text: str = ""             # summary text

    def __setstate__(self, state: dict) -> None:
        """Migrate old un-prefixed field names (loaded from legacy pkl)."""
        migration = {
            "author":          "chapter_summary_author",
            "date":            "chapter_summary_date",
            "prompt":          "chapter_summary_prompt",
            "number_of_words": "chapter_summary_number_of_words",
            "text":            "chapter_summary_text",
        }
        new_state = {migration.get(k, k): v for k, v in state.items()}
        defaults = {
            "chapter_summary_author": "",
            "chapter_summary_date": "",
            "chapter_summary_prompt": "",
            "chapter_summary_number_of_words": 0,
            "chapter_summary_text": "",
        }
        for k, v in defaults.items():
            new_state.setdefault(k, v)
        self.__dict__.update(new_state)


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """
    One text chunk within a chapter, produced by the chunking pipeline.

    Chunks exist at multiple hierarchy levels:
      level 0  — base chunks (fixed-size splits, ~1 000 chars)
      level 1+ — merged chunks (semantically similar neighbours merged
                 by SemanticHierarchicalChunker)

    ``chunk_page_number`` is an approximation computed from char offsets
    and ``Chapter.chapter_number_of_pages``; it counts pages from the
    start of the chapter (not from the start of the PDF).
    """
    chunk_index:           int  = 0    # 0-based index within (book, chapter, level)
    chunk_book_number:     int  = 0    # Book.book_number  (1–22)
    chunk_chapter_number:  int  = 0    # Chapter.chapter_number (1-based)
    chunk_page_number:     int  = 0    # approx. page within chapter
    chunk_text:            str  = ""   # raw chunk text
    chunk_section_heading: str  = ""   # nearest detected section heading (may be empty)
    chunk_level:           int  = 0    # hierarchy level (0 = base)
    chunk_start_offset:    int  = 0    # char offset in Chapter.chapter_text
    chunk_end_offset:      int  = 0    # char offset (exclusive)
    chunk_word_count:      int  = 0
    chunk_token_count:     int  = 0    # filled by tokenizer; 0 until computed

    def __setstate__(self, state: dict) -> None:
        defaults = {
            "chunk_index":           0,
            "chunk_book_number":     0,
            "chunk_chapter_number":  0,
            "chunk_page_number":     0,
            "chunk_text":            "",
            "chunk_section_heading": "",
            "chunk_level":           0,
            "chunk_start_offset":    0,
            "chunk_end_offset":      0,
            "chunk_word_count":      0,
            "chunk_token_count":     0,
        }
        for k, v in defaults.items():
            state.setdefault(k, v)
        self.__dict__.update(state)


@dataclass
class BookSummary:
    """AI-generated summary of a whole book (usually derived from chapter summaries)."""
    book_summary_author: str = ""
    book_summary_date: str = ""
    book_summary_prompt: str = ""
    book_summary_number_of_words: int = 0
    book_summary_text: str = ""

    def __setstate__(self, state: dict) -> None:
        """Migrate old un-prefixed field names (loaded from legacy pkl)."""
        migration = {
            "author":          "book_summary_author",
            "date":            "book_summary_date",
            "prompt":          "book_summary_prompt",
            "number_of_words": "book_summary_number_of_words",
            "text":            "book_summary_text",
        }
        new_state = {migration.get(k, k): v for k, v in state.items()}
        defaults = {
            "book_summary_author": "",
            "book_summary_date": "",
            "book_summary_prompt": "",
            "book_summary_number_of_words": 0,
            "book_summary_text": "",
        }
        for k, v in defaults.items():
            new_state.setdefault(k, v)
        self.__dict__.update(new_state)


# ---------------------------------------------------------------------------
# Chapter dataclass
# ---------------------------------------------------------------------------

@dataclass
class Chapter:
    """
    One chapter (or section / essay / editorial) within a book.

    Numeric metrics are populated either by the PDF parser or by
    the embedding pipeline; they start at 0 until filled in.
    """
    chapter_number: int = 0                              # chapter number within the book
    chapter_title: str = ""
    chapter_author: str = ""
    chapter_keywords: List[str] = field(default_factory=list)
    chapter_reference: str = ""                          # APA citation for the chapter
    chapter_text: str = ""                               # full chapter text
    chapter_number_of_pages: int = 0
    chapter_number_of_words: int = 0
    chapter_number_of_tokens: int = 0
    chapter_number_of_symbols: int = 0
    chapter_summaries: List[ChapterSummary] = field(default_factory=list)
    chapter_chunks: List["Chunk"] = field(default_factory=list)  # all hierarchy levels

    def __setstate__(self, state: dict) -> None:
        """Migrate old un-prefixed field names (loaded from legacy pkl)."""
        migration = {
            "number":           "chapter_number",
            "title":            "chapter_title",
            "author":           "chapter_author",
            "reference":        "chapter_reference",
            "text":             "chapter_text",
            "number_of_pages":  "chapter_number_of_pages",
            "number_of_words":  "chapter_number_of_words",
            "number_of_tokens": "chapter_number_of_tokens",
            "number_of_symbols":"chapter_number_of_symbols",
            "summaries":        "chapter_summaries",
        }
        new_state = {migration.get(k, k): v for k, v in state.items()}
        defaults = {
            "chapter_number": 0,
            "chapter_title": "",
            "chapter_author": "",
            "chapter_keywords": [],
            "chapter_reference": "",
            "chapter_text": "",
            "chapter_number_of_pages": 0,
            "chapter_number_of_words": 0,
            "chapter_number_of_tokens": 0,
            "chapter_number_of_symbols": 0,
            "chapter_summaries": [],
            "chapter_chunks": [],
        }
        for k, v in defaults.items():
            new_state.setdefault(k, v)
        self.__dict__.update(new_state)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def add_summary(self, summary: ChapterSummary) -> None:
        """Append a new summary."""
        self.chapter_summaries.append(summary)

    def latest_summary(self) -> Optional[ChapterSummary]:
        """Return the most recently added summary, or None."""
        return self.chapter_summaries[-1] if self.chapter_summaries else None

    def compute_metrics(self) -> None:
        """
        Derive chapter_number_of_words and chapter_number_of_symbols from chapter_text.
        chapter_number_of_tokens is left for the tokenizer to fill in.
        """
        if self.chapter_text:
            self.chapter_number_of_words = len(self.chapter_text.split())
            self.chapter_number_of_symbols = len(self.chapter_text)

    def __repr__(self) -> str:
        return (
            f"Chapter(chapter_number={self.chapter_number}, "
            f"chapter_title={self.chapter_title!r}, "
            f"words={self.chapter_number_of_words})"
        )


# ---------------------------------------------------------------------------
# Book dataclass
# ---------------------------------------------------------------------------

@dataclass
class Book:
    """
    One academic book in the e-library, backed by a PDF file.

    The ``book_chapters`` list is ordered by chapter number.
    Chapter 0 is the book-level entry (front matter / book overview);
    chapters 1+ are the actual content chapters.
    """
    book_number: int = 0                          # book number 1–22
    book_name: str = ""                           # PDF filename  (e.g. "3.gc.pdf")
    book_title: str = ""                          # full book title
    book_author: str = ""                         # primary author(s)
    book_keywords: List[str] = field(default_factory=list)
    book_reference: str = ""                      # APA book citation
    book_text: str = ""                           # full book text (concatenation of chapters)
    book_number_of_pages: int = 0
    book_kind: str = ""                           # sections | essays | chapters | papers | …
    book_summaries: List[BookSummary] = field(default_factory=list)
    book_chapters: List[Chapter] = field(default_factory=list)

    def __setstate__(self, state: dict) -> None:
        """Migrate old un-prefixed field names (loaded from legacy pkl)."""
        migration = {
            "number":          "book_number",
            "name":            "book_name",
            "title":           "book_title",
            "author":          "book_author",
            "reference":       "book_reference",
            "text":            "book_text",
            "number_of_pages": "book_number_of_pages",
            "kind":            "book_kind",
            "summaries":       "book_summaries",
            "chapters":        "book_chapters",
        }
        new_state = {migration.get(k, k): v for k, v in state.items()}
        defaults = {
            "book_number": 0,
            "book_name": "",
            "book_title": "",
            "book_author": "",
            "book_keywords": [],
            "book_reference": "",
            "book_text": "",
            "book_number_of_pages": 0,
            "book_kind": "",
            "book_summaries": [],
            "book_chapters": [],
        }
        for k, v in defaults.items():
            new_state.setdefault(k, v)
        self.__dict__.update(new_state)

    # ------------------------------------------------------------------
    # Chapter helpers
    # ------------------------------------------------------------------

    def add_chapter(self, chapter: Chapter) -> None:
        """Append a chapter, keeping list sorted by chapter number."""
        self.book_chapters.append(chapter)
        self.book_chapters.sort(key=lambda c: c.chapter_number)

    def get_chapter(self, number: int) -> Optional[Chapter]:
        """Return the chapter with the given number, or None."""
        for ch in self.book_chapters:
            if ch.chapter_number == number:
                return ch
        return None

    def add_summary(self, summary: BookSummary) -> None:
        self.book_summaries.append(summary)

    def latest_summary(self) -> Optional[BookSummary]:
        return self.book_summaries[-1] if self.book_summaries else None

    @property
    def content_chapters(self) -> List[Chapter]:
        """Chapters with chapter_number > 0 (excludes the book-level entry)."""
        return [ch for ch in self.book_chapters if ch.chapter_number > 0]

    @property
    def total_words(self) -> int:
        return sum(ch.chapter_number_of_words for ch in self.content_chapters)

    @property
    def total_tokens(self) -> int:
        return sum(ch.chapter_number_of_tokens for ch in self.content_chapters)

    def __repr__(self) -> str:
        return (
            f"Book(book_number={self.book_number}, book_name={self.book_name!r}, "
            f"chapters={len(self.content_chapters)}, words={self.total_words})"
        )


# ---------------------------------------------------------------------------
# Top-level cache
# ---------------------------------------------------------------------------

@dataclass
class ElibraryCache:
    """
    Root container for the entire e-library cache.

    Usage
    -----
    Build from CSV::

        cache = ElibraryCache()
        cache.add_book(book)
        cache.save()

    Load existing::

        cache = ElibraryCache.load()
        book  = cache.get_book(3)

    Default pickle path: ``~/AI/elibrary/elibrary_cache.pkl``
    """

    DEFAULT_PATH: str = field(
        default="/Users/GStahl2/AI/elibrary/cache/elibrary_cache.pkl",
        init=False,
        repr=False,
        compare=False,
    )

    books: List[Book] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Book helpers
    # ------------------------------------------------------------------

    def add_book(self, book: Book) -> None:
        """Add a book, keeping list sorted by book number."""
        self.books.append(book)
        self.books.sort(key=lambda b: b.book_number)

    def get_book(self, number: int) -> Optional[Book]:
        """Return the book with the given number, or None."""
        for book in self.books:
            if book.book_number == number:
                return book
        return None

    def get_book_by_name(self, name: str) -> Optional[Book]:
        """Return the book whose PDF name matches (case-insensitive)."""
        name_lower = name.lower()
        for book in self.books:
            if book.book_name.lower() == name_lower:
                return book
        return None

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
    def load(cls, path: Optional[str] = None) -> "ElibraryCache":
        """Deserialise the cache from *path* (defaults to DEFAULT_PATH)."""
        target = Path(path or cls.__dataclass_fields__["DEFAULT_PATH"].default)
        if not target.exists():
            raise FileNotFoundError(
                f"Cache file not found: {target}\n"
                "Run  scripts/build_cache.py  to create it."
            )
        import sys as _sys
        # Allow PKLs saved when the module was at the top-level 'elibrary_cache'
        # to be loaded now that it lives at 'cache.elibrary_cache'.
        import cache.elibrary_cache as _self_mod
        _sys.modules.setdefault("elibrary_cache", _self_mod)
        with open(target, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Unexpected pickle type: {type(obj)}")
        return obj

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def total_books(self) -> int:
        return len(self.books)

    @property
    def total_chapters(self) -> int:
        return sum(len(b.content_chapters) for b in self.books)

    @property
    def total_words(self) -> int:
        return sum(b.total_words for b in self.books)

    def summary_stats(self) -> str:
        """Return a human-readable statistics string."""
        lines = [
            "=== ElibraryCache statistics ===",
            f"  Books    : {self.total_books}",
            f"  Chapters : {self.total_chapters}",
            f"  Words    : {self.total_words:,}",
            "",
            f"  {'#':>3}  {'Name':<25}  {'Kind':<12}  {'Chapters':>8}  {'Words':>10}",
            f"  {'-'*3}  {'-'*25}  {'-'*12}  {'-'*8}  {'-'*10}",
        ]
        for b in self.books:
            lines.append(
                f"  {b.book_number:>3}  {b.book_name:<25}  {b.book_kind:<12}  "
                f"{len(b.content_chapters):>8}  {b.total_words:>10,}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ElibraryCache(books={self.total_books}, "
            f"chapters={self.total_chapters}, "
            f"words={self.total_words:,})"
        )
