"""
chunkers/semantic_chunker.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Splits chapter text into overlapping fixed-size chunks using
LangChain's RecursiveCharacterTextSplitter.

The splitter prefers natural paragraph / line-break boundaries,
falling back to word and character breaks only when needed, so
each chunk remains readable and self-contained.

Each returned chunk dict carries metadata that unambiguously
identifies its origin in the e-library::

    {
        "text":     "<chunk text>",
        "metadata": {
            "book_number":     <int>,
            "chapter_number":  <int>,
            "chunk_index":     <int>,   # 0-based within the chapter
            "char_start":      <int>,   # byte offset in chapter_text
            "char_end":        <int>,
            "word_count":      <int>,
            # any extra fields passed via extra_metadata
        }
    }

Usage
-----
    from chunkers.semantic_chunker import TextChunker

    chunker = TextChunker(chunk_size=1_000, overlap=200)
    chunks  = chunker.chunk_chapter(
        text           = chapter.chapter_text,
        book_number    = book.book_number,
        chapter_number = chapter.chapter_number,
        extra_metadata = {
            "book_title":    book.book_title,
            "chapter_title": chapter.chapter_title,
        },
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """
    Split chapter text into overlapping fixed-size chunks.

    Parameters
    ----------
    chunk_size : int
        Target chunk size in characters (default 1 000).
    overlap : int
        Number of characters shared between consecutive chunks (default 200).
        Overlap preserves context across chunk boundaries, which improves
        retrieval quality for passages that span a boundary.
    """

    DEFAULT_CHUNK_SIZE = 1_000
    DEFAULT_OVERLAP    = 200

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_chapter(
        self,
        text: str,
        book_number: int,
        chapter_number: int,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Split *text* and return a list of chunk dicts.

        Parameters
        ----------
        text : str
            Full chapter text (``Chapter.chapter_text``).
        book_number : int
            Book number (1-22).
        chapter_number : int
            Chapter number within the book.
        extra_metadata : dict, optional
            Additional key/value pairs merged into every chunk's
            ``metadata`` dict (e.g. ``book_title``, ``chapter_title``).

        Returns
        -------
        list of dict
            Each element is ``{"text": str, "metadata": dict}``.
        """
        if not text or not text.strip():
            return []

        pieces = self._splitter.split_text(text)
        extra  = extra_metadata or {}
        chunks: List[Dict[str, Any]] = []
        search_pos = 0

        for idx, piece in enumerate(pieces):
            # Locate this piece inside the original text so we can record
            # byte offsets.  Search forward from the previous match to
            # handle cases where the same substring appears multiple times.
            start = text.find(piece, search_pos)
            if start == -1:
                # Overlap or whitespace normalisation shifted it; try from 0.
                start = text.find(piece)
            if start == -1:
                # Last resort: approximate position.
                start = search_pos
            end = start + len(piece)

            chunks.append({
                "text": piece,
                "metadata": {
                    "book_number":    book_number,
                    "chapter_number": chapter_number,
                    "chunk_index":    idx,
                    "char_start":     start,
                    "char_end":       end,
                    "word_count":     len(piece.split()),
                    **extra,
                },
            })

            # Advance search position by 1 past the start of this match
            # so the next call to find() can locate an overlapping piece.
            search_pos = start + 1

        return chunks

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TextChunker(chunk_size={self.chunk_size}, "
            f"overlap={self.overlap})"
        )
