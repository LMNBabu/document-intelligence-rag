"""
Text chunking: cut page text into overlapping, embedding-sized pieces.

Why chunk? One embedding per whole document blurs everything together. Small chunks
give precise matches. Overlap keeps an answer from being cut in half at a boundary.

Public API:
    split_pages(pages, chunk_size, overlap) -> list[Chunk]
    get_chunk_settings() -> (chunk_size, overlap) read from environment
    Chunk  (text + file_name + page_label + chunk_index)
"""

import os
from dataclasses import dataclass

from src.document_loader import PageContent

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Preferred places to cut, best first: paragraph, line, sentence, word.
_SEPARATORS = ("\n\n", "\n", ". ", " ")


@dataclass(frozen=True)
class Chunk:
    """A piece of text that keeps track of exactly where it came from."""

    text: str
    file_name: str
    page_label: str
    chunk_index: int  # 0-based position within the whole file


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def get_chunk_settings() -> tuple[int, int]:
    """Read CHUNK_SIZE and CHUNK_OVERLAP from the environment (with safe defaults)."""
    return (
        _int_env("CHUNK_SIZE", DEFAULT_CHUNK_SIZE),
        _int_env("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP),
    )


def _find_break(text: str, start: int, end: int, chunk_size: int) -> int:
    """Pick a natural cut point near `end`, but never in the first half of the window."""
    window = text[start:end]
    min_pos = chunk_size // 2
    for separator in _SEPARATORS:
        index = window.rfind(separator)
        if index >= min_pos:
            return start + index + len(separator)
    return end  # no good boundary found: hard cut


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split one page of text into overlapping pieces."""
    pieces: list[str] = []
    start, length = 0, len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            end = _find_break(text, start, end, chunk_size)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= length:
            break
        # Step back by `overlap` so neighbouring chunks share context,
        # but always move forward so the loop cannot get stuck.
        previous_start = start
        start = max(end - overlap, previous_start + 1)
        # Avoid starting in the middle of a word.
        while start < end and not text[start - 1].isspace():
            start += 1
    return pieces


def split_pages(
    pages: list[PageContent], chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list[Chunk]:
    """
    Turn pages into chunks. Every chunk carries its file name and page label.
    Chunks never span two pages, so a page citation is always exact.
    """
    if chunk_size < 50:
        raise ValueError("CHUNK_SIZE must be at least 50 characters.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be 0 or more and smaller than CHUNK_SIZE.")

    chunks: list[Chunk] = []
    counters: dict[str, int] = {}  # running chunk_index per file
    for page in pages:
        for piece in _split_text(page.text, chunk_size, overlap):
            index = counters.get(page.file_name, 0)
            counters[page.file_name] = index + 1
            chunks.append(Chunk(piece, page.file_name, page.page_label, index))
    return chunks
