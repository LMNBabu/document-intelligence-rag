"""
Document loading: turn uploaded PDF / DOCX bytes into clean, labelled text.

Public API:
    load_document(file_name, file_bytes) -> list[PageContent]
    DocumentLoadError  (raised for any problem the user should be told about)
    PageContent        (one page of a PDF, or one "section" of a DOCX)
"""

import io
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf  # PyMuPDF; "pymupdf" is the current import name ("fitz" is the legacy one)
from docx import Document  # python-docx

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

# DOCX files have no stored page numbers (pages only exist once Word renders them),
# so we group consecutive paragraphs into "sections" of about one page of text.
DOCX_SECTION_CHARS = 3000

DEFAULT_MAX_FILE_MB = 20


class DocumentLoadError(Exception):
    """Raised when a document cannot be read. The message is safe to show to users."""


@dataclass(frozen=True)
class PageContent:
    """Text from one location in a document, plus where it came from."""

    file_name: str
    page_label: str  # "Page 3" for PDFs, "Section 2" for DOCX
    text: str


def _max_file_bytes() -> int:
    """Read the size limit from the environment at call time (so .env changes apply)."""
    try:
        megabytes = float(os.getenv("MAX_FILE_MB", str(DEFAULT_MAX_FILE_MB)))
    except ValueError:
        megabytes = DEFAULT_MAX_FILE_MB
    return int(megabytes * 1024 * 1024)


def _clean_text(text: str) -> str:
    """Light cleanup: remove null characters, trim lines, collapse long runs of blank lines."""
    text = text.replace("\x00", "")
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_pdf(file_name: str, file_bytes: bytes) -> list[PageContent]:
    """Extract text page by page, keeping real 1-based page numbers."""
    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several different types for bad files
        raise DocumentLoadError(
            f"'{file_name}' could not be opened as a PDF. The file may be corrupt. ({exc})"
        ) from exc

    pages: list[PageContent] = []
    with doc:
        if doc.needs_pass:
            raise DocumentLoadError(
                f"'{file_name}' is password-protected. Remove the password and upload it again."
            )
        try:
            for page in doc:
                text = _clean_text(page.get_text("text"))
                if text:  # skip blank pages (covers, separators, image-only pages)
                    pages.append(PageContent(file_name, f"Page {page.number + 1}", text))
        except Exception as exc:
            raise DocumentLoadError(f"Error while reading pages of '{file_name}': {exc}") from exc

    if not pages:
        raise DocumentLoadError(
            f"'{file_name}' has no extractable text. It may be a scanned document "
            "(images of pages). OCR is not supported in this app."
        )
    return pages


def _load_docx(file_name: str, file_bytes: bytes) -> list[PageContent]:
    """Extract paragraph text and group it into page-sized 'sections'."""
    try:
        doc = Document(io.BytesIO(file_bytes))
    except Exception as exc:  # e.g. not a zip, or an old .doc renamed to .docx
        raise DocumentLoadError(
            f"'{file_name}' could not be opened as a DOCX. The file may be corrupt "
            f"or in an older .doc format. ({exc})"
        ) from exc

    pages: list[PageContent] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        """Close the current section and start a new one."""
        nonlocal current, current_len
        if current:
            text = _clean_text("\n".join(current))
            if text:
                pages.append(PageContent(file_name, f"Section {len(pages) + 1}", text))
        current, current_len = [], 0

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        current.append(text)
        current_len += len(text)
        if current_len >= DOCX_SECTION_CHARS:
            flush()
    flush()  # whatever is left over

    if not pages:
        raise DocumentLoadError(f"'{file_name}' contains no text.")
    return pages


def load_document(file_name: str, file_bytes: bytes) -> list[PageContent]:
    """
    Validate an uploaded file and extract its text.

    Raises DocumentLoadError for unsupported types, empty files, oversized files,
    corrupt files, password-protected PDFs, and documents with no extractable text.
    """
    extension = Path(file_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentLoadError(
            f"'{file_name}' is not supported. Please upload a PDF or DOCX file."
        )
    if not file_bytes:
        raise DocumentLoadError(f"'{file_name}' is empty (0 bytes).")
    if len(file_bytes) > _max_file_bytes():
        raise DocumentLoadError(
            f"'{file_name}' is larger than the {_max_file_bytes() // (1024 * 1024)} MB limit."
        )

    if extension == ".pdf":
        return _load_pdf(file_name, file_bytes)
    return _load_docx(file_name, file_bytes)