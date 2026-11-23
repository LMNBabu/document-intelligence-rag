"""
Project self-check. Run from the project root (virtual environment active):

    python check_project.py          # offline checks (no model download, no API call)
    python check_project.py --live   # also tests the real embedding model and Groq API

Offline checks use a throwaway database in a temp folder, never your real chroma_db.
"""

import io
import os
import sys
import tempfile

from dotenv import load_dotenv

load_dotenv()

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))


def make_pdf(page_texts: list[str]) -> bytes:
    import pymupdf

    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    return doc.tobytes()


def make_docx(paragraphs: list[str]) -> bytes:
    from docx import Document

    doc = Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def fake_embedding(text: str) -> list[float]:
    """Deterministic toy vector (word-hash buckets). Only for OFFLINE plumbing tests."""
    vec = [0.0] * 64
    for word in text.lower().split():
        vec[hash(word.strip(".,?")) % 64] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def check_loader() -> None:
    from src.document_loader import DocumentLoadError, load_document

    print("\n[1] Document loader")
    pages = load_document("a.pdf", make_pdf(["Alpha page one", "", "Gamma page three"]))
    record("PDF page labels (blank page skipped)", [p.page_label for p in pages] == ["Page 1", "Page 3"])
    pages = load_document("a.docx", make_docx(["x" * 1000 for _ in range(7)]))
    record("DOCX grouped into sections", [p.page_label for p in pages] == ["Section 1", "Section 2", "Section 3"])
    bad_inputs = {
        "unsupported type": ("n.txt", b"hello"),
        "empty file": ("e.pdf", b""),
        "corrupt PDF": ("b.pdf", b"not a pdf"),
        "corrupt DOCX": ("b.docx", b"not a docx"),
        "PDF without text": ("blank.pdf", make_pdf(["", ""])),
        "DOCX without text": ("blank.docx", make_docx(["", " "])),
    }
    for label, (name, data) in bad_inputs.items():
        try:
            load_document(name, data)
            record(f"error for {label}", False, "no error raised")
        except DocumentLoadError:
            record(f"error for {label}", True)


def check_splitter() -> None:
    from src.document_loader import PageContent
    from src.text_splitter import split_pages

    print("\n[2] Text splitter")
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    pages = [PageContent("doc.pdf", "Page 1", text), PageContent("doc.pdf", "Page 2", "Short page.")]
    chunks = split_pages(pages, chunk_size=300, overlap=60)
    record("makes several chunks", len(chunks) > 5, f"{len(chunks)} chunks")
    record("chunks within size limit", all(len(c.text) <= 300 for c in chunks))
    record("metadata preserved", chunks[0].file_name == "doc.pdf" and chunks[-1].page_label == "Page 2")
    record("chunk_index runs 0..n-1", [c.chunk_index for c in chunks] == list(range(len(chunks))))
    first_page = [c for c in chunks if c.page_label == "Page 1"]
    overlap_ok = all(
        first_page[i].text[-20:].split()[-1] in first_page[i + 1].text for i in range(len(first_page) - 1)
    )
    record("neighbouring chunks overlap", overlap_ok)
    try:
        split_pages(pages, chunk_size=100, overlap=100)
        record("rejects overlap >= size", False)
    except ValueError:
        record("rejects overlap >= size", True)


def check_vector_store_and_llm_helpers() -> None:
    from src import vector_store
    from src.document_loader import PageContent
    from src.llm import INSUFFICIENT_MESSAGE, LLMError, find_citations, generate_answer, is_insufficient
    from src.retriever import RetrievedChunk
    from src.text_splitter import split_pages

    print("\n[3] Vector store (temporary database, fake vectors)")
    # Windows keeps ChromaDB files open until the process exits, so deleting the temp
    # folder can fail. That is harmless, hence ignore_cleanup_errors=True below.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        os.environ["CHROMA_DIR"] = os.path.join(tmp, "db")
        os.environ["COLLECTION_NAME"] = "check_collection"

        def index(name: str, text: str, data: bytes) -> str:
            chunks = split_pages([PageContent(name, "Page 1", text)], 200, 40)
            doc_hash = vector_store.file_hash(data)
            vector_store.add_chunks(chunks, [fake_embedding(c.text) for c in chunks], doc_hash)
            return doc_hash

        h1 = index("cats.pdf", "Cats are small carnivorous mammals. " * 10, b"cats-v1")
        index("space.pdf", "The planet Jupiter has a giant red storm. " * 10, b"space-v1")
        record("document_exists after indexing", vector_store.document_exists(h1))
        record("document_exists for unknown hash", not vector_store.document_exists("nope"))
        record("list_documents shows 2 files", len(vector_store.list_documents()) == 2)

        hits = vector_store.query(fake_embedding("giant red storm on Jupiter"), 3)
        record("similarity search finds the right file", hits and hits[0]["file_name"] == "space.pdf")
        record("hits carry metadata", hits and {"file_name", "page_label", "chunk_index", "distance"} <= set(hits[0]))

        index("cats.pdf", "Cats sleep a lot. " * 10, b"cats-v2")
        docs = {d["file_name"]: d for d in vector_store.list_documents()}
        record("updated file replaces old version", len(docs) == 2 and docs["cats.pdf"]["doc_hash"] == vector_store.file_hash(b"cats-v2"))

        vector_store.clear_all()
        record("clear_all empties the store", vector_store.count() == 0 and vector_store.list_documents() == [])
        record("query on empty store returns []", vector_store.query(fake_embedding("x"), 3) == [])

    print("\n[4] LLM helpers (no API call)")
    chunk = RetrievedChunk("text", "a.pdf", "Page 1", 0, 0.2)
    record("no chunks -> insufficient message, no API call", generate_answer("q?", []) == INSUFFICIENT_MESSAGE)
    saved = {k: os.environ.pop(k, None) for k in ("GROQ_API_KEY", "GROQ_MODEL")}
    try:
        generate_answer("q?", [chunk])
        record("missing API key gives clear error", False, "no error raised")
    except LLMError as exc:
        record("missing API key gives clear error", "GROQ_API_KEY" in str(exc))
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    record("find_citations separates valid/invalid", find_citations("A [1] and B [2][9].", 2) == ([1, 2], [9]))
    record("is_insufficient detects refusal", is_insufficient(INSUFFICIENT_MESSAGE) and not is_insufficient("Yes [1]"))


def check_live() -> None:
    from src import vector_store
    from src.document_loader import PageContent
    from src.embeddings import embed_texts
    from src.llm import LLMError, generate_answer, missing_config
    from src.retriever import retrieve
    from src.text_splitter import split_pages

    print("\n[5] LIVE: real embedding model + real Groq call")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        os.environ["CHROMA_DIR"] = os.path.join(tmp, "db")
        os.environ["COLLECTION_NAME"] = "live_check"
        text = (
            "The Falcon X drone has a maximum flight time of 42 minutes. "
            "Its battery is a 5000 mAh lithium polymer pack. "
            "The warranty period for all Falcon products is two years."
        )
        chunks = split_pages([PageContent("manual.pdf", "Page 4", text)], 1000, 200)
        try:
            vectors = embed_texts([c.text for c in chunks])
        except Exception as exc:
            record("embedding model loads and encodes", False, str(exc))
            return
        record("embedding model loads and encodes", len(vectors[0]) > 0, f"{len(vectors[0])} dimensions")
        vector_store.add_chunks(chunks, vectors, vector_store.file_hash(b"live"))

        found = retrieve("How long can the drone fly?", 3)
        record("retrieval returns the chunk with metadata", bool(found) and found[0].file_name == "manual.pdf")

        if missing_config():
            record("Groq call", False, f"skipped: set {', '.join(missing_config())} in .env")
            return
        try:
            answer = generate_answer("How long can the drone fly?", found)
            print(f"        Groq answered: {answer}")
            record("Groq call returns an answer mentioning 42", "42" in answer)
        except LLMError as exc:
            record("Groq call", False, str(exc))


def main() -> int:
    print("Document Intelligence RAG Assistant: self-check")
    try:
        check_loader()
        check_splitter()
        check_vector_store_and_llm_helpers()
        if "--live" in sys.argv:
            check_live()
    except ImportError as exc:
        print(f"\nImport problem: {exc}\nDid you run: pip install -r requirements.txt ?")
        return 2

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("Failed: " + ", ".join(name for name, _, _ in failed))
        return 1
    if "--live" not in sys.argv:
        print("Offline checks only. Run with --live to test the real model and Groq.")
    return 0


if __name__ == "__main__":
    sys.exit(main())