"""
Vector store: persist chunks + embeddings + metadata in ChromaDB (on local disk).

Duplicate prevention: every uploaded file gets a SHA-256 hash of its bytes. Chunk IDs
are "<hash>_<chunk_index>", and we skip indexing a hash that is already stored.
If a file with the same NAME but different content is uploaded (an updated version),
the old chunks are removed first so stale text never appears in answers.

Public API:
    file_hash(file_bytes) -> str
    document_exists(doc_hash) -> bool
    add_chunks(chunks, embeddings, doc_hash) -> None
    list_documents() -> list[dict]
    query(query_embedding, top_k) -> list[dict]
    count() -> int
    clear_all() -> None
    VectorStoreError
"""

import hashlib
import os
import sys
import types
from functools import lru_cache


def _work_around_blocked_grpc() -> None:
    """
    Some Windows PCs (Smart App Control / AppLocker / WDAC) refuse to load the unsigned
    `grpc` DLL, which makes `import chromadb` crash with "DLL load failed while importing
    cygrpc". ChromaDB only needs grpc for OPTIONAL telemetry export, which this app does
    not use (the database runs locally). So if grpc cannot be loaded, we register a tiny
    placeholder for that one exporter module and let ChromaDB start normally.
    On machines where grpc loads fine, this function does nothing.
    """
    try:
        import grpc  # noqa: F401
        return
    except ImportError:
        pass

    module_name = "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
    placeholder = types.ModuleType(module_name)

    class OTLPSpanExporter:  # never used: telemetry export is not configured
        def __init__(self, *args, **kwargs):
            raise RuntimeError("OTLP export is unavailable because grpc is blocked on this PC.")

    placeholder.OTLPSpanExporter = OTLPSpanExporter
    sys.modules[module_name] = placeholder


_work_around_blocked_grpc()

import chromadb  # noqa: E402  (must come after the workaround above)
from chromadb.config import Settings  # noqa: E402

from src.text_splitter import Chunk  # noqa: E402

DEFAULT_CHROMA_DIR = "./chroma_db"
DEFAULT_COLLECTION = "documents"  # Chroma needs 3-512 chars: letters, digits . _ -
_BATCH_SIZE = 500


class VectorStoreError(Exception):
    """Raised when the vector database cannot be read or written."""


def file_hash(file_bytes: bytes) -> str:
    """Fingerprint of the file's content (first 16 hex chars of SHA-256 is plenty)."""
    return hashlib.sha256(file_bytes).hexdigest()[:16]


@lru_cache(maxsize=4)
def _get_client(path: str):
    # Telemetry off: keeps the console clean and sends nothing anywhere.
    return chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))


def _collection():
    """Get (or create) the collection. Cosine distance matches normalised embeddings."""
    path = os.getenv("CHROMA_DIR", DEFAULT_CHROMA_DIR)
    name = os.getenv("COLLECTION_NAME", DEFAULT_COLLECTION)
    try:
        client = _get_client(path)
        return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    except Exception as exc:
        raise VectorStoreError(f"Could not open the vector database at '{path}': {exc}") from exc


def document_exists(doc_hash: str) -> bool:
    try:
        found = _collection().get(where={"doc_hash": doc_hash}, limit=1)
        return len(found["ids"]) > 0
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(f"Duplicate check failed: {exc}") from exc


def add_chunks(chunks: list[Chunk], embeddings: list[list[float]], doc_hash: str) -> None:
    """Store chunks with their vectors. Replaces older versions of the same file name."""
    if len(chunks) != len(embeddings):
        raise VectorStoreError("Number of chunks and embeddings do not match.")
    if not chunks:
        raise VectorStoreError("There are no chunks to store.")

    collection = _collection()
    try:
        # Remove an older upload with the same file name (different content).
        collection.delete(where={"file_name": chunks[0].file_name})

        for i in range(0, len(chunks), _BATCH_SIZE):
            batch = chunks[i : i + _BATCH_SIZE]
            collection.upsert(
                ids=[f"{doc_hash}_{c.chunk_index}" for c in batch],
                documents=[c.text for c in batch],
                embeddings=embeddings[i : i + _BATCH_SIZE],
                metadatas=[
                    {
                        "file_name": c.file_name,
                        "page_label": c.page_label,
                        "chunk_index": c.chunk_index,
                        "doc_hash": doc_hash,
                    }
                    for c in batch
                ],
            )
    except Exception as exc:
        raise VectorStoreError(f"Could not store chunks: {exc}") from exc


def list_documents() -> list[dict]:
    """One entry per indexed file: {'file_name', 'doc_hash', 'chunks'}."""
    try:
        data = _collection().get(include=["metadatas"])
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(f"Could not list documents: {exc}") from exc

    summary: dict[str, dict] = {}
    for meta in data["metadatas"] or []:
        entry = summary.setdefault(
            meta["doc_hash"],
            {"file_name": meta["file_name"], "doc_hash": meta["doc_hash"], "chunks": 0},
        )
        entry["chunks"] += 1
    return sorted(summary.values(), key=lambda d: d["file_name"].lower())


def count() -> int:
    """Total number of stored chunks."""
    try:
        return _collection().count()
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(f"Could not count chunks: {exc}") from exc


def query(query_embedding: list[float], top_k: int) -> list[dict]:
    """
    Similarity search. Returns the closest chunks, best first, as dicts:
    {'text', 'file_name', 'page_label', 'chunk_index', 'distance'}.
    """
    collection = _collection()
    try:
        total = collection.count()
        if total == 0:
            return []
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, min(top_k, total)),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise VectorStoreError(f"Similarity search failed: {exc}") from exc

    hits = []
    for text, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append(
            {
                "text": text,
                "file_name": meta["file_name"],
                "page_label": meta["page_label"],
                "chunk_index": meta["chunk_index"],
                "distance": distance,
            }
        )
    return hits


def clear_all() -> None:
    """Delete every indexed document."""
    path = os.getenv("CHROMA_DIR", DEFAULT_CHROMA_DIR)
    name = os.getenv("COLLECTION_NAME", DEFAULT_COLLECTION)
    try:
        client = _get_client(path)
        try:
            client.delete_collection(name)
        except Exception:
            pass  # collection did not exist yet: nothing to clear
    except Exception as exc:
        raise VectorStoreError(f"Could not clear the vector database: {exc}") from exc