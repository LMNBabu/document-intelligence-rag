"""
Retriever: question -> embedding -> most similar chunks (with source metadata).

Public API:
    retrieve(question, top_k=None) -> list[RetrievedChunk]
    RetrievedChunk
"""

import os
from dataclasses import dataclass

from src import vector_store
from src.embeddings import embed_query

DEFAULT_TOP_K = 4


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by similarity search. `text` is the ORIGINAL stored text."""

    text: str
    file_name: str
    page_label: str
    chunk_index: int
    distance: float  # cosine distance: 0 = identical, larger = less similar

    @property
    def similarity(self) -> float:
        """Cosine similarity, roughly 0..1 for text (higher = more relevant)."""
        return 1.0 - self.distance


def default_top_k() -> int:
    try:
        return max(1, int(os.getenv("TOP_K", str(DEFAULT_TOP_K))))
    except ValueError:
        return DEFAULT_TOP_K


def retrieve(question: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """Embed the question with the same model as the documents, then search Chroma."""
    if not question or not question.strip():
        raise ValueError("Please enter a question.")
    k = top_k if top_k is not None else default_top_k()

    query_vector = embed_query(question)
    hits = vector_store.query(query_vector, k)
    return [RetrievedChunk(**hit) for hit in hits]
