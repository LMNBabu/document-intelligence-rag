"""
Embeddings: turn text into vectors with a Sentence Transformers model.

The SAME model must embed both documents and questions, otherwise distances between
them are meaningless. Both functions below use the same cached model.

Public API:
    embed_texts(texts) -> list[list[float]]
    embed_query(text)  -> list[float]
    EmbeddingError
"""

import os
from functools import lru_cache

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingError(Exception):
    """Raised when the embedding model cannot be loaded or fails to encode."""


def _model_name() -> str:
    return os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    """Load the model once per process. The first run downloads it (~90 MB)."""
    try:
        # Imported here so that importing this module is fast and does not need PyTorch.
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception as exc:
        raise EmbeddingError(
            f"Could not load embedding model '{model_name}'. The first run needs an "
            f"internet connection to download it. ({exc})"
        ) from exc


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed many texts. Vectors are normalised, which suits cosine similarity."""
    if not texts:
        return []
    model = _load_model(_model_name())
    try:
        vectors = model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
    except Exception as exc:
        raise EmbeddingError(f"Embedding generation failed: {exc}") from exc
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single question using the same model as the documents."""
    if not text or not text.strip():
        raise EmbeddingError("Cannot embed an empty question.")
    return embed_texts([text.strip()])[0]
