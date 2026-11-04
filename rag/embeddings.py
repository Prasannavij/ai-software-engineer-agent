
from collections.abc import Callable

import ollama

EMBEDDING_MODEL = "nomic-embed-text"
EmbeddingFunction = Callable[[list[str]], list[list[float]]]


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Convert text chunks into numerical embeddings."""

    if not texts:
        return []

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    return response.embeddings


def generate_query_embedding(text: str) -> list[float]:
    """Generate an embedding for a user search query."""

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return response.embeddings[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Provide the batch embedding interface used by the retriever."""
    return generate_embeddings(texts)
