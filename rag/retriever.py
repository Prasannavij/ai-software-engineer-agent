from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.chunker import chunk_documents
from rag.document_loader import Document
from rag.embeddings import EmbeddingFunction, embed_texts
from rag.vector_store import ChromaVectorStore


@dataclass(frozen=True)
class SearchResult:
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float


class Retriever:
    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        embedding_function: EmbeddingFunction = embed_texts,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        persist_directory: str | Path = "chroma_db",
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        self.vector_store = vector_store or ChromaVectorStore(
            persist_directory=persist_directory
        )
        self.embedding_function = embedding_function
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size

    def index_documents(self, documents: list[Document]) -> int:
        chunks = chunk_documents(
            documents,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            embeddings = self.embedding_function([chunk.text for chunk in batch])
            self.vector_store.add_documents(batch, embeddings)
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_embeddings = self.embedding_function([query])
        if len(query_embeddings) != 1:
            raise ValueError("Embedding function must return one vector for one query.")
        matches = self.vector_store.search(query_embeddings[0], n_results=top_k)
        return [
            SearchResult(
                id=match["id"],
                text=match["text"],
                metadata=match["metadata"],
                distance=match["distance"],
            )
            for match in matches
        ]
