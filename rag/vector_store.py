from pathlib import Path
from typing import Any

import chromadb

from rag.document_loader import Document


DB_PATH = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "software_project_code"


class ChromaVectorStore:
    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_directory: str | Path = DB_PATH,
        client: Any | None = None,
    ) -> None:
        self.client = client or chromadb.PersistentClient(
            path=str(persist_directory)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("Each document must have exactly one embedding.")
        if not documents:
            return

        self.collection.upsert(
            ids=[document.id for document in documents],
            documents=[document.text for document in documents],
            metadatas=[document.metadata for document in documents],
            embeddings=embeddings,
        )

    def search(self, embedding: list[float], n_results: int = 5) -> list[dict]:
        if n_results <= 0:
            raise ValueError("n_results must be greater than zero.")
        count = self.collection.count()
        if count == 0:
            return []

        response = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )
        results: list[dict] = []
        ids = response["ids"][0]
        documents = response["documents"][0]
        metadatas = response["metadatas"][0]
        distances = response["distances"][0]
        for item_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            if text is not None:
                results.append(
                    {
                        "id": item_id,
                        "text": text,
                        "metadata": metadata or {},
                        "distance": distance,
                    }
                )
        return results

    def count(self) -> int:
        return self.collection.count()


_default_store = ChromaVectorStore()
client = _default_store.client
collection = _default_store.collection


def store_chunks(chunks: list[dict], embeddings: list[list[float]]) -> int:
    """Store legacy chunk dictionaries in the default collection."""
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings count must match.")
    if not chunks:
        return 0

    collection.upsert(
        ids=[f"{chunk['source']}:{chunk['start']}:{chunk['end']}" for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "source": chunk["source"],
                "start": chunk["start"],
                "end": chunk["end"],
            }
            for chunk in chunks
        ],
    )
    return len(chunks)


def search_chunks(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Search the default collection using the legacy result shape."""
    return [
        {
            "content": result["text"],
            "source": result["metadata"].get("source", ""),
            "distance": result["distance"],
        }
        for result in _default_store.search(query_embedding, n_results=top_k)
    ]


def get_collection_count() -> int:
    return _default_store.count()
