import unittest
import tempfile
from pathlib import Path

from rag.document_loader import Document, load_documents
from rag.retriever import Retriever


class FakeVectorStore:
    def __init__(self) -> None:
        self.documents: list[Document] = []
        self.embeddings: list[list[float]] = []

    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)

    def search(self, embedding: list[float], n_results: int = 5) -> list[dict]:
        return [
            {
                "id": document.id,
                "text": document.text,
                "metadata": document.metadata,
                "distance": 0.1,
            }
            for document in self.documents[:n_results]
        ]


class RetrieverTests(unittest.TestCase):
    def test_loader_reads_utf16_files_when_scanning_a_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "requirements.txt"
            file_path.write_text("chromadb\n", encoding="utf-16")

            documents = load_documents([Path(directory)])

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].text, "chromadb\n")

    def test_indexing_and_retrieval_use_shared_interfaces(self) -> None:
        store = FakeVectorStore()
        retriever = Retriever(
            vector_store=store,
            embedding_function=lambda texts: [[float(len(text))] for text in texts],
            chunk_size=4,
            chunk_overlap=1,
            batch_size=1,
        )
        document = Document("source", "abcdef", {"source": "file.txt"})

        indexed = retriever.index_documents([document])
        results = retriever.retrieve("question", top_k=1)

        self.assertEqual(indexed, 2)
        self.assertEqual(len(store.documents), 2)
        self.assertEqual(len(store.embeddings), 2)
        self.assertEqual(results[0].text, "abcd")
        self.assertEqual(results[0].metadata["source"], "file.txt")


if __name__ == "__main__":
    unittest.main()
