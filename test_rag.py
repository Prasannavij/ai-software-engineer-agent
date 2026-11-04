from pathlib import Path

from rag.document_loader import load_documents
from rag.retriever import Retriever


def main() -> None:
    project_root = Path(__file__).resolve().parent
    print("Loading project documents...")
    documents = load_documents([project_root])
    print("Documents loaded:", len(documents))
    if not documents:
        print("No supported documents found.")
        return

    retriever = Retriever()
    chunk_count = retriever.index_documents(documents)
    print("Chunks indexed:", chunk_count)

    query = "How does the AI agent execute Python tools?"
    print("\nTesting semantic search...")
    results = retriever.retrieve(query, top_k=5)

    for result in results:
        print("\nSource:", result.metadata.get("source", ""))
        print("Distance:", result.distance)
        print("Content:", result.text[:300])


if __name__ == "__main__":
    main()
