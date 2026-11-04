from rag.document_loader import Document


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[Document]:
    chunks: list[Document] = []
    for document in documents:
        for index, text in enumerate(
            chunk_text(document.text, chunk_size=chunk_size, overlap=overlap)
        ):
            chunks.append(
                Document(
                    id=f"{document.id}:{index}",
                    text=text,
                    metadata={**document.metadata, "chunk_index": index},
                )
            )
    return chunks
