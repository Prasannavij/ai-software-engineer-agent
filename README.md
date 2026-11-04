# AI Software Engineer Agent

An Ollama-powered software engineering assistant with a retrieval-augmented
generation (RAG) pipeline backed by ChromaDB.

## Project structure

```text
agent/
  __init__.py
  llm_client.py
  tools.py
  harness.py
mcp_server/
  __init__.py
  server.py
rag/
  __init__.py
  document_loader.py
  chunker.py
  embeddings.py
  vector_store.py
  retriever.py
chroma_db/             Persistent ChromaDB data
main.py                MCP server entry point
test_rag.py            RAG indexing and search example
tests/                 Automated tests
requirements.txt       Python dependencies
```

## MCP server

Start the server over stdio with:

```powershell
python main.py
```

The server provides `ask_project_question`, `list_project_files`, and
`read_project_file` tools. The question tool uses the indexed ChromaDB documents
and Ollama models; file tools are available independently.

## RAG pipeline

`load_documents` reads UTF-8 Markdown, Python, and text files. `chunk_documents`
splits them into overlapping chunks, `embed_texts` creates embeddings with the
locally served Ollama `nomic-embed-text` model, and `Retriever` indexes and
searches those chunks in the persistent `chroma_db/` collection.

Start Ollama and ensure the embedding model is available before calling the
default embedding function. RAG components can also be given a custom embedding
function and vector store for testing or alternate backends.

Run the RAG tests with:

```powershell
python -m unittest test_rag
```
