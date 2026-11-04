from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SUPPORTED_EXTENSIONS = {".md", ".py", ".txt"}
IGNORED_DIRECTORIES = {".git", "__pycache__", "chroma_db", "venv"}


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: dict[str, str | int]


def load_documents(
    sources: str | Path | Iterable[str | Path],
) -> list[Document]:
    """Load UTF-8 text files from paths or directories."""
    if isinstance(sources, (str, Path)):
        paths = [Path(sources)]
    else:
        paths = [Path(source) for source in sources]

    documents: list[Document] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Document path does not exist: {path}")

        candidates = (
            sorted(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and not IGNORED_DIRECTORIES.intersection(
                    candidate.relative_to(path).parts
                )
            )
            if path.is_dir()
            else [path]
        )
        for candidate in candidates:
            if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
                if path.is_file():
                    raise ValueError(
                        f"Unsupported document type: {candidate.suffix or '(none)'}"
                    )
                continue

            resolved_path = candidate.resolve()
            content = resolved_path.read_bytes()
            encoding = (
                "utf-16"
                if content.startswith((b"\xff\xfe", b"\xfe\xff"))
                else "utf-8-sig"
            )
            with resolved_path.open("r", encoding=encoding) as source_file:
                text = source_file.read()
            documents.append(
                Document(
                    id=str(resolved_path),
                    text=text,
                    metadata={
                        "source": str(resolved_path),
                        "filename": resolved_path.name,
                    },
                )
            )
    return documents
