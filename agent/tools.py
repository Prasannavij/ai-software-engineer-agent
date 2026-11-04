
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent


def list_files() -> list[str]:
    ignored_directories = {".git", "__pycache__", "venv", "chroma_db"}
    return sorted(
        str(path.relative_to(WORKSPACE))
        for path in WORKSPACE.rglob("*")
        if path.is_file()
        and not ignored_directories.intersection(path.relative_to(WORKSPACE).parts)
    )


def read_file(filename: str) -> str:
    """Read a text file located inside the project workspace."""
    allowed_extensions = {
        ".py", ".txt", ".md", ".json", ".yaml", ".yml",
        ".toml", ".ini", ".html", ".css", ".js", ".tsx", ".ts"
    }

    try:
        target = (WORKSPACE / filename).resolve()
        target.relative_to(WORKSPACE)

        if not target.is_file():
            return "Error: File not found."

        if target.suffix.lower() not in allowed_extensions:
            return "Error: File type is not allowed."

        if target.stat().st_size > 200_000:
            return "Error: File is too large to read."

        return target.read_text(encoding="utf-8")

    except ValueError:
        return "Error: Access outside the project workspace is forbidden."
    except UnicodeDecodeError:
        return "Error: File is not valid UTF-8 text."
    except OSError:
        return "Error: Unable to read this file."
