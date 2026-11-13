from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def write_file(filename: str, content: str) -> str:
    """Create or update a project file safely."""

    root = PROJECT_ROOT.resolve()
    file_path = (root / filename).resolve()

    if not file_path.is_relative_to(root):
        raise PermissionError("Access outside project directory is not allowed.")

    if file_path.suffix.lower() not in {
        ".py", ".txt", ".md", ".json", ".yaml", ".yml", ".toml", ".sql"
    }:
        raise ValueError("Unsupported file type.")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")

    return f"Successfully saved: {file_path.relative_to(root)}"