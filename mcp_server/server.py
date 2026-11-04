
from pathlib import Path
from mcp.server.mcpserver import MCPServer

# Initialize MCP server
mcp = MCPServer("AI Software Engineer Tools")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "chroma_db",
    "node_modules",
}


@mcp.tool()
def list_project_files() -> list[str]:
    """List files available in the project workspace."""

    files = []

    for path in PROJECT_ROOT.rglob("*"):
        if any(part in IGNORED_DIRS for part in path.parts):
            continue

        if path.is_file():
            files.append(str(path.relative_to(PROJECT_ROOT)))

    return files


@mcp.tool()
def read_project_file(filename: str) -> str:
    """Read a text file from the project workspace."""

    root = PROJECT_ROOT.resolve()
    file_path = (root / filename).resolve()

    if not file_path.is_relative_to(root):
        return "Error: Access outside project directory is not allowed."

    if not file_path.is_file():
        return f"Error: File not found: {filename}"

    allowed_extensions = {
        ".py", ".txt", ".md", ".json",
        ".yaml", ".yml", ".toml", ".sql",
    }

    if file_path.suffix.lower() not in allowed_extensions:
        return "Error: Unsupported file type."

    if file_path.stat().st_size > 200_000:
        return "Error: File exceeds 200 KB limit."

    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return "Error: File is not valid UTF-8 text."


if __name__ == "__main__":
    mcp.run(transport="stdio")
