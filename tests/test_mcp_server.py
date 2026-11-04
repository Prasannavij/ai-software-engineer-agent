import asyncio
import unittest

from mcp_server.server import (
    ask_project_question,
    list_project_files,
    read_project_file,
    server,
)


class MCPServerTests(unittest.TestCase):
    def test_server_registers_project_tools(self) -> None:
        tools = asyncio.run(server.list_tools())
        tool_names = {tool.name for tool in tools}

        self.assertEqual(
            tool_names,
            {
                "ask_project_question",
                "list_project_files",
                "read_project_file",
            },
        )

    def test_list_project_files_includes_entry_point(self) -> None:
        self.assertIn("main.py", list_project_files())

    def test_read_project_file_uses_workspace_guard(self) -> None:
        self.assertEqual(
            read_project_file("..\\outside.txt"),
            "Error: Access outside the project workspace is forbidden.",
        )

    def test_question_rejects_empty_input_before_agent_creation(self) -> None:
        with self.assertRaisesRegex(ValueError, "question must not be empty"):
            ask_project_question("  ")


if __name__ == "__main__":
    unittest.main()
