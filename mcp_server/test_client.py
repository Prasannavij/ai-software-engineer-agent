
import anyio
import sys

from mcp import Client, StdioServerParameters


async def main():

    # 1. Configure MCP server subprocess
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
    )

    print("Connecting to MCP server...")

    # 2. Connect to MCP server
    async with Client(server_params) as client:

        print("\nMCP server connected successfully!")

        # 3. Discover available tools
        tools = await client.list_tools()

        print("\n--- Available MCP Tools ---")

        for tool in tools.tools:
            print(f"Tool: {tool.name}")
            print(f"Description: {tool.description}")

        # 4. Execute list_project_files
        print("\n--- Testing list_project_files ---")

        result = await client.call_tool(
            "list_project_files",
            {},
        )

        for item in result.content:
            if hasattr(item, "text"):
                print(item.text)

        # 5. Execute read_project_file
        print("\n--- Testing read_project_file ---")

        result = await client.call_tool(
            "read_project_file",
            {"filename": "agent/harness.py"},
        )

        for item in result.content:
            if hasattr(item, "text"):
                print(item.text)


if __name__ == "__main__":
    anyio.run(main)
