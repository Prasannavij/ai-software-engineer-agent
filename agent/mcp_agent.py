import json
import sys
import anyio
import ollama

from mcp import Client, StdioServerParameters
from agent.harness import AgentHarness


MODEL_NAME = "qwen2.5:3b"


def convert_mcp_tools(mcp_tools):
    """Convert MCP tool definitions into Ollama tool definitions."""

    ollama_tools = []

    for tool in mcp_tools:
        schema = getattr(tool, "inputSchema", None)

        if schema is None:
            schema = getattr(tool, "input_schema", None)

        if schema is None:
            schema = {"type": "object", "properties": {}}

        ollama_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": schema,
            },
        })

    return ollama_tools


def extract_tool_result(result):
    """Convert MCP response content into readable text."""

    output = []

    for item in result.content:
        if hasattr(item, "text"):
            output.append(item.text)
        else:
            output.append(str(item))

    return "\n".join(output)


async def run_agent(question: str):
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
    )

    harness = AgentHarness(
        max_iterations=5,
        max_tool_calls=10,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI software engineer assistant. "
                "Use the available MCP tools when you need project files "
                "or project information. Do not invent file contents. "
                "Explain your findings clearly."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    async with Client(server_params) as client:
        # Discover tools dynamically from the MCP server.
        mcp_response = await client.list_tools()
        mcp_tools = mcp_response.tools

        ollama_tools = convert_mcp_tools(mcp_tools)

        print("\nAvailable MCP tools:")
        for tool in mcp_tools:
            print(f"- {tool.name}: {tool.description}")

        for iteration in range(harness.max_iterations):
            response = ollama.chat(
                model=MODEL_NAME,
                messages=messages,
                tools=ollama_tools,
            )

            assistant_message = response["message"]
            messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls", [])

            # No more tool calls: the model has produced its answer.
            if not tool_calls:
                print("\nAgent answer:")
                print(assistant_message.get("content", ""))
                break

            for tool_call in tool_calls:
                function_info = tool_call["function"]
                tool_name = function_info["name"]
                arguments = function_info.get("arguments", {})

                if isinstance(arguments, str):
                    arguments = json.loads(arguments)

                # Only execute tools discovered from the MCP server.
                allowed_names = {tool.name for tool in mcp_tools}

                if tool_name not in allowed_names:
                    tool_output = f"Error: Unknown MCP tool: {tool_name}"
                else:
                    result = await harness.execute_async_tool(
                        tool_name,
                        client.call_tool,
                        name=tool_name,
                        arguments=arguments,
                    )

                    if result.success:
                        tool_output = extract_tool_result(result.output)
                    else:
                        tool_output = f"Tool error: {result.error}"

                messages.append({
                    "role": "tool",
                    "content": tool_output,
                })

        else:
            print("\nAgent stopped: maximum iterations reached.")

    print("\nHarness logs:")
    for log in harness.logs:
        print(log)

    print(f"\nTotal tool calls: {harness.tool_calls}")


if __name__ == "__main__":
    question = input("Ask your AI software engineer agent: ").strip()

    if question:
        anyio.run(run_agent, question)
    else:
        print("Please enter a question.")