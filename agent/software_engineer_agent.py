
import json
import sys
import anyio
import ollama

from mcp import Client, StdioServerParameters

from agent.llm_client import MODEL_NAME
from agent.rag_agent import RAGAgent
from agent.harness import AgentHarness


class SoftwareEngineerAgent:

    def __init__(self):
        self.rag_agent = RAGAgent()
        self.harness = AgentHarness(
            max_iterations=5,
            max_tool_calls=10,
        )

    async def ask(self, question: str):

        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
        )

        async with Client(server_params) as client:

            # Discover MCP tools
            mcp_response = await client.list_tools()
            mcp_tools = mcp_response.tools

            # Add RAG as an agent tool
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "search_project_knowledge",
                        "description": (
                            "Search indexed project documentation and "
                            "code knowledge using semantic retrieval."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": (
                                        "Question to search in indexed project knowledge"
                                    )
                                }
                            },
                            "required": ["question"]
                        }
                    }
                }
            ]

            # Convert discovered MCP tools to Ollama format
            for tool in mcp_tools:
                schema = getattr(tool, "inputSchema", None)

                if schema is None:
                    schema = getattr(tool, "input_schema", None)

                tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": schema or {
                            "type": "object",
                            "properties": {}
                        }
                    }
                })

            allowed_mcp_tools = {
                tool.name for tool in mcp_tools
            }

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an AI Software Engineer Agent with access to tools.\n"
                        "IMPORTANT: You MUST call a tool before answering "
                        "project-specific questions.\n"
                        "Tool selection rules:\n"
                        "1. For questions about project architecture, components, "
                        "RAG pipeline, implementation, or documentation, call "
                        "search_project_knowledge first.\n"
                        "2. For questions asking to read, inspect, or list current "
                        "files, use the appropriate MCP tool.\n"
                        "3. For questions requiring both indexed knowledge and current "
                        "file contents, use both tools.\n"
                        "4. Never ask the user for project details when the tools can "
                        "retrieve them.\n"
                        "5. Base your answer on tool results. Do not invent information."
                    )
                },
                {
                    "role": "user",
                    "content": question
                }
            ]

            for _ in range(self.harness.max_iterations):

                response = ollama.chat(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=tools,
                )

                assistant_message = response["message"]
                messages.append(assistant_message)

                tool_calls = assistant_message.get("tool_calls", [])

                if not tool_calls:
                    return assistant_message.get("content", "")

                for tool_call in tool_calls:

                    function = tool_call["function"]
                    tool_name = function["name"]
                    arguments = function.get("arguments", {})

                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)

                    # RAG tool
                    if tool_name == "search_project_knowledge":

                        result = await self.harness.execute_async_tool(
                            tool_name,
                            self._run_rag,
                            **arguments,
                        )

                        tool_output = (
                            result.output if result.success
                            else f"Tool error: {result.error}"
                        )

                    # MCP tools
                    elif tool_name in allowed_mcp_tools:

                        result = await self.harness.execute_async_tool(
                            tool_name,
                            client.call_tool,
                            name=tool_name,
                            arguments=arguments,
                        )

                        if result.success:
                            tool_output = self._extract_mcp_result(
                                result.output
                            )
                        else:
                            tool_output = f"Tool error: {result.error}"

                    else:
                        tool_output = f"Unknown tool: {tool_name}"

                    messages.append({
                        "role": "tool",
                        "content": str(tool_output)
                    })

            return "Agent stopped: maximum iterations reached."

    async def _run_rag(self, question: str):
        return self.rag_agent.ask(question)

    @staticmethod
    def _extract_mcp_result(result):
        output = []

        for item in result.content:
            if hasattr(item, "text"):
                output.append(item.text)
            else:
                output.append(str(item))

        return "\n".join(output)


if __name__ == "__main__":

    question = input(
        "Ask your AI Software Engineer Agent: "
    ).strip()

    if question:
        agent = SoftwareEngineerAgent()

        answer = anyio.run(agent.ask, question)

        print("\nAgent Answer:")
        print(answer)

        print("\nHarness Logs:")
        for log in agent.harness.logs:
            print(log)

        print(
            "\nTotal Tool Calls:",
            agent.harness.tool_calls
        )
