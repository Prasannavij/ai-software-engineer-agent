
import json
import ollama

from agent.harness import AgentHarness, ToolResult
from agent.tools import list_files, read_file



MODEL_NAME = "qwen2.5:3b"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the project workspace.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read a source or documentation file in the project. "
            "Use when the user asks to inspect or explain a file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Relative path, such as main.py or agent/llm_client.py"
                }
            },
            "required": ["filename"],
        },
    },
}
]


def ask_llm(prompt: str) -> str:
    harness = AgentHarness(
        max_iterations=5,
        max_tool_calls=10,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI software engineering assistant. "
                "Use list_files to inspect the project directory. "
                "Use read_file to inspect actual source files. "
                "Base explanations on tool results. "
                "Never claim an action succeeded unless it did."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    for iteration in range(harness.max_iterations):
        print(f"\nAgent iteration: {iteration + 1}")
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
        )
        tool_calls = response.message.tool_calls or []

        if not tool_calls:
            answer = response.message.content or ""
            print("\n--- Agent Harness Logs ---")
            for log in harness.logs:
                print(log)
            print(f"Total tool calls: {harness.tool_calls}")
            return answer

        messages.append(response.message)
        for tool_call in tool_calls:
            name = tool_call.function.name
            args = tool_call.function.arguments or {}
            parse_error = None
            try:
                if isinstance(args, str):
                    args = json.loads(args)
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError as error:
                parse_error = str(error)
                args = {}

            if parse_error is not None:
                tool_result = ToolResult(
                    success=False,
                    output=None,
                    error=f"Invalid tool arguments: {parse_error}",
                )
            elif name == "list_files":
                tool_result = harness.execute_tool(
                    "list_files",
                    list_files,
                )
            elif name == "read_file":
                filename = args.get("filename", "")
                tool_result = harness.execute_tool(
                    "read_file",
                    read_file,
                    filename=filename,
                )
            else:
                result = {"error": f"Tool '{name}' is not allowed."}
                messages.append(
                    {
                        "role": "tool",
                        "name": name,
                        "content": json.dumps(result),
                    }
                )
                continue

            if tool_result.success:
                result = {"success": True, "output": tool_result.output}
            else:
                result = {
                    "success": False,
                    "error": tool_result.error,
                }

            messages.append(
                {
                    "role": "tool",
                    "name": name,
                    "content": json.dumps(result),
                }
            )

    print("\n--- Agent Harness Logs ---")
    for log in harness.logs:
        print(log)
    print(f"Total tool calls: {harness.tool_calls}")
    return "I stopped because the agent reached its iteration limit before producing a final answer."
