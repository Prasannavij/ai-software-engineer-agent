
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    success: bool
    output: Any
    error: str | None = None


@dataclass
class AgentHarness:
    max_iterations: int = 5
    max_tool_calls: int = 10
    tool_calls: int = 0
    logs: list[str] = field(default_factory=list)

    def execute_tool(self, name: str, function, **kwargs) -> ToolResult:
        """Execute an approved tool with basic limits and logging."""

        if self.tool_calls >= self.max_tool_calls:
            return ToolResult(
                success=False,
                output=None,
                error="Tool-call limit reached."
            )

        self.tool_calls += 1
        self.logs.append(f"Executing tool: {name}")

        try:
            output = function(**kwargs)
            self.logs.append(f"Tool completed: {name}")
            return ToolResult(success=True, output=output)

        except Exception as exc:
            self.logs.append(f"Tool failed: {name}: {exc}")
            return ToolResult(
                success=False,
                output=None,
                error=str(exc)
            )

    async def execute_async_tool(self, tool_name: str, function, **kwargs) -> ToolResult:
        """Execute an asynchronous tool with the same limits and logging."""

        if self.tool_calls >= self.max_tool_calls:
            return ToolResult(
                success=False,
                output=None,
                error="Tool-call limit reached."
            )

        self.tool_calls += 1
        self.logs.append(f"Executing tool: {tool_name}")

        try:
            output = await function(**kwargs)
            self.logs.append(f"Tool completed: {tool_name}")
            return ToolResult(success=True, output=output)

        except Exception as exc:
            self.logs.append(f"Tool failed: {tool_name}: {exc}")
            return ToolResult(
                success=False,
                output=None,
                error=str(exc)
            )
    