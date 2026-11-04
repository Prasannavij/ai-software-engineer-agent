import json

import ollama

from agent.llm_client import MODEL_NAME


class PlannerAgent:
    def __init__(self) -> None:
        self.model = MODEL_NAME

    def create_plan(self, task: str) -> list[str]:
        if not task.strip():
            raise ValueError("Task cannot be empty.")

        system_prompt = """You are a senior software engineering planning agent.
Your responsibility is to break a software engineering task into clear, logical and actionable steps.
Rules:
- Understand the task before creating the plan.
- Arrange steps in the correct execution order.
- Keep each step specific and actionable.
- Do not execute code or modify files.
- Do not invent project details.
- Return only valid JSON.
Required JSON structure:
{
    "plan": [
        "Step 1",
        "Step 2",
        "Step 3"
    ]
}"""
        response = ollama.chat(
            model=self.model,
            format="json",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Create an implementation plan for: {task}",
                },
            ],
        )

        try:
            result = json.loads(response.message.content)
        except json.JSONDecodeError as error:
            raise ValueError("Planner returned invalid JSON.") from error

        if not isinstance(result, dict):
            raise ValueError("Planner response must be a JSON object.")

        plan = result.get("plan")
        if (
            not isinstance(plan, list)
            or not plan
            or any(not isinstance(step, str) or not step.strip() for step in plan)
        ):
            raise ValueError("Planner response must contain a non-empty list of steps.")

        return [step.strip() for step in plan]


if __name__ == "__main__":
    planner = PlannerAgent()
    task = input("Enter a software engineering task: ").strip()
    try:
        plan = planner.create_plan(task)
        print("\nImplementation Plan:\n")
        for index, step in enumerate(plan, start=1):
            print(f"{index}. {step}")
    except Exception as exc:
        print(f"Planner error: {exc}")
