
import ollama

from agent.llm_client import MODEL_NAME


class CodingAgent:

    def __init__(self):
        self.model = MODEL_NAME

    def generate_solution(
        self,
        task: str,
        research_findings: str
    ) -> str:

        if not task.strip():
            raise ValueError("Coding task cannot be empty.")

        if not research_findings.strip():
            raise ValueError("Research findings cannot be empty.")

        prompt = f"""
You are a Senior AI Software Engineer and Coding Agent.

Your responsibility is to propose implementation changes
based on the task and research findings.

RULES:
- Analyse the existing architecture first.
- Use only confirmed information from the research findings.
- Do not invent existing filenames, functions or classes.
- Provide complete code examples where appropriate.
- Explain which files need modification.
- Preserve existing interfaces unless a change is necessary.
- Do not claim that code has been executed or tested.
- Identify assumptions and potential risks.

TASK:
{task}

RESEARCH FINDINGS:
{research_findings}

Provide your response in these sections:

1. Implementation Approach
2. Files to Modify
3. Proposed Code Changes
4. Explanation
5. Potential Risks
6. Testing Recommendations
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]


if __name__ == "__main__":

    coding_agent = CodingAgent()

    task = input(
        "Enter a coding task: "
    ).strip()

    research = input(
        "\nEnter research findings: "
    ).strip()

    try:
        solution = coding_agent.generate_solution(
            task,
            research
        )

        print("\nCoding Agent Solution:\n")
        print(solution)

    except Exception as exc:
        print(f"Coding Agent error: {exc}")
