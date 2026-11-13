import ollama

from agent.llm_client import MODEL_NAME
from agent.file_editor import write_file


class CodingAgent:

    def __init__(self):
        self.model = MODEL_NAME

    def generate_solution(
        self,
        task: str,
        existing_code: str,
        research_findings: str,
    ) -> str:

        system_prompt = """
You are an autonomous senior Python software engineer.

Your task is to modify existing source code according to the user's request.

Rules:
1. Understand the requested change.
2. Preserve existing functionality.
3. Do not invent project APIs or dependencies.
4. Return the COMPLETE updated source file.
5. Do not include Markdown code fences.
6. Return only valid Python source code.
"""

        user_prompt = f"""
USER TASK:
{task}

EXISTING SOURCE CODE:
{existing_code}

RESEARCH FINDINGS:
{research_findings}

Generate the complete updated Python source file.
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return response["message"]["content"]

    def apply_solution(self, filename: str, updated_code: str) -> str:
        return write_file(filename, updated_code)