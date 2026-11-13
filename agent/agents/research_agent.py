
import ollama

from agent.llm_client import MODEL_NAME
from agent.rag_agent import RAGAgent


class ResearchAgent:
    def __init__(self):
        self.rag_agent = RAGAgent()
        self.model = MODEL_NAME

    def research(self, task: str, existing_code: str) -> str:
        if not task.strip():
            raise ValueError("Task cannot be empty.")

        if not existing_code.strip():
            raise ValueError("Existing source code cannot be empty.")

        rag_findings = self.rag_agent.ask(
            question=task,
            top_k=5
        )

        system_prompt = """
You are a senior software engineering research agent.

Your job is to investigate the requested task using the supplied source code
and supplementary RAG findings.

STRICT RULES:
1. The supplied existing source code is the primary source of truth.
2. Never invent classes, functions, imports, variables, or APIs.
3. Do not claim that a feature exists unless it appears in the source code.
4. RAG findings are supplementary and may be outdated or inaccurate.
5. If RAG findings conflict with the source code, trust the source code.
6. Clearly distinguish existing implementation from recommendations.
7. Identify missing information instead of guessing.

Return your findings using these sections:

1. Existing Implementation
2. Current Error Handling
3. Missing Validation
4. Potential Risks
5. Recommended Improvements
6. Information That Cannot Be Verified
"""

        user_prompt = f"""
TASK:
{task}

AUTHORITATIVE EXISTING SOURCE CODE:
{existing_code}

SUPPLEMENTARY RAG FINDINGS (NOT AUTHORITATIVE):
{rag_findings}

Analyze the task using the exact source code provided.
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return response.message.content
