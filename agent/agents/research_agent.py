
from agent.rag_agent import RAGAgent
from agent.llm_client import MODEL_NAME
import ollama


class ResearchAgent:

    def __init__(self):
        self.rag_agent = RAGAgent()
        self.model = MODEL_NAME

    def research(self, task: str) -> str:

        if not task.strip():
            raise ValueError("Research task cannot be empty.")

        # Retrieve project knowledge through RAG
        retrieved_context = self.rag_agent.ask(
            question=task,
            top_k=5
        )

        prompt = f"""
You are a Senior Software Engineering Research Agent.

Your responsibility is to investigate the given task
using the retrieved project information.

Rules:
- Use only the retrieved information.
- Identify relevant project components and files.
- Explain the existing implementation.
- Identify possible gaps or risks.
- Do not invent implementation details.
- Clearly mention when information is insufficient.

RESEARCH TASK:
{task}

RETRIEVED PROJECT INFORMATION:
{retrieved_context}

Provide your research findings with these sections:
1. Relevant Components
2. Existing Implementation
3. Potential Issues
4. Recommendations
5. Missing Information
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

    researcher = ResearchAgent()

    task = input(
        "Enter a software engineering research task: "
    ).strip()

    try:
        findings = researcher.research(task)

        print("\nResearch Findings:\n")
        print(findings)

    except Exception as exc:
        print(f"Research error: {exc}")
