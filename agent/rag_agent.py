
from rag.retriever import Retriever
from agent.llm_client import MODEL_NAME
import ollama


class RAGAgent:
    def __init__(self, relevance_threshold: float = 0.45):
        self.retriever = Retriever()
        self.relevance_threshold = relevance_threshold

    def ask(self, question: str, top_k: int = 5) -> str:

        # 1. Retrieve relevant project information
        results = self.retriever.retrieve(
            query=question,
            top_k=top_k
        )

        # 2. Filter irrelevant results
        relevant_results = [
            result for result in results
            if result.distance <= self.relevance_threshold
        ]

        if not relevant_results:
            return (
                "I couldn't find sufficiently relevant information "
                "in the indexed project files to answer this question."
            )

        # 3. Build context
        context_parts = []
        sources = []

        for result in relevant_results:
            source = result.metadata.get("source", "Unknown")

            context_parts.append(
                f"Source: {source}\n"
                f"Content:\n{result.text}"
            )

            if source not in sources:
                sources.append(source)

        context = "\n\n".join(context_parts)

        # 4. Build prompt
        prompt = f"""
You are an AI Software Engineer Assistant.

Answer the user's question using ONLY the project context below.

Rules:
- Do not invent functions, classes, filenames, or features.
- Use only information supported by the retrieved context.
- If information is missing, clearly mention that.
- Explain technical details accurately.
- Distinguish confirmed implementation from assumptions.

PROJECT CONTEXT:
{context}

USER QUESTION:
{question}
"""

        # 5. Generate answer
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response["message"]["content"]

        # 6. Attach source filenames
        source_text = "\n".join(
            f"- {source}" for source in sources
        )

        return (
            f"{answer}\n\n"
            f"--- Retrieved Sources ---\n"
            f"{source_text}"
        )
