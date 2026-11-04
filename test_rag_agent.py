
from agent.rag_agent import RAGAgent


def main():
    agent = RAGAgent()

    question = input("Ask about your project: ")

    answer = agent.ask(question)

    print("\n--- AI SOFTWARE ENGINEER ANSWER ---")
    print(answer)


if __name__ == "__main__":
    main()
