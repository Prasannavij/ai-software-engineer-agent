
from pathlib import Path

from agent.agents.planner_agent import PlannerAgent
from agent.agents.research_agent import ResearchAgent
from agent.agents.coding_agent import CodingAgent
from agent.agents.reviewer_agent import ReviewerAgent


class SoftwareEngineerOrchestrator:

    def __init__(self):
        self.planner = PlannerAgent()
        self.researcher = ResearchAgent()
        self.coder = CodingAgent()
        self.reviewer = ReviewerAgent()


    def run(self, task: str, source_file: str) -> dict:
        if not task.strip():
            raise ValueError("Task cannot be empty.")

        print("\nReading existing source code...")

        project_root = Path(__file__).resolve().parents[2]
        file_path = (project_root / source_file).resolve()

        if not file_path.is_relative_to(project_root):
            raise ValueError("Access outside project directory is not allowed.")

        if not file_path.is_file():
            raise FileNotFoundError(f"Source file not found: {source_file}")

        existing_code = file_path.read_text(encoding="utf-8")

        if not existing_code.strip():
            raise ValueError("Source file is empty.")

        print("\n[1/4] Planner Agent: Creating implementation plan...")
        plan = self.planner.create_plan(task)

        print("\n[2/4] Research Agent: Investigating actual source code...")
        research = self.researcher.research(task, existing_code)

        print("\n[3/4] Coding Agent: Generating source-grounded solution...")
        proposal = self.coder.generate_solution(
            task,
            existing_code,
            research
        )

        print("\n[4/4] Reviewer Agent: Reviewing proposed solution...")
        review = self.reviewer.review_solution(
            task,
            existing_code,
            research,
            proposal
        )

        return {
            "task": task,
            "plan": plan,
            "source_file": source_file,
            "existing_code": existing_code,
            "research": research,
            "proposal": proposal,
            "review": review,
        }



if __name__ == "__main__":

    orchestrator = SoftwareEngineerOrchestrator()

    task = input("Enter your software engineering task: ").strip()

    source_file = input(
        "Enter the project file to review (example: rag/document_loader.py): "
    ).strip()

    try:
        result = orchestrator.run(task, source_file)

        print("\n" + "=" * 60)
        print("FINAL SOFTWARE ENGINEERING REPORT")
        print("=" * 60)

        print("\nIMPLEMENTATION PLAN:")
        for index, step in enumerate(result["plan"], start=1):
            print(f"{index}. {step}")

        print("\nRESEARCH FINDINGS:")
        print(result["research"])

        print("\nCODING PROPOSAL:")
        print(result["proposal"])

        print("\nCODE REVIEW:")
        print(result["review"])

    except Exception as exc:
        print(f"\nOrchestrator error: {exc}")
