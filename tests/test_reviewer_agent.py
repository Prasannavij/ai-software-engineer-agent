
from agent.agents.reviewer_agent import ReviewerAgent


def test_reviewer_detects_incorrect_path_handling():
    reviewer = ReviewerAgent()

    existing_code = """
from pathlib import Path

def load_documents(sources):
    if isinstance(sources, (str, Path)):
        paths = [Path(sources)]
    else:
        paths = [Path(source) for source in sources]

    return paths
"""

    proposal = """
from pathlib import Path

def load_documents(sources):
    paths = [Path(source) for source in sources]
    return paths
"""

    review = reviewer.review_solution(
        task="Preserve support for single string and Path inputs.",
        existing_code=existing_code,
        research_findings="The function accepts a single string or Path.",
        proposed_solution=proposal,
    )

    review_text = review.lower()

    assert any(
        keyword in review_text
        for keyword in ["string", "path", "single", "iterable"]
    ), "Reviewer failed to identify the input-handling issue."

    print("\nReviewer evaluation passed.")


if __name__ == "__main__":
    test_reviewer_detects_incorrect_path_handling()
