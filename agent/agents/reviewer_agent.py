
import ollama
from agent.llm_client import MODEL_NAME


class ReviewerAgent:
    def __init__(self):
        self.model = MODEL_NAME

    def review_solution(
        self,
        task: str,
        existing_code: str,
        research_findings: str,
        proposed_solution: str,
    ) -> str:

        if not task.strip():
            raise ValueError("Review task cannot be empty.")

        if not existing_code.strip():
            raise ValueError("Existing code cannot be empty.")

        if not proposed_solution.strip():
            raise ValueError("Proposed solution cannot be empty.")

        system_prompt = """
You are a strict Senior Software Engineer and Code Reviewer.

Your responsibility is to verify proposed code changes against
the actual existing source code.

SOURCE OF TRUTH:
1. Existing project code is the primary source of truth.
2. The coding proposal is untrusted and must be verified.
3. Research findings are supplementary and may contain mistakes.
4. Never accept a claim just because another agent made it.

STRICT REVIEW RULES:

1. Verify every proposed import against the existing code
   or standard Python libraries.
2. Verify that existing functions, classes and variables
   are used correctly.
3. Check whether the proposed code preserves existing interfaces.
4. Check every claimed validation and exception handler
   by finding it in the proposed code.
5. Never claim that code catches an exception unless an
   actual exception handler exists.
6. Identify incorrect type handling and missing conversions.
7. Check whether the proposed implementation introduces bugs.
8. Do not invent problems that are not supported by the code.
9. Do not assume code has been executed or tested.
10. If something cannot be verified, explicitly say so.

EVIDENCE REQUIREMENT:

For every issue:
- Quote or identify the exact relevant code statement.
- Explain why it is incorrect or risky.
- Give a concrete correction.

For every claimed improvement:
- Verify that the implementation actually performs it.
- If it only claims to perform an action without implementing it,
  mark it as NOT IMPLEMENTED.

VERDICT RULES:

APPROVE:
No critical bugs, unsupported APIs, or required corrections.

NEEDS CHANGES:
The proposal is compatible but contains issues that must be fixed.

REJECT:
The proposal is fundamentally incompatible, introduces serious bugs,
or relies on fabricated project components.

Do not approve code merely because its explanation sounds correct.
"""

        user_prompt = f"""
TASK:
{task}

ACTUAL EXISTING SOURCE CODE:
{existing_code}

SUPPLEMENTARY RESEARCH FINDINGS:
{research_findings}

PROPOSED CODE CHANGES:
{proposed_solution}

Perform a strict comparison.

In particular:
- Check whether the proposal preserves the existing function signature.
- Check whether string and Path inputs are handled correctly.
- Check whether iterable inputs are handled correctly.
- Check whether file and directory handling remains compatible.
- Check whether existing encoding detection is preserved.
- Verify every claimed error handler against the actual code.

Return your review using these sections:

1. Overall Assessment
2. Correct Parts
3. Incorrect or Unsupported Code
4. Architecture Compatibility
5. Missing Error Handling
6. Required Corrections
7. Evidence-Based Findings
8. Final Verdict (APPROVE / NEEDS CHANGES / REJECT)
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return response.message.content


if __name__ == "__main__":
    reviewer = ReviewerAgent()

    task = input("Enter the review task: ").strip()
    existing_code = input("\nEnter existing project code: ").strip()
    research = input("\nEnter research findings: ").strip()
    proposal = input("\nEnter coding agent proposal: ").strip()

    try:
        review = reviewer.review_solution(
            task,
            existing_code,
            research,
            proposal,
        )

        print("\nCode Review:\n")
        print(review)

    except Exception as exc:
        print(f"Reviewer error: {exc}")
