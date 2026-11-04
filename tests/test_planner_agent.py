import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.agents.planner_agent import PlannerAgent


class PlannerAgentTests(unittest.TestCase):
    def test_create_plan_returns_steps_from_json_response(self) -> None:
        response = SimpleNamespace(
            message=SimpleNamespace(
                content='{"plan":[" Inspect the RAG pipeline. ","Run relevant tests."]}'
            )
        )

        with patch("agent.agents.planner_agent.ollama.chat", return_value=response):
            plan = PlannerAgent().create_plan("Improve RAG error handling.")

        self.assertEqual(
            plan,
            ["Inspect the RAG pipeline.", "Run relevant tests."],
        )

    def test_create_plan_rejects_empty_task_without_calling_model(self) -> None:
        with patch("agent.agents.planner_agent.ollama.chat") as chat:
            with self.assertRaisesRegex(ValueError, "Task cannot be empty"):
                PlannerAgent().create_plan("  ")

        chat.assert_not_called()

    def test_create_plan_rejects_invalid_json(self) -> None:
        response = SimpleNamespace(
            message=SimpleNamespace(content="not JSON")
        )

        with patch("agent.agents.planner_agent.ollama.chat", return_value=response):
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                PlannerAgent().create_plan("Improve RAG error handling.")

    def test_create_plan_rejects_invalid_plan_shape(self) -> None:
        response = SimpleNamespace(
            message=SimpleNamespace(content='{"plan":["Inspect the code.", 2]}')
        )

        with patch("agent.agents.planner_agent.ollama.chat", return_value=response):
            with self.assertRaisesRegex(ValueError, "non-empty list of steps"):
                PlannerAgent().create_plan("Improve RAG error handling.")


if __name__ == "__main__":
    unittest.main()
