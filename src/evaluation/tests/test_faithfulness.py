
from unittest.mock import patch

from src.evaluation.evaluation_context import EvaluationContext
from src.evaluation.generation.faithfulness import Faithfulness


def test_faithfulness_score():
    context = EvaluationContext(
        question="What is Kubernetes?",
        answer="Kubernetes orchestrates containers.",
        retrieved_context="Kubernetes is a container orchestration system.",
        reference_answer="Kubernetes orchestrates containers.",
    )

    mocked_scores = {
        "faithfulness": 0.9,
        "answer_completeness": 1.0,
        "hallucination_rate": 0.1,
    }

    with patch(
        "src.evaluation.generation.faithfulness.get_judge_scores",
        return_value=mocked_scores,
    ):
        assert Faithfulness().compute(context) == 0.9


def test_faithfulness_missing_context():
    context = EvaluationContext(answer="Some answer.")

    assert Faithfulness().compute(context) is None
