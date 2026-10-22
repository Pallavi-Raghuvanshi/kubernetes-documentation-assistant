
from unittest.mock import patch

from src.evaluation.evaluation_context import EvaluationContext
from src.evaluation.generation.answer_completeness import AnswerCompleteness


def test_answer_completeness_score():
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
        "src.evaluation.generation.answer_completeness.get_judge_scores",
        return_value=mocked_scores,
    ):
        assert AnswerCompleteness().compute(context) == 1.0


def test_answer_completeness_missing_context():
    context = EvaluationContext(answer="Some answer.")

    assert AnswerCompleteness().compute(context) is None
