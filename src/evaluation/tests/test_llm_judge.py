
import pytest
from unittest.mock import patch

from src.evaluation.evaluation_context import EvaluationContext
from src.evaluation.generation.faithfulness import Faithfulness
from src.evaluation.generation.answer_completeness import AnswerCompleteness
from src.evaluation.generation.hallucination_rate import HallucinationRate
from src.evaluation.generation.llm_judge import get_judge_scores


VALID_RESPONSE = {
    "answer": (
        '{"faithfulness": 0.9, '
        '"answer_completeness": 0.8, '
        '"hallucination_rate": 0.1}'
    )
}


def make_context():
    return EvaluationContext(
        question="What is Kubernetes?",
        answer="Kubernetes orchestrates containers.",
        retrieved_context="Kubernetes is a container orchestration system.",
        reference_answer="Kubernetes orchestrates containers.",
    )


def test_judge_called_once_for_all_three_metrics():
    context = make_context()

    with patch(
        "src.evaluation.generation.llm_judge.judge_generation",
        return_value=VALID_RESPONSE,
    ) as mocked_judge:
        assert Faithfulness().compute(context) == 0.9
        assert AnswerCompleteness().compute(context) == 0.8
        assert HallucinationRate().compute(context) == 0.1

        assert mocked_judge.call_count == 1


def test_cached_scores_are_reused():
    context = make_context()

    with patch(
        "src.evaluation.generation.llm_judge.judge_generation",
        return_value=VALID_RESPONSE,
    ) as mocked_judge:
        first = get_judge_scores(context)
        second = get_judge_scores(context)

        assert first is second
        assert mocked_judge.call_count == 1


def test_invalid_json_raises_value_error():
    context = make_context()

    with patch(
        "src.evaluation.generation.llm_judge.judge_generation",
        return_value={"answer": "not valid JSON"},
    ):
        with pytest.raises(ValueError, match="valid JSON"):
            get_judge_scores(context)


def test_missing_score_raises_value_error():
    context = make_context()

    with patch(
        "src.evaluation.generation.llm_judge.judge_generation",
        return_value={
            "answer": '{"faithfulness": 0.9, "answer_completeness": 0.8}'
        },
    ):
        with pytest.raises(ValueError, match="missing scores"):
            get_judge_scores(context)



@pytest.mark.parametrize("bad_score", [-0.1, 1.1])
def test_out_of_range_score_raises_value_error(bad_score):
    import json

    context = make_context()

    response = {
        "answer": json.dumps({
            "faithfulness": bad_score,
            "answer_completeness": 0.8,
            "hallucination_rate": 0.1,
        })
    }

    with patch(
        "src.evaluation.generation.llm_judge.judge_generation",
        return_value=response,
    ):
        with pytest.raises(
            ValueError,
            match="Expected a value from 0 to 1",
        ):
            get_judge_scores(context)
