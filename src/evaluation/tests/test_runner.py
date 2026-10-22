from src.evaluation.evaluation_context import EvaluationContext
from src.evaluation.runner import run_metrics


from unittest.mock import patch

from src.evaluation.evaluation_context import EvaluationContext
from src.evaluation.runner import run_metrics


def test_run_selected_retrieval_metrics():
    context = EvaluationContext(
        retrieved_results=[
            {"chunk_id": "chunk_1"},
            {"chunk_id": "chunk_2"},
            {"chunk_id": "chunk_3"},
        ],
        relevant_chunk_ids={"chunk_2", "chunk_4"},
        top_k=3,
    )

    scores = run_metrics(
        context=context,
        metric_names=["recall_at_k", "precision_at_k", "mrr", "hit_rate_at_k"],
        category="retrieval",
    )

    assert scores == {
        "recall_at_k": 0.5,
        "precision_at_k": 1 / 3,
        "mrr": 0.5,
        "hit_rate_at_k": 1.0,
    }

def make_generation_context():
    return EvaluationContext(
        query_id="q1",
        question="What is Kubernetes?",
        retrieved_results=[],
        relevant_chunk_ids=set(),
        top_k=5,
        answer="Kubernetes orchestrates containers.",
        reference_answer="Kubernetes orchestrates containers.",
        retrieved_context=(
            "Kubernetes is a container orchestration system."
        ),
    )


def test_run_selected_generation_metrics():
    context = make_generation_context()

    mocked_scores = {
        "faithfulness": 0.9,
        "answer_completeness": 0.8,
        "hallucination_rate": 0.1,
    }

    with patch(
        "src.evaluation.generation.llm_judge.judge_generation",
        return_value={
            "answer": (
                '{"faithfulness": 0.9, '
                '"answer_completeness": 0.8, '
                '"hallucination_rate": 0.1}'
            )
        },
    ) as mocked_judge:
        results = run_metrics(
            context=context,
            metric_names=[
                "faithfulness",
                "answer_completeness",
                "hallucination_rate",
            ],
            category="generation",
        )

    assert results == {
        "faithfulness": 0.9,
        "answer_completeness": 0.8,
        "hallucination_rate": 0.1,
    }
    assert mocked_judge.call_count == 1

def make_generation_context():
    return EvaluationContext(
        query_id="q1",
        question="What is Kubernetes?",
        answer="Kubernetes orchestrates containers.",
        reference_answer="Kubernetes orchestrates containers.",
        retrieved_context="Kubernetes is a container orchestration system.",
    )

def test_run_sentence_integrity_metric():
    context = make_generation_context()

    results = run_metrics(
        context=context,
        metric_names=["sentence_integrity"],
        category="generation",
    )

    assert results == {"sentence_integrity": 1.0}


def test_run_all_generation_metrics():
    context = make_generation_context()

    with patch(
        "src.evaluation.generation.llm_judge.judge_generation",
        return_value={
            "answer": (
                '{"faithfulness": 0.9, '
                '"answer_completeness": 0.8, '
                '"hallucination_rate": 0.1}'
            )
        },
    ) as mocked_judge:
        results = run_metrics(
            context=context,
            metric_names=[
                "faithfulness",
                "answer_completeness",
                "hallucination_rate",
                "sentence_integrity",
            ],
            category="generation",
        )

    assert results == {
        "faithfulness": 0.9,
        "answer_completeness": 0.8,
        "hallucination_rate": 0.1,
        "sentence_integrity": 1.0,
    }
    assert mocked_judge.call_count == 1