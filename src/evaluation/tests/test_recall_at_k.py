# src.evaluation.retrieval.test_recall_at_k
from src.evaluation.evaluation_context import EvaluationContext
from src.evaluation.retrieval.recall_at_k import RecallAtK


def test_recall_at_k():
    context = EvaluationContext(
        retrieved_results=[
            {"chunk_id": "chunk_1"},
            {"chunk_id": "chunk_2"},
            {"chunk_id": "chunk_3"},
        ],
        relevant_chunk_ids={"chunk_2", "chunk_4"},
        top_k=3,
    )

    metric = RecallAtK()

    assert metric.compute(context) == 0.5


def test_recall_at_k_no_relevant_chunks():
    context = EvaluationContext(
        retrieved_results=[],
        relevant_chunk_ids=set(),
        top_k=10,
    )

    assert RecallAtK().compute(context) is None