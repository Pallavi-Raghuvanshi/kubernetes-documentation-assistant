from src.evaluation.evaluation_context import EvaluationContext
from src.evaluation.retrieval.hit_rate_at_k import HitRateAtK


def test_hit_rate_when_relevant_chunk_found():
    context = EvaluationContext(
        retrieved_results=[
            {"chunk_id": "chunk_1"},
            {"chunk_id": "chunk_2"},
        ],
        relevant_chunk_ids={"chunk_2"},
        top_k=2,
    )

    assert HitRateAtK().compute(context) == 1.0


def test_hit_rate_when_no_relevant_chunk_found():
    context = EvaluationContext(
        retrieved_results=[
            {"chunk_id": "chunk_1"},
        ],
        relevant_chunk_ids={"chunk_2"},
        top_k=1,
    )

    assert HitRateAtK().compute(context) == 0.0


def test_hit_rate_when_relevance_labels_missing():
    context = EvaluationContext(
        retrieved_results=[],
        relevant_chunk_ids=set(),
        top_k=10,
    )

    assert HitRateAtK().compute(context) is None