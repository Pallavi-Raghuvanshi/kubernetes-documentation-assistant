from src.evaluation.evaluation_context import EvaluationContext
from src.evaluation.retrieval.precision_at_k import PrecisionAtK


def test_precision_at_k():
    context = EvaluationContext(
        retrieved_results=[
            {"chunk_id": "chunk_1"},
            {"chunk_id": "chunk_2"},
            {"chunk_id": "chunk_3"},
        ],
        relevant_chunk_ids={"chunk_2", "chunk_4"},
        top_k=3,
    )

    assert PrecisionAtK().compute(context) == 1 / 3


def test_precision_at_k_no_relevant_chunks():
    context = EvaluationContext(
        retrieved_results=[],
        relevant_chunk_ids=set(),
        top_k=10,
    )

    assert PrecisionAtK().compute(context) is None