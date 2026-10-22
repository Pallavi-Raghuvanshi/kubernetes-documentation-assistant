from src.evaluation.evaluation_context import EvaluationContext
from src.evaluation.retrieval.mrr import MRR


def test_mrr_first_result_relevant():
    context = EvaluationContext(
        retrieved_results=[
            {"chunk_id": "chunk_1"},
            {"chunk_id": "chunk_2"},
        ],
        relevant_chunk_ids={"chunk_1"},
        top_k=2,
    )

    assert MRR().compute(context) == 1.0


def test_mrr_second_result_relevant():
    context = EvaluationContext(
        retrieved_results=[
            {"chunk_id": "chunk_1"},
            {"chunk_id": "chunk_2"},
        ],
        relevant_chunk_ids={"chunk_2"},
        top_k=2,
    )

    assert MRR().compute(context) == 0.5


def test_mrr_no_relevant_result():
    context = EvaluationContext(
        retrieved_results=[
            {"chunk_id": "chunk_1"},
        ],
        relevant_chunk_ids={"chunk_2"},
        top_k=1,
    )

    assert MRR().compute(context) == 0.0