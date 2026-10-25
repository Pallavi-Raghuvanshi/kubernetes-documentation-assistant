# src/evaluation/retrieval/mrr.py
from src.evaluation.base import RetrievalMetric


class MRR(RetrievalMetric):
    """Calculate reciprocal rank of the first relevant result."""

    name = "mrr"

    def compute(self, context) -> float | None:
        if not context.relevant_chunk_ids:
            return None

        retrieved = context.retrieved_results[:context.top_k]

        for rank, result in enumerate(retrieved, start=1):
            if result["chunk_id"] in context.relevant_chunk_ids:
                return 1.0 / rank

        return 0.0