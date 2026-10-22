# src/evaluation/retrieval/precision_at_k.py

from src.evaluation.base import RetrievalMetric


class PrecisionAtK(RetrievalMetric):
    """Calculate Precision@K for retrieved chunks."""

    name = "precision_at_k"

    def compute(self, context) -> float | None:
        if not context.relevant_chunk_ids:
            return None

        retrieved = context.retrieved_results[:context.top_k]

        retrieved_ids = {
            result["chunk_id"]
            for result in retrieved
        }

        relevant_retrieved = (
            retrieved_ids & context.relevant_chunk_ids
        )

        return len(relevant_retrieved) / context.top_k