# src/evaluation/retrieval/hit_rate_at_k.py
from src.evaluation.base import RetrievalMetric


class HitRateAtK(RetrievalMetric):
    """Return 1 if any relevant chunk appears in the top K."""

    name = "hit_rate_at_k"

    def compute(self, context) -> float | None:
        if not context.relevant_chunk_ids:
            return None

        retrieved = context.retrieved_results[:context.top_k]

        return float(
            any(
                result["chunk_id"] in context.relevant_chunk_ids
                for result in retrieved
            )
        )