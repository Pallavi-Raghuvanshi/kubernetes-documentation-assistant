# src/evaluation/retrieval/hit_rate_at_k.py

from src.evaluation.base import RetrievalMetric
from src.evaluation.retrieval.evidence_utils import is_result_relevant


class HitRateAtK(RetrievalMetric):
    """Return 1 if any evidence-relevant result appears in the top K."""

    name = "hit_rate_at_k"

    def compute(self, context) -> float | None:
        if not context.ground_truth_context.strip():
            return None

        retrieved = context.retrieved_results[:context.top_k]

        return float(
            any(
                is_result_relevant(result, context)
                for result in retrieved
            )
        )
