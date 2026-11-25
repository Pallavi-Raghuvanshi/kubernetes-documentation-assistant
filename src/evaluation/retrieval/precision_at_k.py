# src/evaluation/retrieval/precision_at_k.py

from src.evaluation.base import RetrievalMetric
from src.evaluation.retrieval.evidence_utils import (
    calculate_evidence_overlap,
)


class PrecisionAtK(RetrievalMetric):
    """Calculate the proportion of retrieved text supported by evidence."""

    name = "precision_at_k"

    def compute(self, context) -> float | None:
        retrieved = context.retrieved_results[:context.top_k]

        matched, _, total_retrieved = calculate_evidence_overlap(
            context.ground_truth_context,
            retrieved,
        )

        if total_retrieved == 0:
            return None

        return matched / total_retrieved
