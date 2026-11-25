# src/evaluation/retrieval/mrr.py

from src.evaluation.base import RetrievalMetric
from src.evaluation.retrieval.evidence_utils import is_result_relevant


class MRR(RetrievalMetric):
    """Calculate reciprocal rank of the first evidence-relevant result."""

    name = "mrr"

    def compute(self, context) -> float | None:
        if not context.ground_truth_context.strip():
            return None

        retrieved = context.retrieved_results[:context.top_k]

        for rank, result in enumerate(retrieved, start=1):
            if is_result_relevant(result, context):
                return 1.0 / rank

        return 0.0
