# src/evaluation/retrieval/recall_at_k.py


from src.evaluation.base import RetrievalMetric
from src.evaluation.retrieval.evidence_utils import calculate_evidence_overlap


class RecallAtK(RetrievalMetric):
    """Calculate ground-truth evidence coverage in the top K results."""

    name = "recall_at_k"

    def compute(self, context) -> float | None:
        retrieved = context.retrieved_results[:context.top_k]

        matched, total_gt, _ = calculate_evidence_overlap(
            context.ground_truth_context,
            retrieved,
        )

        if total_gt == 0:
            return None

        return matched / total_gt
