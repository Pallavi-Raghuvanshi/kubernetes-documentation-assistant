# src/evaluation/retrieval/mrr.py

from src.evaluation.base import RetrievalMetric


class MRR(RetrievalMetric):

    name = "mrr"

    def compute(self, context) -> float | None:

        if not context.ground_truth_context.strip():
            return None

        results = context.retrieved_results[:context.top_k]

        for rank, result in enumerate(results, start=1):
            if (
                result.get("sequence_coverage", 0.0)
                >= context.sequence_coverage_threshold
            ):
                return 1.0 / rank

        return 0.0