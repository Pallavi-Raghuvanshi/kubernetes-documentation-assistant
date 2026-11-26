# src/evaluation/retrieval/sequence_coverage.py
# -------------------------------------------------------------------
# This metric reports the average sequence coverage of the final reranked top-K chunks.

from src.evaluation.base import RetrievalMetric


class SequenceCoverage(RetrievalMetric):
    name = "sequence_coverage"

    def compute(self, context):
        results = context.retrieved_results[:context.top_k]

        if not results:
            return 0.0

        return max(
            result.get("sequence_coverage", 0.0)
            for result in results
        )