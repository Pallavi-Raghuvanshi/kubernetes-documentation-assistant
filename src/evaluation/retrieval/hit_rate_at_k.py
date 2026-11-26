# src/evaluation/retrieval/hit_rate_at_k.py

from src.evaluation.base import RetrievalMetric

class HitRateAtK(RetrievalMetric):
    name = "hit_rate_at_k"

    def compute(self, context):
        results = context.retrieved_results[:context.top_k]

        return float(
            any(
                result.get("sequence_coverage", 0.0)
                >= context.sequence_coverage_threshold
                for result in results
            )
        )