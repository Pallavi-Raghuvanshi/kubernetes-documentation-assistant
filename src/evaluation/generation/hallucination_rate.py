# src/evaluation/generation/hallucination_rate.py
from src.evaluation.base import GenerationMetric
from src.evaluation.generation.llm_judge import get_judge_scores


class HallucinationRate(GenerationMetric):
    name = "hallucination_rate"

    def compute(self, context) -> float | None:
        if not context.answer.strip() or not context.retrieved_context.strip():
            return None

        scores = get_judge_scores(context)
        return scores["hallucination_rate"]
