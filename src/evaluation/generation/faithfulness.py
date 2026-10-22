# src/evaluation/generation/faithfulness.py
from src.evaluation.base import GenerationMetric
from src.evaluation.generation.llm_judge import get_judge_scores


class Faithfulness(GenerationMetric):
    """Measure how much of the generated answer is supported by evidence."""

    name = "faithfulness"

    def compute(self, context) -> float | None:
        if not context.answer.strip() or not context.retrieved_context.strip():
            return None

        scores = get_judge_scores(context)
        return scores["faithfulness"]