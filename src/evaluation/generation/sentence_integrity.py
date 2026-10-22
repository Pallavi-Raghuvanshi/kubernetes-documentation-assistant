import re

from src.evaluation.base import GenerationMetric


class SentenceIntegrity(GenerationMetric):
    """Measure the proportion of sentences ending with punctuation."""

    name = "sentence_integrity"

    def compute(self, context) -> float | None:
        answer = context.answer.strip()

        sentences = re.findall(
            r"[^.!?]+(?:[.!?]+|$)",
            answer,
        )
        sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

        if not sentences:
            return None

        complete = sum(
            bool(re.search(r"[.!?][\"')\]]*$", sentence))
            for sentence in sentences
        )

        return complete / len(sentences)