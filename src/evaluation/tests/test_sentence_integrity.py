from src.evaluation.evaluation_context import EvaluationContext
from src.evaluation.generation.sentence_integrity import SentenceIntegrity


def test_all_sentences_complete():
    context = EvaluationContext(
        answer="Kubernetes manages workloads. It exposes an API!"
    )

    assert SentenceIntegrity().compute(context) == 1.0


def test_incomplete_sentence():
    context = EvaluationContext(
        answer="Kubernetes manages workloads. It exposes an API"
    )

    assert SentenceIntegrity().compute(context) == 0.5


def test_empty_answer():
    context = EvaluationContext(answer="   ")

    assert SentenceIntegrity().compute(context) is None