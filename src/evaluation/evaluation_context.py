# src/evaluation/evaluation_context.py

from dataclasses import dataclass, field


@dataclass
class EvaluationContext:
    """Inputs shared with evaluation metrics."""
    query_id: str = ""
    question: str = ""
    # Retrieval inputs
    retrieved_results: list[dict] = field(default_factory=list)
    ground_truth_context: str = ""
    top_k: int = 10
    relevance_threshold: float = 0.15
    # Reranking inputs
    sequence_coverage_threshold: float = 0.75
    # Generation inputs
    answer: str = ""
    reference_answer: str = ""
    retrieved_context: str = ""

    judge_scores: dict[str, float] | None = None
