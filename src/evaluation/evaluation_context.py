# src/evaluation/evaluation_context.py
# ------------------------------------------------------------------------------------------------------------
# gives every metric a consistent input object, so the runner doesn't need metric-specific function arguments
from dataclasses import dataclass, field


@dataclass
class EvaluationContext:
    """Inputs shared with evaluation metrics."""

    query_id: str = ""
    question: str = ""

    # Retrieval inputs
    retrieved_results: list[dict] = field(default_factory=list)
    relevant_chunk_ids: set[str] = field(default_factory=set)
    top_k: int = 10

    # Generation inputs
    answer: str = ""
    reference_answer: str = ""
    retrieved_context: str = ""

    judge_scores: dict[str, float] | None = None # cache the judge's scores for the current evaluation context