from src.evaluation.evaluation_context import EvaluationContext
from src.evaluation.registry import create_metrics, discover_metrics


def run_metrics(
    context: EvaluationContext,
    metric_names: list[str],
    category: str,
) -> dict[str, float | None]:
    """Execute selected metrics and return their scores."""

    discover_metrics()
    metrics = create_metrics(metric_names, category)

    return {
        metric.name: metric.compute(context)
        for metric in metrics
    }