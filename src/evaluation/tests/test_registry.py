from src.evaluation.registry import discover_metrics, get_metrics, create_metrics


def test_retrieval_metrics_register_automatically():
    discover_metrics()

    registered = get_metrics("retrieval")

    expected = {
        "recall_at_k",
        "precision_at_k",
        "mrr",
        "hit_rate_at_k",
    }

    assert expected.issubset(registered.keys())


def test_create_selected_retrieval_metrics():
    discover_metrics()

    metrics = create_metrics(
        names=["recall_at_k", "mrr"],
        category="retrieval",
    )

    assert [metric.name for metric in metrics] == [
        "recall_at_k",
        "mrr",
    ]