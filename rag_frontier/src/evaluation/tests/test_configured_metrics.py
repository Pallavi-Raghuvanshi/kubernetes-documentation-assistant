
from unittest.mock import patch

import pytest

from src.evaluation.evaluation_context import EvaluationContext
from src.evaluation.evaluate import (
    get_selected_metrics,
    run_configured_metrics,
)


def test_get_selected_retrieval_metrics():
    config = {
        "mode": "both",
        "metrics": {
            "retrieval": ["recall_at_k", "mrr"],
            "generation": ["faithfulness"],
        },
    }

    assert get_selected_metrics(config, "retrieval") == [
        "recall_at_k",
        "mrr",
    ]


def test_generation_mode_disables_retrieval_metrics():
    config = {
        "mode": "generation",
        "metrics": {
            "retrieval": ["recall_at_k"],
            "generation": ["faithfulness"],
        },
    }

    assert get_selected_metrics(config, "retrieval") == []


def test_invalid_mode_raises_error():
    config = {
        "mode": "unknown",
        "metrics": {},
    }

    with pytest.raises(ValueError, match="Invalid evaluation mode"):
        get_selected_metrics(config, "retrieval")


def test_runner_executes_only_configured_metrics():
    context = EvaluationContext(
        retrieved_results=[
            {"chunk_id": "chunk_1"},
            {"chunk_id": "chunk_2"},
        ],
        relevant_chunk_ids={"chunk_2"},
        top_k=2,
    )

    config = {
        "mode": "retrieval",
        "metrics": {
            "retrieval": ["recall_at_k"],
            "generation": ["faithfulness"],
        },
    }

    with patch(
        "src.evaluate.evaluate.run_metrics",
        return_value={"recall_at_k": 1.0},
    ) as mocked_runner:
        result = run_configured_metrics(
            context=context,
            evaluation_config=config,
            category="retrieval",
        )

    assert result == {"recall_at_k": 1.0}

    mocked_runner.assert_called_once_with(
        context=context,
        metric_names=["recall_at_k"],
        category="retrieval",
    )
