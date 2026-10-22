# # src/evaluation/base.py
# ----------------------------------------------------------------------------
# defines the shared abstract class and the two category-specific abstractions
from abc import ABC, abstractmethod
from inspect import isabstract

from src.evaluation.registry import register_metric


class Metric(ABC):
    """Abstract base class for all evaluation metrics."""

    name: str
    category: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Register concrete metric implementations automatically.
        if not isabstract(cls) and cls.__dict__.get("name"):
            register_metric(cls)

    @abstractmethod
    def compute(self, context) -> float | None:
        """Calculate the metric score."""
        raise NotImplementedError


class RetrievalMetric(Metric):
    """Base class for retrieval evaluation metrics."""

    category = "retrieval"


class GenerationMetric(Metric):
    """Base class for generation evaluation metrics."""

    category = "generation"