# src/evaluationregistry.py
# ---------------------------------------------------------------------------------------------
# stores metric implementations and discovers modules in the retrieval and generation folders
import importlib
import pkgutil


_METRICS = {}


def register_metric(metric_class) -> None:
    """Register a concrete metric class by its unique name."""

    name = getattr(metric_class, "name", None)
    category = getattr(metric_class, "category", None)

    if not name or not category:
        raise ValueError(
            "Every metric must define a name and category."
        )

    if category not in {"retrieval", "generation"}:
        raise ValueError(f"Unsupported metric category: {category}")

    if name in _METRICS:
        existing = _METRICS[name]

        if existing is not metric_class:
            raise ValueError(f"Duplicate metric name: {name}")

    _METRICS[name] = metric_class


def discover_metrics() -> None:
    """Import metric modules so their classes register automatically."""

    for category in ("retrieval", "generation"):
        package_name = f"src.evaluation.{category}"
        package = importlib.import_module(package_name)

        for module in pkgutil.iter_modules(package.__path__):
            if not module.ispkg and not module.name.startswith("_"):
                importlib.import_module(
                    f"{package_name}.{module.name}"
                )


def get_metric(name: str):
    """Return a registered metric class by name."""

    try:
        return _METRICS[name]
    except KeyError:
        raise KeyError(
            f"Unknown metric '{name}'. "
            f"Available metrics: {sorted(_METRICS)}"
        ) from None


def get_metrics(category: str | None = None) -> dict:
    """Return registered metric classes, optionally by category."""

    if category is None:
        return dict(_METRICS)

    return {
        name: metric_class
        for name, metric_class in _METRICS.items()
        if metric_class.category == category
    }

def create_metrics(names: list[str], category: str) -> list:
    """Create instances of the selected metrics."""

    metrics = []

    for name in names:
        metric_class = get_metric(name)

        if metric_class.category != category:
            raise ValueError(
                f"Metric '{name}' belongs to "
                f"'{metric_class.category}', not '{category}'."
            )

        metrics.append(metric_class())

    return metrics