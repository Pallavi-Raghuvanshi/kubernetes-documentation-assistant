# src.evaluate.retrieval_metrics.py
from statistics import mean, median
import threading
import psutil


def average(values: list):
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def median_value(values: list):
    values = [value for value in values if value is not None]
    return median(values) if values else None


def round_metric(value, digits: int = 4):
    return round(value, digits) if value is not None else None


class PeakMemoryMonitor:
    """Sample process RSS and record the maximum observed value."""

    def __init__(self, interval: float = 0.01):
        self.process = psutil.Process()
        self.interval = interval
        self.peak_mb = 0.0
        self._stop_event = threading.Event()
        self._thread = None

    def _sample(self):
        while not self._stop_event.is_set():
            try:
                memory_mb = (
                    self.process.memory_info().rss / (1024 ** 2)
                )
                self.peak_mb = max(self.peak_mb, memory_mb)
            except psutil.Error:
                pass

            self._stop_event.wait(self.interval)

    def start(self):
        self.peak_mb = (
            self.process.memory_info().rss / (1024 ** 2)
        )
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._sample,
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> float:
        self._stop_event.set()

        if self._thread:
            self._thread.join()

        try:
            self.peak_mb = max(
                self.peak_mb,
                self.process.memory_info().rss / (1024 ** 2),
            )
        except psutil.Error:
            pass

        return self.peak_mb


def evaluate_retrieval(
    results: list[dict],
    relevant_chunks: set[str],
    top_k: int = 10,
) -> dict:
    """Calculate chunk-level retrieval quality metrics."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if not relevant_chunks:
        return {
            "recall_at_k": None,
            "precision_at_k": None,
            "mrr": None,
            "hit_rate_at_k": None,
        }

    retrieved = results[:top_k]
    retrieved_ids = [
        result["chunk_id"] for result in retrieved
    ]

    relevant_retrieved = set(retrieved_ids) & relevant_chunks

    relevant_ranks = [
        rank
        for rank, chunk_id in enumerate(retrieved_ids, start=1)
        if chunk_id in relevant_chunks
    ]

    return {
        "recall_at_k": len(relevant_retrieved) / len(relevant_chunks),
        "precision_at_k": len(relevant_retrieved) / top_k,
        "mrr": 1.0 / relevant_ranks[0] if relevant_ranks else 0.0,
        "hit_rate_at_k": 1.0 if relevant_ranks else 0.0,
    }
