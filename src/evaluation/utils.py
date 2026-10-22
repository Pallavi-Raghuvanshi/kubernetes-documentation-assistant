
import statistics
import threading

import psutil


def average(values):
    """Return the arithmetic mean, or None if no values exist."""
    return statistics.mean(values) if values else None


def median_value(values):
    """Return the median, or None if no values exist."""
    return statistics.median(values) if values else None


def round_metric(value, digits=4):
    """Round a metric while preserving missing values."""
    return round(value, digits) if value is not None else None


class PeakMemoryMonitor:
    """Monitor peak process memory usage in MB."""

    def __init__(self, interval=0.01):
        self.interval = interval
        self.process = psutil.Process()
        self.peak_memory_mb = 0.0
        self._stop_event = threading.Event()
        self._thread = None

    def _monitor(self):
        while not self._stop_event.is_set():
            try:
                memory_mb = self.process.memory_info().rss / (1024 * 1024)
                self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
            except psutil.Error:
                pass

            self._stop_event.wait(self.interval)

    def start(self):
        """Start monitoring process memory."""
        self.peak_memory_mb = (
            self.process.memory_info().rss / (1024 * 1024)
        )
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """Stop monitoring and return peak memory in MB."""
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join()

        return round(self.peak_memory_mb, 2)

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False
