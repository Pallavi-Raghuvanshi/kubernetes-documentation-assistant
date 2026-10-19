
import json
import re
import subprocess
import time
import threading
from pathlib import Path
from statistics import mean, median

import psutil
import yaml

from src.retrieval._4bm25_index import BM25Index
from src.rag_pipeline import run_pipeline


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs.yaml"
QUERIES_PATH = ROOT / "queries.yaml"
OUTPUT_PATH = ROOT / "experiments" / "evaluation_results.json"

EVAL_TOP_K = 10


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def get_git_commit() -> str:
    """Return the current Git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_next_version(saved_results: dict) -> str:
    """Return the next sequential version name."""
    version_numbers = []

    for key in saved_results:
        match = re.fullmatch(r"version(\d+)", key)
        if match:
            version_numbers.append(int(match.group(1)))

    return f"version{max(version_numbers, default=0) + 1}"


def run_git(args: list[str]) -> subprocess.CompletedProcess:
    """Run a Git command from the project root."""
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def commit_code_changes() -> None:
    """Stage and commit all non-ignored project changes."""
    run_git(["add", "-A"])

    staged_changes = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    if staged_changes.returncode == 1:
        run_git([
            "commit",
            "-m",
            "Update RAG implementation before evaluation",
        ])
    elif staged_changes.returncode != 0:
        raise RuntimeError(
            f"Could not inspect staged changes: "
            f"{staged_changes.stderr}"
        )


def get_queries(config: dict) -> list[dict]:
    """Load queries from queries.yaml, falling back to configs.yaml."""
    if QUERIES_PATH.exists():
        queries = load_yaml(QUERIES_PATH).get("queries", [])
    else:
        queries = config.get("queries", [])

    if not queries:
        raise ValueError("No benchmark queries found.")

    return queries


def sentence_integrity_rate(answer: str):
    """
    Heuristic: proportion of detected sentences ending in
    '.', '!' or '?'.

    This detects possible truncation, not factual correctness.
    """
    answer = answer.strip()

    if not answer:
        return None

    sentences = re.findall(
        r"[^.!?]+(?:[.!?]+|$)",
        answer,
    )
    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    if not sentences:
        return None

    complete = sum(
        bool(re.search(r"[.!?][\"')\]]*$", sentence))
        for sentence in sentences
    )

    return complete / len(sentences)


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


def main() -> None:
    config = load_yaml(CONFIG_PATH)
    queries = get_queries(config)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results to determine the next version.
    if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0:
        saved_results = json.loads(
            OUTPUT_PATH.read_text(encoding="utf-8")
        )
    else:
        saved_results = {}

    version = get_next_version(saved_results)

    # Commit code/configuration changes before evaluation.
    commit_code_changes()
    git_commit = get_git_commit()

    print(f"Starting evaluation: {version}")
    print(f"Git commit: {git_commit}")

    retrieval_config = config.get("retrieval", {})
    generation_config = config.get("generation", {})

    generation_top_k = int(
        retrieval_config.get("top_k", EVAL_TOP_K)
    )
    max_new_tokens = int(
        generation_config.get("max_new_tokens", 160)
    )

    # Build the index once and reuse it for all queries.
    index = BM25Index.from_corpus()

    retrieval_latencies = []
    recall_values = []
    precision_values = []
    mrr_values = []
    hit_rate_values = []

    generation_latencies = []
    input_token_counts = []
    output_token_counts = []
    sir_values = []

    overall_latencies = []

    retrieval_peak_memory_values = []
    generation_peak_memory_values = []

    # Measures overall evaluation-query processing after index creation.
    overall_memory_monitor = PeakMemoryMonitor()
    overall_memory_monitor.start()

    for query_item in queries:
        query_id = query_item["query_id"]
        question = query_item["query"]

        overall_start = time.perf_counter()

        # Run the RAG pipeline and monitor its memory usage.
        pipeline_memory_monitor = PeakMemoryMonitor()
        pipeline_memory_monitor.start()


        pipeline_result = run_pipeline(
            question=question,
            config={
                **config,
                "retrieval": {
                    **config.get("retrieval", {}),
                    "top_k": max(EVAL_TOP_K, generation_top_k),
                },
                "generation": {
                    **config.get("generation", {}),
                    "max_new_tokens": max_new_tokens,
                },
            },
            index=index,
            memory_monitor_factory=PeakMemoryMonitor,
        )

        results = pipeline_result["results"]
        generation_result = pipeline_result["generation_result"]
        retrieval_latency = pipeline_result["retrieval_latency_seconds"]

        retrieval_latencies.append(retrieval_latency)
        retrieval_peak_memory_values.append(
            pipeline_result["retrieval_peak_memory_mb"]
        )
        generation_peak_memory_values.append(
            pipeline_result["generation_peak_memory_mb"]
        )

        # Calculate retrieval metrics using source-level relevance.
        relevant_sources = set(
            query_item.get("relevant_sources", [])
        )

        if relevant_sources:
            retrieved_at_10 = results[:EVAL_TOP_K]
            retrieved_sources = [
                result["source"]
                for result in retrieved_at_10
            ]

            relevant_ranks = [
                rank
                for rank, source in enumerate(
                    retrieved_sources,
                    start=1,
                )
                if source in relevant_sources
            ]

            relevant_retrieved = (
                set(retrieved_sources) & relevant_sources
            )

            recall_values.append(
                len(relevant_retrieved) / len(relevant_sources)
            )
            precision_values.append(
                len(relevant_retrieved) / EVAL_TOP_K
            )
            mrr_values.append(
                1.0 / relevant_ranks[0]
                if relevant_ranks
                else 0.0
            )
            hit_rate_values.append(
                1.0 if relevant_ranks else 0.0
            )

        # Record generation metrics from the pipeline result.
        generation_latencies.append(
            generation_result["generation_latency_seconds"]
        )
        input_token_counts.append(
            generation_result["input_tokens_count"]
        )
        output_token_counts.append(
            generation_result["generated_tokens"]
        )

        sir = sentence_integrity_rate(
            generation_result["answer"]
        )
        if sir is not None:
            sir_values.append(sir)

        overall_latencies.append(
            time.perf_counter() - overall_start
        )

        print(
            f"{query_id}: "
            f"retrieval={retrieval_latency:.4f}s, "
            f"generation="
            f"{generation_result['generation_latency_seconds']:.3f}s"
        )

    overall_peak_memory_mb = overall_memory_monitor.stop()

    version_metrics = {
        "git_commit": git_commit,
        "retrieval": {
            "recall_at_10": round_metric(
                average(recall_values)
            ),
            "precision_at_10": round_metric(
                average(precision_values)
            ),
            "mrr": round_metric(average(mrr_values)),
            "hit_rate_at_10": round_metric(
                average(hit_rate_values)
            ),
            "latency_s": round_metric(
                average(retrieval_latencies)
            ),
            "peak_memory_mb": round_metric(
                average(retrieval_peak_memory_values)
            ),
        },
        "generation": {
            "sentence_integrity_rate": round_metric(
                average(sir_values)
            ),
            "latency_s": round_metric(
                average(generation_latencies)
            ),
            "input_tokens_count": round_metric(
                average(input_token_counts), 2
            ),
            "output_tokens_count": round_metric(
                average(output_token_counts), 2
            ),
            "peak_memory_mb": round_metric(
                average(generation_peak_memory_values)
            ),
        },
        "overall": {
            "latency_s": round_metric(
                average(overall_latencies)
            ),
            "peak_memory_mb": round_metric(
                overall_peak_memory_mb
            ),
        },
    }

    # Append the new version while preserving previous results.
    saved_results[version] = version_metrics

    OUTPUT_PATH.write_text(
        json.dumps(
            saved_results,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    # Do not commit after evaluation.
    print(f"\n                  Version: {version}")
    print(f"          Queries processed: {len(queries)}")
    print(
        f"Retrieval metrics evaluated: "
        f"{len(recall_values)} queries"
    )
    print(
        f"     Mean retrieval latency: "
        f"{version_metrics['retrieval']['latency_s']} s"
    )
    print(
        f"    Mean generation latency: "
        f"{version_metrics['generation']['latency_s']} s"
    )
    print(
        f"       Mean overall latency: "
        f"{version_metrics['overall']['latency_s']} s"
    )
    print(
        f"       Retrieval peak memory: "
        f"{version_metrics['retrieval']['peak_memory_mb']} MB"
    )
    print(
        f"      Generation peak memory: "
        f"{version_metrics['generation']['peak_memory_mb']} MB"
    )
    print(
        f"          Overall peak memory: "
        f"{version_metrics['overall']['peak_memory_mb']} MB"
    )
    print(f"                 Git commit: {git_commit}")
    print(f"                      Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
