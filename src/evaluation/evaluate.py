# evaluate.py
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


def commit_code_changes() -> str:
    """Stage and commit changes, then return the current Git commit hash."""
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

    return run_git(["rev-parse", "HEAD"]).stdout.strip()

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

    evaluation_config = config.get("evaluation", {})
    evaluation_mode = evaluation_config.get("mode", "both").lower()
    num_queries = int(evaluation_config.get("num_queries", 0))

    if num_queries > 0:
        queries = queries[:num_queries]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results to determine the next version.
    if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0:
        saved_results = json.loads(
            OUTPUT_PATH.read_text(encoding="utf-8")
        )
    else:
        saved_results = {}

    # Commit code/configuration changes before evaluation.
    evaluation_config = config.get("evaluation", {})
    commit_changes = evaluation_config.get("commit_changes", True)
    print('starting evaluation...')
    if commit_changes:
        version = get_next_version(saved_results)
        git_commit = commit_code_changes()
        print(f"Project Version: {version}")
        print(f"Git commit: {git_commit}")
    else:
        version = ''
        git_commit = "-"
        print("Skipping Git commit.")

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
            run_generation=evaluation_mode != "retrieval"
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

        # Calculate retrieval metrics using chunk-level relevance.
        relevant_chunks = set(query_item["relevant_chunks"])

        if relevant_chunks:
            retrieved_at_10 = results[:EVAL_TOP_K]
            retrieved_chunks = [
                result["chunk_id"]
                for result in retrieved_at_10
            ]

            relevant_ranks = [
                rank
                for rank, chunk_id in enumerate(
                    retrieved_chunks,
                    start=1,
                )
                if chunk_id in relevant_chunks
            ]

            relevant_retrieved = (
                set(retrieved_chunks) & relevant_chunks
            )

            recall_values.append(
                len(relevant_retrieved) / len(relevant_chunks)
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
        if generation_result is not None:
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

        if generation_result is not None:
            print(
                f"{query_id}: "
                f"retrieval={retrieval_latency:.4f}s, "
                f"generation="
                f"{generation_result['generation_latency_seconds']:.3f}s"
            )
        else:
            print(
                f"{query_id}: retrieval={retrieval_latency:.4f}s"
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
    if evaluation_mode == "both":
        print(f"                    Queries: {len(queries)}")
        print(f"     Mean retrieval latency: {version_metrics['retrieval']['latency_s']} s")
        print(f"    Mean generation latency: {version_metrics['generation']['latency_s']} s")
        print(f"       Mean overall latency: {version_metrics['overall']['latency_s']} s")
        print(f"      Retrieval peak memory: {version_metrics['retrieval']['peak_memory_mb']} MB")
        print(f"     Generation peak memory: {version_metrics['generation']['peak_memory_mb']} MB")
        print(f"        Overall peak memory: {version_metrics['overall']['peak_memory_mb']} MB")
        print(f"                 Git commit: {git_commit}")
        print(f"                      Saved: {OUTPUT_PATH}")
    elif evaluation_mode == "retrieval":
        print(f"                    Queries: {len(queries)}")
        print(f"                  Recall@10: {version_metrics['retrieval']['recall_at_10']}")
        print(f"               Precision@10: {version_metrics['retrieval']['precision_at_10']}")
        print(f"                        MRR: {version_metrics['retrieval']['mrr']}")
        print(f"                Hit Rate@10: {version_metrics['retrieval']['hit_rate_at_10']}")
        print(f"     Mean retrieval latency: {version_metrics['retrieval']['latency_s']} s")
        print(f"      Retrieval peak memory: {version_metrics['retrieval']['peak_memory_mb']} MB")
    elif evaluation_mode == "generation":
        print(f"                    Queries: {len(queries)}")
        print(f"  Mean sentence integrity: {version_metrics['generation']['sentence_integrity_rate']}")
        print(f"  Mean fluency: {version_metrics['generation']['fluency_rate']}")
        print(f"  Mean factuality: {version_metrics['generation']['factuality_rate']}")
        print(f"  Mean coherence: {version_metrics['generation']['coherence_rate']}")
        print(f"  Mean relevance: {version_metrics['generation']['relevance_rate']}")
        print(f"  Mean informativeness: {version_metrics['generation']['informativeness_rate']}")
        print(f"  Mean conciseness: {version_metrics['generation']['conciseness_rate']}")
        print(f"  Mean creativity: {version_metrics['generation']['creativity_rate']}")
        print(f"  Mean engagement: {version_metrics['generation']['engagement_rate']}")
        print(f"  Mean emotional impact: {version_metrics['generation']['emotional_impact_rate']}")
        print(f"     Mean generation latency: {version_metrics['generation']['latency_s']} s")
        print(f"     Generation peak memory: {version_metrics['generation']['peak_memory_mb']} MB")
        print(f"                 Git commit: {git_commit}")
        print(f"                      Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
