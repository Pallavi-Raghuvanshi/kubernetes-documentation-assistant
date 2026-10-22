
import json
import subprocess
import time
from pathlib import Path

import yaml

from src.retrieval.__4bm25_index import load_index
from src.rag_pipeline import run_pipeline
from src.evaluation.evaluation_context import EvaluationContext
from src.evaluation.runner import run_metrics
from src.evaluation.utils import (
    average,
    round_metric,
    PeakMemoryMonitor,
)
from sentence_transformers import SentenceTransformer
from src.embeddings.embeddings import MODEL_NAME
from src.embeddings.embeddings import EMBEDDINGS_PATH
from src.retrieval.dense_retrieval import load_dense_index


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs.yaml"
QUERIES_PATH = ROOT / "queries.yaml"
OUTPUT_PATH = ROOT / "experiments" / "evaluation_results.json"


def load_yaml(path: Path) -> dict:
    """Load a YAML file."""
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def get_queries(config: dict) -> list[dict]:
    """Load benchmark queries, preferring queries.yaml."""

    if QUERIES_PATH.exists():
        queries = load_yaml(QUERIES_PATH).get("queries", [])
    else:
        queries = config.get("queries", [])

    if not queries:
        raise ValueError("No benchmark queries found.")

    return queries


def get_selected_metrics(
    evaluation_config: dict,
    category: str,
) -> list[str]:
    """Return configured metrics for retrieval-only or combined evaluation."""

    mode = evaluation_config.get("mode", "both")

    if not isinstance(mode, str):
        raise ValueError("evaluation.mode must be a string.")

    mode = mode.lower()

    if mode not in {"retrieval", "both"}:
        raise ValueError(
            f"Invalid evaluation mode: {mode!r}. "
            "Expected retrieval or both."
        )

    if category not in {"retrieval", "generation"}:
        raise ValueError(f"Invalid metric category: {category!r}")

    if mode == "retrieval":
        if category == "retrieval":
            metrics_config = evaluation_config.get("metrics", {})
            return metrics_config.get("retrieval", [])
        return []

    else:  # mode == "both"
        metrics_config = evaluation_config.get("metrics", {})
        return metrics_config.get(category, [])


def run_configured_metrics(
    context: EvaluationContext,
    evaluation_config: dict,
    category: str,
) -> dict[str, float | None]:
    """Execute only the configured metrics for one category."""

    metric_names = get_selected_metrics(
        evaluation_config,
        category,
    )

    if not metric_names:
        return {}

    return run_metrics(
        context=context,
        metric_names=metric_names,
        category=category,
    )


def summarize_metrics(
    values: dict[str, list[float]],
) -> dict[str, float | None]:
    """Calculate the mean score for each configured metric."""

    return {
        name: round_metric(average(scores))
        for name, scores in values.items()
    }


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
    """Stage and commit changes before evaluation."""

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


def main() -> None:
    # ---------------------------------------------------------
    # 1. Load configuration and benchmark queries
    # ---------------------------------------------------------
    config = load_yaml(CONFIG_PATH)
    evaluation_config = config.get("evaluation", {})

    retrieval_metric_names = get_selected_metrics(
        evaluation_config,
        "retrieval",
    )
    generation_metric_names = get_selected_metrics(
        evaluation_config,
        "generation",
    )

    if not retrieval_metric_names and not generation_metric_names:
        raise ValueError(
            "No evaluation metrics selected. Check evaluation.mode "
            "and evaluation.metrics in configs.yaml."
        )

    queries = get_queries(config)
    num_queries = int(evaluation_config.get("num_queries", 0))

    if num_queries < 0:
        raise ValueError("evaluation.num_queries cannot be negative.")

    if num_queries > 0:
        queries = queries[:num_queries]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 2. Load previously saved experiment results
    # ---------------------------------------------------------
    if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0:
        saved_results = json.loads(
            OUTPUT_PATH.read_text(encoding="utf-8")
        )
    else:
        saved_results = {}

    if not isinstance(saved_results, dict):
        raise ValueError(
            f"Expected a JSON object in {OUTPUT_PATH}."
        )

    # ---------------------------------------------------------
    # 3. Capture code version before evaluation
    # ---------------------------------------------------------
    commit_changes = bool(
        evaluation_config.get("commit_changes", False)
    )
    version = evaluation_config.get("version_name", "baseline")

    print("Starting evaluation...")
    print(f"Evaluation mode: {evaluation_config.get('mode', 'both')}")
    print(f"Queries selected: {len(queries)}")
    print(f"Version name: {version}")

    if commit_changes:
        git_commit = commit_code_changes()
        print(f"Git commit: {git_commit}")
    else:
        git_commit = get_git_commit()
        print("Skipping Git commit.")

    # ---------------------------------------------------------
    # 4. Read retrieval and generation configuration
    # ---------------------------------------------------------
    retrieval_config = config.get("retrieval", {})
    generation_config = config.get("generation", {})

    configured_top_k = int(retrieval_config["top_k"])
    max_new_tokens = int(generation_config.get("max_new_tokens", 160))

    if configured_top_k < 1:
        raise ValueError("retrieval.top_k must be at least 1.")

    # ---------------------------------------------------------
    # 5. Initialize metric and performance accumulators
    # ---------------------------------------------------------
    metric_values = {
        "retrieval": {name: [] for name in retrieval_metric_names},
        "generation": {name: [] for name in generation_metric_names},
    }

    retrieval_latencies = []
    retrieval_peak_memory_values = []
    generation_peak_memory_values = []

    generation_latencies = []
    ttft_values = []
    tpot_values = []
    input_token_counts = []
    output_token_counts = []

    overall_latencies = []

    # ---------------------------------------------------------
    # 6. Load the selected retrieval index once
    retriever = retrieval_config.get("default_retriever", "bm25").lower()

    if retriever == "bm25":
        chunks, bm25 = load_index()
        index = (chunks, bm25)

    elif retriever == "dense":
        chunks, embeddings = load_dense_index()
        model = SentenceTransformer(MODEL_NAME)
        index = (chunks, embeddings, model)

    else:
        raise ValueError(f"Unsupported retriever: {retriever}")

    print(f"Retriever: {retriever}")

    overall_memory_monitor = PeakMemoryMonitor()
    overall_memory_monitor.start()

    try:
        # -----------------------------------------------------
        # 7. Evaluate each benchmark query
        # -----------------------------------------------------
        for query_item in queries:
            query_id = query_item["query_id"]
            question = query_item["query"]

            overall_start = time.perf_counter()

            # Retrieval is needed if retrieval metrics are enabled.
            # Generation mode also needs retrieval for the RAG answer.
            pipeline_result = run_pipeline(
                question=question,
                config={
                    **config,
                    "retrieval": {
                        **retrieval_config,
                        "top_k": configured_top_k,
                    },
                    "generation": {
                        **generation_config,
                        "max_new_tokens": max_new_tokens,
                    },
                },
                index=index,
                memory_monitor_factory=PeakMemoryMonitor,
                run_generation=bool(generation_metric_names),
            )

            results = pipeline_result["results"]
            retrieved_context = pipeline_result["context"]
            generation_result = pipeline_result["generation_result"]

            retrieval_latency = pipeline_result[
                "retrieval_latency_seconds"
            ]

            retrieval_latencies.append(retrieval_latency)
            retrieval_peak_memory_values.append(
                pipeline_result["retrieval_peak_memory_mb"]
            )

            generation_peak_memory_values.append(
                pipeline_result["generation_peak_memory_mb"]
            )

            # -------------------------------------------------
            # 8. Run configured retrieval metrics
            # -------------------------------------------------
            if retrieval_metric_names:
                # ----------------------------------------------------------------------------
                print("\n" + "=" * 60)
                print("RETRIEVAL GROUND-TRUTH DIAGNOSTIC")
                print("=" * 60)
                print("Query ID:", query_id)
                print("Question:", question)
                print("Relevant chunk IDs:", query_item.get("relevant_chunks"))
                print(
                    "Retrieved chunk IDs:",
                    [item.get("chunk_id", item.get("id")) for item in results[:configured_top_k]],
                )
                relevant_chunk_ids = set(
                    query_item.get("relevant_chunks", [])
                )

                print("\nRelevant chunk ranks:")
                relevant_ranks = {
                    item.get("chunk_id", item.get("id")): rank
                    for rank, item in enumerate(
                        results[:configured_top_k],
                        start=1,
                    )
                    if item.get("chunk_id", item.get("id"))
                    in relevant_chunk_ids
                }

                for chunk_id in sorted(relevant_chunk_ids):
                    if chunk_id in relevant_ranks:
                        print(f"  {chunk_id}: rank {relevant_ranks[chunk_id]}")
                    else:
                        print(f"  {chunk_id}: rank not available")
                print("=" * 60)
                # ----------------------------------------------------------------------------
                retrieval_context = EvaluationContext(
                    query_id=query_id,
                    question=question,
                    retrieved_results=results,
                    relevant_chunk_ids=set(
                        query_item["relevant_chunks"]
                    ),
                    top_k=configured_top_k,
                )

                retrieval_scores = run_configured_metrics(
                    context=retrieval_context,
                    evaluation_config=evaluation_config,
                    category="retrieval",
                )

                for name, score in retrieval_scores.items():
                    if score is not None:
                        metric_values["retrieval"][name].append(score)

            # -------------------------------------------------
            # 9. Run configured generation metrics
            # -------------------------------------------------
            if generation_metric_names:
                generation_latencies.append(
                    generation_result["generation_latency_seconds"]
                )

                ttft = generation_result.get("ttft_seconds")
                if ttft is not None:
                    ttft_values.append(ttft)

                tpot = generation_result.get("tpot_ms")
                if tpot is not None:
                    tpot_values.append(tpot)

                input_token_counts.append(
                    generation_result["input_tokens_count"]
                )
                output_token_counts.append(
                    generation_result["generated_tokens"]
                )

                generation_context = EvaluationContext(
                    query_id=query_id,
                    question=question,
                    answer=generation_result["answer"],
                    reference_answer=query_item["answer"],
                    retrieved_context=retrieved_context,
                )

                generation_scores = run_configured_metrics(
                    context=generation_context,
                    evaluation_config=evaluation_config,
                    category="generation",
                )

                for name, score in generation_scores.items():
                    if score is not None:
                        metric_values["generation"][name].append(score)

                print(
                    f"{query_id}: "
                    f"retrieval={retrieval_latency:.4f}s, "
                    f"generation="
                    f"{generation_result['generation_latency_seconds']:.4f}s"
                )

            else:
                print(
                    f"{query_id}: retrieval={retrieval_latency:.4f}s"
                )

            overall_latencies.append(
                time.perf_counter() - overall_start
            )

    finally:
        overall_peak_memory_mb = overall_memory_monitor.stop()

    # ---------------------------------------------------------
    # 10. Summarize scores and performance measurements
    # ---------------------------------------------------------
    version_metrics = {
        "git_commit": git_commit,
        "retrieval": {
            **summarize_metrics(metric_values["retrieval"]),
            "latency_s": round_metric(
                average(retrieval_latencies)
            ),
            "peak_memory_mb": round_metric(
                average([
                    value
                    for value in generation_peak_memory_values
                    if value is not None
                ])
            ),
        },
        "generation": {
            **summarize_metrics(metric_values["generation"]),
            "latency_s": round_metric(
                average(generation_latencies)
            ),
            "ttft_s": round_metric(average(ttft_values)),
            "tpot_ms": round_metric(average(tpot_values)),
            "input_tokens_count": round_metric(
                average(input_token_counts), 2
            ),
            "output_tokens_count": round_metric(
                average(output_token_counts), 2
            ),
            "peak_memory_mb": round_metric(
                average([
                    value
                    for value in generation_peak_memory_values
                    if value is not None
                ])
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

    # ---------------------------------------------------------
    # 11. Save results without deleting previous versions
    # ---------------------------------------------------------
    saved_results[version] = version_metrics

    OUTPUT_PATH.write_text(
        json.dumps(
            saved_results,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # 12. Print the evaluation summary
    # ---------------------------------------------------------
    print("\n========== Evaluation Summary ==========")
    print(f"Queries evaluated: {len(queries)}")
    print(f"Evaluation mode: {evaluation_config.get('mode', 'both')}")
    print("Retrieval metrics:", version_metrics["retrieval"])
    print("Generation metrics:", version_metrics["generation"])
    print("Overall metrics:", version_metrics["overall"])
    if evaluation_config.get("commit_changes", False):
        print(f"Git commit: {git_commit}")
    else:
        print("Git commit: skipped")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
