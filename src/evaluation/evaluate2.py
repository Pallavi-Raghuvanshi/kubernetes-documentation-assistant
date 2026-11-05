# src/evaluation/evaluate2.py
import json
import re
import threading
import time
from pathlib import Path
import webbrowser
from collections import Counter
import subprocess
import sys

from src.rag_pipeline import run_pipeline, load_retrieval_resources

from src.evaluation.evaluation_context import EvaluationContext
from src.evaluation.runner import run_metrics
from src.evaluation.utils import average, round_metric, PeakMemoryMonitor

from src.utils.helpers import load_config, get_queries
from src.models.load_models import load_model


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "experiments" / "evaluation_results.json"

# double-check locking pattern to ensure thread-safe lazy initialization of the model and tokenizer
tokenizer = None
model = None
_model_lock = threading.Lock()
    
def get_model():
    global tokenizer, model

    if tokenizer is None or model is None:
        with _model_lock:
            if tokenizer is None or model is None:
                tokenizer, model = load_model()

    return tokenizer, model

def calculate_text_proportion_metrics(
    gt_context_tokens: list[str],
    retrieved_results: list[dict],
    top_k: int,
) -> dict[str, float]:

    remaining_gt_tokens = list(gt_context_tokens)

    retrieved_tokens = [
        token
        for result in retrieved_results[:top_k]
        for token in result["chunk_tokens"]
    ]

    matching_tokens = 0

    for token in retrieved_tokens:
        if token in remaining_gt_tokens:
            remaining_gt_tokens.remove(token)
            matching_tokens += 1

    total_gt_tokens = len(gt_context_tokens)
    total_retrieved_tokens = len(retrieved_tokens)

    reference_text_coverage = (
        (total_gt_tokens - len(remaining_gt_tokens)) / total_gt_tokens
        if total_gt_tokens else 0.0
    )

    retrieved_text_relevance = (
        matching_tokens / total_retrieved_tokens
        if total_retrieved_tokens else 0.0
    )

    return {
        "reference_text_coverage_at_k": reference_text_coverage,
        "retrieved_text_relevance_at_k": retrieved_text_relevance,
    }

def get_selected_metrics(
    evaluation_config: dict,
    category: str,
) -> list[str]:

    mode = evaluation_config["mode"].lower()

    if mode not in {"retrieval", "both"}:
        raise ValueError(
            f"Invalid evaluation mode: {mode!r}. "
            "Expected retrieval or both."
        )

    if mode == "retrieval":
        if category == "retrieval":
            metrics_config = evaluation_config["metrics"]
            return metrics_config.get("retrieval", [])
        return []

    else:  # mode == "both"
        metrics_config = evaluation_config["metrics"]
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

def calculate_chunk_relevance(
    chunk_text: str,
    ground_truth_context: str,
) -> float:
    """
    Calculate the proportion of chunk tokens that overlap with the ground-truth context.
    """
    chunk_tokens = re.findall(r"\b\w+\b", chunk_text.lower())
    ground_truth_tokens = set(re.findall(r"\b\w+\b", ground_truth_context.lower()))

    if not chunk_tokens:
        return 0.0

    matching_tokens = sum(1 for token in chunk_tokens if token in ground_truth_tokens)
    return matching_tokens / len(chunk_tokens)

def main() -> None:
    # ---------------------------------------------------------
    # 1. Load configuration and benchmark queries
    # ---------------------------------------------------------
    config = load_config()
    evaluation_config = config["evaluation"]

    retrieval_metric_names = get_selected_metrics(evaluation_config, "retrieval")
    generation_metric_names = get_selected_metrics(evaluation_config, "generation")

    queries, _, _= get_queries()

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
    version = evaluation_config["version_name"]
    print(f"version before capture: {version}")

    query_output_dir = ROOT / "experiments" / "query_outputs"
    query_output_dir.mkdir(parents=True, exist_ok=True)

    query_output_path = query_output_dir / f"{version}.jsonl"
    query_output_path.write_text("", encoding="utf-8")

    print("Starting evaluation...")
    print(f"Evaluation mode: {evaluation_config['mode']}")
    print(f"Queries selected: {len(queries)}")
    print(f"Version name: {version}")

    # ---------------------------------------------------------
    # 4. Read retrieval and generation configuration
    # ---------------------------------------------------------
    retrieval_config = config["retrieval"]
    generation_config = config["generation"]

    configured_top_k = int(retrieval_config["top_k"])
    max_new_tokens = int(generation_config["max_new_tokens"])

    # ---------------------------------------------------------
    # 5. Initialize metric and performance accumulators
    # ---------------------------------------------------------
    metric_values = {
        "retrieval": {name: [] for name in retrieval_metric_names},
        "generation": {name: [] for name in generation_metric_names},
    }

    reference_coverage_values = []
    retrieved_relevance_values = []

    retrieval_latencies = []
    retrieval_peak_memory_values = []
    generation_peak_memory_values = []

    generation_latencies = []
    ttft_values = []
    tpot_values = []
    input_token_counts = []
    output_token_counts = []

    overall_latencies = []

    retriever = retrieval_config["retriever"].lower()
    print(f"Retriever: {retriever}", flush=True)

    index = load_retrieval_resources(config)

    overall_memory_monitor = PeakMemoryMonitor()
    overall_memory_monitor.start()

    try:
        # -----------------------------------------------------
        # 7. Evaluate each benchmark query
        # -----------------------------------------------------
        for query in queries:
            query_id = query["query_id"]
            question = query["query"]
            overall_start = time.perf_counter()

            pipeline_result = run_pipeline(
                question=question,
                gt_context = query["ground_truth_context"],
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
                memory_monitor_factory=PeakMemoryMonitor
            )
            results = pipeline_result["results"]
            retrieved_context = pipeline_result["context"]
            gt_context_tokens = pipeline_result["gt_context_tokens"]
            generation_result = pipeline_result["generation_result"]
            query_tokens = pipeline_result["query_tokens"]

            ground_truth_context = str(query["ground_truth_context"])
            text_proportion_scores = calculate_text_proportion_metrics(
                gt_context_tokens=gt_context_tokens,
                retrieved_results=results,
                top_k=configured_top_k,
            )
            # print("\n--- TEXT COVERAGE DEBUG ---")
            # print("GT token count:", len(gt_context_tokens))
            # print(
            #     "Retrieved token count:",
            #     sum(len(r["chunk_tokens"]) for r in results[:configured_top_k])
            # )
            # print("Calculated scores:", text_proportion_scores)
            # print("---------------------------\n")
            coverage = text_proportion_scores["reference_text_coverage_at_k"]
            relevance = text_proportion_scores["retrieved_text_relevance_at_k"]

            if coverage is not None:
                reference_coverage_values.append(coverage)

            if relevance is not None:
                retrieved_relevance_values.append(relevance)

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
            if retrieval_metric_names and "relevant_chunks" in query:
                # Preserve the existing chunk-relevance metrics for legacy query files.
                retrieval_context = EvaluationContext(
                    query_id=query_id,
                    question=question,
                    retrieved_results=results,
                    relevant_chunk_ids=set(query["relevant_chunks"]),
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
                generation_latencies.append(generation_result["generation_latency_seconds"])

                ttft = generation_result.get("ttft_seconds")
                if ttft is not None:
                    ttft_values.append(ttft)

                tpot = generation_result.get("tpot_ms")
                if tpot is not None:
                    tpot_values.append(tpot)

                input_token_counts.append(generation_result["input_tokens_count"])
                output_token_counts.append(generation_result["generated_tokens"])

                generation_context = EvaluationContext(
                    query_id=query_id,
                    question=question,
                    answer=generation_result["answer"],
                    reference_answer=query.get("ground_truth_answer", query.get("answer", "")),
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
                print(f"{query_id}: retrieval={retrieval_latency:.4f}s")

            overall_latencies.append(time.perf_counter() - overall_start)
            source_files = [str(query["source_file"])]

            # Save results for this query.
            query_record = {
                "query_id": query_id,
                "query": question,
                "query_tokens": query_tokens,

                "ground_truth_context": ground_truth_context,
                "gt_context_tokens": gt_context_tokens,
                "ground_truth_answer": query["ground_truth_answer"],

                "source_file": str(query["source_file"]),

                "retrieval_latency_seconds": round(retrieval_latency, 6),
                "retrieval_peak_memory_mb": pipeline_result["retrieval_peak_memory_mb"],

                "text_proportions": text_proportion_scores,

                "retrieved_chunk_ids": [
                    item["chunk_id"]
                    for item in results[:configured_top_k]
                ],

                "relevant_chunk_ranks": {
                    item["chunk_id"]: {
                        "rank": rank,
                        "relevance_score": calculate_chunk_relevance(
                            chunk_text=item["text"],
                            ground_truth_context=ground_truth_context,
                        ),
                    }
                    for rank, item in enumerate(results[:configured_top_k], start=1)
                },
                "retrieved_results": results[:configured_top_k],

                "generation": (
                    {
                        "answer": generation_result["answer"],
                        "latency_seconds": generation_result["generation_latency_seconds"],
                    }
                    if generation_result is not None
                    else None
                ),
            }

            with query_output_path.open("a", encoding="utf-8") as output_file:
                output_file.write(json.dumps(query_record, ensure_ascii=False, default=str) + "\n")

    finally:
        overall_peak_memory_mb = overall_memory_monitor.stop()
    # ---------------------------------------------------------
    # 10. Summarize scores and performance measurements
    # ---------------------------------------------------------
    version_metrics = {
        "query_count": len(queries),
        "retrieval": {
            **summarize_metrics(metric_values["retrieval"]),
            "reference_text_coverage_at_k": round_metric(average(reference_coverage_values)),
            "retrieved_text_relevance_at_k": round_metric(average(retrieved_relevance_values)),
            "latency_s": round_metric(average(retrieval_latencies)),
            "peak_memory_mb": round_metric(
                average([
                    value
                    for value in retrieval_peak_memory_values
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
            "latency_s": round_metric(average(overall_latencies)),
            "peak_memory_mb": round_metric(overall_peak_memory_mb),
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
    print(f"Queries evaluated : {len(queries)}")
    print(f"Evaluation mode   : {evaluation_config.get('mode', 'both')}")
    print("Retrieval metrics :", version_metrics["retrieval"])
    print("Generation metrics:", version_metrics["generation"])
    print("Overall metrics   :", version_metrics["overall"])
    print(f"\nResults saved to          : {OUTPUT_PATH}")
    print(f"Per-query results saved to: {query_output_path}")

    html_report_path = query_output_path.with_suffix(".html")
    report_script = ROOT / "scripts" / "view_query_results.py"

    subprocess.run(
        [
            sys.executable,
            str(report_script),
            "--input",
            str(query_output_path.resolve()),
            "--output",
            str(html_report_path.resolve()),
        ],
        check=True,
        cwd=ROOT,
    )

    # Open in the currently active Edge window
    subprocess.run(
        [
            "cmd",
            "/c",
            "start",
            "msedge",
            str(html_report_path.resolve()),
        ],
        check=True,
    )
if __name__ == "__main__":
    main()
