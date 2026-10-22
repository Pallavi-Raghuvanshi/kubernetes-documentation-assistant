# src.evaluate.evaluate.py
import json
import subprocess
import time
from pathlib import Path
import yaml

from src.retrieval.__4bm25_index import load_index
from src.rag_pipeline import run_pipeline
from src.evaluation.generation_metrics import (
    evaluate_generation,
    sentence_integrity_rate,
)
from src.evaluation.retrieval_metrics import (
    average,
    round_metric,
    PeakMemoryMonitor,
    evaluate_retrieval,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs.yaml"
QUERIES_PATH = ROOT / "queries.yaml"
OUTPUT_PATH = ROOT / "experiments" / "evaluation_results.json"

EVAL_TOP_K = 10

def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}

def get_queries(config: dict) -> list[dict]:
    """Load queries from queries.yaml, falling back to configs.yaml."""
    if QUERIES_PATH.exists():
        queries = load_yaml(QUERIES_PATH).get("queries", [])
    else:
        queries = config.get("queries", [])

    if not queries:
        raise ValueError("No benchmark queries found.")

    return queries

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
    evaluation_config = config["evaluation"]
    commit_changes = evaluation_config["commit_changes"]
    version = evaluation_config["version_name"]
    print('starting evaluation...')
    if commit_changes:
        git_commit = commit_code_changes()
        print(f"Project Version: {version}")
        print(f"Git commit: {git_commit}")
    else:
        git_commit = ""
        print("Skipping Git commit.")

    retrieval_config = config.get("retrieval", {})
    generation_config = config.get("generation", {})

    generation_top_k = int(
        retrieval_config.get("top_k", EVAL_TOP_K)
    )
    max_new_tokens = int(
        generation_config.get("max_new_tokens", 160)
    )

    # Load the prebuilt BM25 index once and reuse it for all queries.
    chunks, bm25 = load_index()

    retrieval_latencies = []
    retrieval_peak_memory_values = []
    generation_peak_memory_values = []

    recall_values = []
    precision_values = []
    mrr_values = []
    hit_rate_values = []

    generation_latencies = []
    input_token_counts = []
    output_token_counts = []

    faithfulness_values = []
    answer_completeness_values = []
    hallucination_rate_values = []
    sir_values = []

    overall_latencies = []

    # Measures overall evaluation-query processing after index creation.
    overall_memory_monitor = PeakMemoryMonitor()
    overall_memory_monitor.start()

    for query_item in queries:
        query_id = query_item["query_id"]
        question = query_item["query"]

        overall_start = time.perf_counter()

        # Run the RAG pipeline and monitor its memory usage.    
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
            index=(chunks, bm25),
            memory_monitor_factory=PeakMemoryMonitor,
            run_generation=evaluation_mode != "retrieval",
        )

        context = pipeline_result["context"]

        results = pipeline_result["results"]

        #----------------------------------------------
        print("\n--- Retrieval Debug: Q001 ---")
        print("Expected relevant chunks:", query_item["relevant_chunks"])

        for result in results[:10]:
            print(
                f"Rank: {result['rank']} | "
                f"Chunk ID: {result['chunk_id']} | "
                f"Score: {result['score']:.4f} | "
                f"Source: {result['source']}"
            )
        #----------------------------------------------

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
        retrieval_scores = evaluate_retrieval(
            results=results,
            relevant_chunks=set(query_item["relevant_chunks"]),
            top_k=EVAL_TOP_K,
        )

        if retrieval_scores["recall_at_k"] is not None:
            recall_values.append(retrieval_scores["recall_at_k"])
            precision_values.append(retrieval_scores["precision_at_k"])
            mrr_values.append(retrieval_scores["mrr"])
            hit_rate_values.append(retrieval_scores["hit_rate_at_k"])

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

            print("\n--- Question ---")
            print(question)

            print("\n--- Retrieved Context ---")
            print(pipeline_result["context"])

            print("\n--- Reference Answer ---")
            print(query_item["answer"])
            print("\n--- Generated Answer ---")
            print(generation_result["answer"])
            print("--- End Generated Answer ---\n")

            judge_scores = evaluate_generation(
                question=question,
                answer=generation_result["answer"],
                context=pipeline_result["context"],
                reference_answer=query_item["answer"],
                max_new_tokens=max_new_tokens,
            )

            if judge_scores["faithfulness"] is not None:
                faithfulness_values.append(
                    judge_scores["faithfulness"]
                )

            if judge_scores["answer_completeness"] is not None:
                answer_completeness_values.append(
                    judge_scores["answer_completeness"]
                )

            if judge_scores["hallucination_rate"] is not None:
                hallucination_rate_values.append(
                    judge_scores["hallucination_rate"]
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
            "faithfulness": round_metric(
                average(faithfulness_values)
            ),
            "answer_completeness": round_metric(
                average(answer_completeness_values)
            ),
            "hallucination_rate": round_metric(
                average(hallucination_rate_values)
            ),
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
        print(f"                Queries: {len(queries)}")
        print(f"           Faithfulness: {version_metrics['generation']['faithfulness']}")
        print(f"    Answer completeness: {version_metrics['generation']['answer_completeness']}")
        print(f"     Hallucination rate: {version_metrics['generation']['hallucination_rate']}")
        print(f"     Sentence integrity: {version_metrics['generation']['sentence_integrity_rate']}")
        print(f"     Generation latency: {version_metrics['generation']['latency_s']} s")
        print(f"           Input tokens: {version_metrics['generation']['input_tokens_count']}")
        print(f"          Output tokens: {version_metrics['generation']['output_tokens_count']}")
        print(f" Peak generation memory: {version_metrics['generation']['peak_memory_mb']} MB")
        print(f"             Git commit: {git_commit}")
        print(f"                  Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
