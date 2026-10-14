
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
QUERIES_CONFIG_PATH = ROOT / "configs" / "queries.yaml"
RESULTS_PATH = ROOT / "experiments" / "bm25" / "results.jsonl"
OUTPUT_PATH = ROOT / "experiments" / "bm25" / "evaluation.json"


def main() -> None:
    with QUERIES_CONFIG_PATH.open(encoding="utf-8") as file:
        queries = yaml.safe_load(file)["queries"]

    results_by_query = {}

    with RESULTS_PATH.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                results_by_query[record["query_id"]] = record["results"]

    query_metrics = []

    for query_item in queries:
        query_id = query_item["query_id"]
        relevant_sources = set(query_item["relevant_sources"])
        results = results_by_query.get(query_id, [])

        if not relevant_sources:
            continue

        retrieved_sources = [
            result["source"] for result in results
        ]

        relevant_ranks = [
            rank
            for rank, source in enumerate(retrieved_sources, start=1)
            if source in relevant_sources
        ]

        relevant_retrieved = len(
            set(retrieved_sources) & relevant_sources
        )

        recall_at_10 = relevant_retrieved / len(relevant_sources)
        reciprocal_rank = (
            1 / relevant_ranks[0] if relevant_ranks else 0.0
        )
        hit_rate = 1.0 if relevant_ranks else 0.0

        query_metrics.append({
            "query_id": query_id,
            "recall_at_10": recall_at_10,
            "mrr": reciprocal_rank,
            "hit_rate": hit_rate,
        })

    if not query_metrics:
        raise ValueError("No queries could be evaluated.")

    count = len(query_metrics)

    summary = {
        "retriever": "bm25",
        "relevance_level": "source",
        "evaluated_queries": count,
        "recall_at_10": sum(
            item["recall_at_10"] for item in query_metrics
        ) / count,
        "mrr": sum(
            item["mrr"] for item in query_metrics
        ) / count,
        "hit_rate": sum(
            item["hit_rate"] for item in query_metrics
        ) / count,
        "per_query": query_metrics,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=== BM25 Evaluation ===")
    print(f"Queries evaluated: {count}")
    print(f"Recall@10: {summary['recall_at_10']:.4f}")
    print(f"MRR:       {summary['mrr']:.4f}")
    print(f"Hit Rate:  {summary['hit_rate']:.4f}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()