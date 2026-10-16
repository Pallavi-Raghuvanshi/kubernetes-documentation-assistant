import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
QUERIES_CONFIG_PATH = ROOT / "configs" / "queries.yaml"
RESULTS_PATH = ROOT / "experiments" / "bm25" / "results.jsonl"
OUTPUT_PATH = ROOT / "experiments" / "bm25" / "evaluation.json"
RETRIEVAL_CONFIG_PATH = ROOT / "configs" / "retrieval.yaml"


def main() -> None:
    with QUERIES_CONFIG_PATH.open(encoding="utf-8") as file:
        queries = yaml.safe_load(file)["queries"]

    with RETRIEVAL_CONFIG_PATH.open(encoding="utf-8") as file:
        retrieval_config = yaml.safe_load(file)["retrieval"]

    retriever = retrieval_config["default_retriever"]
    top_k = int(retrieval_config["top_k"])

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
        results = results_by_query.get(query_id, [])[:top_k]

        if not relevant_sources:
            continue

        retrieved_sources = [result["source"] for result in results]

        relevant_ranks = [
            rank
            for rank, source in enumerate(retrieved_sources, start=1)
            if source in relevant_sources
        ]

        relevant_retrieved = len(set(retrieved_sources) & relevant_sources)

        recall_at_k = relevant_retrieved / len(relevant_sources)

        precision_at_k = sum(
            source in relevant_sources for source in retrieved_sources # produces boolean valuesx
            ) / top_k

        reciprocal_rank = (1 / relevant_ranks[0] if relevant_ranks else 0.0) # get rank of 1st relevant retrieved chunk
        hit_rate = 1.0 if relevant_ranks else 0.0

        query_metrics.append({
            "query_id": query_id,
            f"precision_at_{top_k}": precision_at_k,
            f"recall_at_{top_k}": recall_at_k,
            "mrr": reciprocal_rank,
            "hit_rate": hit_rate,
        })

    if not query_metrics:
        raise ValueError("No queries could be evaluated.")

    count = len(query_metrics)

    summary = {
        "retriever": retriever,
        "relevance_level": "source",
        "evaluated_queries": count,
        f"precision_at_{top_k}": sum(
            item[f"precision_at_{top_k}"]
            for item in query_metrics
        ) / count,
        f"recall_at_{top_k}": sum(
            item[f"recall_at_{top_k}"]
            for item in query_metrics
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

    print(f"=== {retriever.upper()} Evaluation ===")
    print(f"Queries evaluated: {count}")
    print(
        f"Precision@{top_k}: "
        f"{summary[f'precision_at_{top_k}']:.4f}"
    )
    print(
        f"Recall@{top_k}:    "
        f"{summary[f'recall_at_{top_k}']:.4f}"
    )
    print(f"MRR:               {summary['mrr']:.4f}")
    print(f"Hit Rate:          {summary['hit_rate']:.4f}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()