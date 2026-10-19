
import time

import yaml

from src.retrieval._4bm25_index import BM25Index
from src.generation._6llm_inference import generate_answer

with open("queries.yaml", "r", encoding="utf-8") as file:
    queries = yaml.safe_load(file)["queries"]

with open("configs.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)



def run_pipeline(
    question: str,
    config: dict,
    index=None,
    memory_monitor_factory=None,
    run_generation: bool = True,
) -> dict:
    if not question:
        raise ValueError("Question cannot be empty.")

    if index is None:
        index = BM25Index.from_corpus()

    retrieval_memory_monitor = (
        memory_monitor_factory()
        if memory_monitor_factory is not None
        else None
    )

    if retrieval_memory_monitor:
        retrieval_memory_monitor.start()

    retrieval_start = time.perf_counter()
    results = index.search(
        question,
        top_k=config["retrieval"]["top_k"],
    )
    retrieval_latency = time.perf_counter() - retrieval_start

    retrieval_peak_memory_mb = (
        retrieval_memory_monitor.stop()
        if retrieval_memory_monitor
        else None
    )

    context_parts = []

    for result in results:
        context_parts.append(
            f"[Source {result['rank']}] {result['source']}\n"
            f"{result['text']}"
        )

    context = "\n\n".join(context_parts)

    if not run_generation:
        return {
            "results": results,
            "generation_result": None,
            "retrieval_latency_seconds": retrieval_latency,
            "retrieval_peak_memory_mb": retrieval_peak_memory_mb,
            "generation_peak_memory_mb": None,
        }

    generation_memory_monitor = (
        memory_monitor_factory()
        if memory_monitor_factory is not None
        else None
    )

    if generation_memory_monitor:
        generation_memory_monitor.start()

    generation_result = generate_answer(
        question=question,
        context=context,
        max_new_tokens=config["generation"]["max_new_tokens"],
    )

    generation_peak_memory_mb = (
        generation_memory_monitor.stop()
        if generation_memory_monitor
        else None
    )

    return {
        "results": results,
        "generation_result": generation_result,
        "retrieval_latency_seconds": retrieval_latency,
        "retrieval_peak_memory_mb": retrieval_peak_memory_mb,
        "generation_peak_memory_mb": generation_peak_memory_mb,
    }



def main():
    question = queries[0]["query"]  # Use the first query for demonstration

    if not question:
        print("Question cannot be empty.")
        return

    pipeline_result = run_pipeline(question, config)

    results = pipeline_result["results"]
    generation_result = pipeline_result["generation_result"]
    retrieval_latency = pipeline_result["retrieval_latency_seconds"]

    print("\n--- Answer ---")
    print(generation_result["answer"])

    print("\n--- Retrieved Sources ---")
    seen_sources = set()

    for result in results:
        if result["source"] not in seen_sources:
            seen_sources.add(result["source"])
            print(f"- {result['source']}")

    print("\n--- Latency ---")
    print(f"Retrieval: {retrieval_latency:.4f} seconds")
    print(
        "Generation: "
        f"{generation_result['generation_latency_seconds']:.3f} seconds"
    )


if __name__ == "__main__":
    main()
