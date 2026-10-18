import time

import yaml

from src.retrieval._4bm25_index import BM25Index
from src.generation._6llm_inference import generate_answer

with open("queries.yaml", "r", encoding="utf-8") as file:
    queries = yaml.safe_load(file)["queries"]

with open("configs.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

def main():
    question = queries[0]["query"]  # Use the first query for demonstration

    if not question:
        print("Question cannot be empty.")
        return

    # Build the BM25 index and measure retrieval separately.
    index = BM25Index.from_corpus()

    retrieval_start = time.perf_counter()
    results = index.search(question, top_k=config["retrieval"]["top_k"])
    retrieval_latency = time.perf_counter() - retrieval_start

    # Prepare retrieved context with source metadata.
    context_parts = []

    for result in results:
        context_parts.append(
            f"[Source {result['rank']}] {result['source']}\n"
            f"{result['text']}"
        )

    context = "\n\n".join(context_parts)

    # Generate an answer using the retrieved chunks.
    generation_result = generate_answer(
        question=question,
        context=context,
        max_new_tokens=config["generation"]["max_new_tokens"],
    )

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
