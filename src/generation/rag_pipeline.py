
import time

from src.retrieval._4bm25_index import BM25Index
from src.generation._6llm_inference import generate_answer


def main():
    question = input("Ask a Kubernetes question: ").strip()

    if not question:
        print("Question cannot be empty.")
        return

    # Build the BM25 index and measure retrieval separately.
    index = BM25Index.from_corpus()

    retrieval_start = time.perf_counter()
    results = index.search(question, top_k=10)
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
        max_new_tokens=160,
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
