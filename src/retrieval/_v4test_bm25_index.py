
import time

from retrieval._4bm25_index import BM25Index


def main():
    start = time.perf_counter()
    index = BM25Index.from_corpus()
    index_build_seconds = time.perf_counter() - start

    query = "How do Kubernetes Deployments manage application replicas?"

    start = time.perf_counter()
    results = index.search(query, top_k=10)
    retrieval_seconds = time.perf_counter() - start

    print(f"Indexed chunks: {len(index.chunks)}")
    print(f"Index build time: {index_build_seconds:.4f} seconds")
    print(f"Retrieval time: {retrieval_seconds:.4f} seconds")
    print(f"\nQuery: {query}\n")

    for result in results:
        print(
            f"Rank {result['rank']} | "
            f"Score {result['score']:.4f} | "
            f"Source: {result['source']}"
        )
        print(result["text"][:250].replace("\n", " "))
        print("-" * 60)


if __name__ == "__main__":
    main()
