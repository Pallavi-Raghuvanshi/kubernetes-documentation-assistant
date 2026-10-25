# src/retrieval/hybrid_retrieval.py

from src.retrieval.sparse_retrieval_bm25 import search as bm25_search
from src.retrieval.dense_retrieval_minilm import search as dense_search


def search(
    query: str,
    chunks: list[dict],
    bm25,
    embeddings,
    model,
    top_k: int = 10,
    candidate_k: int = 100,
    rrf_k: int = 60,
) -> list[dict]:
    """Fuse BM25 and dense rankings using Reciprocal Rank Fusion."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if candidate_k < 1:
        raise ValueError("candidate_k must be at least 1.")

    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1.")

    candidate_k = min(candidate_k, len(chunks))

    bm25_results = bm25_search(
        query=query,
        chunks=chunks,
        bm25=bm25,
        top_k=candidate_k,
    )

    dense_results = dense_search(
        query=query,
        chunks=chunks,
        embeddings=embeddings,
        model=model,
        top_k=candidate_k,
    )

    # Reciprocal Rank Fusion: combine rankings, not raw scores.
    fused = {}

    for results in (bm25_results, dense_results):
        for result in results:
            chunk_id = result["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "source": result["source"],
                    "text": result["text"],
                    "score": 0.0,
                }

            fused[chunk_id]["score"] += (
                1.0 / (rrf_k + result["rank"])
            )

    ranked_results = sorted(
        fused.values(),
        key=lambda item: item["score"],
        reverse=True,
    )[:top_k]

    for rank, result in enumerate(ranked_results, start=1):
        result["rank"] = rank

    return ranked_results
