# scripts/test_reranker.py
from src.reranker.cross_encoder import CrossEncoderReranker

reranker = CrossEncoderReranker()

query = "How does Kubernetes handle pod scheduling?"

# Replace these examples with actual candidates returned
# by your existing BM25 or dense retriever.
candidates = [
    {
        "chunk_id": "chunk_001",
        "text": "The scheduler assigns Pods to suitable Nodes.",
        "retrieval_score": 0.82,
    },
    {
        "chunk_id": "chunk_002",
        "text": "Kubernetes Services expose network endpoints.",
        "retrieval_score": 0.76,
    },
    {
        "chunk_id": "chunk_003",
        "text": "The kube-scheduler selects a Node for a Pod.",
        "retrieval_score": 0.79,
    },
]

results = reranker.rerank(
    query=query,
    candidates=candidates,
    top_k=2,
)

for rank, result in enumerate(results, start=1):
    print(
        f"Rank: {rank} | "
        f"ID: {result['chunk_id']} | "
        f"Score: {result['reranker_score']:.4f}"
    )
