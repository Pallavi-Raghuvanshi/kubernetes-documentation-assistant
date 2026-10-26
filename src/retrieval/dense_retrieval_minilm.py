# src/retrieval/dense_retrieval.py
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.helpers import load_config, load_chunks

ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_PATH = ROOT / "data" / "embeddings" / "kubernetes_minilm_embeddings.npy"


def load_embeddings() -> np.ndarray:
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {EMBEDDINGS_PATH}. "
            "Run src.embeddings.embeddings first."
        )
    return np.load(EMBEDDINGS_PATH)

def load_dense_index():
    return load_chunks(), load_embeddings()

def search(
    query: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    model: str,
    top_k: int,
) -> list[dict]:
    
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    
    model_org = model.split('/')[0]
    query_embedding = model_org.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Embeddings are normalized, so dot product gives cosine similarity.
    scores = embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []

    for rank, index in enumerate(top_indices, start=1):
        chunk = chunks[int(index)]
        results.append(
            {
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "score": float(scores[index]),
                "text": chunk["text"],
            }
        )
    return results


def main():
    chunks, embeddings = load_dense_index()
    config = load_config()

    print(f"Loaded {len(chunks)} chunks.")
    print(f"Embedding shape: {embeddings.shape}")

    model = config["retrieval"]["model"]
    query = input("\nEnter a search query: ").strip()
    results = search(query, chunks, embeddings, model, config["retrieval"]["top_k"])

    print("\n--- Dense Retrieval Results ---")

    for result in results:
        print(f"\nRank: {result['rank']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Source: {result['source']}")
        print(f"Cosine similarity: {result['score']:.4f}")
        print(f"Text: {result['text'][:500]}")


if __name__ == "__main__":
    main()
