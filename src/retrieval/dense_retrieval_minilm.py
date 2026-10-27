# src/retrieval/dense_retrieval_minilm.py
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.helpers import load_config, load_chunks

ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_PATH = ROOT / "data" / "embeddings" / "kubernetes_minilm_initialized_embeddings.npy"

def load_embeddings(path: Path = EMBEDDINGS_PATH) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Generate initialized-model embeddings first using "
            f"src.embeddings.new_minilm_vocab_embeddings."
        )
    return np.load(path)

def load_dense_index(embeddings_path: Path = EMBEDDINGS_PATH):
    return load_chunks(), load_embeddings(embeddings_path)

def search(
    query: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int,
) -> list[dict]:
    
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    
    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if embeddings.shape[1] != query_embedding.shape[0]:
        raise ValueError(
            f"Embedding dimension mismatch: document embeddings have "
            f"{embeddings.shape[1]} dimensions, but query embedding has "
            f"{query_embedding.shape[0]}. Ensure both use the same model."
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

    model = SentenceTransformer(config["retrieval"]["model"])
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
