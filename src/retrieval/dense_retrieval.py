# src/retrieval/dense_retrieval.py
import json
from pathlib import Path
import yaml

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs.yaml"
CHUNKS_PATH = ROOT / "data" / "processed" / "kubernetes" / "chunks.jsonl"
EMBEDDINGS_PATH = ROOT / "data" / "embeddings" / "kubernetes_minilm_embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_yaml(path: Path) -> dict:
    """Load a YAML file."""
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}

def load_chunks() -> list[dict]:
    chunks = []

    with CHUNKS_PATH.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    if not chunks:
        raise ValueError(f"No chunks found at {CHUNKS_PATH}")

    return chunks


def load_embeddings() -> np.ndarray:
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {EMBEDDINGS_PATH}. "
            "Run src.embeddings.embeddings first."
        )

    return np.load(EMBEDDINGS_PATH)


def load_dense_index():
    chunks = load_chunks()
    embeddings = load_embeddings()

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunk count ({len(chunks)}) does not match "
            f"embedding count ({len(embeddings)})."
        )

    return chunks, embeddings


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
    config = load_yaml(CONFIG_PATH)

    print(f"Loaded {len(chunks)} chunks.")
    print(f"Embedding shape: {embeddings.shape}")

    model = SentenceTransformer(MODEL_NAME)

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
