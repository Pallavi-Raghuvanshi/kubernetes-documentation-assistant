# src/retrieval/dense_retrieval_minilm.py
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.helpers import load_config, load_chunks


def load_embeddings() -> np.ndarray:
    ROOT = Path(__file__).resolve().parents[2]
    config = load_config()
    MODEL_NAME = config["retrieval"]["model"]
    if MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2":
        EMBEDDINGS_PATH = ROOT / "data" / "embeddings" / "kubernetes_minilm_embeddings.npy"
    elif MODEL_NAME == "models/kubernetes_updated_minilm_initialized":
        EMBEDDINGS_PATH = ROOT / "data" / "embeddings" / "kubernetes_minilm_initialized_embeddings.npy"
    else:
        raise ValueError(f"Invalid model selected in configs: {MODEL_NAME}")
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Generate initialized-model embeddings first using "
            f"src.embeddings.new_minilm_vocab_embeddings."
        )
    return np.load(EMBEDDINGS_PATH)


def load_dense_index():
    return load_chunks(), load_embeddings()


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
    
    # generate tokens
    tokenizer = model.tokenizer
    query_tokens = tokenizer.tokenize(query)

    # generate embeddings
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
        chunk_tokens = tokenizer.tokenize(chunk["text"])
        results.append(
            {
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "score": float(scores[index]),
                "text": chunk["text"],
                "chunk_tokens": chunk_tokens,
            }
        )
    return results, query_tokens


def main():
    chunks, embeddings = load_dense_index()
    config = load_config()
    MODEL_NAME = config["retrieval"]["model"]
    print(f"Loaded {len(chunks)} chunks.")
    print(f"Embedding shape: {embeddings.shape}")
    model = SentenceTransformer(MODEL_NAME)
    query = input("\nEnter a search query: ").strip()
    results, query_tokens = search(query, chunks, embeddings, model, config["retrieval"]["top_k"])

    print("\n--- Dense Retrieval Results ---")

    for result in results:
        print(f"\nRank: {result['rank']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Source: {result['source']}")
        print(f"Cosine similarity: {result['score']:.4f}")
        print(f"Text: {result['text'][:500]}")
        print(f"Query embeddings: {query_tokens}")


if __name__ == "__main__":
    main()
