# src/retrieval/dense_retrieval_minilm.py
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

from src.utils.helpers import load_config, load_chunks


def load_dense_index():
    import chromadb
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    client = chromadb.PersistentClient(path=str(root / "data" / "chroma_db_l2"))
    return client.get_collection(name="kubernetes_docs")

def search(
    query: str,
    collection,
    model,
    top_k: int,
) -> tuple[list[dict], list[str]]:

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_tokens = model.tokenizer.tokenize(query)

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    count = collection.count()

    if count == 0:
        return [], query_tokens

    response = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    results = []

    ids = response["ids"][0]
    documents = response["documents"][0]
    metadatas = response["metadatas"][0]
    distances = response["distances"][0]

    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances),
        start=1,
    ):
        results.append({
            "rank": rank,
            "chunk_id": metadata.get("chunk_id", chunk_id),
            "source": metadata.get("source", "unknown"),
            "score": 1.0 - float(distance),
            "text": document,
            "chunk_tokens": model.tokenizer.tokenize(document),
        })

    return results, query_tokens

def main():
    collection = load_dense_index()

    config = load_config()
    model_name = config["retrieval"]["model"]

    print(f"Loaded {collection.count()} chunks from ChromaDB.")

    model = SentenceTransformer(model_name)

    query = input("\nEnter a search query: ").strip()

    results, query_tokens = search(
        query=query,
        collection=collection,
        model=model,
        top_k=config["retrieval"]["top_k"],
    )

    print("\n--- Dense Retrieval Results ---")

    for result in results:
        print(f"\nRank: {result['rank']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Cosine similarity: {result['score']:.4f}")
        print(f"Text: {result['text'][:500]}")
        print(f"Query tokens: {query_tokens}")


if __name__ == "__main__":
    main()
