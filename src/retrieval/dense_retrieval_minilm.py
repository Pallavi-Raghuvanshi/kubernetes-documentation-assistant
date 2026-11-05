# src/retrieval/dense_retrieval_minilm.py
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

from src.utils.helpers import load_config, get_queries


def load_dense_index():
    import chromadb
    from pathlib import Path

    config = load_config()
    ss = config["vector_store"]["similarity_search"]
    chunk_size = config["chunking"]["chunk_size"]
    overlap = config["chunking"]["chunk_overlap"]
    ROOT = Path(__file__).resolve().parents[2]
    CHROMA_DB_PATH = ROOT / "data" / "vector_store" / f"chromaDB_{ss}_{chunk_size}_{overlap}"

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    return client.get_collection(name="kubernetes_docs")

def search(
    query: str,
    gt_context: str,
    collection,
    model,
    top_k: int,
) -> tuple[list[dict], list[str]]:

    query_tokens = model.tokenizer.tokenize(query)
    gt_context_tokens = model.tokenizer.tokenize(gt_context)

    query_embedding = model.encode(
        query,
        convert_to_numpy=True, # else converted to PyTorch tensor
        normalize_embeddings=True,
    ).tolist()

    count = collection.count()

    if count == 0:
        return [], query_tokens, gt_context_tokens

    response = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    results = []
    # searching one query at a time, so only accessing one result in the response
    ids = response["ids"][0]
    documents = response["documents"][0]
    metadatas = response["metadatas"][0]
    #-----------------------------------
    # print("\n--- Retrieved Metadata ---")

    # for chunk_id, metadata in zip(ids, metadatas):
    #     print(f"\nChunk ID: {chunk_id}")
    #     print(f"Metadata: {metadata}")
    #     print(f"Available keys: {metadata.keys() if metadata else None}")
    #-----------------------------------
    distances = response["distances"][0]

    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances),
        start=1,
    ):
        results.append({
            "rank": rank,
            "chunk_id": chunk_id,
            "source": metadata["source"],
            "score": 1.0 - float(distance),
            "text": document,
            "chunk_tokens": model.tokenizer.tokenize(document),
        })

    return results, query_tokens, gt_context_tokens

def main():    
    
    collection = load_dense_index()
    #---------------------------------------------
    data = collection.get(
        ids=["kubernetes_18055"],
        include=["metadatas", "documents"]
    )

    print("ID:", data["ids"])
    print("Metadata:", data["metadatas"])
    print("Document:", data["documents"][0][:500])
    #---------------------------------------------
    config = load_config()
    model_name = config["retrieval"]["model"]

    print(f"Loaded {collection.count()} chunks from ChromaDB.")

    model = SentenceTransformer(model_name)
    queries, start, end = get_queries()
    query = queries[-1]
    # query = input("\nEnter a search query: ").strip()
    
    results, query_tokens, gt_context_tokens = search(
        query=query["query"],
        gt_context=query["ground_truth_context"],
        collection=collection,
        model=model,
        top_k= 3 #config["retrieval"]["top_k"],
    )

    print("\n--- Dense Retrieval Results ---")
    print(f"Query: {query}")
    for result in results:
        print(f"\nRank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Source: {result['source']}")
        print(f"Query tokens: {query_tokens}")
        print(f"Text: {result['text'][:500]}")


if __name__ == "__main__":
    main()
