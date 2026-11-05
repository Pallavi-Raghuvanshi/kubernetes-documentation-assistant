# scripts/vector_store/index_chroma.py
from pathlib import Path
import json
import time
import chromadb
import numpy as np

from src.utils.helpers import load_chunks, load_config


ROOT = Path(__file__).resolve().parents[2]

configs = load_config()
chunk_size = configs["chunking"]["chunk_size"]
chunk_overlap = configs["chunking"]["chunk_overlap"]
EMBEDDINGS_PATH = (ROOT / "data" / "embeddings" / f"kubernetes_minilm_emb_{chunk_size}_{chunk_overlap}.npy")

COLLECTION_NAME = "kubernetes_docs"


# --------------------------------------------------
# Prepare ChromaDB records
# --------------------------------------------------

def prepare_records(chunks: list[dict]):
    """Prepare document text, IDs, and metadata for ChromaDB."""

    documents = []
    ids = []
    metadatas = []

    for index, chunk in enumerate(chunks):

        # Extract actual documentation text.
        text = next(
            (
                chunk[field]
                for field in ("text", "page_content", "content", "document")
                if isinstance(chunk.get(field), str)
                and chunk[field].strip()
            ),
            None,
        )

        if text is None:
            raise ValueError(
                f"Missing document text at record {index}. "
                f"Available fields: {list(chunk.keys())}"
            )

        # Extract the chunk ID.
        chunk_id = chunk.get("chunk_id", chunk.get("id", index))

        if str(chunk_id) == text:
            raise ValueError(
                f"Chunk {chunk_id} contains its ID as document text."
            )

        ids.append(str(chunk_id))
        documents.append(text)

        # Preserve existing metadata, including source if available.
        metadata = {
            key: value
            for key, value in chunk.items()
            if key not in {
                "text",
                "page_content",
                "content",
                "document",
                "chunk_id",
                "id",
            }
            and isinstance(value, (str, int, float, bool))
        }

        metadata["chunk_index"] = index
        metadatas.append(metadata)

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate chunk IDs found.")

    return ids, documents, metadatas

# --------------------------------------------------
# Index embeddings in ChromaDB
# --------------------------------------------------

def main():
    start_time = time.time()
    configs = load_config()
    ss = configs["vector_store"]["similarity_search"]
    chunk_size = configs["chunking"]["chunk_size"]
    overlap = configs["chunking"]["chunk_overlap"]
    CHROMA_DB_PATH = ROOT / "data" / "vector_store" / f"chromaDB_{ss}_{chunk_size}_{overlap}"

    # Load existing embeddings; do not regenerate them.
    embeddings = np.load(EMBEDDINGS_PATH)
    #------------------------------------------------
    chunks = load_chunks()

    if not chunks:
        raise ValueError("No chunks were loaded.")

    print("\n--- Data alignment check ---")
    print("Embedding file:", EMBEDDINGS_PATH)
    print("Number of chunks:", len(chunks))
    print("Embedding shape:", embeddings.shape)

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"Chunk and embedding count mismatch: "
            f"{len(chunks)} chunks vs {embeddings.shape[0]} embeddings."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    print("Alignment: PASSED")

    print("\n--- First source chunk ---")
    print(chunks[0])

    ids, documents, metadatas = prepare_records(chunks)
    print("\n--- First prepared record ---")
    print("ID:", ids[0])
    print("Document:", documents[0][:300])
    print("Metadata:", metadatas[0])

    # return
    #------------------------------------------------

    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH)
    )

    if COLLECTION_NAME in [
        collection.name for collection in client.list_collections()
    ]:
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": configs["vector_store"]["similarity_search"]},
    )

    # Insert or update records in batches.
    batch_size = configs["vector_store"]["batch_size"]

    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))

        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

        print(f"Indexed {end}/{len(chunks)} chunks")

    print("\nIndexing completed.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Stored records: {collection.count()}")
    print(f"Database location: {CHROMA_DB_PATH}")
    print(f"Execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()