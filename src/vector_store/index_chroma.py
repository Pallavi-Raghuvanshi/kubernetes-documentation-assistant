from pathlib import Path
import json
import time
import chromadb
import numpy as np

from src.utils.helpers import load_chunks, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "kubernetes_minilm_new_embeddings.npy"
)

CHROMA_DB_PATH = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "kubernetes_docs"


# --------------------------------------------------
# Prepare ChromaDB records
# --------------------------------------------------

def prepare_records(chunks: list[dict]):
    """Extract text, stable IDs, and scalar metadata."""

    documents = []
    ids = []
    metadatas = []

    # Adjust these field names to match your chunk schema.
    text_fields = ("chunk_id", "chunk_size", "source", "text")

    for index, chunk in enumerate(chunks):
        text = next(
            (
                chunk[field]
                for field in text_fields
                if isinstance(chunk.get(field), str)
                and chunk[field].strip()
            ),
            None,
        )

        if text is None:
            raise ValueError(
                f"Could not find chunk text at record {index}. "
                f"Available fields: {list(chunk.keys())}"
            )

        # Prefer an existing stable identifier.
        chunk_id = chunk.get("chunk_id", chunk.get("id", index))

        ids.append(str(chunk_id))
        documents.append(text)

        # ChromaDB metadata must contain supported scalar values.
        metadata = {}

        for key, value in chunk.items():
            if key in text_fields or key in {"chunk_id", "id"}:
                continue

            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value

        metadatas.append(metadata or {"chunk_index": index})

    if len(set(ids)) != len(ids):
        raise ValueError(
            "Duplicate chunk IDs found. Generate unique, stable IDs "
            "before indexing."
        )

    return ids, documents, metadatas


# --------------------------------------------------
# Index embeddings in ChromaDB
# --------------------------------------------------

def main():
    start_time = time.time()

    # Load existing embeddings; do not regenerate them.
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"Embedding shape: {embeddings.shape}")

    # Load the corresponding chunks in their original order.
    chunks = load_chunks()

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"Chunk count ({len(chunks)}) does not match "
            f"embedding count ({embeddings.shape[0]}). "
            "Ensure both files use exactly the same chunk ordering."
        )

    ids, documents, metadatas = prepare_records(chunks)

    # Create persistent ChromaDB storage.
    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH)
    )

    # Explicitly configure cosine distance.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Insert or update records in batches.
    configs = load_config()
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