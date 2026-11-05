# scripts/vector_store/verify_chroma.py
from pathlib import Path
import chromadb
import numpy as np

from src.utils.helpers import load_config


ROOT = Path(__file__).resolve().parents[2]
COLLECTION_NAME = "kubernetes_docs"


def main():
    configs = load_config()

    ss = configs["vector_store"]["similarity_search"]
    chunk_size = configs["chunking"]["chunk_size"]
    overlap = configs["chunking"]["chunk_overlap"]

    CHROMA_DB_PATH = ROOT / "data" / "vector_store" / f"chromaDB_{ss}_{chunk_size}_{overlap}"

    print(f"Database path: data / vector_store / chromaDB_{ss}_{chunk_size}_{overlap}")

    # Connect to the existing database.
    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    # Verify collection count.
    print("\n--- Collection verification ---")
    print("Collection:", COLLECTION_NAME)
    print("Stored records:", collection.count())

    # Retrieve one sample record.
    sample = collection.get(
        ids=["kubernetes_00000"],
        include=["documents", "metadatas", "embeddings"],
    )

    if not sample["ids"]:
        raise ValueError("Sample record not found.")

    print("\n--- Sample record verification ---")
    print("ID:", sample["ids"][0])
    print("Document:", sample["documents"][0][:500])
    print("Metadata:", sample["metadatas"][0])

    embedding = np.asarray(sample["embeddings"])

    print("\n--- Embedding verification ---")
    print("Embedding shape:", embedding.shape)
    print("Contains NaN:", np.isnan(embedding).any())
    print("Contains Inf:", np.isinf(embedding).any())

    print("\nVerification completed.")


if __name__ == "__main__":
    main()