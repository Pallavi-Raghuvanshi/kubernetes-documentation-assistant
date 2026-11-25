
# scripts/vector_store/index_parent_child.py
# -----------------------------------------------------------
# generates embeddings for child chunks and stores them in a separate ChromaDB collection

import time
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.helpers import (
    load_config,
    load_parent_child_records,
)


ROOT = Path(__file__).resolve().parents[2]


def main():

    start_time = time.perf_counter()

    config = load_config()

    model_name = config["retrieval"]["model"]
    similarity = config["vector_store"]["similarity_search"]
    batch_size = config["vector_store"]["batch_size"]

    settings = config["chunking"]["parent_child"]

    child_size = settings["child_size"]
    child_overlap = settings["child_overlap"]

    vector_store_path = (
        ROOT
        / "data"
        / "vector_store"
        / f"chromaDB_parent_child_{similarity}_{child_size}_{child_overlap}"
    )

    collection_name = "kubernetes_child_docs"

    parents, children = load_parent_child_records()

    if not parents or not children:
        raise ValueError("Parent or child records are empty.")

    parent_ids = {parent["parent_id"] for parent in parents}

    for child in children:
        if child["parent_id"] not in parent_ids:
            raise ValueError(
                f"Missing parent for child {child['child_id']}"
            )

    child_ids = [child["child_id"] for child in children]

    if len(child_ids) != len(set(child_ids)):
        raise ValueError("Duplicate child IDs found.")

    print("Loading embedding model:", model_name)

    model = SentenceTransformer(model_name)

    documents = [child["text"] for child in children]

    print("Generating child embeddings...")
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    if len(embeddings) != len(children):
        raise ValueError("Child and embedding counts do not match.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid values.")

    client = chromadb.PersistentClient(
        path=str(vector_store_path)
    )

    existing = [
        collection.name
        for collection in client.list_collections()
    ]

    if collection_name in existing:
        client.delete_collection(name=collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": similarity},
    )

    for start in range(0, len(children), batch_size):

        end = min(start + batch_size, len(children))

        batch = children[start:end]

        ids = [item["child_id"] for item in batch]

        batch_documents = [
            item["text"] for item in batch
        ]

        metadatas = [
            {
                "parent_id": item["parent_id"],
                "document_id": item["document_id"],
                "source": item["source"],
                "child_index": item["child_index"],
                "token_count": item["token_count"],
            }
            for item in batch
        ]

        collection.upsert(
            ids=ids,
            embeddings=embeddings[start:end].tolist(),
            documents=batch_documents,
            metadatas=metadatas,
        )

        print(f"Indexed {end}/{len(children)} children")

    if collection.count() != len(children):
        raise RuntimeError(
            "ChromaDB record count does not match child count."
        )

    print("\nParent-child indexing completed.")
    print("Parents:", len(parents))
    print("Children:", len(children))
    print("Collection:", collection_name)
    print("Stored records:", collection.count())
    print("Vector store:", vector_store_path)
    print(
        "Execution time:",
        f"{time.perf_counter() - start_time:.2f} seconds",
    )


if __name__ == "__main__":
    main()
