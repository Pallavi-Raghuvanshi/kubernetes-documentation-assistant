
# src/retrieval/parent_child_retrieval.py
# ----------------------------------------------------------
# retrieves child chunks from ChromaDB, maps them to their parents and removes duplicate parents

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.utils.helpers import (
    load_config,
    load_parent_child_records,
)


ROOT = Path(__file__).resolve().parents[2]


def load_parent_child_index():
    """Load the child vector index, embedding model and parent store."""

    config = load_config()

    similarity = config["vector_store"]["similarity_search"]
    model_name = config["retrieval"]["model"]

    settings = config["chunking"]["parent_child"]

    child_size = settings["child_size"]
    child_overlap = settings["child_overlap"]

    vector_store_path = (
        ROOT
        / "data"
        / "vector_store"
        / f"chromaDB_parent_child_{similarity}_{child_size}_{child_overlap}"
    )

    client = chromadb.PersistentClient(
        path=str(vector_store_path)
    )

    collection = client.get_collection(
        name="kubernetes_child_docs"
    )

    parents, _ = load_parent_child_records()

    parent_lookup = {
        parent["parent_id"]: parent
        for parent in parents
    }

    model = SentenceTransformer(model_name)

    return collection, model, parent_lookup


def search(
    query: str,
    gt_context: str,
    collection,
    model,
    parent_lookup: dict,
    child_top_k: int = 40,
    parent_top_k: int = 20,
) -> tuple[list[dict], list[str], list[str]]:
    """
    Retrieve child chunks and return their unique parent chunks.
    """

    query_tokens = model.tokenizer.tokenize(query)
    gt_context_tokens = model.tokenizer.tokenize(gt_context)

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    count = collection.count()

    if count == 0:
        return [], query_tokens, gt_context_tokens

    response = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(child_top_k, count),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    child_ids = response["ids"][0]
    child_documents = response["documents"][0]
    child_metadatas = response["metadatas"][0]
    distances = response["distances"][0]

    results = []
    seen_parent_ids = set()

    for child_rank, (
        child_id,
        child_document,
        metadata,
        distance,
    ) in enumerate(
        zip(
            child_ids,
            child_documents,
            child_metadatas,
            distances,
        ),
        start=1,
    ):

        parent_id = metadata["parent_id"]

        # Multiple children can map to the same parent.
        if parent_id in seen_parent_ids:
            continue

        parent = parent_lookup.get(parent_id)

        if parent is None:
            raise ValueError(
                f"Parent not found for child {child_id}: {parent_id}"
            )

        seen_parent_ids.add(parent_id)

        results.append({
            "rank": len(results) + 1,
            "chunk_id": parent_id,
            "parent_id": parent_id,
            "matched_child_id": child_id,
            "child_rank": child_rank,
            "source": parent["source"],
            "score": 1.0 - float(distance),
            "text": parent["text"],
            "chunk_tokens": model.tokenizer.tokenize(
                parent["text"]
            ),
        })

        if len(results) >= parent_top_k:
            break

    return results, query_tokens, gt_context_tokens
