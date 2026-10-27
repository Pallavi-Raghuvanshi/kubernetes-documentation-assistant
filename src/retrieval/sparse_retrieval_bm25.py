# _4bm25_index.py
#----------------------------------------------------
# Run once per dataset
# to build bm25 index for it
import json
import pickle
from pathlib import Path
import yaml
from rank_bm25 import BM25Okapi

from src.tokenization.tokenization import tokenize
from src.utils.helpers import load_config, load_chunks


ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = ROOT / "data" / "index" / "bm25_index.pkl"


def get_chunks_path() -> Path:
    config = load_config()
    corpus_name = config["corpus"]["name"]
    return ROOT / "data" / "processed" / corpus_name / "chunks.jsonl"


def build_index() -> None:
    """Build and persist a BM25 index from processed chunks."""
    chunks_path = get_chunks_path()
    chunks = load_chunks(chunks_path)

    tokenized_corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    with INDEX_PATH.open("wb") as file:
        pickle.dump(
            {
                "chunks": chunks,
                "bm25": bm25,
            },
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(f"Indexed {len(chunks)} chunks.")
    print(f"BM25 index saved to: {INDEX_PATH}")


def load_index() -> tuple[list[dict], BM25Okapi]:
    """Load the persisted BM25 index."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {INDEX_PATH}. "
            "Run _4bm25_index.py first."
        )

    with INDEX_PATH.open("rb") as file:
        data = pickle.load(file)

    return data["chunks"], data["bm25"]


def search(
    query: str,
    chunks: list[dict],
    bm25: BM25Okapi,
    top_k: int = 10,
) -> tuple[list[dict], list[str]]:
    """Retrieve the top-k chunks and return query tokens."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:top_k]

    results = []

    for rank, index in enumerate(ranked_indices, start=1):
        chunk = chunks[index]

        results.append({
            "rank": rank,
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "score": float(scores[index]),
            "text": chunk["text"],
        })

    return results, query_tokens

if __name__ == "__main__":
    build_index()