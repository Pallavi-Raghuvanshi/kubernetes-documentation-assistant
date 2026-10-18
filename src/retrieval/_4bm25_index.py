
import json
import re
from pathlib import Path

import yaml
from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parents[2]


def tokenize(text: str) -> list[str]:
    """Text split into lowercase alphanumeric tokens, ignoring punctuation and whitespace."""
    return re.findall(
        r"\b\w+\b",  # word boundary + alphanumeric characters + word boundary
        text.lower()
    ) 


def load_chunks(path: Path) -> list[dict]:
    chunks = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip(): # Ignore empty lines
                chunks.append(json.loads(line))

    if not chunks:
        raise ValueError(f"No chunks found in {path}")

    return chunks


def get_chunks_path() -> Path:
    config_path = ROOT / "configs.yaml"

    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    corpus_name = config["corpus"]["name"]
    return ROOT / "data" / "processed" / corpus_name / "chunks.jsonl"


class BM25Index:
    """Build a BM25 index once for the corpus."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
        # build doc stats for each unique token in the corpus
        # doc stats: idf, term frequency, doc length etc.
        self.bm25 = BM25Okapi(tokenized_corpus) 

    @classmethod
    def from_corpus(cls):
        chunks = load_chunks(get_chunks_path())
        return cls(chunks)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        scores = self.bm25.get_scores(tokenize(query))
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results = []

        for rank, index in enumerate(ranked_indices, start=1):
            chunk = self.chunks[index]

            results.append({
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "score": float(scores[index]),
                "text": chunk["text"],
            })

        return results
