import json
import re
from pathlib import Path
from collections import Counter

import yaml
from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parents[2]
CORPUS_CONFIG_PATH = ROOT / "configs" / "corpus.yaml"
QUERIES_CONFIG_PATH = ROOT / "configs" / "queries.yaml"


def tokenize(text: str) -> list[str]:
    """Basic lowercase tokenization for the BM25 baseline."""
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks(path: Path) -> list[dict]:
    chunks = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    if not chunks:
        raise ValueError(f"No chunks found in {path}")

    return chunks


def main() -> None:
    with CORPUS_CONFIG_PATH.open(encoding="utf-8") as file:
        corpus_config = yaml.safe_load(file)

    with QUERIES_CONFIG_PATH.open(encoding="utf-8") as file:
        query_config = yaml.safe_load(file)

    corpus_name = corpus_config["corpus"]["name"]

    chunks_path = (
        ROOT / "data" / "processed" /
        corpus_name / "chunks.jsonl"
    )
    output_path = ROOT / "experiments" / "bm25" / "results.jsonl"

    chunks = load_chunks(chunks_path)
    queries = query_config["queries"]
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]

    bm25 = BM25Okapi(tokenized_corpus)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for query_item in queries:
            query_id = query_item["query_id"]
            query_text = query_item["query"]
            query_tokens = tokenize(query_text)
            # Compute BM25 scores for the query against each chunk
            scores = bm25.get_scores(query_tokens)
            # Rank chunks by score
            ranked_indices = sorted(
                range(len(scores)),
                key=lambda index: scores[index],
                reverse=True,
            )[:10] # select top 10 chunks

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

            record = {
                "query_id": query_id,
                "query": query_text,
                "retriever": "bm25",
                "top_k": 10,
                "results": results,
            }

            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            print(
                f"{query_id}: retrieved {len(results)} chunks"
            )

    print(f"\nQueries processed: {len(queries)}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Results saved: {output_path}")


if __name__ == "__main__":
    main()
