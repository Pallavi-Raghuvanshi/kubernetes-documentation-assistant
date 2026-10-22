
import json
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "kubernetes_minilm_embeddings.npy"
CHUNKS_PATH = ROOT / "data" / "processed" / "kubernetes" / "chunks.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks() -> list[dict]:
    chunks = []

    with CHUNKS_PATH.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    if not chunks:
        raise ValueError(f"No chunks found at {CHUNKS_PATH}")

    return chunks


def build_embeddings() -> None:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks()
    texts = [chunk["text"] for chunk in chunks]

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(EMBEDDINGS_PATH, embeddings)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved embeddings to: {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    build_embeddings()
