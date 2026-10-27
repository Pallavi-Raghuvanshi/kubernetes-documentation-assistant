# src\embeddings\original_minilm_vocab_embeddings.py
import json
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from src.utils.helpers import load_config, load_chunks


ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "kubernetes_minilm_embeddings.npy"

def build_embeddings() -> None:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    chunks = load_chunks()
    texts = [chunk["text"] for chunk in chunks]
    MODEL_NAME = config["retrieval"]["model"]

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
