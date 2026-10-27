# src\embeddings\new_minilm_vocab_embeddings.py
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.helpers import load_config, load_chunks


ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "kubernetes_minilm_initialized_embeddings.npy"

def build_embeddings() -> None:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    chunks = load_chunks()
    texts = [chunk["text"] for chunk in chunks]
    model_name = config["retrieval"]["model"]
    model_path = ROOT / model_name

    print(f"Loading model: {model_path}")
    model = SentenceTransformer(str(model_path))

    print(f"Model vocabulary size: {model[0].auto_model.get_input_embeddings().num_embeddings}")
    print(f"Sentence embedding dimension: {model.get_sentence_embedding_dimension()}")

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