# src/embeddings/new_minilm_vocab_embeddings.py
from pathlib import Path
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from src.utils.helpers import load_chunks


ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "kubernetes_minilm_initialized_embeddings.npy"

def build_embeddings() -> None:
    start_time = time.perf_counter()
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()
    texts = [chunk["text"] for chunk in chunks]
    model_path = ROOT / "models" / "kubernetes_updated_minilm_initialized"

    print(f"Loading model: {model_path}")
    model = SentenceTransformer(str(model_path))

    model_vocab_size = (
        model[0].auto_model.get_input_embeddings().num_embeddings
    )

    print(f"Model vocabulary size: {model_vocab_size}")
    print(f"Sentence embedding dimension: {model.get_embedding_dimension()}")

    tokenizer_path = model_path / "0_Transformer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if model_vocab_size != len(tokenizer):
        raise ValueError(
            "Model and tokenizer vocabulary sizes do not match: "
            f"{model_vocab_size} != {len(tokenizer)}"
        )

    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            f"Expected {len(chunks)} embeddings, got {embeddings.shape[0]}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Generated embeddings contain NaN or infinite values.")

    np.save(EMBEDDINGS_PATH, embeddings)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved embeddings to: {EMBEDDINGS_PATH}")
    
    elapsed_time = time.perf_counter() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"Total execution time: {elapsed_time / 60:.2f} minutes")


if __name__ == "__main__":
    build_embeddings()