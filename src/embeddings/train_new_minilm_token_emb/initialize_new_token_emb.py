# src/embeddings/train_new_minilm_token_emb/initialize_new_token_emb.py
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from src.utils.helpers import load_config


ROOT = Path(__file__).resolve().parents[3]

TOKENIZER_PATH = ROOT / "models" / "kubernetes_upd_tokenizer_vocab"
OUTPUT_DIR = ROOT / "models" / "kubernetes_updated_minilm_initialized"


def main() -> None:
    config = load_config()
    model_name = config["retrieval"]["model"]

    # 1. Load the original tokenizer and expanded tokenizer.
    print(f"Loading original tokenizer: {model_name}")
    original_tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading expanded tokenizer: {TOKENIZER_PATH}")
    expanded_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    expanded_vocab = expanded_tokenizer.get_vocab()

    # 2. Identify tokens that were not in the original vocabulary.
    original_vocab_size = len(original_tokenizer)

    new_tokens = [
        token
        for token, token_id in expanded_vocab.items()
        if token_id >= original_vocab_size
    ]

    print(f"Original vocabulary size: {len(original_tokenizer)}")
    print(f"Expanded vocabulary size: {len(expanded_tokenizer)}")
    print(f"New tokens identified: {len(new_tokens)}")

    if not new_tokens:
        raise ValueError(
            "No new tokens found. Check the expanded tokenizer."
        )

    if len(new_tokens) != len(set(new_tokens)):
        raise ValueError("Duplicate new tokens detected.")

    # Validate token IDs before resizing the model
    new_token_ids = [expanded_vocab[token] for token in new_tokens]

    if len(new_token_ids) != len(set(new_token_ids)):
        raise ValueError("Multiple new tokens share the same token ID.")

    if any(token_id < original_vocab_size for token_id in new_token_ids):
        raise ValueError("A new token has an ID inside the original vocabulary.")

    if any(token_id >= len(expanded_tokenizer) for token_id in new_token_ids):
        raise ValueError("A new token ID is outside the expanded vocabulary.")

    # 3. Load the original pretrained SentenceTransformer model.
    print(f"Loading pretrained model: {model_name}")
    model = SentenceTransformer(model_name)

    transformer = model[0]
    transformer_model = transformer.auto_model

    # 4. Resize the embedding matrix for the expanded vocabulary.
    transformer_model.resize_token_embeddings(len(expanded_tokenizer))

    embedding_layer = transformer_model.get_input_embeddings()

    if embedding_layer.num_embeddings != len(expanded_tokenizer):
        raise ValueError(
            "Embedding matrix size does not match tokenizer vocabulary: "
            f"{embedding_layer.num_embeddings} != {len(expanded_tokenizer)}"
        )

    embedding_weights = embedding_layer.weight

    # 5. Initialize each new token using its original subword embeddings.
    initialized_tokens = []
    uninitialized_tokens = []

    with torch.no_grad():
        for token in new_tokens:
            # Reject obvious scraping artifacts.
            if (
                len(token) >= 80
                and all(c.isalnum() or c in "_+/=-" for c in token)
            ):
                uninitialized_tokens.append(token)
                continue

            if "<" in token or ">" in token:
                uninitialized_tokens.append(token)
                continue

            # Tokenize the term using the ORIGINAL tokenizer.
            subword_ids = original_tokenizer.encode(
                token,
                add_special_tokens=False,
            )

            # Avoid initializing from an unknown token alone.
            if (
                not subword_ids
                or original_tokenizer.unk_token_id in subword_ids
            ):
                uninitialized_tokens.append(token)
                continue

            # Calculate the mean of the original subword embeddings.
            subword_ids_tensor = torch.tensor(
                subword_ids,
                dtype=torch.long,
                device=embedding_weights.device,
            )

            subword_embeddings = embedding_weights[
                subword_ids_tensor
            ]

            mean_embedding = subword_embeddings.mean(dim=0)

            # Find the new token's ID in the expanded vocabulary.
            new_token_id = expanded_vocab[token]

            # Copy the mean embedding into its new embedding row.
            embedding_weights[new_token_id].copy_(mean_embedding)

            initialized_tokens.append(
                {
                    "token": token,
                    "token_id": new_token_id,
                    "original_subwords": original_tokenizer.convert_ids_to_tokens(
                        subword_ids
                    ),
                }
            )

    # 6. Stop before saving if any token failed initialization.
    if uninitialized_tokens:
        print("\nTokens not initialized:")
        for token in uninitialized_tokens:
            print(f"  - {token}")

        raise ValueError(
            f"{len(uninitialized_tokens)} tokens could not be initialized. "
            "Clean the vocabulary and rerun initialization."
        )

    # Save only after every new token has been initialized.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(OUTPUT_DIR))

    # Save the expanded tokenizer where SentenceTransformer loads it.
    expanded_tokenizer.save_pretrained(OUTPUT_DIR)

    transformer_dir = OUTPUT_DIR / "0_Transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)

    # Save the expanded tokenizer.
    expanded_tokenizer.save_pretrained(transformer_dir)

    # Save the Transformer model configuration.
    transformer_model.config.vocab_size = len(expanded_tokenizer)
    transformer_model.config.save_pretrained(transformer_dir)

    print(f"Expanded tokenizer saved to: {transformer_dir}")

    # Save a record of the initialization for reproducibility.
    metadata = {
        "base_model": model_name,
        "original_vocab_size": len(original_tokenizer),
        "expanded_vocab_size": len(expanded_tokenizer),
        "new_tokens_count": len(new_tokens),
        "initialized_count": len(initialized_tokens),
        "uninitialized_count": len(uninitialized_tokens),
        "initialized_tokens": initialized_tokens,
        "uninitialized_tokens": uninitialized_tokens,
    }

    metadata_path = OUTPUT_DIR / "initialization_metadata.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    # 7. Report the results.
    print("\nInitialization complete.")
    print(f"Initialized tokens: {len(initialized_tokens)}")
    print(f"Tokens needing attention: {len(uninitialized_tokens)}")
    print(f"Model embedding rows: {embedding_layer.num_embeddings}")
    print(f"Tokenizer vocabulary size: {len(expanded_tokenizer)}")

    assert (embedding_layer.num_embeddings == len(expanded_tokenizer)), "Model and tokenizer vocabulary sizes do not match."
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    main()