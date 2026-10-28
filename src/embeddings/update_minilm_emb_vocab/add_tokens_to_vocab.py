# src/embeddings/update_minilm_emb_vocab/add_tokens_to_vocab.py
import csv
import re
from pathlib import Path
from transformers import AutoTokenizer

from src.utils.helpers import load_config

ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = ROOT / "data" / "ranked_keywords.csv"
OUTPUT_DIR = ROOT / "models" / "kubernetes_upd_tokenizer_vocab"

config = load_config()
MODEL_NAME = config["retrieval"]["model"]

TOP_FREQUENCY = 500
TOP_FRAGMENTATION = 500

def main():
    with INPUT_PATH.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        records = list(reader)

    for row in records:
        row["occurrences"] = int(row["occurrences"])
        row["subword_count"] = int(row["subword_count"])

    # Select by frequency.
    top_frequency = sorted(
        records,
        key=lambda row: row["occurrences"],
        reverse=True,
    )[:TOP_FREQUENCY]

    # Select by subword count.
    top_fragmentation = sorted(
        records,
        key=lambda row: row["subword_count"],
        reverse=True,
    )[:TOP_FRAGMENTATION]

    # Preserve each selection separately.
    frequency_tokens = list(dict.fromkeys(
        row["keyword"].strip()
        for row in top_frequency
        if row["keyword"].strip()
    ))

    fragmentation_tokens = list(dict.fromkeys(
        row["keyword"].strip()
        for row in top_fragmentation
        if row["keyword"].strip()
    ))

    # Load the original tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    original_vocab_size = len(tokenizer)

    # Add each group separately, reporting its additions.
    existing_tokens = set(tokenizer.get_vocab())

    frequency_new = [
        token for token in frequency_tokens
        if token not in existing_tokens
    ]
    tokenizer.add_tokens(frequency_new)
    existing_tokens.update(frequency_new)

    fragmentation_new = [
        token for token in fragmentation_tokens
        if token not in existing_tokens
    ]
    tokenizer.add_tokens(fragmentation_new)

    print(f"Original vocabulary size: {original_vocab_size}")
    print(f"New tokens added from frequency group: {len(frequency_new)}")
    print(f"New tokens added from fragmentation group: {len(fragmentation_new)}")
    print(f"Final vocabulary size: {len(tokenizer)}")
    print(f"Actual new vocabulary entries: {len(tokenizer) - original_vocab_size}")
    print(f"Tokenizer saved to: {OUTPUT_DIR}")

    added_tokens = [
        token
        for token, token_id in tokenizer.get_vocab().items()
        if token_id >= original_vocab_size
    ]

    print("\nAdded tokens for manual inspection:")
    for token in added_tokens:
        print(f"  - {token}")
    # Final validation check
    if len(added_tokens) != len(set(added_tokens)):
        raise ValueError("Duplicate added tokens detected.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__": 
    main()