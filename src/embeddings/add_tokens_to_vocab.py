import csv
from pathlib import Path
from transformers import AutoTokenizer

from src.utils.helpers import load_config

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data" / "ranked_keywords.csv"
OUTPUT_DIR = ROOT / "models" / "kubernetes_tokenizer_updated_vocab"

config = load_config()
MODEL_NAME = config["retrieval"]["model"]

TOP_FREQUENCY = 150
TOP_FRAGMENTATION = 50


def main():
    with INPUT_PATH.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        records = list(reader)

    for row in records:
        row["occurrences"] = int(row["occurrences"])
        row["subword_count"] = int(row["subword_count"])

    # Select top 150 by frequency.
    top_frequency = sorted(
        records,
        key=lambda row: row["occurrences"],
        reverse=True,
    )[:TOP_FREQUENCY]

    # Select top 50 by subword count.
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Original vocabulary size: {original_vocab_size}")
    print(f"Top-frequency candidates: {len(frequency_tokens)}")
    print(f"New tokens added from frequency group: {len(frequency_new)}")
    print(f"Top-fragmentation candidates: {len(fragmentation_tokens)}")
    print(
        "New tokens added from fragmentation group: "
        f"{len(fragmentation_new)}"
    )
    print(f"Final vocabulary size: {len(tokenizer)}")
    print(f"Tokenizer saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()