# src/embeddings/extract_keywords.py
import json
import re
from collections import Counter
from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.utils.helpers import load_chunks, load_config


# Match technical identifiers containing uppercase letters,
# lowercase letters, digits, or hyphens.
IDENTIFIER_PATTERN = re.compile(
    r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b"
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "data" / "processed" / "kubernetes" / "candidate_keywords.txt"

def main():
    config = load_config()
    model = SentenceTransformer(config["retrieval"]["model"])
    tokenizer = model.tokenizer

    keyword_counts = Counter()

    chunks = load_chunks()

    for chunk in chunks:
        text = chunk["text"]

        for match in IDENTIFIER_PATTERN.finditer(text):
            word = match.group()

            if not (
                any(c.isupper() for c in word)
                or any(c.isdigit() for c in word)
            ):
                continue

            tokens = model.tokenizer.tokenize(word)

            if len(tokens) > 1:
                keyword_counts[word] += 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        file.write("keyword,occurrences,subword_count\n")

        for word, count in keyword_counts.most_common():
            subword_count = len(tokenizer.tokenize(word))
            file.write(f"{word},{count},{subword_count}\n")

    print(f"Candidate keywords saved to: {OUTPUT_PATH}")
    print(f"Unique candidates: {len(keyword_counts)}")

    print("\nTop 30 candidates:")
    for word, count in keyword_counts.most_common(30):
        print(
            f"{word:35} "
            f"occurrences={count:5} "
            f"subwords={len(tokenizer.tokenize(word))}"
        )


if __name__ == "__main__":
    main()