
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import yaml
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "corpus.yaml"


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    corpus = config["corpus"]
    settings = config["chunking"]

    corpus_name = corpus["name"]
    source_urls = set(corpus["source_urls"])
    tokenizer_name = settings["tokenizer"]
    chunk_size = int(settings["chunk_size"])

    chunks_path = (
        ROOT / "data" / "processed" /
        corpus_name / "chunks.jsonl"
    )

    if not chunks_path.is_file():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}"
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    records = []
    errors = []
    seen_ids = set()
    source_counts = Counter()
    token_counts = []

    with chunks_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(
                    f"Line {line_number}: Invalid JSON ({error})"
                )
                continue

            records.append(record)

            required_fields = {
                "chunk_id", "source", "tokenizer",
                "token_count", "text"
            }

            missing = required_fields - record.keys()
            if missing:
                errors.append(
                    f"Line {line_number}: Missing fields {sorted(missing)}"
                )
                continue

            chunk_id = record["chunk_id"]
            source = record["source"]
            text = record["text"]

            if not isinstance(chunk_id, str) or not chunk_id:
                errors.append(
                    f"Line {line_number}: Invalid chunk_id"
                )
            elif chunk_id in seen_ids:
                errors.append(
                    f"Line {line_number}: Duplicate chunk_id {chunk_id}"
                )
            else:
                seen_ids.add(chunk_id)

            if not isinstance(text, str) or not text.strip():
                errors.append(
                    f"Line {line_number}: Empty or invalid text"
                )
                continue

            if source not in source_urls:
                errors.append(
                    f"Line {line_number}: Unexpected source {source}"
                )

            parsed_url = urlparse(source)
            if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
                errors.append(
                    f"Line {line_number}: Invalid source URL {source}"
                )

            actual_tokens = len(
                tokenizer.encode(text, add_special_tokens=False)
            )
            token_counts.append(actual_tokens)
            source_counts[source] += 1

            if record["tokenizer"] != tokenizer_name:
                errors.append(
                    f"Line {line_number}: Unexpected tokenizer"
                )

            if not isinstance(record["token_count"], int):
                errors.append(
                    f"Line {line_number}: token_count must be an integer"
                )
            elif record["token_count"] != actual_tokens:
                errors.append(
                    f"Line {line_number}: Stored token count "
                    f"{record['token_count']} != actual {actual_tokens}"
                )

            if actual_tokens > chunk_size:
                errors.append(
                    f"Line {line_number}: {actual_tokens} tokens "
                    f"exceeds chunk_size={chunk_size}"
                )

    print("=== Chunk Validation Report ===")
    print(f"Corpus: {corpus_name}")
    print(f"Chunks found: {len(records)}")
    print(f"Unique chunk IDs: {len(seen_ids)}")
    print(f"Configured chunk size: {chunk_size}")

    if token_counts:
        print(f"Minimum tokens: {min(token_counts)}")
        print(f"Maximum tokens: {max(token_counts)}")
        print(
            f"Average tokens: "
            f"{sum(token_counts) / len(token_counts):.2f}"
        )

    print("\nChunks per source:")
    for source in sorted(source_urls):
        print(f"  {source}: {source_counts[source]}")

    print(f"\nValidation errors: {len(errors)}")

    for error in errors[:50]:
        print(f"  - {error}")

    if len(errors) > 50:
        print(f"  ... and {len(errors) - 50} more errors")

    if errors:
        raise SystemExit("Validation failed. Fix the reported errors.")

    print("\nValidation passed.")


if __name__ == "__main__":
    main()