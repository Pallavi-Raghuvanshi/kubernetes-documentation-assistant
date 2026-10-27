# src/embeddings/update_minilm_emb_vocab/add_tokens_to_vocab.py
import csv
import re
from pathlib import Path
from transformers import AutoTokenizer

from src.utils.helpers import load_config

ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = ROOT / "data" / "ranked_keywords.csv"
OUTPUT_DIR = ROOT / "models" / "kubernetes_updated_tokenizer_vocab"

config = load_config()
MODEL_NAME = config["retrieval"]["model"]

TOP_FREQUENCY = 150
TOP_FRAGMENTATION = 50


def is_valid_keyword(keyword: str) -> bool:
    """Reject empty keywords and obvious encoded or markup artifacts."""
    keyword = keyword.strip()

    if not keyword:
        return False

    # # Reject long Base64-like strings.
    # if len(keyword) >= 80 and re.fullmatch(
    #     r"[A-Za-z0-9_+/=-]+", keyword
    # ):
    #     return False

    # Reject HTML tags and Mermaid diagram syntax.
    if re.search(r"<[^>]+>", keyword):
        return False

    if re.search(
        r"\b(graph\s+(LR|RL|TB|TD|BT)|flowchart\s+(LR|RL|TB|TD|BT))\b",
        keyword,
        re.IGNORECASE,
    ):
        return False

    # Reject tokens containing letters and digits with at least 3 digit groups separated by non-digit characters.
    digit_groups = re.findall(r"\d+", keyword)
    
    if (
        len(digit_groups) >= 3
        and re.search(r"[A-Za-z]", keyword)
        and re.search(r"\D", keyword)
    ):
        return False

    # Reject UUIDs.
    if re.fullmatch(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{12}",
        keyword,
    ):
        return False

    # Reject long hexadecimal hashes.
    if re.fullmatch(r"[0-9a-fA-F]{32,}", keyword):
        return False

    # Reject Base64/JWT-like encoded strings.
    if (
        len(keyword) >= 16
        and re.fullmatch(r"[A-Za-z0-9_+/=-]+", keyword)
        and (
            re.search(r"[+/=]", keyword)
            or (
                re.search(r"[a-z]", keyword)
                and re.search(r"[A-Z]", keyword)
                and re.search(r"\d", keyword)
            )
        )
    ):
        return False

    # Reject runtime-generated Kubernetes resource names.
    if re.fullmatch(
        r"(?:nginx-deployment|my-nginx|hello-world|snowflake|"
        r"wordpress-mysql|redis-follower|dra-example-driver-kubeletplugin|"
        r"fluentd-cloud-logging-kubernetes-node|kube-proxy-kubernetes-node)"
        r"-[a-z0-9]+(?:-[a-z0-9]+)+",
        keyword,
        re.IGNORECASE,
    ):
        return False

    # Reject generated pod/job names with numeric identifiers and suffixes.
    if re.fullmatch(
        r"(?:job-backoff-limit-per-index-failindex|"
        r"job-pod-failure-policy-failjob|"
        r"job-pod-failure-policy-config)-[a-z0-9-]+",
        keyword,
        re.IGNORECASE,
    ):
        return False

    return True


def main():
    with INPUT_PATH.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        records = list(reader)

    for row in records:
        row["occurrences"] = int(row["occurrences"])
        row["subword_count"] = int(row["subword_count"])
    # Filter valid tokens
    records = [
        row for row in records
        if is_valid_keyword(row.get("keyword", ""))
    ]
    print(f"Valid keyword records: {len(records)}")

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

    invalid_tokens = [
        token for token in added_tokens
        if not is_valid_keyword(token)
    ]

    if invalid_tokens:
        raise ValueError(
            "Invalid tokens found in the added vocabulary: "
            f"{invalid_tokens}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__": 
    main()