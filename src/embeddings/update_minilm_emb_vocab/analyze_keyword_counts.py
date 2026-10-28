# src/embeddings/analyze_keyword_counts.py
import csv
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = ROOT / "data" / "ranked_keywords.csv"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Ranked keyword file not found: {INPUT_PATH}"
        )

    total_keywords = 0
    subword_counts = {3: 0, 10: 0, 15: 0}
    frequency_thresholds = {10: 0, 20: 0, 30: 0}

    with INPUT_PATH.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {"keyword", "occurrences", "subword_count"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                "Input CSV must contain columns: "
                "keyword, occurrences, subword_count"
            )

        for row in reader:
            try:
                frequency = int(row["occurrences"])
                subwords = int(row["subword_count"])
            except (ValueError, TypeError):
                continue

            total_keywords += 1

            for threshold in frequency_thresholds:
                if frequency > threshold:
                    frequency_thresholds[threshold] += 1

            for threshold in subword_counts:
                if subwords > threshold:
                    subword_counts[threshold] += 1

    # Load keyword records for overlap and noise analysis.
    with INPUT_PATH.open("r", encoding="utf-8", newline="") as file:
        records = list(csv.DictReader(file))

    # Define candidate groups.
    high_frequency = {
        row["keyword"].strip()
        for row in records
        if int(row["occurrences"]) > 100
    }

    high_fragmentation = {
        row["keyword"].strip()
        for row in records
        if int(row["subword_count"]) > 20
    }

    # Calculate overlap.
    overlap = high_frequency & high_fragmentation
    frequency_only = high_frequency - high_fragmentation
    fragmentation_only = high_fragmentation - high_frequency

    print("\n--- Candidate Group Overlap ---")
    print(f"Frequency > 100: {len(high_frequency)}")
    print(f"Subwords > 20: {len(high_fragmentation)}")
    print(f"In both groups: {len(overlap)}")

    print("\n--- Overlapping Candidates ---")
    for keyword in sorted(overlap):
        print(keyword)

    # Identify suspicious strings for manual inspection.
    uuid_pattern = re.compile(
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{12}$"
    )

    hex_hash_pattern = re.compile(r"^[0-9a-fA-F]{32,}$")
    long_opaque_pattern = re.compile(r"^[A-Za-z0-9_-]{40,}={0,2}$")

    suspicious = []

    for row in records:
        keyword = row["keyword"].strip()

        if (
            uuid_pattern.fullmatch(keyword)
            or hex_hash_pattern.fullmatch(keyword)
            or long_opaque_pattern.fullmatch(keyword)
        ):
            suspicious.append(row)

    print("\n--- Suspicious Candidates for Manual Review ---")
    print(f"Potentially suspicious strings: {len(suspicious)}")

    for row in suspicious[:100]:
        print(
            f"{row['keyword'][:60]} | "
            f"frequency={row['occurrences']} | "
            f"subwords={row['subword_count']}"
        )

    # Save suspicious candidates for review.
    suspicious_path = INPUT_PATH.parent / "suspicious_keywords.csv"

    with suspicious_path.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(suspicious)

    print(f"\nSuspicious candidates saved to: {suspicious_path}")

    print(f"Input file: {INPUT_PATH}")
    print(f"Total keywords analyzed: {total_keywords}\n")

    for threshold in (30, 20, 10):
        print(
            f"Keywords with frequency > {threshold}: "
            f"{frequency_thresholds[threshold]}"
        )
    print()
    for threshold in (15, 10, 3):
        print(
            f"Keywords with > {threshold} subwords: "
            f"{subword_counts[threshold]}"
        )


if __name__ == "__main__":
    main()
