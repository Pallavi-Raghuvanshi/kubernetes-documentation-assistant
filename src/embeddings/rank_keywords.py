# src/embeddings/rank_keywords.py

import csv
import math
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.utils.helpers import load_config


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data" / "candidate_keywords.txt"
OUTPUT_DIR = ROOT / "data" / "kubernetes"

RANKED_OUTPUT = OUTPUT_DIR / "ranked_keywords.csv"
SHORTLIST_OUTPUT = OUTPUT_DIR / "keyword_shortlist.csv"
HIGH_FREQUENCY_OUTPUT = OUTPUT_DIR / "high_frequency_keywords.csv"

# Ranking criteria
HIGH_FREQUENCY_THRESHOLD = 100
MIN_OCCURRENCES = 2
SHORTLIST_SIZE = 200

# Exclude likely UUIDs, hashes, and machine-generated strings.
UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}$"
)

HEX_HASH_PATTERN = re.compile(r"^[0-9a-fA-F]{32,}$")

BASE64LIKE_PATTERN = re.compile(
    r"^[A-Za-z0-9_-]{35,}={0,2}$"
)

# Long, opaque identifiers with mixed case and digits.
OPAQUE_ID_PATTERN = re.compile(
    r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[A-Za-z0-9_-]{30,}$"
)


def is_valid_candidate(keyword: str) -> bool:
    """Reject obvious noise while retaining useful technical terms."""

    if not keyword or len(keyword) > 80:
        return False

    if UUID_PATTERN.fullmatch(keyword):
        return False

    if HEX_HASH_PATTERN.fullmatch(keyword):
        return False

    if BASE64LIKE_PATTERN.fullmatch(keyword):
        return False

    if OPAQUE_ID_PATTERN.fullmatch(keyword):
        return False

    # Require an alphabetic character.
    if not any(char.isalpha() for char in keyword):
        return False

    return True


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Candidate file not found: {INPUT_PATH}"
        )

    config = load_config()
    model = SentenceTransformer(config["retrieval"]["model"])
    tokenizer = model.tokenizer

    candidates = []
    seen = set()

    with INPUT_PATH.open(
        "r", encoding="utf-8", newline=""
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "keyword",
            "occurrences",
            "subword_count",
        }

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                "Expected columns: keyword, occurrences, subword_count"
            )

        for row in reader:
            keyword = row["keyword"].strip()

            try:
                occurrences = int(row["occurrences"])
            except (ValueError, TypeError):
                continue

            if occurrences < MIN_OCCURRENCES:
                continue

            if not is_valid_candidate(keyword):
                continue

            # Prevent duplicate entries without merging case variants.
            if keyword in seen:
                continue
            seen.add(keyword)

            # Recalculate using the active tokenizer.
            subword_count = len(tokenizer.tokenize(keyword))

            if subword_count <= 1:
                continue

            # High-frequency terms qualify independently of fragmentation.
            high_frequency = (
                occurrences >= HIGH_FREQUENCY_THRESHOLD
            )

            # High-fragmentation terms are also retained.
            high_fragmentation = subword_count >= 6

            if not (high_frequency or high_fragmentation):
                continue

            # Two independent ranking signals.
            frequency_score = math.log1p(occurrences)
            fragmentation_score = subword_count - 1

            # Balanced score: frequency matters even for short terms.
            ranking_score = (
                0.6 * frequency_score
                + 0.4 * math.log1p(fragmentation_score)
            )

            candidates.append(
                {
                    "keyword": keyword,
                    "occurrences": occurrences,
                    "subword_count": subword_count,
                    "high_frequency": high_frequency,
                    "high_fragmentation": high_fragmentation,
                    "ranking_score": round(ranking_score, 4),
                }
            )

    # Overall ranking: frequency first, then fragmentation.
    candidates.sort(
        key=lambda item: (
            item["ranking_score"],
            item["occurrences"],
            item["subword_count"],
        ),
        reverse=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "keyword",
        "occurrences",
        "subword_count",
        "high_frequency",
        "high_fragmentation",
        "ranking_score",
    ]

    def save_csv(path, rows):
        with path.open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=fieldnames
            )
            writer.writeheader()
            writer.writerows(rows)

    # All eligible candidates.
    save_csv(RANKED_OUTPUT, candidates)

    # A manageable shortlist for manual review.
    save_csv(
        SHORTLIST_OUTPUT,
        candidates[:SHORTLIST_SIZE],
    )

    # Every candidate meeting the frequency threshold.
    high_frequency_candidates = [
        item for item in candidates
        if item["high_frequency"]
    ]

    high_frequency_candidates.sort(
        key=lambda item: (
            item["occurrences"],
            item["subword_count"],
        ),
        reverse=True,
    )

    save_csv(
        HIGH_FREQUENCY_OUTPUT,
        high_frequency_candidates,
    )

    print(f"Eligible candidates: {len(candidates)}")
    print(
        "High-frequency candidates: "
        f"{len(high_frequency_candidates)}"
    )
    print(
        "Top-ranked shortlist: "
        f"{min(SHORTLIST_SIZE, len(candidates))}"
    )

    print(f"\nAll candidates: {RANKED_OUTPUT}")
    print(f"Shortlist: {SHORTLIST_OUTPUT}")
    print(f"High-frequency terms: {HIGH_FREQUENCY_OUTPUT}")

    print("\nTop 25 candidates:")

    for item in candidates[:25]:
        print(
            f"{item['keyword'][:35]:35} "
            f"freq={item['occurrences']:5} "
            f"subwords={item['subword_count']:3} "
            f"high_freq={item['high_frequency']}"
        )


if __name__ == "__main__":
    main()