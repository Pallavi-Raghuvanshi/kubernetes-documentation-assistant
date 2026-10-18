
import json
import sqlite3
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs.yaml"

CORPUS_NAME = "daiict_faculty_data"

DB_PATH = (
    ROOT
    / "data"
    / "raw"
    / "faculty_data"
    / "daiict_faculty_data.db"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "faculty_data"
    / CORPUS_NAME
)

OUTPUT_PATH = OUTPUT_DIR / "chunks.jsonl"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """Split text into overlapping word-based chunks."""
    words = text.split()

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]

        if not chunk_words:
            break

        chunks.append({
            "start_word": start,
            "word_count": len(chunk_words),
            "text": " ".join(chunk_words),
        })

        if start + chunk_size >= len(words):
            break

    return chunks


def main() -> None:
    config = load_config()
    chunk_config = config["chunking"]

    chunk_size = int(chunk_config["chunk_size"])
    overlap = int(chunk_config["chunk_overlap"])

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if not 0 <= overlap < chunk_size:
        raise ValueError(
            "Overlap must be >= 0 and smaller than chunk size."
        )

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_chunks = 0

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            "SELECT * FROM Faculty ORDER BY id"
        ).fetchall()

        with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
            for row in rows:
                faculty = dict(row)

                text_parts = []

                for field, value in faculty.items():
                    if field.lower() in {
                        "id",
                        "email",
                        "phone",
                        "contact",
                        "contact_number",
                        "mobile",
                    }:
                        continue

                    if value is None or not str(value).strip():
                        continue

                    field_name = field.replace("_", " ").title()
                    text_parts.append(f"{field_name}: {value}")

                text = "\n".join(text_parts).strip()

                if not text:
                    continue

                chunks = chunk_text(
                    text=text,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )

                for chunk in chunks:
                    record = {
                        "chunk_id": (
                            f"{CORPUS_NAME}_{total_chunks:05d}"
                        ),
                        "source": str(faculty.get("id", "")),
                        "word_count": chunk["word_count"],
                        "start_word": chunk["start_word"],
                        "text": chunk["text"],
                    }

                    output_file.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )

                    total_chunks += 1

    print(f"Faculty records processed: {len(rows)}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Chunk size: {chunk_size} words")
    print(f"Overlap: {overlap} words")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()