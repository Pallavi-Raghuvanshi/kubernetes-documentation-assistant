
import json
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs.yaml"


def split_into_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split text into overlapping character-based chunks."""
    sections = [
        section.strip()
        for section in re.split(r"\n\s*\n", text)
        if section.strip()
    ]

    chunks = []
    current_parts = []

    def save_chunk(parts: list[str]) -> None:
        content = "\n\n".join(parts).strip()
        if content:
            chunks.append(content)

    for section in sections:
        if len(section) > chunk_size:
            if current_parts:
                save_chunk(current_parts)
                current_parts = []

            step = chunk_size - overlap

            for start in range(0, len(section), step):
                window = section[start : start + chunk_size]
                if not window:
                    break

                chunks.append(window)

                if start + chunk_size >= len(section):
                    break

            continue

        candidate = "\n\n".join(current_parts + [section])

        if current_parts and len(candidate) > chunk_size:
            previous = "\n\n".join(current_parts)
            save_chunk(current_parts)

            carried = previous[-overlap:] if overlap else ""
            current_parts = [carried] if carried else []

        current_parts.append(section)

    if current_parts:
        save_chunk(current_parts)

    return chunks


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    corpus_name = config["corpus"]["name"]
    settings = config["chunking"]

    chunk_size = int(settings["chunk_size"])
    overlap = int(settings["chunk_overlap"])

    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid chunk_size or chunk_overlap.")

    input_dir = ROOT / "data" / "processed" / corpus_name
    sources_path = input_dir / "sources.jsonl"
    output_path = ROOT / "chunks.jsonl"

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    if not sources_path.exists():
        raise FileNotFoundError(
            f"Source mapping not found: {sources_path}. "
            "Run the extraction script first."
        )

    # Map each extracted text filename to its original URL.
    source_urls = {}

    with sources_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                source_urls[record["filename"]] = record["url"]
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(
                    f"Invalid source mapping at line {line_number}: {exc}"
                ) from exc

    text_files = sorted(
        path for path in input_dir.glob("*.txt")
        if path.is_file()
    )

    if not text_files:
        raise FileNotFoundError(
            f"No extracted .txt files found in {input_dir}"
        )

    records_written = 0
    files_processed = 0
    empty_files = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for input_path in text_files:
            text = input_path.read_text(
                encoding="utf-8"
            ).strip()

            if not text:
                empty_files += 1
                print(f"Skipping empty file: {input_path.name}")
                continue

            original_url = source_urls.get(input_path.name)

            if not original_url:
                raise ValueError(
                    f"No original URL found for {input_path.name} "
                    f"in {sources_path}"
                )

            chunks = split_into_chunks(
                text,
                chunk_size,
                overlap,
            )

            for content in chunks:
                record = {
                    "chunk_id": f"{corpus_name}_{records_written:05d}",
                    "chunk_size": len(content),
                    "source": original_url,
                    "text": content,
                }

                output_file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                records_written += 1

            files_processed += 1

            print(
                f"Processed: {input_path.name} | "
                f"URL: {original_url} | "
                f"Characters: {len(text)} | "
                f"Chunks: {len(chunks)}"
            )

    print("\nChunking unit: characters")
    print(f"Chunk size: {chunk_size}")
    print(f"Overlap: {overlap}")
    print(f"Text files found: {len(text_files)}")
    print(f"Files processed: {files_processed}")
    print(f"Empty files skipped: {empty_files}")
    print(f"Total chunks created: {records_written}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
