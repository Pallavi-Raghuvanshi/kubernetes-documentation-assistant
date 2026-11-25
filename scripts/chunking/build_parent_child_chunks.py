
# scripts/chunking/build_parent_child_chunks.py
# ------------------------------------------------------
# reads your extracted text files and generates two separate JSONL files: parents_512_26.jsonl, children_128_16.jsonl

import json
from pathlib import Path

from transformers import AutoTokenizer

from src.utils.helpers import load_config
from src.chunking.parent_child_chunker import (
    create_parent_child_chunks,
)


ROOT = Path(__file__).resolve().parents[2]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def load_sources(sources_path: Path) -> dict:
    """Map extracted filenames to their original URLs."""

    sources = {}

    with sources_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            sources[record["filename"]] = record["url"]

    return sources


def main():

    config = load_config()

    corpus_name = config["corpus"]["name"]
    model_name = config["retrieval"]["model"]

    settings = config["chunking"]["parent_child"]

    parent_size = settings["parent_size"]
    parent_overlap = settings["parent_overlap"]
    child_size = settings["child_size"]
    child_overlap = settings["child_overlap"]

    input_dir = (
        ROOT / "data" / "processed" / corpus_name
    )

    sources_path = input_dir / "sources.jsonl"

    output_dir = ROOT / "data" / "parent_child"

    parent_path = (
        output_dir /
        f"parents_{parent_size}_{parent_overlap}.jsonl"
    )

    child_path = (
        output_dir /
        f"children_{child_size}_{child_overlap}.jsonl"
    )

    if not sources_path.exists():
        raise FileNotFoundError(sources_path)

    print("Loading tokenizer:", model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True,
    )

    source_mapping = load_sources(sources_path)

    all_parents = []
    all_children = []

    text_files = sorted(input_dir.glob("*.txt"))

    for file_path in text_files:

        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            continue

        source = source_mapping.get(file_path.name)

        if source is None:
            raise ValueError(
                f"Source URL missing for {file_path.name}"
            )

        document_id = file_path.stem

        parents, children = create_parent_child_chunks(
            text=text,
            source=source,
            document_id=document_id,
            tokenizer=tokenizer,
            parent_size=parent_size,
            parent_overlap=parent_overlap,
            child_size=child_size,
            child_overlap=child_overlap,
        )

        all_parents.extend(parents)
        all_children.extend(children)

        print(
            f"{file_path.name}: "
            f"{len(parents)} parents, "
            f"{len(children)} children"
        )

    write_jsonl(parent_path, all_parents)
    write_jsonl(child_path, all_children)

    print("\nParent-child chunking completed.")
    print("Parent count:", len(all_parents))
    print("Child count:", len(all_children))
    print("Parent output:", parent_path)
    print("Child output:", child_path)


if __name__ == "__main__":
    main()
