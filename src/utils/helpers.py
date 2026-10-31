# src/utils/helpers.py
import re
from urllib.parse import urlparse
import yaml
from pathlib import Path
import json
import csv

ROOT = Path(__file__).resolve().parents[2]

def filename_from_url(url: str) -> str:
    """Generate HTML filename from a documentation URL."""
    path = urlparse(url).path.rstrip("/") # https://example.com/docs/setup@first/ -> /docs/setup
    name = path.split("/")[-1] # /docs/setup@first -> docs, setup@first -> setup@first
    # Replace any character that is not alphanumeric(Cap, small), dot, underscore, or hyphen with an underscore
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name) # e.g. setup@first -> setup_first.html

    return f"{name or 'index'}.html"

def load_yaml(path: Path) -> dict:
    """Load a YAML file."""
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}

def load_config():
    return load_yaml(ROOT / "configs.yaml")

def get_queries():
    query_path = Path("data/queries.csv")

    if not query_path.exists():
        raise FileNotFoundError(
            f"Evaluation queries not found: {query_path.resolve()}"
        )

    with query_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        queries = list(reader)

    # required_columns = {
    #     "query_id",
    #     "query",
    #     "ground_truth_answer",
    #     "source_file",
    #     "ground_truth_context",
    # }

    # missing_columns = required_columns - set(reader.fieldnames or [])

    # if missing_columns:
    #     raise ValueError(
    #         f"Missing columns in queries.csv: {sorted(missing_columns)}"
    #     )

    # if not queries:
    #     raise ValueError("queries.csv contains no queries.")

    # for row in queries:
    #     for column in required_columns:
    #         if not row[column] or not row[column].strip():
    #             raise ValueError(
    #                 f"Empty value in column '{column}' "
    #                 f"for query {row.get('query_id')}"
    #             )

    return queries

def load_chunks() -> list[dict]:
    chunks = []
    config = load_config()
    chunk_size = config["chunking"]["chunk_size"]
    overlap = config["chunking"]["chunk_overlap"]
    CHUNKS_PATH = ROOT / "data" / "chunks" / f"chunks_{chunk_size}_{overlap}.jsonl"

    with CHUNKS_PATH.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))
                
    return chunks
