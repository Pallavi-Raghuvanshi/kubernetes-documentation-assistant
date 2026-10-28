# src/utils/helpers.py
import re
from urllib.parse import urlparse
import yaml
from pathlib import Path
import json

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

def get_queries() -> list[dict]:
    return load_yaml(ROOT / "queries.yaml")

def load_chunks() -> list[dict]:
    chunks = []
    CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"

    with CHUNKS_PATH.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))
                
    return chunks