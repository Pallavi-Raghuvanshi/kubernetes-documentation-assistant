import argparse
import json
from pathlib import Path
import requests
import yaml

from src.utils.helpers import filename_from_url

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs.yaml"
DEFAULT_URLS = PROJECT_ROOT / "data" / "raw" / "kubernetes_urls.jsonl"


def load_config(config_path: Path) -> dict:
    """Load and validate the corpus configuration."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_urls(urls_path: Path) -> list[str]:
    """Load documentation URLs from a JSONL file."""
    urls = []

    with urls_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                url = record["url"]

                if not isinstance(url, str):
                    raise ValueError("URL must be a string.")

                urls.append(url)

            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                raise ValueError(
                    f"Invalid record at line {line_number} in {urls_path}: {exc}"
                ) from exc

    return list(dict.fromkeys(urls))


def fetch_documentation(url: str, timeout_sec: int, output_path: Path) -> bool:
    """Fetch one HTML page and save it locally."""
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "RAGFrontierResearch/0.1"},
            timeout=timeout_sec,
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type.lower():
            raise ValueError(f"Expected HTML, received: {content_type}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(response.text, encoding="utf-8")

        print(f"Source URL: {url}")
        print(f"Saved to: {output_path}")
        print(f"Characters downloaded: {len(response.text)}")
        return True

    except (requests.RequestException, ValueError) as exc:
        print(f"Failed to fetch {url}: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch documentation pages from a JSONL URL file."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--urls",
        type=Path,
        default=DEFAULT_URLS,
        help="Path to the JSONL file containing documentation URLs.",
    )
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    corpus = config["corpus"]
    timeout_sec = config["ingestion"]["timeout_seconds"]

    source_urls = load_urls(args.urls.resolve())

    output_dir = PROJECT_ROOT / "data" / "raw" / corpus["name"]

    successful = 0
    failed = 0
    used_filenames = set()

    for url in source_urls:
        filename = filename_from_url(url)

        if filename in used_filenames:
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            filename = f"{stem}_{len(used_filenames)}{suffix}"

        used_filenames.add(filename)

        used_filenames.add(filename)
        output_path = output_dir / filename

        if fetch_documentation(url, timeout_sec, output_path):
            successful += 1
        else:
            failed += 1

    print(f"\nUnique URLs loaded: {len(source_urls)}")
    print(f"Download summary: {successful} succeeded, {failed} failed.")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()