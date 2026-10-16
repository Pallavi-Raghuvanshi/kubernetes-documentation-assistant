
import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "corpus.yaml"


def load_config(config_path: Path) -> dict:
    """Load and validate the corpus configuration."""
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    corpus = config.get("corpus", {})
    name = corpus.get("name")
    urls = corpus.get("source_urls")
    source_type = corpus.get("source_type")

    if (
        not name
        or not isinstance(urls, list)
        or not urls
        or not all(isinstance(url, str) and url.strip() for url in urls)
        or source_type != "html"
    ):
        raise ValueError(
            "corpus.yaml must define corpus.name, "
            "a non-empty corpus.source_urls list, "
            "and source_type: html."
        )

    return config


def filename_from_url(url: str) -> str:
    """Generate an HTML filename from a documentation URL."""
    path = urlparse(url).path.rstrip("/")
    name = path.split("/")[-1] if path else "index"

    # Keep filenames safe for Windows and other filesystems.
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)

    return f"{name or 'index'}.html"


def fetch_documentation(url: str, output_path: Path) -> bool:
    """Fetch one HTML page and save it locally."""
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "RAGFrontierResearch/0.1"},
            timeout=30,
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type.lower():
            raise ValueError(
                f"Expected HTML, received: {content_type}"
            )

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
        description="Fetch documentation pages from corpus.yaml."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the corpus configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    corpus = config["corpus"]

    output_dir = (
        PROJECT_ROOT / "data" / "raw" / corpus["name"]
    )

    successful = 0
    failed = 0
    used_filenames = set()

    for url in corpus["source_urls"]:
        filename = filename_from_url(url)

        if filename in used_filenames:
            raise ValueError(
                f"Duplicate output filename {filename!r}. "
                "Use distinct URLs or filenames."
            )

        used_filenames.add(filename)
        output_path = output_dir / filename

        if fetch_documentation(url, output_path):
            successful += 1
        else:
            failed += 1

    print(f"\nDownload summary: {successful} succeeded, {failed} failed.")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()