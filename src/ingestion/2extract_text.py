
from pathlib import Path
from urllib.parse import urlparse
import re

import yaml
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/corpus.yaml"


def filename_from_url(url: str) -> str:
    """Generate the HTML filename used by 1fetch_docs.py."""
    name = urlparse(url).path.rstrip("/").split("/")[-1] or "index"
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    return f"{name}.html"


def extract_text(input_path: Path) -> str:
    """Extract structured documentation text from an HTML file."""
    soup = BeautifulSoup(
        input_path.read_text(encoding="utf-8"),
        "html.parser",
    )

    for element in soup.select(
        "script, style, nav, footer, header, noscript"
    ):
        element.decompose()

    main_content = soup.find("main") or soup.body or soup
    lines = []

    for element in main_content.find_all(
        ["h1", "h2", "h3", "h4", "p", "li", "pre"]
    ):
        text = element.get_text(" ", strip=True)

        if not text:
            continue

        if element.name.startswith("h"):
            lines.append(f"\n{'#' * int(element.name[1])} {text}\n")
        else:
            lines.append(text)

    return "\n\n".join(lines).strip()


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    corpus_name = config["corpus"]["name"]
    source_urls = config["corpus"]["source_urls"]

    raw_dir = ROOT / "data" / "raw" / corpus_name
    output_dir = ROOT / "data" / "processed" / corpus_name
    output_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0

    for source_url in source_urls:
        filename = filename_from_url(source_url)
        input_path = raw_dir / filename
        output_path = output_dir / f"{Path(filename).stem}.txt"

        if not input_path.exists():
            print(f"Missing HTML file: {input_path}")
            failed += 1
            continue

        try:
            clean_text = extract_text(input_path)

            if not clean_text:
                print(f"No text extracted: {input_path}")
                failed += 1
                continue

            output_path.write_text(clean_text, encoding="utf-8")

            print(f"Saved: {output_path}")
            print(f"Characters extracted: {len(clean_text)}")
            successful += 1

        except (OSError, UnicodeError) as error:
            print(f"Failed to process {input_path}: {error}")
            failed += 1

    print(f"\nExtraction complete: {successful} succeeded, {failed} failed.")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()