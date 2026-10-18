
import json
from pathlib import Path

import yaml
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs.yaml"


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
            lines.append(
                f"\n{'#' * int(element.name[1])} {text}\n"
            )
        else:
            lines.append(text)

    return "\n\n".join(lines).strip()


def main() -> None:
    config = yaml.safe_load(
        CONFIG_PATH.read_text(encoding="utf-8")
    )

    corpus_name = config["corpus"]["name"]

    raw_dir = ROOT / "data" / "raw" / corpus_name
    output_dir = ROOT / "data" / "processed" / corpus_name

    manifest_path = raw_dir / "manifest.jsonl"
    sources_path = output_dir / "sources.jsonl"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Fetch manifest not found: {manifest_path}. "
            "Run the fetch script first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = []

    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                url = record["url"]
                filename = record["filename"]

                if not isinstance(url, str) or not isinstance(filename, str):
                    raise ValueError(
                        "URL and filename must be strings."
                    )

                manifest_records.append(record)

            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                raise ValueError(
                    f"Invalid manifest record at line {line_number}: {exc}"
                ) from exc

    successful = 0
    failed = 0
    source_records = []

    for record in manifest_records:
        source_url = record["url"]
        filename = record["filename"]

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

            output_path.write_text(
                clean_text,
                encoding="utf-8",
            )

            source_records.append({
                "filename": output_path.name,
                "url": source_url,
            })

            print(f"Saved: {output_path}")
            print(f"Source URL: {source_url}")
            print(f"Characters extracted: {len(clean_text)}")
            successful += 1

        except (OSError, UnicodeError) as error:
            print(f"Failed to process {input_path}: {error}")
            failed += 1

    # Save the exact URL for each extracted text file.
    with sources_path.open("w", encoding="utf-8") as file:
        for record in source_records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print(f"\nExtraction complete: {successful} succeeded, {failed} failed.")
    print(f"Source mapping saved: {sources_path}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
