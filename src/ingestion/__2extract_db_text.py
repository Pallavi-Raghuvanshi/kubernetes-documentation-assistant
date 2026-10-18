import re
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DB_PATH = ROOT / "data" / "raw" / "daiict_faculty_data.db"
OUTPUT_DIR = ROOT / "data" / "processed" / "daiict_faculty_data"


def clean_text(value) -> str:
    """Normalize whitespace in extracted field values."""
    if value is None:
        return ""

    text = str(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_faculty_text(row: sqlite3.Row) -> str:
    """Convert a faculty database record into structured text."""

    fields = [
        ("Name", "name"),
        ("Qualification", "qualification"),
        ("Biography", "biography"),
        ("Specialization", "specialization"),
        ("Publications", "publications"),
        ("Teaching", "teaching"),
        ("Research", "research"),
    ]

    sections = []

    for label, column in fields:
        value = clean_text(row[column])

        if value:
            sections.append(f"{label}: {value}")

    return "\n\n".join(sections)


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT
                id,
                name,
                qualification,
                biography,
                specialization,
                publications,
                teaching,
                research
            FROM Faculty
            ORDER BY id
            """
        ).fetchall()

    successful = 0
    empty = 0

    for row in rows:
        text = extract_faculty_text(row)

        if not text:
            empty += 1
            continue

        output_path = OUTPUT_DIR / f"faculty_{row['id']:04d}.txt"
        output_path.write_text(text, encoding="utf-8")

        successful += 1

    print(f"Faculty records: {len(rows)}")
    print(f"Text files created: {successful}")
    print(f"Empty records skipped: {empty}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()