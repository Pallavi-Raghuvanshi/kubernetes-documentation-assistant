# scripts/view_query_results.py
"""Generate a dark, readable HTML report from per-query RAG evaluation JSONL."""

import argparse
import html
import json
from pathlib import Path
from typing import Any


def first(record: dict, *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return default


def pretty(value: Any) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, (dict, list, tuple, set)):
        if isinstance(value, set):
            value = sorted(value, key=str)
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    return str(value)


def esc(value: Any) -> str:
    return html.escape(pretty(value))


def normalize_ids(values: Any) -> set[str]:
    """Convert ground-truth IDs to strings for reliable ID comparisons."""
    if values is None:
        return set()
    if isinstance(values, dict):
        values = list(values.keys())
    elif isinstance(values, (str, int)):
        values = [values]
    try:
        return {str(value) for value in values if value is not None}
    except TypeError:
        return {str(values)}


def chunk_id_of(chunk: Any) -> str | None:
    if not isinstance(chunk, dict):
        return None
    value = first(chunk, "chunk_id", "id", "document_id", "doc_id")
    return None if value is None else str(value)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object on line {line_number}")
            records.append(record)
    return records


def get_retrieval_metrics(relevant_ids: set[str], retrieved: list) -> dict:
    """Calculate per-query metrics from saved ground truth and retrieved results."""
    retrieved_ids = [chunk_id_of(item) for item in retrieved]
    retrieved_ids = [chunk_id for chunk_id in retrieved_ids if chunk_id is not None]
    hits = [chunk_id for chunk_id in retrieved_ids if chunk_id in relevant_ids]
    first_relevant_rank = next(
        (rank for rank, chunk_id in enumerate(retrieved_ids, start=1)
         if chunk_id in relevant_ids),
        None,
    )
    # The JSONL contains the retrieved top-k list. Use its actual length as k.
    k = len(retrieved_ids)
    return {
        f"Recall@{k}": len(set(hits)) / len(relevant_ids) if relevant_ids else None,
        f"Precision@{k}": len(hits) / k if k else None,
        "MRR": 1 / first_relevant_rank if first_relevant_rank else 0.0,
        f"Hit Rate@{k}": 1.0 if hits else 0.0,
    }


def render_query(record: dict, index: int) -> str:
    query_id = first(record, "query_id", "id", "qid", default=f"Query {index + 1}")
    question = first(record, "question", "query", "user_query", default="Not available")
    generation = first(record, "generation", default={}) or {}
    if not isinstance(generation, dict):
        generation = {}
    answer = first(
        record, "answer", "generated_answer", "response",
        default=generation.get("answer"),
    )

    ground_truth_raw = first(
        record, "relevant_chunk_ids", "ground_truth_chunk_ids",
        "relevant_ids", "ground_truth", default=[],
    )
    relevant_ids = normalize_ids(ground_truth_raw)

    retrieved = first(
        record, "retrieved_results", "retrieved_chunks", "retrieved_documents",
        "results", "documents", default=[],
    )
    if not isinstance(retrieved, list):
        retrieved = []

    # Prefer saved metrics if present; otherwise compute them from this record.
    saved_metrics = first(record, "metrics", "retrieval_metrics", default={})
    metrics = saved_metrics if isinstance(saved_metrics, dict) and saved_metrics else (
        get_retrieval_metrics(relevant_ids, retrieved)
    )

    metric_html = ""
    for name, value in metrics.items():
        display_value = "N/A" if value is None else (
            f"{value:.4f}" if isinstance(value, (int, float)) else pretty(value)
        )
        metric_html += (
            '<div class="metric"><span class="metric-name">'
            f'{esc(name)}</span><strong>{html.escape(display_value)}</strong></div>'
        )

    # Build ID -> rank map from retrieved results, preserving first occurrence.
    ranks_by_id = {}
    for rank, chunk in enumerate(retrieved, start=1):
        chunk_id = chunk_id_of(chunk)
        if chunk_id is not None and chunk_id not in ranks_by_id:
            ranks_by_id[chunk_id] = rank

    if relevant_ids:
        ground_truth_rows = []
        # Sort IDs for stable, easy-to-scan output.
        for gt_id in sorted(relevant_ids):
            rank = ranks_by_id.get(gt_id)
            status = (
                f"Rank {rank}" if rank is not None else "Not in top-k"
            )
            status_class = "found" if rank is not None else "missing"
            ground_truth_rows.append(
                f'<div class="gt-row"><code>{esc(gt_id)}</code>'
                f'<span class="gt-rank {status_class}">{html.escape(status)}</span></div>'
            )
        ground_truth_html = "\n".join(ground_truth_rows)
    else:
        ground_truth_html = '<p class="muted">No ground-truth relevant chunk IDs stored.</p>'

    chunks_html = ""
    for rank, chunk in enumerate(retrieved, start=1):
        if not isinstance(chunk, dict):
            chunk = {"content": chunk}

        chunk_id = chunk_id_of(chunk) or "Unknown"
        score = first(chunk, "score", "similarity_score", "relevance_score")
        content = first(
            chunk, "text", "content", "page_content", "chunk_text",
            default="No chunk text stored in this record.",
        )
        metadata = first(chunk, "metadata", default={})

        # Derive the relevance label from annotated ground truth, not from score
        # or optional flags that may not exist in the stored result.
        is_relevant = chunk_id in relevant_ids
        if is_relevant:
            relevance_label = '<span class="badge relevant">Relevant</span>'
        else:
            relevance_label = '<span class="badge not-relevant">Not relevant</span>'

        chunks_html += f"""
        <details class="chunk">
          <summary>
            <span class="rank">Rank {rank}</span>
            <strong>{esc(chunk_id)}</strong>
            <span class="score">Score: {esc(score)}</span>
            {relevance_label}
          </summary>
          <div class="chunk-body">
            <h4>Chunk text</h4>
            <pre>{esc(content)}</pre>
            <h4>Metadata</h4>
            <pre>{esc(metadata)}</pre>
          </div>
        </details>
        """

    if not chunks_html:
        chunks_html = '<p class="muted">No retrieved chunks found in this record.</p>'

    answer_html = (
        f"<pre>{esc(answer)}</pre>" if answer is not None
        else '<p class="muted">No generated answer stored in this record.</p>'
    )

    return f"""
    <details class="query" {'open' if index == 0 else ''}>
      <summary>
        <span class="query-id">{esc(query_id)}</span>
        <span class="query-question">{esc(question)}</span>
      </summary>
      <div class="query-body">
        <h3>Question</h3>
        <p class="question">{esc(question)}</p>

        <h3>Ground-truth relevant chunks ({len(relevant_ids)})</h3>
        <div class="ground-truth">{ground_truth_html}</div>

        <h3>Per-query retrieval metrics</h3>
        <div class="metrics">
          {metric_html or '<p class="muted">Metrics unavailable: no ground truth or retrieved IDs to calculate them.</p>'}
        </div>

        <h3>Retrieved chunks ({len(retrieved)})</h3>
        {chunks_html}

        <h3>Generated answer</h3>
        {answer_html}
      </div>
    </details>
    """


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a dark HTML report from per-query RAG evaluation JSONL."
    )
    parser.add_argument("--input", required=True, help="Path to per-query JSONL file")
    parser.add_argument(
        "--output", default=None,
        help="HTML output path (defaults to the input filename with .html)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".html")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    records = load_jsonl(input_path)
    query_html = "\n".join(
        render_query(record, index) for index, record in enumerate(records)
    )

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG Query Results</title>
<style>
  :root {{
    color-scheme: dark;
    font-family: "Segoe UI", Arial, sans-serif;
    background: #0b1120;
    color: #e5e7eb;
  }}
  * {{ box-sizing: border-box; }}
  body {{ max-width: 1250px; margin: 0 auto; padding: 26px; background: #0b1120; }}
  h1 {{ margin: 0 0 6px; color: #f8fafc; }}
  h3 {{ margin-top: 26px; color: #cbd5e1; font-size: 1.05rem; }}
  h4 {{ margin-bottom: 8px; color: #cbd5e1; }}
  .subtitle, .muted {{ color: #94a3b8; }}
  .toolbar {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 22px 0; }}
  button, input {{
    font: inherit; padding: 10px 12px; border: 1px solid #334155;
    border-radius: 8px; background: #111c2f; color: #e5e7eb;
  }}
  button {{ cursor: pointer; }}
  button:hover {{ background: #1e293b; }}
  input {{ flex: 1; min-width: 220px; }}
  input::placeholder {{ color: #94a3b8; }}
  details.query {{
    background: #111827; border: 1px solid #273449; border-radius: 12px;
    margin: 12px 0; overflow: hidden;
  }}
  details.query > summary {{
    display: flex; gap: 14px; align-items: baseline; padding: 17px;
    cursor: pointer; background: #151f31;
  }}
  details.query > summary:hover {{ background: #1b2940; }}
  .query-id {{ font-weight: 700; min-width: 90px; color: #93c5fd; }}
  .query-question {{ overflow-wrap: anywhere; color: #f1f5f9; }}
  .query-body {{ padding: 0 18px 22px; }}
  .question {{ white-space: pre-wrap; overflow-wrap: anywhere; color: #f1f5f9; }}
  pre {{
    white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word;
    background: #0b1220; color: #dbeafe; padding: 14px; border-radius: 8px;
    border: 1px solid #263449; font: 13px/1.6 Consolas, monospace;
  }}
  .metrics {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .metric {{ background: #172338; border: 1px solid #2b3a52; padding: 11px 14px; border-radius: 8px; min-width: 115px; }}
  .metric-name {{ display: block; color: #94a3b8; font-size: 12px; margin-bottom: 5px; }}
  .metric strong {{ color: #f8fafc; font-size: 16px; }}
  .ground-truth {{ display: grid; gap: 7px; }}
  .gt-row {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; background: #0f1a2b; border: 1px solid #273449; padding: 9px 12px; border-radius: 7px; }}
  code {{ color: #bfdbfe; overflow-wrap: anywhere; }}
  .gt-rank {{ font-size: 12px; font-weight: 700; padding: 4px 8px; border-radius: 6px; white-space: nowrap; }}
  .found {{ color: #86efac; background: #123524; }}
  .missing {{ color: #fca5a5; background: #451a1a; }}
  details.chunk {{ border: 1px solid #2b394e; border-radius: 8px; margin: 8px 0; background: #0f1726; }}
  details.chunk > summary {{ cursor: pointer; padding: 12px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }}
  details.chunk > summary:hover {{ background: #172338; }}
  .chunk-body {{ padding: 0 12px 12px; }}
  .rank, .score {{ color: #94a3b8; font-size: 13px; }}
  .badge {{ padding: 4px 8px; border-radius: 6px; font-size: 12px; font-weight: 700; }}
  .relevant {{ background: #14532d; color: #bbf7d0; }}
  .not-relevant {{ background: #3f1d2a; color: #fecdd3; }}
  @media (max-width: 650px) {{
    body {{ padding: 12px; }}
    details.query > summary {{ flex-direction: column; gap: 7px; }}
    .gt-row {{ align-items: flex-start; flex-direction: column; }}
  }}
</style>
</head>
<body>
  <h1>RAG Query Results</h1>
  <p class="subtitle">Source: {esc(input_path)} · Queries: {len(records)}</p>
  <div class="toolbar">
    <input id="search" type="search" placeholder="Filter by query ID or question...">
    <button type="button" id="expand">Expand all</button>
    <button type="button" id="collapse">Collapse all</button>
  </div>
  <main id="queries">
    {query_html or '<p class="muted">No query records found.</p>'}
  </main>
<script>
  const queryCards = [...document.querySelectorAll("details.query")];
  document.getElementById("search").addEventListener("input", function () {{
    const term = this.value.toLowerCase();
    queryCards.forEach(card => {{
      card.hidden = !card.innerText.toLowerCase().includes(term);
    }});
  }});
  document.getElementById("expand").addEventListener("click", () => {{
    queryCards.forEach(card => {{ if (!card.hidden) card.open = true; }});
  }});
  document.getElementById("collapse").addEventListener("click", () => {{
    queryCards.forEach(card => {{ card.open = false; }});
  }});
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    print(f"HTML report created       : {output_path.resolve()}")


if __name__ == "__main__":
    main()
