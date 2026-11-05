"""Generate a dark HTML report from per-query RAG evaluation JSONL."""

import argparse
import html
import json
from pathlib import Path
from typing import Any

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.evaluate2 import calculate_text_proportion_metrics

def get_text_proportion_metrics(record: dict) -> dict:
    """Calculate text-proportion metrics using evaluate2.py."""

    gt_context_tokens = record["gt_context_tokens"]
    retrieved_results = record["retrieved_results"]

    return calculate_text_proportion_metrics(
        gt_context_tokens=gt_context_tokens,
        retrieved_results=retrieved_results,
        top_k=len(retrieved_results),
    )

def format_metric(value: Any) -> str:
    """Format a metric as a percentage."""

    if value is None:
        return "Not available"

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return pretty(value)

    return f"{numeric_value:.2%}"


def esc(value: Any) -> str:
  """Convert a value to a readable, HTML-escaped string."""
  # print(f".................esc......................")
  return html.escape(pretty(value))


def pretty(value: Any) -> str:
  """Convert a value to a readable string."""
  if isinstance(value, (dict, list, tuple, set)):
      if isinstance(value, set):
          value = sorted(value, key=str)
      return json.dumps(
          value,
          indent=2,
          ensure_ascii=False,
          default=str,
      )
  # print(f".................pretty......................")
  return str(value)


def load_jsonl(path: Path) -> list[dict]:
  """Load evaluation records from a JSONL file."""
  records = []

  # open jsonl file and read line by line
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
              raise ValueError(
                  f"Expected a JSON object on line {line_number}"
              )

          records.append(record)
  # print(records)
  # print(f".................load_jsonl.....................")
  return records


def render_query(record: dict, index: int) -> str:

  # print(record.keys())
  query_id = record["query_id"]
  question = record["query"]
  # query_tokens = record["query_tokens"]
  # query_token_set = {token.lower() for token in query_tokens}
  gt_context_tokens = record["gt_context_tokens"]
  gt_context_token_set = {token.lower() for token in gt_context_tokens}
  generation = record["generation"]

  if not isinstance(generation, dict):
      generation = {}

  answer = record["ground_truth_answer"]

  # Read the ground-truth reference text.
  ground_truth_text = record["ground_truth_context"]

  # Read the retrieved top-k results.
  retrieved = record["retrieved_results"]

  if not isinstance(retrieved, list):
    retrieved = []

  # Collect tokens appearing in the retrieved top-k chunks.
  retrieved_token_set = {
    token.lower()
    for chunk in retrieved
    for token in chunk["chunk_tokens"]
  }

  # Read the character-based text-proportion metrics.
  metrics = get_text_proportion_metrics(record)

  metric_html = ""

  for name, value in metrics.items():
    display_value = format_metric(value)

    metric_html += (
      '<div class="metric">'
      '<span class="metric-name">'
      f"{esc(name)}"
      "</span>"
      f"<strong>{html.escape(display_value)}</strong>"
      "</div>"
    )

  # Display the ground-truth reference text.
  if ground_truth_text:
    ground_truth_html = f"<pre>{esc(ground_truth_text)}</pre>"
  else:
    ground_truth_html = (
      '<p class="muted">'
      "Ground-truth reference text is not stored "
      "in this query record."
      "</p>"
    )

  # Render the retrieved chunks.
  chunks_html = ""

  for rank, chunk in enumerate(retrieved, start=1):
    #----------------
    if not isinstance(chunk, dict):
      chunk = {"content": chunk}

    chunk_id = chunk["chunk_id"]
    score = chunk["score"]

    if isinstance(score, (int, float)):
      score = f"{score:.4f}"

    content = chunk["text"]
    chunk_tokens = chunk["chunk_tokens"]

    chunks_html += f"""
    <details class="chunk">
      <summary>
        <span class="rank">Rank {rank}</span>
        <strong>{esc(chunk_id)}</strong>
        <span class="score">Score: {esc(score)}</span>
      </summary>

      <div class="chunk-body">
        <h4>Chunk text</h4>
        <pre>{esc(content)}</pre>

        <h4>Chunk tokens ({len(chunk_tokens)})</h4>
        <div class="tokens">
            {
                "".join(
                    f'<span class="token {"matched-token" if token.lower() in gt_context_token_set else ""}">'
                    f'{esc(token)}</span>'
                    for token in chunk_tokens
                )
            }
        </div>
      </div>
    </details>
    """

    if not chunks_html:
        chunks_html = (
            '<p class="muted">'
            "No retrieved chunks found in this record."
            "</p>"
        )

  # Render the generated answer.
  if answer is not None:
    # convert characters such as <, >, and & into HTML-safe representations
    answer_html = f"<pre>{esc(answer)}</pre>" 
  else:
    answer_html = (
      '<p class="muted">'
      "No generated answer stored in this record."
      "</p>"
    )
  # # print(f".................render_query......................")
  return f"""
    <details class="query" {"open" if index == 0 else ""}>
      <summary>
        <span class="query-id">{esc(query_id)}</span>
        <span class="query-question">{esc(question)}</span>
      </summary>

      <div class="query-body">
        <h3>Question</h3>
        <p class="question">{esc(question)}</p>

        <h3>Ground-truth Answer</h3>
        {answer_html}

        <h3>Ground-truth Context</h3>
        <div class="ground-truth">
          {ground_truth_html}
        </div>

        <h3>Ground-truth Context Tokens ({len(gt_context_tokens)})</h3>
          <div class="tokens">
            {
              "".join(
                  f'<span class="token {"matched-token" if token.lower() in retrieved_token_set else ""}">'
                  f'{esc(token)}</span>'
                  for token in gt_context_tokens
              )
            }
          </div>

        <h3>Text-Proportion Metrics</h3>
        <div class="metrics">
          {
              metric_html
              or '<p class="muted">Text-proportion metrics unavailable.</p>'
          }
        </div>

        <h3>Retrieved Chunks ({len(retrieved)})</h3>
        {chunks_html}

      </div>
    </details>
    """


def main() -> None:
    """Generate the HTML report from a JSONL evaluation file."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate a dark HTML report from per-query "
            "RAG evaluation JSONL."
        )
    )
    # print(1)

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the per-query JSONL file",
    )
    # print(2)
    parser.add_argument(
        "--output",
        default=None,
        help="HTML output path. Defaults to the input filename with a .html extension.",
    )
    # # print(3)
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".html")
    # print(4)
    # if not input_path.is_file():
    #     raise FileNotFoundError(
    #         f"Input JSONL not found: {input_path}"
    #     )
    # print(5)
    records = load_jsonl(input_path)
    # #------------------------
    # from collections import Counter

    # records = load_jsonl(input_path)

    # query_counts = Counter(
    #     (record.get("query_id"), record.get("query"))
    #     for record in records
    # )

    # print(f"Total records: {len(records)}")

    # for query, count in query_counts.items():
    #     if count > 1:
    #         print(f"Duplicate query ({count} times): {query}")
            
    # print(f"Input file: {input_path.resolve()}")
    # print((5.5)
    query_html = "\n".join(render_query(record, index) for index, record in enumerate(records))
    # print(6)

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

  * {{
    box-sizing: border-box;
  }}

  body {{
    max-width: 1250px;
    margin: 0 auto;
    padding: 26px;
    background: #0b1120;
  }}

  h1 {{
    margin: 0 0 6px;
    color: #60A5FA;
  }}

  h3 {{
    margin-top: 26px;
    color: #D6B878;
    font-size: 1.05rem;
    padding-bottom: 6px;
    border-bottom: 1px solid #374151;
  }}

  h4 {{
    margin-bottom: 8px;
    color: #34D399;
  }}

  .subtitle,
  .muted {{
    color: #94a3b8;
  }}

  .toolbar {{
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin: 22px 0;
  }}

  button,
  input {{
    font: inherit;
    padding: 10px 12px;
    border: 1px solid #334155;
    border-radius: 8px;
    background: #111c2f;
    color: #e5e7eb;
  }}

  button {{
    cursor: pointer;
  }}

  button:hover {{
    background: #1e293b;
  }}

  input {{
    flex: 1;
    min-width: 220px;
  }}

  input::placeholder {{
    color: #94a3b8;
  }}

  details.query {{
    background: #111827;
    border: 1px solid #273449;
    border-radius: 12px;
    margin: 12px 0;
    overflow: hidden;
  }}

  details.query > summary {{
    display: flex;
    gap: 14px;
    align-items: baseline;
    padding: 17px;
    cursor: pointer;
    background: #151f31;
  }}

  details.query > summary:hover {{
    background: #1b2940;
  }}

  .query-id {{
    font-weight: 700;
    min-width: 90px;
    color: #93c5fd;
  }}

  .query-question {{
    overflow-wrap: anywhere;
    color: #f1f5f9;
  }}

  .query-body {{
    padding: 0 18px 22px;
  }}

  .question {{
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    color: #f1f5f9;
  }}

  pre {{
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    word-break: break-word;
    background: #0b1220;
    color: #dbeafe;
    padding: 14px;
    border-radius: 8px;
    border: 1px solid #263449;
    font: 13px/1.6 Consolas, monospace;
  }}

  .metrics {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
  }}

  .metric {{
    background: #172338;
    border: 1px solid #2b3a52;
    padding: 11px 14px;
    border-radius: 8px;
    min-width: 180px;
  }}

  .metric-name {{
    display: block;
    color: #94a3b8;
    font-size: 12px;
    margin-bottom: 5px;
  }}

  .metric strong {{
    color: #f8fafc;
    font-size: 16px;
  }}

  .ground-truth {{
    overflow-wrap: anywhere;
  }}

  .ground-truth pre {{
    max-height: 500px;
    overflow-y: auto;
  }}

  details.chunk {{
    border: 1px solid #2b394e;
    border-radius: 8px;
    margin: 8px 0;
    background: #0f1726;
  }}

  details.chunk > summary {{
    cursor: pointer;
    padding: 12px;
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
  }}

  details.chunk > summary:hover {{
    background: #172338;
  }}

  .chunk-body {{
    padding: 0 12px 12px;
  }}

  .rank,
  .score {{
    color: #94a3b8;
    font-size: 13px;
  }}

  code {{
    color: #bfdbfe;
    overflow-wrap: anywhere;
  }}

  @media (max-width: 650px) {{
    body {{
      padding: 12px;
    }}

    details.query > summary {{
      flex-direction: column;
      gap: 7px;
    }}

    .metric {{
      width: 100%;
    }}
  }}
  .tokens {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      padding: 12px;
      background: #0b1220;
      border: 1px solid #263449;
      border-radius: 8px;
  }}

  .token {{
      display: inline-block;
      padding: 4px 8px;
      background: #172338;
      color: #93c5fd;
      border: 1px solid #2b3a52;
      border-radius: 5px;
      font: 12px/1.5 Consolas, monospace;
      overflow-wrap: anywhere;
  }}
  .matched-token {{
      background: #14532d;
      color: #86efac;
      border-color: #22c55e;
  }}
</style>
</head>

<body>
  <h1>RAG Query Results</h1>

  <p class="subtitle">
    Source: {esc(input_path)} · Queries: {len(records)}
  </p>

  <div class="toolbar">
    <input
      id="search"
      type="search"
      placeholder="Filter by query ID or question..."
    >

    <button type="button" id="expand">
      Expand all
    </button>

    <button type="button" id="collapse">
      Collapse all
    </button>
  </div>

  <main id="queries">
    {query_html or '<p class="muted">No query records found.</p>'}
  </main>

<script>
  const queryCards = [
    ...document.querySelectorAll("details.query")
  ];

  document.getElementById("search").addEventListener(
    "input",
    function () {{
      const term = this.value.toLowerCase();

      queryCards.forEach(card => {{
        card.hidden = !card.innerText.toLowerCase().includes(term);
      }});
    }}
  );

  document.getElementById("expand").addEventListener(
    "click",
    () => {{
      queryCards.forEach(card => {{
        if (!card.hidden) {{
          card.open = true;
        }}
      }});
    }}
  );

  document.getElementById("collapse").addEventListener(
    "click",
    () => {{
      queryCards.forEach(card => {{
        card.open = false;
      }});
    }}
  );
</script>
</body>
</html>
"""
    # print(7)
    # Save the generated HTML report
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        document,
        encoding="utf-8",
    )

    print(f"HTML report saved successfully: {output_path.resolve()}")
    print(f"Queries included: {len(records)}")

if __name__ == "__main__":
    main()