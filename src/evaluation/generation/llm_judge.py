# src/generation/llm_judge.py
import json
import re

from src.generation._6llm_inference import generate_answer


REQUIRED_SCORES = {
    "faithfulness",
    "answer_completeness",
    "hallucination_rate",
}


def judge_generation(
    question: str,
    answer: str,
    context: str,
    reference_answer: str,
    max_new_tokens: int = 160,
) -> dict:
    judge_context = f"""
QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

REFERENCE ANSWER:
{reference_answer}

GENERATED ANSWER:
{answer}
"""

    judge_question = """
Evaluate the GENERATED ANSWER using the supplied evidence.

Return ONLY a valid JSON object with these numeric fields:
{
  "faithfulness": 0.0,
  "answer_completeness": 0.0,
  "hallucination_rate": 0.0
}

Scoring rules:
1. faithfulness: Fraction of factual claims supported by the retrieved context.
2. answer_completeness: Fraction of essential reference facts correctly covered.
3. hallucination_rate: Fraction of factual claims unsupported by or contradicted
   by the retrieved context.

Score each metric from 0 to 1.
Return only JSON without Markdown or explanations.
"""

    return generate_answer(
        question=judge_question,
        context=judge_context,
        max_new_tokens=max_new_tokens,
    )



def get_judge_scores(context) -> dict[str, float]:
    """Return cached judge scores, generating them only once per context."""

    if context.judge_scores is not None:
        return context.judge_scores

    result = judge_generation(
        question=context.question,
        answer=context.answer,
        context=context.retrieved_context,
        reference_answer=context.reference_answer,
    )

    # Extract the raw answer before using it in diagnostic output.
    if not isinstance(result, dict):
        raise ValueError(
            f"LLM judge returned an unexpected result type: "
            f"{type(result).__name__}"
        )

    raw_output = result.get("answer", "")

    if not isinstance(raw_output, str):
        raise ValueError(
            f"LLM judge answer must be a string, got "
            f"{type(raw_output).__name__}"
        )

    raw_output = raw_output.strip()

    # Temporary diagnostics: inspect the model's actual response.
    print("\n" + "=" * 70)
    print("LLM JUDGE DIAGNOSTIC")
    print("=" * 70)
    print("Query:", context.question)
    print("Generated answer:", context.answer)
    print("Retrieved context:", context.retrieved_context[:1500])
    print("Raw judge output:", raw_output)
    print("=" * 70)

    match = re.search(r"\{.*\}", raw_output, re.DOTALL)

    if not match:
        raise ValueError(
            f"LLM judge did not return valid JSON: {raw_output}"
        )

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM judge returned invalid JSON: {raw_output}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError("LLM judge response must be a JSON object.")

    missing = REQUIRED_SCORES - parsed.keys()
    if missing:
        raise ValueError(
            f"LLM judge response is missing scores: {sorted(missing)}"
        )

    scores = {}

    for name in REQUIRED_SCORES:
        value = parsed[name]

        # Reject booleans and non-numeric values.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"Invalid {name} score: expected a number, got {value!r}"
            )

        value = float(value)

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Invalid {name} score: {value}. "
                "Expected a value from 0 to 1."
            )

        scores[name] = value

    print("Parsed judge scores:", scores)
    print("=" * 70 + "\n")

    context.judge_scores = scores
    return scores
