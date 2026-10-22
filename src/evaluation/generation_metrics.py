# src.evaluation.generation_metrics.py
import json
import re

from src.generation._6llm_inference import generate_answer

def sentence_integrity_rate(answer: str):
    """
    Heuristic: proportion of detected sentences ending in
    '.', '!' or '?'.

    This detects possible truncation, not factual correctness.
    """
    answer = answer.strip()

    sentences = re.findall(
        r"[^.!?]+(?:[.!?]+|$)",
        answer,
    )
    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    if not sentences:
        return None

    complete = sum(
        bool(re.search(r"[.!?][\"')\]]*$", sentence))
        for sentence in sentences
    )

    return complete / len(sentences)


def evaluate_generation(
    question: str,
    answer: str,
    context: str,
    reference_answer: str,
    max_new_tokens: int = 160,
) -> dict:
    """Evaluate faithfulness, completeness, and hallucination rate."""

    if not answer.strip() or not context.strip():
        return {
            "faithfulness": None,
            "answer_completeness": None,
            "hallucination_rate": None,
        }

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

1. faithfulness:
   Fraction of factual claims in the generated answer supported
   by the retrieved context. Score from 0 to 1.

2. answer_completeness:
   Fraction of essential facts from the reference answer correctly
   covered by the generated answer. Paraphrases are acceptable.
   Score from 0 to 1.

3. hallucination_rate:
   Fraction of factual claims in the generated answer that are
   unsupported by, or contradicted by, the retrieved context.
   Score from 0 to 1.

Treat the retrieved context as the evidence source.
Do not assume a claim is supported merely because it appears
in the reference answer.

Return only JSON. Do not include Markdown or explanations.
"""

    result = generate_answer(
        question=judge_question,
        context=judge_context,
        max_new_tokens=max_new_tokens,
    )

    raw_output = result["answer"].strip()
    print("\n--- LLM Judge Raw Output ---")
    print(raw_output)
    print("--- End Judge Output ---\n")

    # Extract JSON if the model adds surrounding text.
    match = re.search(r"\{.*\}", raw_output, re.DOTALL)

    if not match:
        raise ValueError(
            f"LLM judge did not return valid JSON: {raw_output}"
        )

    scores = json.loads(match.group(0))

    metric_names = (
        "faithfulness",
        "answer_completeness",
        "hallucination_rate",
    )

    for metric in metric_names:
        value = float(scores[metric])

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Invalid {metric} score: {value}"
            )

        scores[metric] = value

    return scores