# src/evaluation/retrieval/evidence_utils.py
# --------------------------------------------------------
# will contain the shared evidence-matching logic used by the retrieval metrics

import re
from collections import Counter


def tokenize_evidence(text: str) -> list[str]:
    """Tokenize text for lexical evidence comparison."""
    return re.findall(r"\b\w+\b", text.lower())


def calculate_evidence_overlap(
    ground_truth_context: str,
    retrieved_results: list[dict],
) -> tuple[int, int, int]:
    """
    Return:
        matched_tokens,
        total_ground_truth_tokens,
        total_retrieved_tokens

    Token frequencies are capped by their frequency in the
    ground-truth evidence, preventing repeated retrieval text
    from artificially increasing evidence coverage.
    """

    gt_tokens = Counter(tokenize_evidence(ground_truth_context))

    retrieved_tokens = [
        token
        for result in retrieved_results
        for token in tokenize_evidence(result.get("text", ""))
    ]

    retrieved_counts = Counter(retrieved_tokens)

    matched_tokens = sum(
        min(count, retrieved_counts[token])
        for token, count in gt_tokens.items()
    )

    return (
        matched_tokens,
        sum(gt_tokens.values()),
        len(retrieved_tokens),
    )


def calculate_result_relevance(
    result_text: str,
    ground_truth_context: str,
) -> float:
    """
    Calculate lexical F1 between one retrieved result
    and the ground-truth evidence.
    """

    result_tokens = set(tokenize_evidence(result_text))
    gt_tokens = set(tokenize_evidence(ground_truth_context))

    if not result_tokens or not gt_tokens:
        return 0.0

    overlap = len(result_tokens & gt_tokens)

    precision = overlap / len(result_tokens)
    recall = overlap / len(gt_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def is_result_relevant(
    result: dict,
    context,
) -> bool:
    """Determine whether a retrieved result is evidence-relevant."""

    score = calculate_result_relevance(
        result.get("text", ""),
        context.ground_truth_context,
    )

    return score >= context.relevance_threshold

import re
from collections import Counter


def calculate_ground_truth_similarity(
    ground_truth_context: str,
    retrieved_text: str,
) -> float:
    """
    Calculate token-based similarity between ground-truth
    context and a retrieved chunk.

    Uses ground-truth token coverage.
    """

    gt_tokens = Counter(
        re.findall(r"\b\w+\b", ground_truth_context.lower())
    )

    retrieved_tokens = Counter(
        re.findall(r"\b\w+\b", retrieved_text.lower())
    )

    if not gt_tokens:
        return 0.0

    matching_tokens = sum(
        min(count, retrieved_tokens[token])
        for token, count in gt_tokens.items()
    )

    return matching_tokens / sum(gt_tokens.values())