# src/evaluation/retrieval/evidence_utils.py
# --------------------------------------------------------
# will contain the shared evidence-matching logic used by the retrieval metrics

import re
from collections import Counter, defaultdict


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


def calculate_sequence_coverage(
    ground_truth_context: str,
    retrieved_text: str,
) -> dict:
    """
    Find the longest consecutive token sequence shared by
    ground truth and a retrieved chunk.
    """

    gt_tokens = tokenize_evidence(ground_truth_context)
    retrieved_tokens = tokenize_evidence(retrieved_text)

    empty_result = {
        "sequence_coverage": 0.0,
        "longest_match": 0,
        "ground_truth_start": None,
        "retrieved_start": None,
        "matching_sequence": [],
    }

    if not gt_tokens or not retrieved_tokens:
        return empty_result

    token_positions = defaultdict(list)

    for index, token in enumerate(gt_tokens):
        token_positions[token].append(index)

    # Stores matches ending at each GT position
    # for the previous retrieved token.
    previous = {}

    max_match = 0
    best_gt_end = None
    best_retrieved_end = None

    for retrieved_index, token in enumerate(retrieved_tokens):
        current = {}

        for gt_index in token_positions.get(token, []):
            if gt_index == 0:
                match_length = 1
            else:
                match_length = previous.get(gt_index - 1, 0) + 1

            current[gt_index] = match_length

            if match_length > max_match:
                max_match = match_length
                best_gt_end = gt_index
                best_retrieved_end = retrieved_index

        previous = current

    if max_match == 0:
        return empty_result

    gt_start = best_gt_end - max_match + 1
    retrieved_start = best_retrieved_end - max_match + 1

    return {
        "sequence_coverage": max_match / len(gt_tokens),
        "longest_match": max_match,
        "ground_truth_start": gt_start,
        "retrieved_start": retrieved_start,
        "matching_sequence": gt_tokens[
            gt_start:best_gt_end + 1
        ],
    }