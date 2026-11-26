from src.evaluation.retrieval.evidence_utils import (
    calculate_sequence_coverage
)

ground_truth = """
a b x a b c
"""

retrieved_text = """
a b c
"""

result = calculate_sequence_coverage(
    ground_truth,
    retrieved_text
)

print("Ground Truth:", ground_truth)
print("Retrieved Text:", retrieved_text)
print("\nResult:")
print(result)