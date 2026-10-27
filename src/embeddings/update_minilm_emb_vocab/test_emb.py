# src/embeddings/test_embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

from src.utils.helpers import load_config, get_queries
from src.retrieval.dense_retrieval_minilm import load_embeddings

config = load_config()
model = SentenceTransformer(config["retrieval"]["model"])
print("Maximum sequence length:", model.max_seq_length)

query = get_queries()[0]["query"]
print('\nquery')
print("Tokens:", model.tokenizer.tokenize(query))
norms = np.linalg.norm(load_embeddings(), axis=1)

print("Minimum document embedding norm:", norms.min())
print("Maximum document embedding norm:", norms.max())

print()
technical_terms = [
    "Kubernetes",
    "CrashLoopBackOff",
    "APIService",
    "API Priority and Fairness",
    "ResourceQuota",
    "HorizontalPodAutoscaler",
    "kubectl",
    "etcd",
    "RBAC",
    "CustomResourceDefinition",
]

for term in technical_terms:
    tokens = model.tokenizer.tokenize(term)
    print(f"{term:30} -> {tokens}")