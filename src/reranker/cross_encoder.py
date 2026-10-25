# src/reranker/cross_encoder
from typing import Any

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Rerank retrieved document chunks using query-document relevance scores."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
    ) -> None:
        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
        )

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 20,
        text_key: str = "text",
    ) -> list[dict[str, Any]]:
        """
        Rescore candidate chunks and return the top-k by relevance.

        Each candidate must contain its text under text_key.
        Existing candidate metadata is preserved.
        """
        if not candidates:
            return []

        pairs = [
            (query, str(candidate[text_key]))
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        reranked = []

        for candidate, score in zip(candidates, scores):
            result = candidate.copy()
            result["reranker_score"] = float(score)
            reranked.append(result)

        reranked.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

        return reranked[:top_k]
