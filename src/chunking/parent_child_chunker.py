
# src/chunking/parent_child_chunker.py
# ---------------------------------------------------------------------------------------
# handles token-based splitting and parent-child relationships.

from typing import Any


def split_token_ids(
    token_ids: list[int],
    chunk_size: int,
    overlap: int,
) -> list[list[int]]:
    """Split token IDs into overlapping windows."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(token_ids), step):
        window = token_ids[start:start + chunk_size]

        if not window:
            break

        chunks.append(window)

        if start + chunk_size >= len(token_ids):
            break

    return chunks


def decode_chunks(
    token_chunks: list[list[int]],
    tokenizer: Any,
) -> list[str]:
    """Convert token ID chunks back into text."""

    return [
        tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        for token_ids in token_chunks
    ]


def create_parent_child_chunks(
    text: str,
    source: str,
    document_id: str,
    tokenizer: Any,
    parent_size: int = 512,
    parent_overlap: int = 26,
    child_size: int = 128,
    child_overlap: int = 16,
) -> tuple[list[dict], list[dict]]:
    """
    Create parent chunks and their associated child chunks.

    Child chunks are always generated within their parent.
    """

    if not text.strip():
        return [], []

    if child_size > parent_size:
        raise ValueError("child_size cannot exceed parent_size")

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    parent_token_chunks = split_token_ids(
        token_ids,
        parent_size,
        parent_overlap,
    )

    parents = []
    children = []

    for parent_index, parent_ids in enumerate(parent_token_chunks):

        parent_id = f"{document_id}_parent_{parent_index:05d}"

        parent_text = tokenizer.decode(
            parent_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        if not parent_text:
            continue

        parent_record = {
            "parent_id": parent_id,
            "document_id": document_id,
            "parent_index": parent_index,
            "source": source,
            "text": parent_text,
            "token_count": len(parent_ids),
        }

        parents.append(parent_record)

        child_token_chunks = split_token_ids(
            parent_ids,
            child_size,
            child_overlap,
        )

        child_texts = decode_chunks(
            child_token_chunks,
            tokenizer,
        )

        for child_index, (child_ids, child_text) in enumerate(
            zip(child_token_chunks, child_texts)
        ):

            if not child_text:
                continue

            child_id = (
                f"{parent_id}_child_{child_index:03d}"
            )

            child_record = {
                "child_id": child_id,
                "parent_id": parent_id,
                "document_id": document_id,
                "child_index": child_index,
                "source": source,
                "text": child_text,
                "token_count": len(child_ids),
            }

            children.append(child_record)

    return parents, children
