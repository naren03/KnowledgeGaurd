from typing import Any, Iterable

import numpy as np


def retrieve_chunks(
    question: str,
    embedding_model: Any,
    index: Any,
    chunks: list[dict[str, Any]],
    user_groups: Iterable[str],
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Return the highest-ranked chunks the user's groups are allowed to read."""
    if limit <= 0 or index.ntotal == 0 or not chunks:
        return []

    question_vector = embedding_model.encode(
        [question], normalize_embeddings=True
    )
    question_vector = np.asarray(question_vector, dtype="float32")

    _, chunk_indexes = index.search(question_vector, k=index.ntotal)
    groups = set(user_groups)
    selected_chunks = []

    for chunk_index in chunk_indexes[0]:
        chunk_index = int(chunk_index)
        if chunk_index < 0 or chunk_index >= len(chunks):
            continue

        chunk = chunks[chunk_index]
        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            continue

        allowed_groups = metadata.get("allowed_groups", [])
        if not isinstance(allowed_groups, list):
            continue

        if any(
            isinstance(group, str) and group in groups for group in allowed_groups
        ):
            selected_chunks.append(chunk)
            if len(selected_chunks) == limit:
                break

    return selected_chunks