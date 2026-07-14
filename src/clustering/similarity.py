"""Vector similarity helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity for equal-length vectors."""

    if len(left) != len(right):
        raise ValueError("Embedding vectors must have the same dimensions.")
    if not left:
        return 0.0
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot_product / (left_norm * right_norm)))


def mean_embedding(vectors: Sequence[Sequence[float]]) -> list[float]:
    """Return the dimension-wise centroid for a set of embeddings."""

    if not vectors:
        raise ValueError("At least one embedding is required.")
    dimensions = len(vectors[0])
    if dimensions == 0 or any(len(vector) != dimensions for vector in vectors):
        raise ValueError("Embedding vectors must be non-empty and equally sized.")
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimensions)]
