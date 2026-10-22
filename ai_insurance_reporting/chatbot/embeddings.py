"""Embedding models for chatbot document representations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class EmbeddingModel(Protocol):
    """Protocol for pluggable embedding models."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts into vector representations."""


@dataclass(slots=True)
class HashingEmbeddingModel:
    """Stateless hashing-based embedding model using scikit-learn."""

    n_features: int = 256
    _vectorizer: HashingVectorizer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._vectorizer = HashingVectorizer(
            n_features=self.n_features,
            alternate_sign=False,
            norm="l2",
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().astype(float).tolist()


def get_default_embedding_model() -> EmbeddingModel:
    """Return the default local embedding model."""

    return HashingEmbeddingModel()
