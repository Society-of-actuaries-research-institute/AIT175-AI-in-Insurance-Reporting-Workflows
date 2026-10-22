"""Vector store implementations for chatbot retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(slots=True)
class VectorQueryResult:
    """A semantic retrieval result."""

    document_id: str
    score: float


class VectorStore:
    """Abstract vector store interface."""

    def add_documents(self, docs: pd.DataFrame, embeddings: list[list[float]]) -> None:
        raise NotImplementedError

    def query(self, question_embedding: list[float], k: int = 5) -> list[VectorQueryResult]:
        raise NotImplementedError


class SklearnVectorStore(VectorStore):
    """Cosine-similarity vector store using NumPy and scikit-learn."""

    def __init__(self) -> None:
        self._documents = pd.DataFrame()
        self._matrix = np.empty((0, 0))

    def add_documents(self, docs: pd.DataFrame, embeddings: list[list[float]]) -> None:
        self._documents = docs.reset_index(drop=True).copy()
        self._matrix = np.asarray(embeddings, dtype=float)

    def query(self, question_embedding: list[float], k: int = 5) -> list[VectorQueryResult]:
        if self._documents.empty or self._matrix.size == 0:
            return []
        similarities = cosine_similarity([question_embedding], self._matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:k]
        results: list[VectorQueryResult] = []
        for index in top_indices:
            results.append(
                VectorQueryResult(
                    document_id=str(self._documents.iloc[index]["document_id"]),
                    score=float(similarities[index]),
                )
            )
        return results

    def save(self, directory: Path, *, file_format: str = "csv") -> dict[str, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        metadata_path = directory / f"vector_store_documents.{file_format}"
        embeddings_path = directory / "vector_store_embeddings.npy"
        self._documents.to_csv(metadata_path, index=False)
        np.save(embeddings_path, self._matrix)
        return {"documents": metadata_path, "embeddings": embeddings_path}

    @classmethod
    def load(cls, directory: Path, *, file_format: str = "csv") -> "SklearnVectorStore":
        metadata_path = directory / f"vector_store_documents.{file_format}"
        embeddings_path = directory / "vector_store_embeddings.npy"
        if not metadata_path.exists() or not embeddings_path.exists():
            raise FileNotFoundError("Persisted vector store not found.")

        store = cls()
        store._documents = pd.read_csv(metadata_path)
        store._matrix = np.load(embeddings_path)
        return store


class FaissVectorStore(SklearnVectorStore):
    """Optional FAISS-backed vector store, falling back to sklearn behavior."""

    def __init__(self) -> None:
        super().__init__()
        self._faiss_index = None
        try:
            import faiss  # type: ignore

            self._faiss = faiss
        except ImportError:
            self._faiss = None

    def add_documents(self, docs: pd.DataFrame, embeddings: list[list[float]]) -> None:
        super().add_documents(docs, embeddings)
        if self._faiss is None or self._matrix.size == 0:
            return

        dimension = self._matrix.shape[1]
        index = self._faiss.IndexFlatIP(dimension)
        normalized = self._matrix.astype("float32")
        self._faiss.normalize_L2(normalized)
        index.add(normalized)
        self._faiss_index = index

    def query(self, question_embedding: list[float], k: int = 5) -> list[VectorQueryResult]:
        if self._faiss is None or self._faiss_index is None:
            return super().query(question_embedding, k=k)

        vector = np.asarray([question_embedding], dtype="float32")
        self._faiss.normalize_L2(vector)
        scores, indices = self._faiss_index.search(vector, k)
        results: list[VectorQueryResult] = []
        for idx, score in zip(indices[0], scores[0], strict=False):
            if idx < 0:
                continue
            results.append(
                VectorQueryResult(
                    document_id=str(self._documents.iloc[int(idx)]["document_id"]),
                    score=float(score),
                )
            )
        return results


def get_default_vector_store() -> VectorStore:
    """Return the preferred vector store implementation."""

    return FaissVectorStore()
