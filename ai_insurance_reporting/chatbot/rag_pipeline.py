"""RAG pipeline for grounded chatbot answers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from ai_insurance_reporting.chatbot.embeddings import EmbeddingModel, get_default_embedding_model
from ai_insurance_reporting.chatbot.indexing import ChatbotIndexResult, ChatbotIndexer
from ai_insurance_reporting.chatbot.llm_client import LLMClient, get_default_llm_client
from ai_insurance_reporting.chatbot.vector_store import (
    SklearnVectorStore,
    VectorQueryResult,
    VectorStore,
    get_default_vector_store,
)
from ai_insurance_reporting.config.loader import AppConfig, load_config
from ai_insurance_reporting.utils.artifacts import ensure_artifact_dirs


NO_EVIDENCE_MESSAGE = "The reporting artifacts do not contain enough information to answer this question."


@dataclass(slots=True)
class RAGSource:
    """Grounding source for an answer."""

    document: str
    score: float
    document_id: str
    source_dataset: str
    source_filters: str


@dataclass(slots=True)
class RAGAnswer:
    """Structured RAG response."""

    answer: str
    sources: list[RAGSource]
    citations: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "sources": [
                {
                    "document": source.document,
                    "score": source.score,
                    "document_id": source.document_id,
                    "source_dataset": source.source_dataset,
                    "source_filters": source.source_filters,
                }
                for source in self.sources
            ],
        }


class RAGPipeline:
    """Combine retrieval, prompt construction, and grounded answer generation.

    The pipeline keeps document indexing and retrieval separate from answer
    generation so the same evidence base can support both direct RAG usage and
    the higher-level analytical agent.
    """

    PROMPT_MODES = ("default", "management_qa")

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        embedding_model: EmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.config = config or load_config()
        self.indexer = ChatbotIndexer(self.config)
        self.embedding_model = embedding_model or get_default_embedding_model()
        self.vector_store = vector_store or get_default_vector_store()
        self.llm_client = llm_client or get_default_llm_client()

    def build_knowledge_base(
        self,
        *,
        file_format: str = "csv",
    ) -> tuple[ChatbotIndexResult, dict[str, Path]]:
        """Build the lexical index and semantic vector store."""

        index_result, output_paths = self.indexer.run(file_format=file_format)
        index_frame = index_result.chatbot_index.copy()
        index_frame["document_text"] = index_frame.apply(
            lambda row: " ".join(
                [
                    str(row["title"]),
                    str(row["content"]),
                    str(row["keywords"]),
                    str(row["source_dataset"]),
                    str(row["source_filters"]),
                ]
            ),
            axis=1,
        )
        embeddings = self.embedding_model.embed(index_frame["document_text"].tolist())
        self.vector_store.add_documents(index_frame, embeddings)

        artifact_paths = ensure_artifact_dirs(self.config)
        store_dir = artifact_paths.root / "chatbot" / "vector_store"
        if isinstance(self.vector_store, SklearnVectorStore):
            vector_paths = self.vector_store.save(store_dir, file_format=file_format)
        else:
            vector_paths = {}

        output_paths.update({f"vector_store_{key}": value for key, value in vector_paths.items()})
        return index_result, output_paths

    def load_index(self, *, file_format: str = "csv") -> pd.DataFrame:
        artifact_paths = ensure_artifact_dirs(self.config)
        path = artifact_paths.root / "chatbot" / f"chatbot_index.{file_format}"
        if not path.exists():
            raise FileNotFoundError(f"Chatbot index not found: {path}")
        return pd.read_csv(path)

    def load_vector_store(self, *, file_format: str = "csv") -> VectorStore:
        artifact_paths = ensure_artifact_dirs(self.config)
        store_dir = artifact_paths.root / "chatbot" / "vector_store"
        if isinstance(self.vector_store, SklearnVectorStore):
            self.vector_store = SklearnVectorStore.load(store_dir, file_format=file_format)
            return self.vector_store
        raise FileNotFoundError("Persisted vector store is not available for this backend.")

    def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
        file_format: str = "csv",
        index_frame: pd.DataFrame | None = None,
        prompt_mode: str = "management_qa",
    ) -> RAGAnswer:
        """Retrieve context, build a prompt, and generate a grounded answer."""

        if prompt_mode not in self.PROMPT_MODES:
            raise ValueError(f"Unsupported prompt mode: {prompt_mode}")
        index_frame = index_frame if index_frame is not None else self.load_index(file_format=file_format)
        if isinstance(self.vector_store, SklearnVectorStore) and self.vector_store._documents.empty:
            try:
                self.load_vector_store(file_format=file_format)
            except FileNotFoundError:
                self.build_knowledge_base(file_format=file_format)
                index_frame = self.load_index(file_format=file_format)

        lexical_hits = self._lexical_retrieve(question, index_frame=index_frame, top_k=top_k * 2)
        if not lexical_hits:
            return RAGAnswer(answer=NO_EVIDENCE_MESSAGE, sources=[], citations=[])
        semantic_hits = self._semantic_retrieve(question, top_k=top_k * 2)
        combined = self._combine_hits(index_frame, lexical_hits, semantic_hits, top_k=top_k)

        if not combined:
            return RAGAnswer(answer=NO_EVIDENCE_MESSAGE, sources=[], citations=[])

        prompt = self._build_prompt(question, combined, index_frame=index_frame, prompt_mode=prompt_mode)
        raw_answer = self.llm_client.generate(prompt).strip()
        if not raw_answer:
            raw_answer = NO_EVIDENCE_MESSAGE

        citations = [source.document_id for source in combined[:3]]
        return RAGAnswer(answer=raw_answer, sources=combined, citations=citations)

    def _lexical_retrieve(self, question: str, *, index_frame: pd.DataFrame, top_k: int) -> list[tuple[str, float]]:
        from ai_insurance_reporting.chatbot.retrieval import ReportingChatbot

        retriever = ReportingChatbot(config=self.config)
        hits = retriever.retrieve(question, index_frame=index_frame, top_k=top_k)
        return [(hit.document_id, hit.score) for hit in hits]

    def _semantic_retrieve(self, question: str, *, top_k: int) -> list[tuple[str, float]]:
        embedding = self.embedding_model.embed([question])[0]
        hits = self.vector_store.query(embedding, k=top_k)
        return [(hit.document_id, hit.score) for hit in hits if hit.score >= 0.05]

    def _combine_hits(
        self,
        index_frame: pd.DataFrame,
        lexical_hits: list[tuple[str, float]],
        semantic_hits: list[tuple[str, float]],
        *,
        top_k: int,
    ) -> list[RAGSource]:
        score_map: dict[str, float] = {}
        for document_id, score in lexical_hits:
            score_map[document_id] = score_map.get(document_id, 0.0) + (score * 1.0)
        for document_id, score in semantic_hits:
            score_map[document_id] = score_map.get(document_id, 0.0) + (score * 3.0)

        ranked_ids = sorted(score_map.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        sources: list[RAGSource] = []
        for document_id, score in ranked_ids:
            match = index_frame.loc[index_frame["document_id"] == document_id]
            if match.empty:
                continue
            row = match.iloc[0]
            sources.append(
                RAGSource(
                    document=str(row["title"]),
                    score=round(float(score), 4),
                    document_id=str(row["document_id"]),
                    source_dataset=str(row["source_dataset"]),
                    source_filters=str(row["source_filters"]),
                )
            )
        return sources

    def _build_prompt(
        self,
        question: str,
        sources: list[RAGSource],
        *,
        index_frame: pd.DataFrame,
        prompt_mode: str,
    ) -> str:
        context_chunks: list[str] = []
        for source in sources:
            row = index_frame.loc[index_frame["document_id"] == source.document_id].iloc[0]
            context_chunks.append(
                f"[{source.document_id}] {row['title']} | dataset={row['source_dataset']} | "
                f"filters={row['source_filters']} | content={row['content']}"
            )

        if prompt_mode == "management_qa":
            header = (
                "You are assisting senior management reviewing an insurance reporting package.\n"
                "Explain results clearly and reference supporting metrics.\n"
                "Use only the information from the context below to answer the question.\n"
                f"If the context is insufficient, respond exactly with: {NO_EVIDENCE_MESSAGE}\n\n"
            )
            footer = (
                "\n\nProvide:\n"
                "- a concise management summary\n"
                "- explanation of key drivers where applicable\n"
                "- references to the supporting artifacts"
            )
        else:
            header = (
                "You are an actuarial reporting assistant.\n\n"
                "Use only the information from the context below to answer the question.\n"
                "If the context is insufficient, respond exactly with: "
                f"{NO_EVIDENCE_MESSAGE}\n\n"
            )
            footer = (
                "\n\nProvide:\n"
                "- a concise answer\n"
                "- explanation of drivers if applicable\n"
                "- references to the source artifacts."
            )

        return (
            header
            + 
            "Context:\n"
            + "\n".join(context_chunks)
            + "\n\nQuestion:\n"
            + question
            + footer
        )
