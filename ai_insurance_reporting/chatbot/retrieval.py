"""Retrieval-based chatbot for reporting artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from ai_insurance_reporting.chatbot.indexing import ChatbotIndexResult, ChatbotIndexer
from ai_insurance_reporting.chatbot.rag_pipeline import NO_EVIDENCE_MESSAGE, RAGPipeline
from ai_insurance_reporting.config.loader import AppConfig, load_config
from ai_insurance_reporting.utils.artifacts import ensure_artifact_dirs


@dataclass(slots=True)
class RetrievalHit:
    """A ranked retrieval result."""

    document_id: str
    document_type: str
    title: str
    content: str
    score: float
    source_dataset: str
    source_filters: str
    source_value: str


@dataclass(slots=True)
class ChatbotResponse:
    """Chatbot answer with supporting retrieval context."""

    question: str
    answer: str
    answer_text: str
    retrieved_hits: list[RetrievalHit]
    sources: list[dict[str, object]]
    citations: list[str]


class ReportingChatbot:
    """Answer questions over indexed reporting artifacts using lexical retrieval."""

    STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "be",
        "do",
        "does",
        "for",
        "how",
        "i",
        "in",
        "is",
        "me",
        "my",
        "of",
        "on",
        "or",
        "show",
        "series",
        "tell",
        "the",
        "to",
        "what",
        "which",
    }
    SYNONYM_MAP = {
        "claims": {"claim", "claims", "loss", "losses"},
        "premium": {"premium", "premiums", "revenue"},
        "reserve": {"reserve", "reserves", "liability", "liabilities"},
        "capital": {"capital", "solvency", "ratio"},
        "validation": {"validation", "quality", "check", "checks", "issue", "issues", "anomaly", "anomalies"},
        "explainability": {"explain", "explains", "explainability", "driver", "drivers", "importance", "feature"},
        "narrative": {"narrative", "commentary", "comment", "report"},
        "charts": {"chart", "charts", "figure", "figures", "plot", "plots", "visual", "visuals", "visualization"},
        "forecast": {"forecast", "forecasts", "projected", "projection", "outlook", "future", "next"},
    }
    CATEGORY_KEYWORDS = {
        "report": {"report", "management", "content", "summary", "narrative"},
        "forecast": {"forecast", "projected", "outlook", "future"},
        "validation": {"validation", "anomaly", "quality", "issue", "reconciliation"},
        "explainability": {"shap", "lime", "explainability", "feature", "importance", "pdp", "ice"},
        "narrative": {"narrative", "commentary", "statement"},
        "charts": {"chart", "figure", "visualization", "plot", "graph"},
    }
    CATEGORY_TYPES = {
        "report": {"narrative"},
        "forecast": {"forecast", "forecast_evaluation", "forecast_backtest"},
        "validation": {"validation", "validation_anomaly"},
        "explainability": {"explainability_global", "explainability_local"},
        "narrative": {"narrative"},
        "charts": {"chart"},
    }

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.indexer = ChatbotIndexer(self.config)
        self.rag_pipeline = RAGPipeline(self.config)

    def build_index(
        self,
        *,
        file_format: str = "csv",
    ) -> tuple[ChatbotIndexResult, dict[str, Path]]:
        """Build a retrieval index from generated artifacts."""

        return self.rag_pipeline.build_knowledge_base(file_format=file_format)

    def load_index(self, *, file_format: str = "csv") -> pd.DataFrame:
        """Load the persisted chatbot index."""

        artifact_paths = ensure_artifact_dirs(self.config)
        path = artifact_paths.root / "chatbot" / f"chatbot_index.{file_format}"
        if not path.exists():
            raise FileNotFoundError(f"Chatbot index not found: {path}")
        return pd.read_csv(path)

    def answer_question(
        self,
        question: str,
        *,
        top_k: int = 5,
        file_format: str = "csv",
        index_frame: pd.DataFrame | None = None,
        prompt_mode: str = "management_qa",
    ) -> ChatbotResponse:
        """Answer a user question from the indexed documents using RAG."""

        index_frame = index_frame if index_frame is not None else self.load_index(file_format=file_format)
        rag_answer = self.rag_pipeline.answer(
            question,
            top_k=top_k,
            file_format=file_format,
            index_frame=index_frame,
            prompt_mode=prompt_mode,
        )
        retrieved_hits = self._sources_to_hits(rag_answer.sources, index_frame=index_frame)
        formatted_answer = rag_answer.answer
        if rag_answer.answer != NO_EVIDENCE_MESSAGE:
            lines = [f"Answer: {rag_answer.answer}"]
            if rag_answer.citations:
                lines.append(f"Citations: {' '.join(f'[{citation}]' for citation in rag_answer.citations)}")
            lines.append("Sources:")
            for source in rag_answer.sources[:3]:
                lines.append(
                    f"- {source.document} ({source.source_dataset}) "
                    f"[score={source.score}]"
                )
            formatted_answer = "\n".join(lines)
        return ChatbotResponse(
            question=question,
            answer=formatted_answer,
            answer_text=rag_answer.answer,
            retrieved_hits=retrieved_hits,
            sources=rag_answer.to_dict()["sources"],
            citations=rag_answer.citations,
        )

    def retrieve(
        self,
        question: str,
        *,
        index_frame: pd.DataFrame,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        """Retrieve the most relevant documents by lexical overlap."""

        tokens = self._tokenize(question)
        category = self._detect_category(tokens)
        rows: list[RetrievalHit] = []

        for row in index_frame.itertuples(index=False):
            haystack_text = " ".join(
                [
                    str(row.title),
                    str(row.content),
                    str(row.keywords),
                    str(row.source_dataset),
                    str(row.section),
                ]
            ).lower()
            haystack_tokens = self._tokenize(haystack_text)
            score = 0.0
            for token in tokens:
                if token in haystack_tokens:
                    score += 2.0
                elif len(token) >= 5 and any(part.startswith(token) or token.startswith(part) for part in haystack_tokens):
                    score += 0.5
            if category and str(row.document_type) in self.CATEGORY_TYPES[category]:
                score += 2.0
            if score <= 0:
                continue
            rows.append(
                RetrievalHit(
                    document_id=str(row.document_id),
                    document_type=str(row.document_type),
                    title=str(row.title),
                    content=str(row.content),
                    score=score,
                    source_dataset=str(row.source_dataset),
                    source_filters=str(row.source_filters),
                    source_value=str(row.source_value),
                )
            )

        if not rows and category:
            rows = self._fallback_category_hits(index_frame, category=category)

        rows.sort(key=lambda hit: (-hit.score, hit.document_id))
        return rows[:top_k]

    def _compose_answer(self, question: str, hits: list[RetrievalHit]) -> str:
        if not hits:
            return NO_EVIDENCE_MESSAGE

        lines = [f"Answer: {hits[0].content}"]
        if len(hits) > 1:
            lines.append("Supporting context:")
            for hit in hits[1: min(3, len(hits))]:
                lines.append(f"- {hit.content}")
        lines.append("Sources:")
        for hit in hits[: min(3, len(hits))]:
            lines.append(f"- {hit.source_dataset} [{hit.source_filters}] -> {hit.source_value}")
        return "\n".join(lines)

    def _sources_to_hits(self, sources: list[object], *, index_frame: pd.DataFrame) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        for source in sources:
            match = index_frame.loc[index_frame["document_id"] == source.document_id]
            if match.empty:
                continue
            row = match.iloc[0]
            hits.append(
                RetrievalHit(
                    document_id=str(source.document_id),
                    document_type=str(row["document_type"]),
                    title=str(row["title"]),
                    content=str(row["content"]),
                    score=float(source.score),
                    source_dataset=str(source.source_dataset),
                    source_filters=str(source.source_filters),
                    source_value=str(row["source_value"]),
                )
            )
        return hits

    def _detect_category(self, tokens: set[str]) -> str | None:
        best_category: str | None = None
        best_overlap = 0
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            overlap = len(tokens & keywords)
            if overlap > best_overlap:
                best_category = category
                best_overlap = overlap
        return best_category

    def _tokenize(self, text: str) -> set[str]:
        cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
        tokens = {
            token
            for token in cleaned.split()
            if token and token not in self.STOPWORDS and not token.isdigit()
        }
        normalized = set(tokens)
        for token in tokens:
            if token.endswith("s") and len(token) > 3:
                normalized.add(token[:-1])
            if token.endswith("ing") and len(token) > 5:
                normalized.add(token[:-3])
            if token.endswith("ed") and len(token) > 4:
                normalized.add(token[:-2])

        expanded = set(normalized)
        for token in normalized:
            for canonical, variants in self.SYNONYM_MAP.items():
                if token == canonical or token in variants:
                    expanded.update(variants)
                    expanded.add(canonical)
        return expanded

    def _fallback_category_hits(self, index_frame: pd.DataFrame, *, category: str) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        allowed_types = self.CATEGORY_TYPES[category]
        subset = index_frame.loc[index_frame["document_type"].isin(allowed_types)].head(5)
        for row in subset.itertuples(index=False):
            hits.append(
                RetrievalHit(
                    document_id=str(row.document_id),
                    document_type=str(row.document_type),
                    title=str(row.title),
                    content=str(row.content),
                    score=1.0,
                    source_dataset=str(row.source_dataset),
                    source_filters=str(row.source_filters),
                    source_value=str(row.source_value),
                )
            )
        return hits
