"""Agentic analytical assistant built on top of the RAG chatbot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ai_insurance_reporting.chatbot.planner import AgentPlanner
from ai_insurance_reporting.chatbot.rag_pipeline import NO_EVIDENCE_MESSAGE, RAGPipeline, RAGSource
from ai_insurance_reporting.chatbot.tools import BaseTool, build_default_tools
from ai_insurance_reporting.config.loader import AppConfig, load_config
from ai_insurance_reporting.utils.artifacts import ensure_artifact_dirs


AGENT_NO_EVIDENCE_MESSAGE = (
    "The available reporting artifacts do not contain enough information to answer this question reliably."
)


@dataclass(slots=True)
class AgentAnswer:
    """Structured response from the analytical assistant."""

    answer: str
    sources: list[dict[str, object]]
    tools_used: list[str]
    tool_outputs: dict[str, Any]
    citations: list[str]
    retrieved_document_ids: list[str]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for logs, UI state, and tests."""
        return {
            "answer": self.answer,
            "sources": self.sources,
            "tools_used": self.tools_used,
            "tool_outputs": self.tool_outputs,
            "citations": self.citations,
            "retrieved_document_ids": self.retrieved_document_ids,
        }


class ReportingAssistantAgent:
    """Tool-augmented question answering layer for reporting artifacts.

    The agent decides which structured tools to call, retrieves supporting
    document evidence, and then builds a grounded prompt for the configured LLM
    client. The final answer always includes source and tool trace data so the
    UI and CLI can present the supporting evidence separately from the answer
    text.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        planner: AgentPlanner | None = None,
        tools: dict[str, BaseTool] | None = None,
        rag_pipeline: RAGPipeline | None = None,
    ) -> None:
        self.config = config or load_config()
        self.planner = planner or AgentPlanner()
        self.tools = tools or build_default_tools(self.config)
        self.rag_pipeline = rag_pipeline or RAGPipeline(self.config)

    def answer(
        self,
        question: str,
        state: dict[str, Any] | None = None,
        *,
        top_k: int = 5,
        file_format: str = "csv",
        prompt_mode: str = "management_qa",
    ) -> AgentAnswer:
        """Answer a question using retrieval and optional structured tools.

        Retrieval remains the common grounding layer. Structured tools are used
        for questions that benefit from direct access to validation, forecast,
        explainability, narrative, figure, run-summary, or scenario outputs.
        """

        state = dict(state or {})
        state.setdefault("file_format", file_format)
        index_frame = self.rag_pipeline.load_index(file_format=file_format)
        planned_tools = self.planner.plan(question)
        tools_used = [tool_name for tool_name in planned_tools if tool_name in self.tools]
        tool_outputs: dict[str, Any] = {}

        for tool_name in tools_used:
            tool_outputs[tool_name] = self.tools[tool_name].run(question, state)

        lexical_hits = self.rag_pipeline._lexical_retrieve(question, index_frame=index_frame, top_k=top_k * 2)
        if not lexical_hits and not tool_outputs:
            answer = AgentAnswer(
                answer=AGENT_NO_EVIDENCE_MESSAGE,
                sources=[],
                tools_used=tools_used,
                tool_outputs=tool_outputs,
                citations=[],
                retrieved_document_ids=[],
            )
            self._log_interaction(question, answer, tool_outputs)
            return answer
        semantic_hits = self.rag_pipeline._semantic_retrieve(question, top_k=top_k * 2)
        strongest_lexical = max((score for _, score in lexical_hits), default=0.0)
        strongest_semantic = max((score for _, score in semantic_hits), default=0.0)
        combined_sources = self.rag_pipeline._combine_hits(index_frame, lexical_hits, semantic_hits, top_k=top_k)

        if (not combined_sources or (strongest_lexical < 2.0 and strongest_semantic < 0.1)) and not tool_outputs:
            answer = AgentAnswer(
                answer=AGENT_NO_EVIDENCE_MESSAGE,
                sources=[],
                tools_used=tools_used,
                tool_outputs=tool_outputs,
                citations=[],
                retrieved_document_ids=[],
            )
            self._log_interaction(question, answer, tool_outputs)
            return answer

        prompt = self._build_agent_prompt(
            question,
            tool_outputs=tool_outputs,
            sources=combined_sources,
            index_frame=index_frame,
            prompt_mode=prompt_mode,
        )
        raw_answer = self.rag_pipeline.llm_client.generate(prompt).strip()
        if not raw_answer:
            raw_answer = AGENT_NO_EVIDENCE_MESSAGE

        citations = [source.document_id for source in combined_sources[:3]]
        sources = [
            {
                "document": source.document,
                "score": source.score,
                "document_id": source.document_id,
                "source_dataset": source.source_dataset,
                "source_filters": source.source_filters,
            }
            for source in combined_sources
        ]
        answer = AgentAnswer(
            answer=raw_answer if raw_answer != NO_EVIDENCE_MESSAGE else AGENT_NO_EVIDENCE_MESSAGE,
            sources=sources,
            tools_used=tools_used,
            tool_outputs=tool_outputs,
            citations=citations,
            retrieved_document_ids=[source.document_id for source in combined_sources],
        )
        self._log_interaction(question, answer, tool_outputs)
        return answer

    def _build_agent_prompt(
        self,
        question: str,
        *,
        tool_outputs: dict[str, Any],
        sources: list[RAGSource],
        index_frame: pd.DataFrame,
        prompt_mode: str,
    ) -> str:
        """Build the final grounded prompt passed to the configured LLM client.

        The prompt combines structured tool outputs and retrieved document text.
        Tool payloads are intentionally kept explicit because they represent the
        highest-trust evidence path for questions about forecasts, scenarios,
        validation, movement analysis, and review items.
        """
        context_chunks: list[str] = []
        for source in sources:
            row = index_frame.loc[index_frame["document_id"] == source.document_id].iloc[0]
            context_chunks.append(
                f"[{source.document_id}] {row['title']} | dataset={row['source_dataset']} | "
                f"filters={row['source_filters']} | content={row['content']}"
            )

        tool_output_text = json.dumps(tool_outputs, indent=2)
        if prompt_mode == "management_qa":
            header = (
                "You are an insurance management reporting assistant.\n"
                "Use only the provided context and tool outputs to answer the question.\n"
                "If tool outputs and retrieved documents differ, prioritize the structured tool outputs.\n"
                f"If the evidence is insufficient, respond exactly with: {AGENT_NO_EVIDENCE_MESSAGE}\n"
            )
        else:
            header = (
                "You are an actuarial reporting assistant.\n"
                "Use only the provided context and tool outputs to answer the question.\n"
                "If tool outputs and retrieved documents differ, prioritize the structured tool outputs.\n"
                f"If the evidence is insufficient, respond exactly with: {AGENT_NO_EVIDENCE_MESSAGE}\n"
            )

        return (
            header
            + "\nQuestion:\n"
            + question
            + "\n\nTool outputs:\n"
            + tool_output_text
            + "\n\nRetrieved context:\n"
            + "\n".join(context_chunks)
            + "\n\nRequirements:\n"
            + "- answer clearly and concisely\n"
            + "- explain relevant drivers where applicable\n"
            + "- reference source artifacts\n"
            + "- do not invent facts"
        )

    def _log_interaction(self, question: str, answer: AgentAnswer, tool_outputs: dict[str, Any]) -> None:
        """Persist a compact audit trail for each assistant interaction."""
        artifact_paths = ensure_artifact_dirs(self.config)
        chatbot_dir = artifact_paths.root / "chatbot"
        chatbot_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).isoformat()
        history_record = {
            "timestamp": timestamp,
            "question": question,
            "retrieved_docs": answer.retrieved_document_ids,
            "tools_used": answer.tools_used,
            "final_answer": answer.answer,
            "source_references": answer.sources,
        }
        self._append_jsonl(chatbot_dir / "chat_history.jsonl", history_record)

        tool_record = {
            "timestamp": timestamp,
            "question": question,
            "tools_used": answer.tools_used,
            "tool_outputs": tool_outputs,
        }
        self._append_jsonl(chatbot_dir / "tool_usage_log.jsonl", tool_record)

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
