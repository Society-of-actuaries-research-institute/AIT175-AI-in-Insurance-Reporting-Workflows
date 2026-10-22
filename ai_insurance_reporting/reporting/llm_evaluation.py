"""LLM output evaluation and feedback logging for reporting use cases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ai_insurance_reporting.chatbot.agent import AGENT_NO_EVIDENCE_MESSAGE, ReportingAssistantAgent
from ai_insurance_reporting.chatbot.tools import build_default_tools
from ai_insurance_reporting.config.loader import AppConfig, load_config
from ai_insurance_reporting.utils.artifacts import ensure_artifact_dirs


@dataclass(slots=True)
class LLMEvaluationResult:
    """Structured outputs from benchmark evaluation and reviewer feedback summary."""

    llm_eval_results: pd.DataFrame
    llm_feedback_summary: pd.DataFrame
    llm_eval_summary: dict[str, Any]

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "llm_eval_results": self.llm_eval_results,
            "llm_feedback_summary": self.llm_feedback_summary,
        }


class LLMEvaluator:
    """Evaluate grounded assistant outputs against benchmark reporting questions.

    The evaluator is intentionally deterministic. It replays a curated benchmark
    set through the reporting assistant, inspects the returned sources, tools,
    citations, and fallback behavior, and then writes structured scores that can
    be reviewed in the dashboard or exported for the research appendix.
    """

    CATEGORY_TOOL_HINTS = {
        "forecast_movement": {"ForecastComparisonTool"},
        "driver_explanation": {"ExplainabilityTool", "ForecastComparisonTool"},
        "movement_analysis": {"MovementAnalysisTool"},
        "validation_review": {"ValidationSummaryTool"},
        "scenario_comparison": {"ScenarioRunTool", "ScenarioSummaryTool"},
        "narrative_retrieval": {"NarrativeLookupTool"},
    }
    TOOL_ARTIFACT_HINTS = {
        "ValidationSummaryTool": {"quarterly_validation_summary", "anomaly_table", "anomaly_investigation"},
        "ForecastComparisonTool": {"forecast_output_table", "model_evaluation", "backtest_predictions", "insight_summary"},
        "ExplainabilityTool": {"shap_global_importance", "shap_local_explanations", "lime_explanations", "pdp_ice_table"},
        "NarrativeLookupTool": {"narrative_statements", "narrative_quality_check"},
        "MovementAnalysisTool": {"movement_bridge_summary", "movement_analysis"},
        "FigureLookupTool": {"figure_metadata"},
        "RunSummaryTool": {"governance_log"},
        "ScenarioRunTool": {"scenario_impact_summary", "scenario_top_impacts", "scenario_narrative_summary"},
        "ScenarioSummaryTool": {"scenario_impact_summary", "scenario_top_impacts", "scenario_narrative_summary"},
        "AnalystReviewTool": {"analyst_review_queue", "analyst_review_summary"},
    }

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.settings = self.config.reporting.llm_evaluation

    def run(self, *, file_format: str = "csv") -> tuple[LLMEvaluationResult, dict[str, Path]]:
        result = self.generate(file_format=file_format)
        return result, self.write(result, file_format=file_format)

    def generate(self, *, file_format: str = "csv") -> LLMEvaluationResult:
        """Run the benchmark set and return evaluation tables plus summary metadata."""
        queries = self._load_queries()
        tools = build_default_tools(self.config)
        tools.pop("WorkflowExecutionTool", None)
        tools.pop("ScenarioRunTool", None)
        available_tool_names = set(tools)
        assistant = ReportingAssistantAgent(config=self.config, tools=tools)
        eval_rows: list[dict[str, Any]] = []
        for index, query in enumerate(queries[: int(self.settings.benchmark_query_limit)], start=1):
            question = str(query.get("question", "")).strip()
            if not question:
                continue
            response = assistant.answer(
                question,
                state={"file_format": file_format},
                file_format=file_format,
                top_k=int(self.config.reporting.chatbot.retrieval_top_k),
                prompt_mode="management_qa",
            )
            eval_rows.append(self._evaluate_response(index, query, response, available_tool_names=available_tool_names))

        eval_frame = pd.DataFrame(eval_rows)
        feedback_summary = self._build_feedback_summary()
        feedback_record_count = self._feedback_record_count()
        summary = self._build_summary(eval_frame, feedback_summary, feedback_record_count=feedback_record_count)
        return LLMEvaluationResult(
            llm_eval_results=eval_frame,
            llm_feedback_summary=feedback_summary,
            llm_eval_summary=summary,
        )

    def write(self, result: LLMEvaluationResult, *, file_format: str = "csv") -> dict[str, Path]:
        artifact_paths = ensure_artifact_dirs(self.config)
        output_dir = artifact_paths.reports / "reporting"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths: dict[str, Path] = {}
        for name, frame in result.as_dict().items():
            destination = output_dir / f"{name}.{file_format}"
            frame.to_csv(destination, index=False)
            output_paths[name] = destination
        summary_path = output_dir / "llm_eval_summary.json"
        summary_path.write_text(json.dumps(result.llm_eval_summary, indent=2, ensure_ascii=True), encoding="utf-8")
        output_paths["llm_eval_summary"] = summary_path
        return output_paths

    def record_feedback(
        self,
        *,
        question: str,
        answer: str,
        rating: str,
        grounded: str,
        helpful: str,
        comment: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        artifact_paths = ensure_artifact_dirs(self.config)
        chatbot_dir = artifact_paths.root / "chatbot"
        chatbot_dir.mkdir(parents=True, exist_ok=True)
        feedback_path = chatbot_dir / "llm_feedback_log.jsonl"
        record = {
            "question": question,
            "answer": answer,
            "rating": rating,
            "grounded": grounded,
            "helpful": helpful,
            "comment": comment,
            "reviewer_status": self.settings.reviewer_default_status,
            "metadata": metadata or {},
        }
        with feedback_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        return feedback_path

    def _load_queries(self) -> list[dict[str, Any]]:
        artifact_paths = ensure_artifact_dirs(self.config)
        query_path = artifact_paths.root / "chatbot" / "chatbot_test_queries.json"
        if not query_path.exists():
            return []
        payload = json.loads(query_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return []
        has_scenarios = (artifact_paths.root / "scenarios").exists() and any((artifact_paths.root / "scenarios").iterdir())
        filtered = []
        for item in payload:
            category = str(item.get("category", ""))
            if category == "scenario_comparison" and not has_scenarios:
                continue
            filtered.append(item)
        return filtered

    def _evaluate_response(
        self,
        index: int,
        query: dict[str, Any],
        response: Any,
        *,
        available_tool_names: set[str],
    ) -> dict[str, Any]:
        """Score one benchmark response against expected evidence and tool usage.

        The intent is not to judge prose quality. We check whether the answer
        stayed grounded, whether it touched the expected artifacts for the
        question, whether the appropriate structured tools were used, and
        whether the final answer preserved source references.
        """
        expected_artifacts = {str(item) for item in query.get("supporting_artifacts", [])}
        expected_tools = ({str(item) for item in query.get("expected_tools", [])} or self.CATEGORY_TOOL_HINTS.get(
            str(query.get("category", "")),
            set(),
        )) & available_tool_names
        source_artifacts = {str(source.get("source_dataset", "")) for source in response.sources}
        used_tools = set(response.tools_used)
        tool_artifacts = self._collect_tool_artifacts(response.tool_outputs, used_tools)
        used_fallback = response.answer == AGENT_NO_EVIDENCE_MESSAGE
        artifact_matches = expected_artifacts & (source_artifacts | tool_artifacts)
        tool_matches = expected_tools & used_tools
        grounded_score = 1.0 if (response.sources and not used_fallback) else 0.0
        artifact_score = len(artifact_matches) / max(len(expected_artifacts), 1)
        tool_score = len(tool_matches) / max(len(expected_tools), 1) if expected_tools else 1.0
        citation_score = min(len(response.citations), 1)
        overall_score = (
            grounded_score * float(self.settings.grounded_weight)
            + artifact_score * float(self.settings.artifact_match_weight)
            + tool_score * float(self.settings.tool_match_weight)
            + citation_score * float(self.settings.citation_weight)
        )
        return {
            "query_id": f"LLM-EVAL-{index:03d}",
            "category": str(query.get("category", "")),
            "question": str(query.get("question", "")),
            "expected_artifacts": ",".join(sorted(expected_artifacts)),
            "expected_tools": ",".join(sorted(expected_tools)),
            "answer": response.answer,
            "used_fallback": used_fallback,
            "source_count": int(len(response.sources)),
            "citation_count": int(len(response.citations)),
            "tools_used": ",".join(response.tools_used),
            "artifact_matches": ",".join(sorted(artifact_matches)),
            "tool_matches": ",".join(sorted(tool_matches)),
            "grounded_score": round(grounded_score, 4),
            "artifact_match_score": round(artifact_score, 4),
            "tool_match_score": round(tool_score, 4),
            "citation_score": round(float(citation_score), 4),
            "overall_score": round(float(overall_score), 4),
            "evaluation_label": "pass" if overall_score >= float(self.settings.passing_score) else "review",
        }

    def _collect_tool_artifacts(self, tool_outputs: dict[str, Any], used_tools: set[str]) -> set[str]:
        """Infer artifact coverage from the structured tool path as well as retrieval.

        Many reporting answers rely on tools that read artifacts directly rather
        than mentioning those artifact names in retrieved documents. This helper
        gives the evaluator credit for that structured evidence path so artifact
        matching reflects how the assistant actually answered.
        """
        artifacts: set[str] = set()
        for tool_name in used_tools:
            artifacts.update(self.TOOL_ARTIFACT_HINTS.get(tool_name, set()))
            payload = tool_outputs.get(tool_name, {}) if isinstance(tool_outputs, dict) else {}
            artifacts.update(self._extract_artifacts_from_payload(payload))
        return {item for item in artifacts if item}

    def _extract_artifacts_from_payload(self, payload: Any) -> set[str]:
        artifacts: set[str] = set()
        if isinstance(payload, dict):
            explicit = payload.get("artifacts_used")
            if isinstance(explicit, list):
                artifacts.update(str(item) for item in explicit)
            for value in payload.values():
                artifacts.update(self._extract_artifacts_from_payload(value))
        elif isinstance(payload, list):
            for item in payload:
                artifacts.update(self._extract_artifacts_from_payload(item))
        return artifacts

    def _feedback_record_count(self) -> int:
        artifact_paths = ensure_artifact_dirs(self.config)
        feedback_path = artifact_paths.root / "chatbot" / "llm_feedback_log.jsonl"
        if not feedback_path.exists():
            return 0
        return sum(1 for line in feedback_path.read_text(encoding="utf-8").splitlines() if line.strip())

    def _build_feedback_summary(self) -> pd.DataFrame:
        artifact_paths = ensure_artifact_dirs(self.config)
        feedback_path = artifact_paths.root / "chatbot" / "llm_feedback_log.jsonl"
        if not feedback_path.exists():
            return pd.DataFrame(columns=["dimension", "value", "count"])
        rows = [json.loads(line) for line in feedback_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            return pd.DataFrame(columns=["dimension", "value", "count"])
        frame = pd.DataFrame(rows)
        summary_rows: list[dict[str, Any]] = []
        for column in ["rating", "grounded", "helpful", "reviewer_status"]:
            counts = frame[column].value_counts(dropna=False)
            for value, count in counts.items():
                summary_rows.append({"dimension": column, "value": str(value), "count": int(count)})
        return pd.DataFrame(summary_rows)

    def _build_summary(
        self,
        eval_frame: pd.DataFrame,
        feedback_summary: pd.DataFrame,
        *,
        feedback_record_count: int,
    ) -> dict[str, Any]:
        if eval_frame.empty:
            return {
                "benchmark_queries": 0,
                "pass_rate": 0.0,
                "average_score": 0.0,
                "average_grounded_score": 0.0,
                "average_artifact_match_score": 0.0,
                "average_tool_match_score": 0.0,
                "review_items": 0,
                "feedback_records": feedback_record_count,
            }
        return {
            "benchmark_queries": int(len(eval_frame)),
            "pass_rate": round(float((eval_frame["evaluation_label"] == "pass").mean()), 4),
            "average_score": round(float(eval_frame["overall_score"].mean()), 4),
            "average_grounded_score": round(float(eval_frame["grounded_score"].mean()), 4),
            "average_artifact_match_score": round(float(eval_frame["artifact_match_score"].mean()), 4),
            "average_tool_match_score": round(float(eval_frame["tool_match_score"].mean()), 4),
            "review_items": int((eval_frame["evaluation_label"] == "review").sum()),
            "feedback_records": feedback_record_count,
        }
