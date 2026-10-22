"""Sequential workflow orchestrator for reporting agents."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from ai_insurance_reporting.config.loader import AppConfig, load_config
from ai_insurance_reporting.utils.file_formats import normalize_tabular_output_format
from ai_insurance_reporting.orchestration.agents import (
    AgentExecutionResult,
    AnalystReviewQueueAgent,
    AnomalyInvestigationAgent,
    ChatbotIndexingAgent,
    ExplainabilityAgent,
    FeatureEngineeringAgent,
    ForecastingAgent,
    FullReportAgent,
    IngestionAgent,
    InsightDetectionAgent,
    LLMEvaluationAgent,
    MovementAnalysisAgent,
    NarrativeAgent,
    NarrativeQualityAgent,
    ValidationAgent,
    VisualizationAgent,
    WorkflowAgent,
)


@dataclass(slots=True)
class WorkflowRunResult:
    """End-to-end workflow execution summary."""

    execution_log: list[AgentExecutionResult]
    final_context: dict[str, object]
    governance_log_path: Path | None = None

    def output_paths(self) -> dict[str, Path]:
        """Flatten output paths from all agents."""

        paths: dict[str, Path] = {}
        for item in self.execution_log:
            for key, value in item.output_paths.items():
                paths[f"{item.agent_name}.{key}"] = value
        if self.governance_log_path is not None:
            paths["governance_log"] = self.governance_log_path
        return paths


class WorkflowOrchestrator:
    """Run workflow agents sequentially."""

    AGENT_DEPENDENCIES: dict[str, tuple[str, ...]] = {
        "IngestionAgent": (),
        "ValidationAgent": ("IngestionAgent",),
        "FeatureEngineeringAgent": ("IngestionAgent",),
        "ForecastingAgent": ("FeatureEngineeringAgent",),
        "ExplainabilityAgent": ("FeatureEngineeringAgent", "ForecastingAgent"),
        "InsightDetectionAgent": ("IngestionAgent", "ForecastingAgent"),
        "AnomalyInvestigationAgent": ("IngestionAgent", "ValidationAgent", "InsightDetectionAgent", "ExplainabilityAgent"),
        "MovementAnalysisAgent": ("IngestionAgent", "ForecastingAgent"),
        "NarrativeAgent": ("IngestionAgent", "ValidationAgent", "ForecastingAgent", "MovementAnalysisAgent"),
        "NarrativeQualityAgent": ("IngestionAgent", "ForecastingAgent", "ExplainabilityAgent", "NarrativeAgent"),
        "VisualizationAgent": ("IngestionAgent", "ValidationAgent", "ForecastingAgent", "ExplainabilityAgent", "MovementAnalysisAgent"),
        "FullReportAgent": ("ValidationAgent", "InsightDetectionAgent", "AnomalyInvestigationAgent", "MovementAnalysisAgent", "NarrativeAgent", "NarrativeQualityAgent", "VisualizationAgent"),
        "ChatbotIndexingAgent": ("FullReportAgent",),
        "LLMEvaluationAgent": ("ChatbotIndexingAgent",),
        "AnalystReviewQueueAgent": ("InsightDetectionAgent", "AnomalyInvestigationAgent", "NarrativeQualityAgent", "LLMEvaluationAgent"),
    }

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.agents: list[WorkflowAgent] = [
            IngestionAgent(self.config),
            ValidationAgent(self.config),
            FeatureEngineeringAgent(self.config),
            ForecastingAgent(self.config),
            ExplainabilityAgent(self.config),
            InsightDetectionAgent(self.config),
            AnomalyInvestigationAgent(self.config),
            MovementAnalysisAgent(self.config),
            NarrativeAgent(self.config),
            NarrativeQualityAgent(self.config),
            VisualizationAgent(self.config),
            FullReportAgent(self.config),
            ChatbotIndexingAgent(self.config),
            LLMEvaluationAgent(self.config),
            AnalystReviewQueueAgent(self.config),
        ]
        self.agent_map = {agent.agent_name: agent for agent in self.agents}

    def run(
        self,
        *,
        file_format: str = "csv",
        initial_context: dict[str, object] | None = None,
    ) -> WorkflowRunResult:
        file_format = normalize_tabular_output_format(file_format)
        context: dict[str, object] = dict(initial_context or {})
        execution_log: list[AgentExecutionResult] = []

        for agent in self.agents:
            result = agent.run(context, file_format=file_format)
            execution_log.append(result)

        governance_log_path = self._write_governance_log(execution_log, file_format=file_format)
        return WorkflowRunResult(
            execution_log=execution_log,
            final_context=context,
            governance_log_path=governance_log_path,
        )

    def run_selected(
        self,
        agent_names: list[str],
        *,
        file_format: str = "csv",
        initial_context: dict[str, object] | None = None,
    ) -> WorkflowRunResult:
        file_format = normalize_tabular_output_format(file_format)
        execution_plan = self.resolve_execution_plan(agent_names)
        context: dict[str, object] = dict(initial_context or {})
        execution_log: list[AgentExecutionResult] = []

        for agent_name in execution_plan:
            agent = self.agent_map[agent_name]
            result = agent.run(context, file_format=file_format)
            execution_log.append(result)

        governance_log_path = self._write_governance_log(execution_log, file_format=file_format)
        return WorkflowRunResult(
            execution_log=execution_log,
            final_context=context,
            governance_log_path=governance_log_path,
        )

    def resolve_execution_plan(self, agent_names: list[str]) -> list[str]:
        if not agent_names:
            return [agent.agent_name for agent in self.agents]

        unknown = [name for name in agent_names if name not in self.agent_map]
        if unknown:
            raise ValueError(f"Unknown workflow agents requested: {', '.join(sorted(unknown))}")

        required: set[str] = set()
        for agent_name in agent_names:
            required.update(self._collect_dependencies(agent_name))
            required.add(agent_name)

        ordered_names = [agent.agent_name for agent in self.agents]
        return [name for name in ordered_names if name in required]

    def _collect_dependencies(self, agent_name: str) -> set[str]:
        dependencies = set(self.AGENT_DEPENDENCIES.get(agent_name, ()))
        expanded = set(dependencies)
        for dependency in dependencies:
            expanded.update(self._collect_dependencies(dependency))
        return expanded

    def _write_governance_log(
        self,
        execution_log: list[AgentExecutionResult],
        *,
        file_format: str,
    ) -> Path:
        rows: list[dict[str, object]] = []
        for item in execution_log:
            rows.append(
                {
                    "agent_name": item.agent_name,
                    "status": item.status,
                    "output_keys": ",".join(sorted(item.output_paths.keys())),
                    "details": str(item.details),
                }
            )

        frame = pd.DataFrame(rows)
        logs_dir = self.config.paths.logs_dir
        root = Path(logs_dir) if Path(logs_dir).is_absolute() else Path(__file__).resolve().parents[2] / logs_dir
        root.mkdir(parents=True, exist_ok=True)
        output_path = root / f"governance_log.{file_format}"
        timestamped_output_path = root / f"governance_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_format}"
        frame.to_csv(output_path, index=False)
        frame.to_csv(timestamped_output_path, index=False)
        return output_path
