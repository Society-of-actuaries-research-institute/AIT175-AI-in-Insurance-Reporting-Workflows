"""Agent definitions for the sequential workflow orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ai_insurance_reporting.chatbot.retrieval import ReportingChatbot
from ai_insurance_reporting.config.loader import AppConfig
from ai_insurance_reporting.data.etl import InsuranceETLPipeline
from ai_insurance_reporting.data.synthetic import SyntheticDataGenerator
from ai_insurance_reporting.data.validation import ReportingValidationEngine
from ai_insurance_reporting.explainability.reporting import ExplainabilityPipeline
from ai_insurance_reporting.models.forecasting import ForecastingPipeline
from ai_insurance_reporting.narrative.generator import NarrativeGenerator
from ai_insurance_reporting.narrative.quality_check import NarrativeQualityChecker
from ai_insurance_reporting.reporting.anomaly_investigation import AnomalyInvestigator
from ai_insurance_reporting.reporting.full_report import FullReportBuilder
from ai_insurance_reporting.reporting.insights import InsightDetector
from ai_insurance_reporting.reporting.llm_evaluation import LLMEvaluator
from ai_insurance_reporting.reporting.movement import MovementAnalysisBuilder
from ai_insurance_reporting.reporting.review_queue import AnalystReviewQueueBuilder
from ai_insurance_reporting.utils.artifacts import ensure_artifact_dirs
from ai_insurance_reporting.visualization.charts import VisualizationGenerator


@dataclass(slots=True)
class AgentExecutionResult:
    """Execution summary for a workflow agent."""

    agent_name: str
    status: str
    output_paths: dict[str, Path] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


class WorkflowAgent:
    """Base class for workflow agents."""

    agent_name = "WorkflowAgent"

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def run(
        self,
        context: dict[str, Any],
        *,
        file_format: str = "csv",
    ) -> AgentExecutionResult:
        raise NotImplementedError


class IngestionAgent(WorkflowAgent):
    """Generate raw synthetic data and curate it through ETL."""

    agent_name = "IngestionAgent"

    def run(self, context: dict[str, Any], *, file_format: str = "csv") -> AgentExecutionResult:
        uploaded_raw_paths = context.get("uploaded_raw_paths", {})
        if uploaded_raw_paths:
            raw_paths = self._stage_uploaded_raw_inputs(uploaded_raw_paths, file_format=file_format)
        else:
            synthetic_overrides = context.get("workflow_assumption_overrides", {})
            scenario_parameters = dict(context.get("scenario_parameters", {}))
            scenario_parameters.update(
                {
                    key: value
                    for key, value in synthetic_overrides.items()
                    if key
                    in {
                        "premium_multiplier",
                        "claims_multiplier",
                        "reserve_multiplier",
                        "csm_multiplier",
                        "asset_return_shift",
                        "capital_multiplier",
                    }
                }
            )
            generator = SyntheticDataGenerator(self.config, **scenario_parameters)
            raw_paths = generator.write(file_format=file_format)

        pipeline = InsuranceETLPipeline(self.config)
        curated_dataset, curated_path = pipeline.run(file_format=file_format)

        context["curated_reporting_dataset"] = curated_dataset
        context["raw_output_paths"] = raw_paths
        context["curated_reporting_dataset_path"] = curated_path

        output_paths = dict(raw_paths)
        output_paths["curated_reporting_dataset"] = curated_path
        return AgentExecutionResult(self.agent_name, "completed", output_paths)

    def _stage_uploaded_raw_inputs(
        self,
        uploaded_raw_paths: dict[str, str],
        *,
        file_format: str,
    ) -> dict[str, Path]:
        artifact_paths = ensure_artifact_dirs(self.config)
        output_dir = artifact_paths.data_input
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths: dict[str, Path] = {}

        for dataset_name, source_path_str in uploaded_raw_paths.items():
            source_path = Path(source_path_str)
            if not source_path.exists():
                raise FileNotFoundError(f"Uploaded raw dataset not found: {source_path}")
            if source_path.suffix != ".csv":
                raise ValueError(f"Unsupported uploaded raw dataset format: {source_path.suffix}")
            frame = pd.read_csv(source_path)

            destination = output_dir / f"{dataset_name}.{file_format}"
            frame.to_csv(destination, index=False)
            output_paths[dataset_name] = destination

        return output_paths


class ValidationAgent(WorkflowAgent):
    """Validate the curated reporting dataset."""

    agent_name = "ValidationAgent"

    def run(self, context: dict[str, Any], *, file_format: str = "csv") -> AgentExecutionResult:
        curated_dataset = context["curated_reporting_dataset"]
        validation_overrides = context.get("validation_override_params", {})
        result, output_paths = ReportingValidationEngine(self.config, **validation_overrides).run(
            curated_dataset,
            file_format=file_format,
        )
        context["validation_result"] = result
        return AgentExecutionResult(
            self.agent_name,
            "completed",
            output_paths,
            {
                "records_with_issues": int(result.validation_flags["has_validation_issue"].sum()),
                "validation_overrides": validation_overrides,
            },
        )


class FeatureEngineeringAgent(WorkflowAgent):
    """Create a supervised training frame for forecasting."""

    agent_name = "FeatureEngineeringAgent"

    def run(self, context: dict[str, Any], *, file_format: str = "csv") -> AgentExecutionResult:
        curated_dataset = context["curated_reporting_dataset"]
        forecasting_pipeline = ForecastingPipeline(self.config)
        training_frame = forecasting_pipeline.prepare_training_frame(curated_dataset)

        artifact_paths = ensure_artifact_dirs(self.config)
        output_dir = artifact_paths.data_processed
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"forecast_training_frame.{file_format}"
        training_frame.to_csv(output_path, index=False)

        context["forecast_training_frame"] = training_frame
        return AgentExecutionResult(
            self.agent_name,
            "completed",
            {"forecast_training_frame": output_path},
            {"rows": len(training_frame)},
        )


class ForecastingAgent(WorkflowAgent):
    """Train forecasting models and persist outputs."""

    agent_name = "ForecastingAgent"

    def run(self, context: dict[str, Any], *, file_format: str = "csv") -> AgentExecutionResult:
        training_frame = context["forecast_training_frame"]
        forecasting_overrides = context.get("forecast_override_params", {})
        pipeline = ForecastingPipeline(self.config, **forecasting_overrides)
        result = pipeline.train(training_frame)
        output_paths = pipeline.write(result, file_format=file_format)

        context["forecasting_result"] = result
        return AgentExecutionResult(
            self.agent_name,
            "completed",
            output_paths,
            {
                "targets": sorted(result.evaluation_table["target_name"].unique().tolist()),
                "forecast_overrides": forecasting_overrides,
            },
        )


class ExplainabilityAgent(WorkflowAgent):
    """Generate explainability artifacts for trained models."""

    agent_name = "ExplainabilityAgent"

    def run(self, context: dict[str, Any], *, file_format: str = "csv") -> AgentExecutionResult:
        training_frame = context["forecast_training_frame"]
        forecasting_result = context["forecasting_result"]
        pipeline = ExplainabilityPipeline(self.config)
        result = pipeline.generate(training_frame, forecasting_result)
        output_paths = pipeline.write(result, file_format=file_format)

        context["explainability_result"] = result
        return AgentExecutionResult(self.agent_name, "completed", output_paths)


class InsightDetectionAgent(WorkflowAgent):
    """Detect material projected movements in key reporting metrics."""

    agent_name = "InsightDetectionAgent"

    def run(self, context: dict[str, Any], *, file_format: str = "csv") -> AgentExecutionResult:
        detector = InsightDetector(self.config)
        result, output_paths = detector.run(
            curated_reporting_dataset=context["curated_reporting_dataset"],
            forecast_output_table=context["forecasting_result"].forecast_output_table,
            file_format=file_format,
        )
        context["insight_result"] = result
        return AgentExecutionResult(
            self.agent_name,
            "completed",
            output_paths,
            {"insight_count": int(len(result.insight_summary))},
        )


class AnomalyInvestigationAgent(WorkflowAgent):
    """Generate first-pass investigation context for validation anomalies."""

    agent_name = "AnomalyInvestigationAgent"

    def run(self, context: dict[str, Any], *, file_format: str = "csv") -> AgentExecutionResult:
        investigator = AnomalyInvestigator(self.config)
        result, output_paths = investigator.run(
            anomaly_table=context["validation_result"].anomaly_table,
            curated_reporting_dataset=context["curated_reporting_dataset"],
            insight_summary=context.get("insight_result").insight_summary if context.get("insight_result") is not None else None,
            shap_global_importance=context.get("explainability_result").shap_global_importance if context.get("explainability_result") is not None else None,
            scenario_impact_summary=context.get("scenario_reporting_result").scenario_impact_summary if context.get("scenario_reporting_result") is not None else None,
            file_format=file_format,
        )
        context["anomaly_investigation_result"] = result
        return AgentExecutionResult(
            self.agent_name,
            "completed",
            output_paths,
            {"anomaly_review_rows": int(len(result.anomaly_investigation))},
        )


class MovementAnalysisAgent(WorkflowAgent):
    """Generate beginning-to-end movement bridge analysis."""

    agent_name = "MovementAnalysisAgent"

    def run(self, context: dict[str, Any], *, file_format: str = "csv") -> AgentExecutionResult:
        builder = MovementAnalysisBuilder(self.config)
        result, output_paths = builder.run(
            curated_reporting_dataset=context["curated_reporting_dataset"],
            forecast_output_table=context["forecasting_result"].forecast_output_table,
            file_format=file_format,
        )
        context["movement_analysis_result"] = result
        return AgentExecutionResult(
            self.agent_name,
            "completed",
            output_paths,
            {"movement_rows": int(len(result.movement_analysis)), "bridge_rows": int(len(result.movement_bridge_summary))},
        )


class NarrativeAgent(WorkflowAgent):
    """Generate traceable management commentary."""

    agent_name = "NarrativeAgent"

    def run(self, context: dict[str, Any], *, file_format: str = "csv") -> AgentExecutionResult:
        curated_dataset = context["curated_reporting_dataset"]
        validation_result = context["validation_result"]
        forecasting_result = context["forecasting_result"]
        generator = NarrativeGenerator(self.config)
        result = generator.generate(
            curated_reporting_dataset=curated_dataset,
            quarterly_validation_summary=validation_result.quarterly_validation_summary,
            forecast_output_table=forecasting_result.forecast_output_table,
            movement_bridge_summary=context.get("movement_analysis_result").movement_bridge_summary if context.get("movement_analysis_result") is not None else None,
        )
        output_paths = generator.write(result, file_format=file_format)

        context["narrative_result"] = result
        return AgentExecutionResult(self.agent_name, "completed", output_paths)


class NarrativeQualityAgent(WorkflowAgent):
    """Review generated narrative statements against supporting data."""

    agent_name = "NarrativeQualityAgent"

    def run(self, context: dict[str, Any], *, file_format: str = "csv") -> AgentExecutionResult:
        checker = NarrativeQualityChecker(self.config)
        result, output_paths = checker.run(
            narrative_statements=context["narrative_result"].narrative_statements,
            curated_reporting_dataset=context["curated_reporting_dataset"],
            forecast_output_table=context["forecasting_result"].forecast_output_table,
            shap_global_importance=context.get("explainability_result").shap_global_importance if context.get("explainability_result") is not None else None,
            scenario_impact_summary=context.get("scenario_reporting_result").scenario_impact_summary if context.get("scenario_reporting_result") is not None else None,
            file_format=file_format,
        )
        context["narrative_quality_result"] = result
        return AgentExecutionResult(
            self.agent_name,
            "completed",
            output_paths,
            {"warnings": int(result.narrative_quality_check["warning_flag"].sum()) if not result.narrative_quality_check.empty else 0},
        )


class VisualizationAgent(WorkflowAgent):
    """Generate figures and visualization metadata."""

    agent_name = "VisualizationAgent"

    def run(self, context: dict[str, Any], *, file_format: str = "csv") -> AgentExecutionResult:
        curated_dataset = context["curated_reporting_dataset"]
        validation_result = context["validation_result"]
        forecasting_result = context["forecasting_result"]
        explainability_result = context["explainability_result"]

        generator = VisualizationGenerator(self.config)
        result = generator.generate(
            curated_reporting_dataset=curated_dataset,
            quarterly_validation_summary=validation_result.quarterly_validation_summary,
            backtest_predictions=forecasting_result.backtest_predictions,
            forecast_output_table=forecasting_result.forecast_output_table,
            shap_global_importance=explainability_result.shap_global_importance,
            movement_bridge_summary=context.get("movement_analysis_result").movement_bridge_summary if context.get("movement_analysis_result") is not None else None,
        )
        output_paths = generator.write(result, file_format=file_format)

        context["visualization_result"] = result
        return AgentExecutionResult(self.agent_name, "completed", output_paths)


class FullReportAgent(WorkflowAgent):
    """Assemble a full management report from workflow outputs."""

    agent_name = "FullReportAgent"

    def run(self, context: dict[str, Any], *, file_format: str = "csv") -> AgentExecutionResult:
        latest_quarter = str(context["curated_reporting_dataset"]["quarter"].max())
        builder = FullReportBuilder(self.config)
        result, output_paths = builder.run(
            latest_quarter=latest_quarter,
            quarterly_validation_summary=context["validation_result"].quarterly_validation_summary,
            insight_summary=context.get("insight_result").insight_summary if context.get("insight_result") is not None else pd.DataFrame(),
            anomaly_investigation=context.get("anomaly_investigation_result").anomaly_investigation if context.get("anomaly_investigation_result") is not None else pd.DataFrame(),
            movement_bridge_summary=context.get("movement_analysis_result").movement_bridge_summary if context.get("movement_analysis_result") is not None else pd.DataFrame(),
            narrative_statements=context["narrative_result"].narrative_statements,
            narrative_quality_check=context.get("narrative_quality_result").narrative_quality_check if context.get("narrative_quality_result") is not None else pd.DataFrame(),
            figure_metadata=context.get("visualization_result").figure_metadata if context.get("visualization_result") is not None else pd.DataFrame(),
            file_format=file_format,
        )
        context["full_report_result"] = result
        return AgentExecutionResult(self.agent_name, "completed", output_paths, {"sections": int(len(result.report_sections))})


class ChatbotIndexingAgent(WorkflowAgent):
    """Create a chatbot retrieval index from generated reporting outputs."""

    agent_name = "ChatbotIndexingAgent"

    def run(self, context: dict[str, Any], *, file_format: str = "csv") -> AgentExecutionResult:
        chatbot = ReportingChatbot(self.config)
        result, output_paths = chatbot.build_index(file_format=file_format)
        output_paths.update(self._write_demo_assets())

        context["chatbot_index_result"] = result
        return AgentExecutionResult(
            self.agent_name,
            "completed",
            output_paths,
            {"documents": len(result.chatbot_index)},
        )

    def _write_demo_assets(self) -> dict[str, Path]:
        artifact_paths = ensure_artifact_dirs(self.config)
        output_dir = artifact_paths.root / "chatbot"
        output_dir.mkdir(parents=True, exist_ok=True)
        limit = int(self.config.reporting.chatbot.demo_query_limit)
        demo_queries = [
            {
                "category": "forecast_movement",
                "question": "Which segment has the largest projected increase in claims?",
                "supporting_artifacts": ["insight_summary", "forecast_output_table"],
                "expected_tools": ["ForecastComparisonTool"],
            },
            {
                "category": "driver_explanation",
                "question": "Which factors drove the capital ratio forecast?",
                "supporting_artifacts": ["shap_global_importance", "model_evaluation", "insight_summary"],
                "expected_tools": ["ExplainabilityTool", "ForecastComparisonTool"],
            },
            {
                "category": "movement_analysis",
                "question": "What drove reserve movement this quarter?",
                "supporting_artifacts": ["movement_bridge_summary", "movement_analysis", "movement_llm_summary"],
                "expected_tools": ["MovementAnalysisTool"],
            },
            {
                "category": "validation_review",
                "question": "Which anomalies appear most material?",
                "supporting_artifacts": ["anomaly_investigation", "anomaly_table"],
                "expected_tools": ["ValidationSummaryTool"],
            },
            {
                "category": "scenario_comparison",
                "question": "How does the adverse scenario affect the capital ratio?",
                "supporting_artifacts": ["scenario_impact_summary", "scenario_narrative_summary"],
                "expected_tools": ["ScenarioSummaryTool", "ScenarioRunTool"],
            },
            {
                "category": "narrative_retrieval",
                "question": "Show the draft commentary for claims.",
                "supporting_artifacts": ["narrative_statements", "narrative_quality_check"],
                "expected_tools": ["NarrativeLookupTool"],
            },
        ][:limit]
        json_path = output_dir / "chatbot_test_queries.json"
        json_path.write_text(__import__("json").dumps(demo_queries, indent=2, ensure_ascii=True), encoding="utf-8")
        markdown_lines = ["# Chatbot Demo Examples", ""]
        for item in demo_queries:
            markdown_lines.append(f"- {item['question']}")
            markdown_lines.append(f"  Supporting artifacts: {', '.join(item['supporting_artifacts'])}")
        markdown_path = output_dir / "chatbot_demo_examples.md"
        markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
        evidence_map_path = output_dir / "chatbot_evidence_map.csv"
        pd.DataFrame(demo_queries).to_csv(evidence_map_path, index=False)
        return {
            "chatbot_test_queries": json_path,
            "chatbot_demo_examples": markdown_path,
            "chatbot_evidence_map": evidence_map_path,
        }



class LLMEvaluationAgent(WorkflowAgent):
    """Run deterministic benchmark evaluation over grounded LLM outputs."""

    agent_name = "LLMEvaluationAgent"

    def run(self, context: dict[str, Any], *, file_format: str = "csv") -> AgentExecutionResult:
        evaluator = LLMEvaluator(self.config)
        result, output_paths = evaluator.run(file_format=file_format)
        context["llm_evaluation_result"] = result
        return AgentExecutionResult(
            self.agent_name,
            "completed",
            output_paths,
            {"benchmark_queries": int(len(result.llm_eval_results)), "review_items": int((result.llm_eval_results["evaluation_label"] == "review").sum()) if not result.llm_eval_results.empty else 0},
        )


class AnalystReviewQueueAgent(WorkflowAgent):
    """Consolidate AI-generated outputs into a single analyst review queue."""

    agent_name = "AnalystReviewQueueAgent"

    def run(self, context: dict[str, Any], *, file_format: str = "csv") -> AgentExecutionResult:
        builder = AnalystReviewQueueBuilder(self.config)
        result, output_paths = builder.run(
            insight_summary=context.get("insight_result").insight_summary if context.get("insight_result") is not None else pd.DataFrame(),
            anomaly_investigation=context.get("anomaly_investigation_result").anomaly_investigation if context.get("anomaly_investigation_result") is not None else pd.DataFrame(),
            narrative_quality_check=context.get("narrative_quality_result").narrative_quality_check if context.get("narrative_quality_result") is not None else pd.DataFrame(),
            llm_eval_results=context.get("llm_evaluation_result").llm_eval_results if context.get("llm_evaluation_result") is not None else pd.DataFrame(),
            file_format=file_format,
        )
        context["analyst_review_result"] = result
        return AgentExecutionResult(
            self.agent_name,
            "completed",
            output_paths,
            {
                "queue_items": int(len(result.analyst_review_queue)),
                "critical_items": int((result.analyst_review_queue["priority_label"] == "critical").sum()) if not result.analyst_review_queue.empty else 0,
            },
        )
