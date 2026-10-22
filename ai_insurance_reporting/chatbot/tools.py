"""Structured analytical tools for the reporting assistant."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ai_insurance_reporting.config.loader import AppConfig, load_config
from ai_insurance_reporting.reporting.review_queue import apply_review_status_updates, load_review_status_updates
from ai_insurance_reporting.utils.artifacts import ensure_artifact_dirs


@dataclass(slots=True)
class BaseTool:
    """Base interface for structured reporting tools.

    Each tool reads existing workflow artifacts and returns a structured payload
    that can be combined with retrieved document evidence. Tools do not generate
    free-form answers; they provide normalized data for the answer layer.
    """

    tool_name: str
    description: str
    config: AppConfig

    def run(self, query: str, state: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _load_processed(self, stem: str) -> pd.DataFrame:
        return _load_artifact_table(self.config, "processed", stem)

    def _load_model(self, stem: str) -> pd.DataFrame:
        return _load_artifact_table(self.config, "models", stem)

    def _load_report(self, subdir: str, stem: str) -> pd.DataFrame:
        return _load_artifact_table(self.config, f"report:{subdir}", stem)

    def _load_chatbot(self, stem: str) -> pd.DataFrame:
        return _load_artifact_table(self.config, "chatbot", stem)

    def _load_logs(self, stem: str) -> pd.DataFrame:
        return _load_artifact_table(self.config, "logs", stem)


class ValidationSummaryTool(BaseTool):
    """Return validation summaries, anomaly counts, and example exceptions."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__(
            tool_name="ValidationSummaryTool",
            description="Summarize validation results, anomaly counts, and material exceptions.",
            config=config,
        )

    def run(self, query: str, state: dict[str, Any]) -> dict[str, Any]:
        summary = self._load_processed("quarterly_validation_summary")
        anomalies = self._load_processed("anomaly_table")
        anomaly_review = self._load_report("reporting", "anomaly_investigation")
        latest_quarter = summary["quarter"].max() if not summary.empty else ""
        latest = summary.loc[summary["quarter"] == latest_quarter].to_dict(orient="records")
        material = anomaly_review.sort_values("support_score", ascending=False).head(10).to_dict(orient="records") if not anomaly_review.empty else anomalies.head(10).to_dict(orient="records") if not anomalies.empty else []
        return {
            "latest_quarter": latest_quarter,
            "validation_summary": latest,
            "material_exceptions": material,
            "anomaly_count_total": int(len(anomalies)),
            "anomaly_investigation": anomaly_review.head(10).to_dict(orient="records") if not anomaly_review.empty else [],
        }


class ForecastComparisonTool(BaseTool):
    """Return forecast outputs, backtest summaries, and model comparisons."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__(
            tool_name="ForecastComparisonTool",
            description="Compare forecasts, backtests, and identify the best model.",
            config=config,
        )

    def run(self, query: str, state: dict[str, Any]) -> dict[str, Any]:
        forecast_output = self._load_model("forecast_output_table")
        evaluation = self._load_model("model_evaluation")
        backtest = self._load_model("backtest_predictions")
        insights = self._load_report("reporting", "insight_summary")
        query_lower = query.lower()
        target = _infer_target(query, forecast_output["target_name"].unique().tolist() if not forecast_output.empty else [])

        target_forecast = forecast_output.loc[forecast_output["target_name"] == target] if target else forecast_output
        target_eval = evaluation.loc[evaluation["target_name"] == target] if target else evaluation
        target_backtest = backtest.loc[backtest["target_name"] == target] if target else backtest
        target_insights = insights.loc[insights["metric"] == target] if target and not insights.empty else insights
        selection_metric = str(target_eval["selection_metric"].iloc[0]) if not target_eval.empty and "selection_metric" in target_eval.columns else "mae"
        best_model = target_eval.sort_values(selection_metric).iloc[0].to_dict() if not target_eval.empty else {}
        backtest_summary = (
            target_backtest.groupby("quarter", as_index=False)[["actual_value", "predicted_value", "absolute_error"]]
            .mean()
            .to_dict(orient="records")
            if not target_backtest.empty
            else []
        )
        largest_segment_change: list[dict[str, Any]]
        if target_insights.empty:
            largest_segment_change = []
        else:
            ranked_insights = target_insights.copy()
            if "increase" in query_lower or "increases" in query_lower or "increased" in query_lower or "rise" in query_lower or "rises" in query_lower:
                positive = ranked_insights.loc[ranked_insights["absolute_change"] > 0].sort_values("absolute_change", ascending=False)
                ranked_insights = positive if not positive.empty else ranked_insights.sort_values("absolute_change", ascending=False)
            elif "decrease" in query_lower or "decreases" in query_lower or "decreased" in query_lower or "fall" in query_lower or "falls" in query_lower or "decline" in query_lower:
                negative = ranked_insights.loc[ranked_insights["absolute_change"] < 0].sort_values("absolute_change", ascending=True)
                ranked_insights = negative if not negative.empty else ranked_insights.sort_values("absolute_change", ascending=True)
            else:
                ranked_insights = ranked_insights.reindex(ranked_insights["absolute_change"].abs().sort_values(ascending=False).index)
            largest_segment_change = ranked_insights.head(5).to_dict(orient="records")
        return {
            "selected_target": target or "",
            "forecast_rows": target_forecast.head(12).to_dict(orient="records"),
            "model_performance": target_eval.to_dict(orient="records"),
            "best_model": best_model,
            "actual_vs_forecast": backtest_summary,
            "insight_candidates": largest_segment_change,
        }


class ExplainabilityTool(BaseTool):
    """Return SHAP, LIME, and PDP or ICE artifacts relevant to a question."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__(
            tool_name="ExplainabilityTool",
            description="Return SHAP, LIME, and PDP/ICE explainability artifacts.",
            config=config,
        )

    def run(self, query: str, state: dict[str, Any]) -> dict[str, Any]:
        shap_global = self._load_report("explainability", "shap_global_importance")
        shap_local = self._load_report("explainability", "shap_local_explanations")
        lime = self._load_report("explainability", "lime_explanations")
        pdp_ice = self._load_report("explainability", "pdp_ice_table")
        target = _infer_target(query, shap_global["target_name"].unique().tolist() if not shap_global.empty else [])
        target_global = shap_global.loc[shap_global["target_name"] == target] if target else shap_global
        target_local = shap_local.loc[shap_local["target_name"] == target] if target else shap_local
        target_lime = lime.loc[lime["target_name"] == target] if target else lime
        target_pdp = pdp_ice.loc[pdp_ice["target_name"] == target] if target else pdp_ice
        return {
            "selected_target": target or "",
            "shap_top_drivers": target_global.sort_values("mean_abs_shap", ascending=False).head(10).to_dict(orient="records"),
            "lime_local_explanations": target_lime.head(10).to_dict(orient="records"),
            "pdp_ice_metadata": target_pdp.head(20).to_dict(orient="records"),
            "shap_local_examples": target_local.head(10).to_dict(orient="records"),
        }


class NarrativeLookupTool(BaseTool):
    """Return relevant narrative sections and traceability facts."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__(
            tool_name="NarrativeLookupTool",
            description="Look up narrative sections and traceability facts.",
            config=config,
        )

    def run(self, query: str, state: dict[str, Any]) -> dict[str, Any]:
        statements = self._load_report("narrative", "narrative_statements")
        quality = self._load_report("narrative", "narrative_quality_check")
        tokens = _simple_tokens(query)
        if statements.empty:
            return {"narrative_sections": [], "traceability_facts": [], "quality_review": []}

        mask = statements["section"].str.replace("_", " ", regex=False).str.lower().apply(
            lambda value: any(token in value for token in tokens)
        )
        matched = statements.loc[mask] if mask.any() else statements.head(6)
        quality_rows = quality.loc[quality["statement_id"].isin(matched["statement_id"])] if not quality.empty else pd.DataFrame()
        return {
            "narrative_sections": matched[["section", "statement_text"]].to_dict(orient="records"),
            "traceability_facts": matched[
                ["statement_id", "source_dataset", "source_columns", "source_filters", "source_value"]
            ].to_dict(orient="records"),
            "quality_review": quality_rows.to_dict(orient="records") if not quality_rows.empty else [],
        }


class MovementAnalysisTool(BaseTool):
    """Return movement bridge summaries and component steps for reporting analysis."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__(
            tool_name="MovementAnalysisTool",
            description="Summarize beginning-to-end movement analysis and bridge components.",
            config=config,
        )

    def run(self, query: str, state: dict[str, Any]) -> dict[str, Any]:
        movement_summary = self._load_report("reporting", "movement_bridge_summary")
        movement_analysis = self._load_report("reporting", "movement_analysis")
        if movement_summary.empty:
            return {"latest_quarter": "", "selected_metric": "", "selected_scope": "portfolio", "portfolio_summary": {}, "top_segments": [], "movement_summary": [], "movement_steps": []}
        latest_quarter = str(movement_summary["quarter"].max())
        latest = movement_summary.loc[movement_summary["quarter"] == latest_quarter].copy()
        query_lower = query.lower()
        metric_aliases = {
            "reserve": "reserves",
            "reserves": "reserves",
            "claim": "claims",
            "claims": "claims",
            "premium": "premium_income",
            "premiums": "premium_income",
            "csm": "csm_closing",
            "capital": "capital_proxy",
        }
        selected_metric = next((value for key, value in metric_aliases.items() if key in query.lower()), "")
        if selected_metric:
            latest = latest.loc[latest["metric"] == selected_metric]
        ranked = latest.assign(abs_change=latest["net_change"].abs()).sort_values("abs_change", ascending=False).head(8)
        has_segment_filter = self._query_mentions_segment(query_lower, latest)
        movement_steps = movement_analysis.loc[
            (movement_analysis["quarter"] == latest_quarter)
            & (movement_analysis["metric"].isin(ranked["metric"]))
            & (movement_analysis["product"].isin(ranked["product"]))
            & (movement_analysis["region"].isin(ranked["region"]))
        ].copy() if not movement_analysis.empty and not ranked.empty else pd.DataFrame()
        portfolio_summary = self._build_portfolio_summary(latest, movement_analysis, latest_quarter=latest_quarter)
        return {
            "latest_quarter": latest_quarter,
            "selected_metric": selected_metric,
            "selected_scope": "segment" if has_segment_filter else "portfolio",
            "portfolio_summary": portfolio_summary,
            "top_segments": ranked.to_dict(orient="records"),
            "movement_summary": ranked.to_dict(orient="records"),
            "movement_steps": movement_steps.sort_values(["metric", "product", "region", "step_order"]).head(40).to_dict(orient="records") if not movement_steps.empty else [],
        }

    def _query_mentions_segment(self, query_lower: str, latest_summary: pd.DataFrame) -> bool:
        if latest_summary.empty:
            return False
        products = {str(value).lower() for value in latest_summary["product"].dropna().unique().tolist()}
        regions = {str(value).lower() for value in latest_summary["region"].dropna().unique().tolist()}
        return any(name in query_lower for name in products | regions)

    def _build_portfolio_summary(
        self,
        latest_summary: pd.DataFrame,
        movement_analysis: pd.DataFrame,
        *,
        latest_quarter: str,
    ) -> dict[str, object]:
        if latest_summary.empty:
            return {}
        metric_name = str(latest_summary["metric"].iloc[0])
        opening_value = float(latest_summary["opening_value"].sum())
        closing_value = float(latest_summary["closing_value"].sum())
        net_change = float(latest_summary["net_change"].sum())
        if movement_analysis.empty:
            top_positive_steps: list[dict[str, object]] = []
            top_negative_steps: list[dict[str, object]] = []
        else:
            relevant_steps = movement_analysis.loc[
                (movement_analysis["quarter"] == latest_quarter)
                & (movement_analysis["metric"] == metric_name)
                & (movement_analysis["movement_step"] != "residual")
            ].copy()
            grouped = (
                relevant_steps.groupby("movement_step", as_index=False)["movement_amount"]
                .sum()
                .sort_values("movement_amount", ascending=False)
            ) if not relevant_steps.empty else pd.DataFrame(columns=["movement_step", "movement_amount"])
            top_positive_steps = grouped.loc[grouped["movement_amount"] > 0].head(3).to_dict(orient="records")
            top_negative_steps = grouped.loc[grouped["movement_amount"] < 0].sort_values("movement_amount").head(3).to_dict(orient="records")
        return {
            "quarter": latest_quarter,
            "metric": metric_name,
            "opening_value": round(opening_value, 6),
            "closing_value": round(closing_value, 6),
            "net_change": round(net_change, 6),
            "top_positive_steps": top_positive_steps,
            "top_negative_steps": top_negative_steps,
        }


class AnalystReviewTool(BaseTool):
    """Return prioritized analyst review queue items and suggested actions."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__(
            tool_name="AnalystReviewTool",
            description="Summarize the unified analyst review queue and recommended actions.",
            config=config,
        )

    def run(self, query: str, state: dict[str, Any]) -> dict[str, Any]:
        """Return a filtered analyst-review view for triage-oriented questions.

        The tool keeps the queue grounded in saved workflow artifacts and then
        overlays any persisted analyst status updates so the chatbot reflects
        the current review state rather than only the last workflow run.
        """
        queue = self._load_report("reporting", "analyst_review_queue")
        summary = self._load_report("reporting", "analyst_review_summary")
        updates = load_review_status_updates(self.config)
        queue = apply_review_status_updates(queue, updates) if not queue.empty else queue
        if queue.empty:
            return {"review_items": [], "summary_by_source": [], "top_actions": []}

        query_lower = query.lower()
        review_view = queue.copy()
        priority_keywords = {"critical", "high", "medium", "monitor"}
        for priority in priority_keywords:
            if priority in query_lower:
                review_view = review_view.loc[review_view["priority_label"] == priority]
                break

        source_map = {
            "insight": "insight_detection",
            "anomaly": "anomaly_investigation",
            "narrative": "narrative_quality",
            "llm": "llm_evaluation",
            "benchmark": "llm_evaluation",
            "review": None,
            "triage": None,
        }
        for token, source_name in source_map.items():
            if token in query_lower and source_name:
                review_view = review_view.loc[review_view["review_source"] == source_name]
                break

        top_actions = (
            review_view.groupby("recommended_action", as_index=False)
            .agg(item_count=("review_item_id", "count"), average_priority=("priority_score", "mean"))
            .sort_values(["item_count", "average_priority"], ascending=[False, False])
            .head(5)
            .to_dict(orient="records")
        ) if not review_view.empty else []

        return {
            "review_items": review_view.head(12).to_dict(orient="records"),
            "summary_by_source": summary.to_dict(orient="records") if not summary.empty else [],
            "top_actions": top_actions,
        }


class FigureLookupTool(BaseTool):
    """Return figure metadata relevant to a question."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__(
            tool_name="FigureLookupTool",
            description="Look up available figures and related metadata.",
            config=config,
        )

    def run(self, query: str, state: dict[str, Any]) -> dict[str, Any]:
        metadata = _load_artifact_table(self.config, "figures", "figure_metadata")
        tokens = _simple_tokens(query)
        if metadata.empty:
            return {"figures": []}
        mask = metadata["chart_title"].str.lower().apply(lambda value: any(token in value for token in tokens))
        matched = metadata.loc[mask] if mask.any() else metadata
        return {"figures": matched.to_dict(orient="records")}


class RunSummaryTool(BaseTool):
    """Return run metadata and governance checkpoints."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__(
            tool_name="RunSummaryTool",
            description="Summarize run metadata, available artifacts, and governance checkpoints.",
            config=config,
        )

    def run(self, query: str, state: dict[str, Any]) -> dict[str, Any]:
        governance = self._load_logs("governance_log")
        artifact_paths = ensure_artifact_dirs(self.config)
        available_artifacts: list[str] = []
        for directory in [artifact_paths.data_processed, artifact_paths.models, artifact_paths.reports, artifact_paths.figures]:
            if Path(directory).exists():
                for path in Path(directory).rglob("*"):
                    if path.is_file():
                        available_artifacts.append(str(path))
        return {
            "completed_stages": governance["agent_name"].tolist() if not governance.empty else [],
            "governance_checkpoints": governance.to_dict(orient="records"),
            "available_artifacts": available_artifacts[:100],
        }


class ScenarioRunTool(BaseTool):
    """Run an isolated what-if or stress scenario and summarize the result."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__(
            tool_name="ScenarioRunTool",
            description="Run a controlled scenario rerun and compare it to the baseline workflow.",
            config=config,
        )
        from ai_insurance_reporting.orchestration.scenario import ScenarioWorkflowRunner

        self.runner = ScenarioWorkflowRunner(config)

    def run(self, query: str, state: dict[str, Any]) -> dict[str, Any]:
        scenario_name, scenario_parameters = self.runner.infer_parameters(query)
        cache_key = f"scenario::{scenario_name}"
        cached = state.get(cache_key)
        if cached is None:
            cached = self.runner.run_scenario(
                scenario_name=scenario_name,
                scenario_parameters=scenario_parameters,
                file_format=str(state.get("file_format", "csv")),
            )
            state[cache_key] = cached

        return {
            "scenario_name": cached.scenario_name,
            "scenario_parameters": cached.scenario_parameters,
            "scenario_artifact_root": str(cached.scenario_artifact_root),
            "scenario_metadata_path": str(cached.metadata_path) if cached.metadata_path is not None else "",
            "summary": cached.summary,
            "comparison": cached.comparison,
            "source_artifacts": {key: str(value) for key, value in cached.output_paths.items()},
        }


class ScenarioSummaryTool(BaseTool):
    """Return saved scenario reporting summaries for comparison questions."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__(
            tool_name="ScenarioSummaryTool",
            description="Look up saved scenario impact summaries and scenario narrative outputs.",
            config=config,
        )

    def run(self, query: str, state: dict[str, Any]) -> dict[str, Any]:
        artifact_paths = ensure_artifact_dirs(self.config)
        scenarios_root = artifact_paths.root / "scenarios"
        rows: list[dict[str, Any]] = []
        query_tokens = _simple_tokens(query)
        if scenarios_root.exists():
            for scenario_root in scenarios_root.iterdir():
                if not scenario_root.is_dir():
                    continue
                impact = _load_table_from_path(scenario_root / "reports" / "scenario", "scenario_impact_summary")
                top_impacts = _load_table_from_path(scenario_root / "reports" / "scenario", "scenario_top_impacts")
                narrative_path = scenario_root / "reports" / "scenario" / "scenario_narrative_summary.json"
                narrative = {}
                if narrative_path.exists():
                    narrative = __import__("json").loads(narrative_path.read_text(encoding="utf-8"))
                searchable_parts = [scenario_root.name]
                if not impact.empty:
                    searchable_parts.extend(impact.get("metric", pd.Series(dtype=str)).astype(str).tolist())
                if not top_impacts.empty:
                    searchable_parts.extend(top_impacts.get("metric", pd.Series(dtype=str)).astype(str).tolist())
                    searchable_parts.extend(top_impacts.get("product", pd.Series(dtype=str)).astype(str).tolist())
                    searchable_parts.extend(top_impacts.get("region", pd.Series(dtype=str)).astype(str).tolist())
                if narrative:
                    searchable_parts.extend(str(value) for value in narrative.values() if isinstance(value, (str, int, float)))
                searchable_text = " ".join(searchable_parts).lower()
                score = sum(1 for token in query_tokens if token in searchable_text)
                rows.append(
                    {
                        "scenario_name": scenario_root.name,
                        "impact_summary": impact.head(10).to_dict(orient="records") if not impact.empty else [],
                        "top_impacts": top_impacts.head(10).to_dict(orient="records") if not top_impacts.empty else [],
                        "narrative_summary": narrative,
                        "match_score": score,
                        "updated_at": scenario_root.stat().st_mtime,
                    }
                )
        ranked = sorted(rows, key=lambda item: (item["match_score"], item["updated_at"]), reverse=True)
        trimmed = [{key: value for key, value in row.items() if key not in {"match_score", "updated_at"}} for row in ranked[:5]]
        return {"scenario_summaries": trimmed}

class WorkflowExecutionTool(BaseTool):
    """Run the full workflow or a selected subset of stages."""

    AGENT_SEQUENCE = [
        "IngestionAgent",
        "ValidationAgent",
        "FeatureEngineeringAgent",
        "ForecastingAgent",
        "ExplainabilityAgent",
        "InsightDetectionAgent",
        "AnomalyInvestigationAgent",
        "MovementAnalysisAgent",
        "NarrativeAgent",
        "NarrativeQualityAgent",
        "VisualizationAgent",
        "FullReportAgent",
        "ChatbotIndexingAgent",
        "LLMEvaluationAgent",
    ]
    STAGE_KEYWORDS = {
        "IngestionAgent": {"ingestion", "ingest", "etl", "data", "generate", "raw"},
        "ValidationAgent": {"validation", "validate", "quality", "anomaly", "reconciliation"},
        "FeatureEngineeringAgent": {"feature", "features", "training", "frame"},
        "ForecastingAgent": {"forecast", "forecasting", "model", "models", "backtest"},
        "ExplainabilityAgent": {"explain", "explainability", "shap", "lime", "pdp", "ice", "driver", "drivers"},
        "InsightDetectionAgent": {"insight", "insights", "largest", "changed", "movement", "projected", "material"},
        "AnomalyInvestigationAgent": {"investigate", "investigation", "anomaly", "anomalies", "issue", "issues", "exception", "exceptions"},
        "MovementAnalysisAgent": {"movement", "bridge", "waterfall", "opening", "closing", "beginning", "ending"},
        "NarrativeAgent": {"narrative", "commentary", "report", "reporting", "management"},
        "NarrativeQualityAgent": {"quality", "governance", "consistency", "review", "warning", "wording"},
        "VisualizationAgent": {"visualization", "visualizations", "chart", "charts", "figure", "figures", "dashboard"},
        "FullReportAgent": {"full report", "report pack", "report package", "management report", "appendix"},
        "ChatbotIndexingAgent": {"chatbot", "index", "retrieval", "vector"},
        "LLMEvaluationAgent": {"llm", "evaluation", "benchmark", "benchmarks", "feedback", "review queue"},
    }

    def __init__(self, config: AppConfig) -> None:
        super().__init__(
            tool_name="WorkflowExecutionTool",
            description="Run the full workflow or selected reporting stages in a controlled sequence.",
            config=config,
        )
        from ai_insurance_reporting.orchestration.workflow import WorkflowOrchestrator

        self.orchestrator = WorkflowOrchestrator(config)

    def run(self, query: str, state: dict[str, Any]) -> dict[str, Any]:
        requested_agents = self._infer_requested_agents(query)
        workflow_assumption_overrides = dict(state.get("workflow_assumption_overrides", {}))
        validation_override_params = dict(state.get("validation_override_params", {}))
        forecast_override_params = dict(state.get("forecast_override_params", {}))
        uploaded_raw_paths = dict(state.get("uploaded_raw_paths", {}))

        effective_requested_agents = requested_agents or self.AGENT_SEQUENCE.copy()
        if uploaded_raw_paths or workflow_assumption_overrides:
            if "IngestionAgent" not in effective_requested_agents:
                effective_requested_agents = ["IngestionAgent", *effective_requested_agents]
        if validation_override_params and "ValidationAgent" not in effective_requested_agents:
            effective_requested_agents = ["ValidationAgent", *effective_requested_agents]
        if forecast_override_params and "ForecastingAgent" not in effective_requested_agents:
            effective_requested_agents = ["ForecastingAgent", *effective_requested_agents]

        deduped_requested_agents: list[str] = []
        for agent_name in effective_requested_agents:
            if agent_name not in deduped_requested_agents:
                deduped_requested_agents.append(agent_name)

        execution_plan = self.orchestrator.resolve_execution_plan(deduped_requested_agents)
        result = self.orchestrator.run_selected(
            deduped_requested_agents,
            file_format=str(state.get("file_format", "csv")),
            initial_context={
                "workflow_execution_query": query,
                "workflow_assumption_overrides": workflow_assumption_overrides,
                "validation_override_params": validation_override_params,
                "forecast_override_params": forecast_override_params,
                "uploaded_raw_paths": uploaded_raw_paths,
            },
        )
        return {
            "mode": "full" if not requested_agents or set(deduped_requested_agents) == set(self.AGENT_SEQUENCE) else "partial",
            "requested_agents": deduped_requested_agents,
            "executed_agents": [item.agent_name for item in result.execution_log],
            "requested_stage_labels": [self._label_for_agent(name) for name in deduped_requested_agents],
            "executed_stage_labels": [self._label_for_agent(name) for name in execution_plan],
            "governance_log_path": str(result.governance_log_path) if result.governance_log_path else "",
            "output_artifacts": {key: str(value) for key, value in result.output_paths().items()},
            "workflow_assumption_overrides": workflow_assumption_overrides,
            "validation_override_params": validation_override_params,
            "forecast_override_params": forecast_override_params,
            "uploaded_raw_datasets": sorted(uploaded_raw_paths.keys()),
        }

    def _infer_requested_agents(self, query: str) -> list[str]:
        query_lower = query.lower()
        if any(phrase in query_lower for phrase in {"full workflow", "entire workflow", "whole workflow", "all stages"}):
            return self.AGENT_SEQUENCE.copy()

        tokens = _simple_tokens(query)
        selected: list[str] = []
        for agent_name, keywords in self.STAGE_KEYWORDS.items():
            if tokens & keywords:
                selected.append(agent_name)

        if "rebuild index" in query_lower or "refresh index" in query_lower:
            if "ChatbotIndexingAgent" not in selected:
                selected.append("ChatbotIndexingAgent")
        return [agent_name for agent_name in self.AGENT_SEQUENCE if agent_name in selected]

    def _label_for_agent(self, agent_name: str) -> str:
        labels = {
            "IngestionAgent": "data ingestion and ETL",
            "ValidationAgent": "validation",
            "FeatureEngineeringAgent": "feature engineering",
            "ForecastingAgent": "forecasting",
            "ExplainabilityAgent": "explainability",
            "InsightDetectionAgent": "insight detection",
            "AnomalyInvestigationAgent": "anomaly investigation",
            "MovementAnalysisAgent": "movement analysis",
            "NarrativeAgent": "narrative reporting",
            "NarrativeQualityAgent": "narrative quality review",
            "VisualizationAgent": "visualization",
            "FullReportAgent": "full report assembly",
            "ChatbotIndexingAgent": "chatbot indexing",
            "LLMEvaluationAgent": "LLM evaluation",
        }
        return labels.get(agent_name, agent_name)


def build_default_tools(config: AppConfig | None = None) -> dict[str, BaseTool]:
    """Build the default tool registry."""

    cfg = config or load_config()
    return {
        "WorkflowExecutionTool": WorkflowExecutionTool(cfg),
        "ScenarioRunTool": ScenarioRunTool(cfg),
        "ValidationSummaryTool": ValidationSummaryTool(cfg),
        "ForecastComparisonTool": ForecastComparisonTool(cfg),
        "ExplainabilityTool": ExplainabilityTool(cfg),
        "MovementAnalysisTool": MovementAnalysisTool(cfg),
        "NarrativeLookupTool": NarrativeLookupTool(cfg),
        "FigureLookupTool": FigureLookupTool(cfg),
        "ScenarioSummaryTool": ScenarioSummaryTool(cfg),
        "RunSummaryTool": RunSummaryTool(cfg),
        "AnalystReviewTool": AnalystReviewTool(cfg),
    }


def _load_artifact_table(config: AppConfig, area: str, stem: str) -> pd.DataFrame:
    artifact_paths = ensure_artifact_dirs(config)
    if area == "processed":
        base = artifact_paths.data_processed
    elif area == "models":
        base = artifact_paths.models
    elif area == "figures":
        base = artifact_paths.figures / "reporting"
    elif area == "chatbot":
        base = artifact_paths.root / "chatbot"
    elif area == "logs":
        base = artifact_paths.logs
    elif area.startswith("report:"):
        base = artifact_paths.reports / area.split(":", 1)[1]
    else:
        raise ValueError(f"Unsupported artifact area: {area}")

    for suffix in (".csv",):
        path = Path(base) / f"{stem}{suffix}"
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def _simple_tokens(query: str) -> set[str]:
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in query)
    return {token for token in cleaned.split() if token}


def _infer_target(query: str, candidates: list[str]) -> str | None:
    query_lower = query.lower()
    normalized = {candidate.lower(): candidate for candidate in candidates}
    aliases = {
        "claims": "claims",
        "claim": "claims",
        "premium": "premium",
        "premiums": "premium",
        "reserve": "reserve_movement",
        "reserves": "reserve_movement",
        "csm": "csm_movement",
        "capital": "capital_ratio",
    }
    for token, mapped in aliases.items():
        if token in query_lower:
            for candidate_lower, original in normalized.items():
                if mapped == candidate_lower:
                    return original
    return candidates[0] if candidates else None






def _load_table_from_path(base: Path, stem: str) -> pd.DataFrame:
    for suffix in (".csv",):
        path = base / f"{stem}{suffix}"
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()
