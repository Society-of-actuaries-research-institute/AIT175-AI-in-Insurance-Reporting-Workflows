"""Indexing utilities for generated management reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from ai_insurance_reporting.config.loader import AppConfig, load_config
from ai_insurance_reporting.utils.artifacts import ensure_artifact_dirs


@dataclass(slots=True)
class ChatbotIndexResult:
    """Indexed records for chatbot retrieval."""

    chatbot_index: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        """Return index outputs keyed by file stem."""

        return {"chatbot_index": self.chatbot_index}


class ChatbotIndexer:
    """Build a retrieval index from generated reports and tables."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()

    def run(
        self,
        *,
        file_format: str = "csv",
    ) -> tuple[ChatbotIndexResult, dict[str, Path]]:
        """Load generated outputs and persist a chatbot index."""

        artifact_paths = ensure_artifact_dirs(self.config)
        narrative_path = artifact_paths.reports / "narrative" / f"narrative_statements.{file_format}"
        if not narrative_path.exists():
            raise FileNotFoundError(f"Narrative statements not found: {narrative_path}")

        narrative_statements = self._load_table(narrative_path, file_format=file_format)
        result = self.generate(
            narrative_statements,
            forecast_output_table=self._load_model_table("forecast_output_table", file_format=file_format),
            model_evaluation=self._load_model_table("model_evaluation", file_format=file_format),
            backtest_predictions=self._load_model_table("backtest_predictions", file_format=file_format),
            quarterly_validation_summary=self._load_processed_table(
                "quarterly_validation_summary",
                file_format=file_format,
            ),
            anomaly_table=self._load_processed_table("anomaly_table", file_format=file_format),
            shap_global_importance=self._load_report_table(
                "explainability",
                "shap_global_importance",
                file_format=file_format,
            ),
            shap_local_explanations=self._load_report_table(
                "explainability",
                "shap_local_explanations",
                file_format=file_format,
            ),
            insight_summary=self._load_report_table(
                "reporting",
                "insight_summary",
                file_format=file_format,
            ),
            anomaly_investigation=self._load_report_table(
                "reporting",
                "anomaly_investigation",
                file_format=file_format,
            ),
            movement_bridge_summary=self._load_report_table(
                "reporting",
                "movement_bridge_summary",
                file_format=file_format,
            ),
            management_report_sections=self._load_report_table(
                "final",
                "management_report_sections",
                file_format=file_format,
            ),
            narrative_quality_check=self._load_report_table(
                "narrative",
                "narrative_quality_check",
                file_format=file_format,
            ),
            analyst_review_queue=self._load_report_table(
                "reporting",
                "analyst_review_queue",
                file_format=file_format,
            ),
            figure_metadata=self._load_figure_metadata(file_format=file_format),
            governance_log=self._load_governance_log(file_format=file_format),
            scenario_documents=self._load_scenario_documents(file_format=file_format),
        )
        output_paths = self.write(result, file_format=file_format)
        return result, output_paths

    def generate(
        self,
        narrative_statements: pd.DataFrame,
        *,
        forecast_output_table: pd.DataFrame | None = None,
        model_evaluation: pd.DataFrame | None = None,
        backtest_predictions: pd.DataFrame | None = None,
        quarterly_validation_summary: pd.DataFrame | None = None,
        anomaly_table: pd.DataFrame | None = None,
        shap_global_importance: pd.DataFrame | None = None,
        shap_local_explanations: pd.DataFrame | None = None,
        insight_summary: pd.DataFrame | None = None,
        anomaly_investigation: pd.DataFrame | None = None,
        movement_bridge_summary: pd.DataFrame | None = None,
        management_report_sections: pd.DataFrame | None = None,
        narrative_quality_check: pd.DataFrame | None = None,
        analyst_review_queue: pd.DataFrame | None = None,
        figure_metadata: pd.DataFrame | None = None,
        governance_log: pd.DataFrame | None = None,
        scenario_documents: list[dict[str, object]] | None = None,
    ) -> ChatbotIndexResult:
        """Create a retrieval index spanning reporting artifacts."""

        document_rows: list[dict[str, object]] = []
        document_rows.extend(self._index_narratives(narrative_statements))

        if forecast_output_table is not None and not forecast_output_table.empty:
            document_rows.extend(self._index_forecasts(forecast_output_table))
        if model_evaluation is not None and not model_evaluation.empty:
            document_rows.extend(self._index_model_evaluation(model_evaluation))
        if backtest_predictions is not None and not backtest_predictions.empty:
            document_rows.extend(self._index_backtests(backtest_predictions))
        if quarterly_validation_summary is not None and not quarterly_validation_summary.empty:
            document_rows.extend(self._index_validation_summary(quarterly_validation_summary))
        if anomaly_table is not None and not anomaly_table.empty:
            document_rows.extend(self._index_anomalies(anomaly_table))
        if shap_global_importance is not None and not shap_global_importance.empty:
            document_rows.extend(self._index_shap_global(shap_global_importance))
        if shap_local_explanations is not None and not shap_local_explanations.empty:
            document_rows.extend(self._index_shap_local(shap_local_explanations))
        if insight_summary is not None and not insight_summary.empty:
            document_rows.extend(self._index_insights(insight_summary))
        if anomaly_investigation is not None and not anomaly_investigation.empty:
            document_rows.extend(self._index_anomaly_investigation(anomaly_investigation))
        if movement_bridge_summary is not None and not movement_bridge_summary.empty:
            document_rows.extend(self._index_movement_summary(movement_bridge_summary))
        if management_report_sections is not None and not management_report_sections.empty:
            document_rows.extend(self._index_management_report(management_report_sections))
        if narrative_quality_check is not None and not narrative_quality_check.empty:
            document_rows.extend(self._index_narrative_quality(narrative_quality_check))
        if analyst_review_queue is not None and not analyst_review_queue.empty:
            document_rows.extend(self._index_analyst_review_queue(analyst_review_queue))
        if figure_metadata is not None and not figure_metadata.empty:
            document_rows.extend(self._index_figures(figure_metadata))
        if governance_log is not None and not governance_log.empty:
            document_rows.extend(self._index_governance_logs(governance_log))
        if scenario_documents:
            document_rows.extend(scenario_documents)

        index_frame = pd.DataFrame(document_rows)
        ordered = [
            "document_id",
            "document_type",
            "reporting_quarter",
            "section",
            "title",
            "content",
            "keywords",
            "source_dataset",
            "source_columns",
            "source_filters",
            "source_value",
        ]
        return ChatbotIndexResult(chatbot_index=index_frame[ordered].reset_index(drop=True))
    def _index_narratives(self, narrative_statements: pd.DataFrame) -> list[dict[str, object]]:
        index_frame = narrative_statements.copy()
        index_frame["document_id"] = index_frame["statement_id"]
        index_frame["document_type"] = "narrative"
        index_frame["title"] = index_frame["section"].str.replace("_", " ", regex=False).str.title()
        index_frame["content"] = index_frame["statement_text"]
        index_frame["keywords"] = (
            "narrative report management commentary "
            + index_frame["section"].str.replace("_", " ", regex=False)
            + " "
            + index_frame["source_columns"].str.replace(",", " ", regex=False)
        )
        return index_frame.to_dict(orient="records")

    def _index_forecasts(self, forecast_output_table: pd.DataFrame) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        grouped = forecast_output_table.groupby(["forecast_quarter", "target_name"], as_index=False)["forecast_value"].mean()
        for row in grouped.itertuples(index=False):
            rows.append(
                {
                    "document_id": f"FCT-{row.forecast_quarter}-{row.target_name}",
                    "document_type": "forecast",
                    "reporting_quarter": row.forecast_quarter,
                    "section": "forecast",
                    "title": f"{row.target_name} forecast",
                    "content": f"Forecast for {row.target_name} in {row.forecast_quarter} is {float(row.forecast_value):.4f}.",
                    "keywords": f"forecast {row.target_name} outlook projected future",
                    "source_dataset": "forecast_output_table",
                    "source_columns": "forecast_quarter,target_name,forecast_value",
                    "source_filters": f"forecast_quarter={row.forecast_quarter};target_name={row.target_name}",
                    "source_value": f"forecast_value={float(row.forecast_value):.4f}",
                }
            )
        return rows

    def _index_model_evaluation(self, model_evaluation: pd.DataFrame) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for row in model_evaluation.itertuples(index=False):
            rows.append(
                {
                    "document_id": f"EVAL-{row.target_name}-{row.model_family}",
                    "document_type": "forecast_evaluation",
                    "reporting_quarter": "",
                    "section": "forecasting",
                    "title": f"{row.target_name} {row.model_family} evaluation",
                    "content": (
                        f"{row.model_family} for {row.target_name} has MAE {float(row.mae):.4f}, "
                        f"RMSE {float(row.rmse):.4f}, MAPE {float(row.mape):.4f}, and R2 {float(row.r2):.4f}."
                    ),
                    "keywords": f"forecast model evaluation {row.target_name} {row.model_family} mae rmse mape r2",
                    "source_dataset": "model_evaluation",
                    "source_columns": "target_name,model_family,mae,rmse,mape,r2",
                    "source_filters": f"target_name={row.target_name};model_family={row.model_family}",
                    "source_value": f"mae={float(row.mae):.4f};rmse={float(row.rmse):.4f};mape={float(row.mape):.4f};r2={float(row.r2):.4f}",
                }
            )
        return rows

    def _index_backtests(self, backtest_predictions: pd.DataFrame) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        grouped = backtest_predictions.groupby(["quarter", "target_name"], as_index=False)["absolute_error"].mean()
        for row in grouped.itertuples(index=False):
            rows.append(
                {
                    "document_id": f"BTC-{row.quarter}-{row.target_name}",
                    "document_type": "forecast_backtest",
                    "reporting_quarter": row.quarter,
                    "section": "forecasting",
                    "title": f"{row.target_name} backtest",
                    "content": f"Average backtest absolute error for {row.target_name} in {row.quarter} is {float(row.absolute_error):.4f}.",
                    "keywords": f"forecast backtest {row.target_name} actual predicted error",
                    "source_dataset": "backtest_predictions",
                    "source_columns": "quarter,target_name,absolute_error",
                    "source_filters": f"quarter={row.quarter};target_name={row.target_name}",
                    "source_value": f"absolute_error={float(row.absolute_error):.4f}",
                }
            )
        return rows

    def _index_validation_summary(self, quarterly_validation_summary: pd.DataFrame) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for row in quarterly_validation_summary.itertuples(index=False):
            rows.append(
                {
                    "document_id": f"VAL-{row.quarter}",
                    "document_type": "validation",
                    "reporting_quarter": row.quarter,
                    "section": "validation",
                    "title": f"Validation summary {row.quarter}",
                    "content": (
                        f"Validation pass rate for {row.quarter} is {float(row.validation_pass_rate):.2%} "
                        f"with {int(row.records_with_issues)} records with issues and {int(row.anomaly_count)} anomalies."
                    ),
                    "keywords": "validation quality checks anomalies issues pass rate",
                    "source_dataset": "quarterly_validation_summary",
                    "source_columns": "quarter,validation_pass_rate,records_with_issues,anomaly_count",
                    "source_filters": f"quarter={row.quarter}",
                    "source_value": f"validation_pass_rate={float(row.validation_pass_rate):.4f};records_with_issues={int(row.records_with_issues)};anomaly_count={int(row.anomaly_count)}",
                }
            )
        return rows

    def _index_anomalies(self, anomaly_table: pd.DataFrame) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for row in anomaly_table.itertuples(index=False):
            rows.append(
                {
                    "document_id": f"ANM-{row.quarter}-{row.rule_name}-{abs(hash((row.product, row.region, row.metric_name))) % 100000}",
                    "document_type": "validation_anomaly",
                    "reporting_quarter": row.quarter,
                    "section": "validation",
                    "title": f"{row.rule_name} anomaly",
                    "content": (
                        f"{row.rule_name} anomaly for {row.product} in {row.region} during {row.quarter}: "
                        f"{row.metric_name} observed at {row.observed_value}. {row.details}"
                    ),
                    "keywords": f"validation anomaly {row.rule_name} {row.metric_name} {row.product} {row.region}",
                    "source_dataset": "anomaly_table",
                    "source_columns": "quarter,product,region,rule_name,metric_name,observed_value,details",
                    "source_filters": f"quarter={row.quarter};product={row.product};region={row.region}",
                    "source_value": f"observed_value={row.observed_value}",
                }
            )
        return rows

    def _index_shap_global(self, shap_global_importance: pd.DataFrame) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        grouped = (
            shap_global_importance.sort_values(["target_name", "mean_abs_shap"], ascending=[True, False])
            .groupby("target_name")
            .head(3)
        )
        for row in grouped.itertuples(index=False):
            rows.append(
                {
                    "document_id": f"SHPG-{row.target_name}-{row.feature_name}",
                    "document_type": "explainability_global",
                    "reporting_quarter": "",
                    "section": "explainability",
                    "title": f"SHAP global importance for {row.target_name}",
                    "content": (
                        f"Feature {row.feature_name} is an important global SHAP driver for {row.target_name} "
                        f"with mean absolute SHAP value {float(row.mean_abs_shap):.6f}."
                    ),
                    "keywords": f"shap explainability feature importance global {row.target_name} {row.feature_name}",
                    "source_dataset": "shap_global_importance",
                    "source_columns": "target_name,feature_name,mean_abs_shap",
                    "source_filters": f"target_name={row.target_name};feature_name={row.feature_name}",
                    "source_value": f"mean_abs_shap={float(row.mean_abs_shap):.6f}",
                }
            )
        return rows
    def _index_shap_local(self, shap_local_explanations: pd.DataFrame) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        grouped = (
            shap_local_explanations.sort_values(["target_name", "row_label", "abs_shap_value"], ascending=[True, True, False])
            .groupby(["target_name", "row_label"])
            .head(1)
            .head(20)
        )
        for row in grouped.itertuples(index=False):
            rows.append(
                {
                    "document_id": f"SHPL-{row.target_name}-{abs(hash(row.row_label)) % 100000}",
                    "document_type": "explainability_local",
                    "reporting_quarter": row.row_label.split("|")[0],
                    "section": "explainability",
                    "title": f"SHAP local explanation for {row.target_name}",
                    "content": (
                        f"For {row.row_label}, feature {row.feature_name} has SHAP value {float(row.shap_value):.6f} "
                        f"for target {row.target_name}."
                    ),
                    "keywords": f"shap local explanation {row.target_name} {row.feature_name} {row.row_label}",
                    "source_dataset": "shap_local_explanations",
                    "source_columns": "target_name,row_label,feature_name,shap_value",
                    "source_filters": f"target_name={row.target_name};row_label={row.row_label}",
                    "source_value": f"shap_value={float(row.shap_value):.6f}",
                }
            )
        return rows

    def _index_insights(self, insight_summary: pd.DataFrame) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        ranking = {"critical": 4, "material": 3, "moderate": 2, "normal": 1}
        ordered_frame = insight_summary.assign(
            severity_order=insight_summary["severity_classification"].map(ranking).fillna(0)
        ).sort_values(["severity_order", "z_score"], ascending=[False, False]).head(100)
        for row in ordered_frame.itertuples(index=False):
            rows.append(
                {
                    "document_id": f"INS-{row.quarter}-{row.metric}-{abs(hash((row.product, row.region))) % 100000}",
                    "document_type": "insight",
                    "reporting_quarter": row.quarter,
                    "section": "insight_detection",
                    "title": str(row.insight_title),
                    "content": (
                        f"{row.metric} for {row.product} in {row.region} moves from {float(row.latest_actual):.4f} "
                        f"to forecast {float(row.forecast):.4f}, a change of {float(row.absolute_change):.4f} "
                        f"or {float(row.percentage_change):.2%}, with z-score {float(row.z_score):.2f}."
                    ),
                    "keywords": f"insight material movement {row.metric} {row.product} {row.region} {row.severity_classification}",
                    "source_dataset": "insight_summary",
                    "source_columns": "quarter,metric,product,region,latest_actual,forecast,absolute_change,percentage_change,z_score,severity_classification",
                    "source_filters": f"quarter={row.quarter};metric={row.metric};product={row.product};region={row.region}",
                    "source_value": f"severity={row.severity_classification};z_score={float(row.z_score):.4f}",
                }
            )
        return rows

    def _index_anomaly_investigation(self, anomaly_investigation: pd.DataFrame) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for row in anomaly_investigation.itertuples(index=False):
            rows.append(
                {
                    "document_id": str(row.anomaly_id),
                    "document_type": "anomaly_investigation",
                    "reporting_quarter": row.quarter,
                    "section": "anomaly_investigation",
                    "title": f"Anomaly investigation for {row.anomaly_type}",
                    "content": str(row.explanation_text),
                    "keywords": f"anomaly investigation {row.anomaly_type} {row.product} {row.region} {' '.join(str(row.likely_drivers).replace(',', ' ').split())}",
                    "source_dataset": "anomaly_investigation",
                    "source_columns": "anomaly_id,anomaly_type,product,region,quarter,likely_drivers,explanation_text,support_score,reviewer_status",
                    "source_filters": f"quarter={row.quarter};product={row.product};region={row.region}",
                    "source_value": f"support_score={float(row.support_score):.4f};reviewer_status={row.reviewer_status}",
                }
            )
        return rows

    def _index_movement_summary(self, movement_bridge_summary: pd.DataFrame) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for row in movement_bridge_summary.itertuples(index=False):
            rows.append(
                {
                    "document_id": f"MOV-{row.quarter}-{row.metric}-{abs(hash((row.product, row.region))) % 100000}",
                    "document_type": "movement_analysis",
                    "reporting_quarter": row.quarter,
                    "section": "movement_analysis",
                    "title": f"Movement bridge for {row.metric}",
                    "content": (
                        f"{row.metric} for {row.product} in {row.region} moved from {float(row.opening_value):.4f} to "
                        f"{float(row.closing_value):.4f}. Dominant step: {row.dominant_step}. Top steps: {row.top_movement_steps}."
                    ),
                    "keywords": f"movement bridge waterfall {row.metric} {row.product} {row.region} {row.dominant_step}",
                    "source_dataset": "movement_bridge_summary",
                    "source_columns": "quarter,metric,product,region,opening_value,closing_value,net_change,dominant_step,top_movement_steps,forecast_value",
                    "source_filters": f"quarter={row.quarter};metric={row.metric};product={row.product};region={row.region}",
                    "source_value": f"net_change={float(row.net_change):.4f};forecast_value={row.forecast_value}",
                }
            )
        return rows

    def _index_management_report(self, management_report_sections: pd.DataFrame) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for row in management_report_sections.itertuples(index=False):
            rows.append(
                {
                    "document_id": f"RPT-{int(row.section_order):02d}",
                    "document_type": "management_report",
                    "reporting_quarter": "",
                    "section": "full_report",
                    "title": str(row.section_title),
                    "content": str(row.section_text),
                    "keywords": f"management report full report {row.section_title}",
                    "source_dataset": "management_report_sections",
                    "source_columns": "section_order,section_title,section_text",
                    "source_filters": f"section_order={int(row.section_order)}",
                    "source_value": str(row.section_title),
                }
            )
        return rows

    def _index_narrative_quality(self, narrative_quality_check: pd.DataFrame) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for row in narrative_quality_check.itertuples(index=False):
            rows.append(
                {
                    "document_id": f"NQC-{row.statement_id}",
                    "document_type": "narrative_quality",
                    "reporting_quarter": "",
                    "section": "narrative_quality",
                    "title": f"Narrative quality review for {row.statement_id}",
                    "content": (
                        f"Narrative review result is {row.consistency_result}. "
                        f"Warning flag is {bool(row.warning_flag)}. Suggested revision: {row.suggestion_for_revised_wording}"
                    ),
                    "keywords": f"narrative quality governance {row.linked_metric} {row.consistency_result}",
                    "source_dataset": "narrative_quality_check",
                    "source_columns": "statement_id,linked_metric,consistency_result,warning_flag,suggestion_for_revised_wording",
                    "source_filters": f"statement_id={row.statement_id}",
                    "source_value": f"warning_flag={bool(row.warning_flag)}",
                }
            )
        return rows

    def _index_analyst_review_queue(self, analyst_review_queue: pd.DataFrame) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for row in analyst_review_queue.itertuples(index=False):
            rows.append(
                {
                    "document_id": f"ARQ-{row.review_item_id}",
                    "document_type": "analyst_review",
                    "reporting_quarter": str(row.quarter),
                    "section": "analyst_review",
                    "title": f"Analyst review item: {row.review_type}",
                    "content": (
                        f"Priority {row.priority_label} item from {row.review_source}: {row.issue_summary}. "
                        f"Recommended action: {row.recommended_action}."
                    ),
                    "keywords": f"analyst review queue triage {row.review_source} {row.review_type} {row.priority_label}",
                    "source_dataset": "analyst_review_queue",
                    "source_columns": "review_item_id,review_source,review_type,quarter,metric,product,region,priority_label,issue_summary,recommended_action",
                    "source_filters": f"review_source={row.review_source};priority_label={row.priority_label};review_item_id={row.review_item_id}",
                    "source_value": f"recommended_action={row.recommended_action}",
                }
            )
        return rows

    def _index_figures(self, figure_metadata: pd.DataFrame) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for row in figure_metadata.itertuples(index=False):
            rows.append(
                {
                    "document_id": f"FIG-{row.chart_name}",
                    "document_type": "chart",
                    "reporting_quarter": "",
                    "section": "charts",
                    "title": row.chart_title,
                    "content": f"Chart {row.chart_title} is available at {row.file_path} and covers {row.source_datasets}.",
                    "keywords": f"chart figure visualization {row.chart_name} {row.chart_title}",
                    "source_dataset": "figure_metadata",
                    "source_columns": "chart_name,chart_title,file_path,source_datasets,source_columns",
                    "source_filters": f"chart_name={row.chart_name}",
                    "source_value": f"file_path={row.file_path}",
                }
            )
        return rows

    def _index_governance_logs(self, governance_log: pd.DataFrame) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for row in governance_log.itertuples(index=False):
            rows.append(
                {
                    "document_id": f"GOV-{row.agent_name}",
                    "document_type": "governance_log",
                    "reporting_quarter": "",
                    "section": "governance",
                    "title": f"Governance log for {row.agent_name}",
                    "content": (
                        f"Agent {row.agent_name} finished with status {row.status}. "
                        f"Output keys: {row.output_keys}. Details: {row.details}."
                    ),
                    "keywords": f"governance log workflow {row.agent_name} {row.status}",
                    "source_dataset": "governance_log",
                    "source_columns": "agent_name,status,output_keys,details",
                    "source_filters": f"agent_name={row.agent_name}",
                    "source_value": f"status={row.status}",
                }
            )
        return rows
    def _load_scenario_documents(self, *, file_format: str) -> list[dict[str, object]]:
        artifact_paths = ensure_artifact_dirs(self.config)
        scenario_root = artifact_paths.root / "scenarios"
        if not scenario_root.exists():
            return []

        documents: list[dict[str, object]] = []
        for scenario_dir in sorted(path for path in scenario_root.iterdir() if path.is_dir()):
            report_dir = scenario_dir / "reports" / "scenario"
            impact_path = report_dir / f"scenario_impact_summary.{file_format}"
            top_path = report_dir / f"scenario_top_impacts.{file_format}"
            narrative_path = report_dir / "scenario_narrative_summary.json"

            if impact_path.exists():
                impact_frame = self._load_table(impact_path, file_format=file_format)
                documents.extend(self._index_scenario_impact_summary(impact_frame, scenario_name=scenario_dir.name))
            if top_path.exists():
                top_frame = self._load_table(top_path, file_format=file_format)
                documents.extend(self._index_scenario_top_impacts(top_frame, scenario_name=scenario_dir.name))
            if narrative_path.exists():
                documents.extend(self._index_scenario_narrative(narrative_path, scenario_name=scenario_dir.name))
        return documents

    def _index_scenario_impact_summary(self, impact_summary: pd.DataFrame, *, scenario_name: str) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for row in impact_summary.itertuples(index=False):
            metric_name = getattr(row, "metric", getattr(row, "metric_name", "scenario_metric"))
            quarter = str(getattr(row, "quarter", ""))
            absolute_delta = float(getattr(row, "absolute_delta", getattr(row, "change", 0.0)))
            percentage_delta = float(getattr(row, "percentage_delta", getattr(row, "change_pct", 0.0)))
            rows.append(
                {
                    "document_id": f"SCN-IMP-{scenario_name}-{metric_name}",
                    "document_type": "scenario_summary",
                    "reporting_quarter": quarter,
                    "section": "scenario_reporting",
                    "title": f"Scenario impact for {metric_name}",
                    "content": (
                        f"Scenario {scenario_name} changes {metric_name} from {float(row.baseline_value):.4f} to "
                        f"{float(row.scenario_value):.4f}, a delta of {absolute_delta:.4f} "
                        f"or {percentage_delta:.2%}."
                    ),
                    "keywords": f"scenario impact {scenario_name} {metric_name}",
                    "source_dataset": "scenario_impact_summary",
                    "source_columns": "scenario_name,quarter,metric_name,baseline_value,scenario_value,change,change_pct",
                    "source_filters": f"scenario_name={scenario_name};metric_name={metric_name}",
                    "source_value": f"change_pct={percentage_delta:.4f}",
                }
            )
        return rows

    def _index_scenario_top_impacts(self, top_impacts: pd.DataFrame, *, scenario_name: str) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for row in top_impacts.itertuples(index=False):
            metric_name = getattr(row, "metric", getattr(row, "metric_name", "scenario_metric"))
            quarter = str(getattr(row, "quarter", ""))
            percentage_delta = float(getattr(row, "percentage_delta", getattr(row, "change_pct", 0.0)))
            rows.append(
                {
                    "document_id": f"SCN-TOP-{scenario_name}-{metric_name}-{abs(hash((row.product, row.region))) % 100000}",
                    "document_type": "scenario_top_impact",
                    "reporting_quarter": quarter,
                    "section": "scenario_reporting",
                    "title": f"Top scenario impact for {metric_name}",
                    "content": (
                        f"Scenario {scenario_name} shows a {percentage_delta:.2%} change in {metric_name} "
                        f"for {row.product} in {row.region}."
                    ),
                    "keywords": f"scenario top impact {scenario_name} {metric_name} {row.product} {row.region}",
                    "source_dataset": "scenario_top_impacts",
                    "source_columns": "scenario_name,quarter,metric_name,product,region,baseline_value,scenario_value,change,change_pct",
                    "source_filters": f"scenario_name={scenario_name};metric_name={metric_name};product={row.product};region={row.region}",
                    "source_value": f"change_pct={percentage_delta:.4f}",
                }
            )
        return rows

    def _index_scenario_narrative(self, narrative_path: Path, *, scenario_name: str) -> list[dict[str, object]]:
        payload = json.loads(narrative_path.read_text(encoding="utf-8"))
        summary_text = str(payload.get("summary", payload.get("summary_text", ""))).strip()
        top_metrics_payload = payload.get("top_impacted_metrics", [])
        top_segments_payload = payload.get("top_impacted_segments", [])
        top_metrics = ", ".join(
            item.get("metric_name", item.get("metric", "")) if isinstance(item, dict) else str(item)
            for item in top_metrics_payload
        )
        top_segments = ", ".join(
            (
                f"{item.get('product', '')} / {item.get('region', '')} for {item.get('metric_name', item.get('metric', ''))}".strip()
                if isinstance(item, dict)
                else str(item)
            )
            for item in top_segments_payload
        )
        return [
            {
                "document_id": f"SCN-NAR-{scenario_name}",
                "document_type": "scenario_narrative",
                "reporting_quarter": str(payload.get("quarter", "")),
                "section": "scenario_reporting",
                "title": f"Scenario narrative summary for {scenario_name}",
                "content": f"{summary_text} Top impacted metrics: {top_metrics}. Top impacted segments: {top_segments}.",
                "keywords": (
                    f"scenario narrative summary {scenario_name} "
                    + " ".join(
                        item.get("metric_name", item.get("metric", "")) if isinstance(item, dict) else str(item)
                        for item in top_metrics_payload
                    )
                ),
                "source_dataset": "scenario_narrative_summary",
                "source_columns": "summary,top_impacted_metrics,top_impacted_segments",
                "source_filters": f"scenario_name={scenario_name}",
                "source_value": summary_text,
            }
        ]
    def write(
        self,
        result: ChatbotIndexResult,
        *,
        file_format: str = "csv",
    ) -> dict[str, Path]:
        """Persist the chatbot index."""

        artifact_paths = ensure_artifact_dirs(self.config)
        output_dir = artifact_paths.root / "chatbot"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"chatbot_index.{file_format}"

        result.chatbot_index.to_csv(output_path, index=False)

        return {"chatbot_index": output_path}

    def _load_processed_table(self, name: str, *, file_format: str) -> pd.DataFrame:
        artifact_paths = ensure_artifact_dirs(self.config)
        path = artifact_paths.data_processed / f"{name}.{file_format}"
        if not path.exists():
            return pd.DataFrame()
        return self._load_table(path, file_format=file_format)

    def _load_model_table(self, name: str, *, file_format: str) -> pd.DataFrame:
        artifact_paths = ensure_artifact_dirs(self.config)
        path = artifact_paths.models / f"{name}.{file_format}"
        if not path.exists():
            return pd.DataFrame()
        return self._load_table(path, file_format=file_format)

    def _load_report_table(
        self,
        subdir: str,
        name: str,
        *,
        file_format: str,
    ) -> pd.DataFrame:
        artifact_paths = ensure_artifact_dirs(self.config)
        path = artifact_paths.reports / subdir / f"{name}.{file_format}"
        if not path.exists():
            return pd.DataFrame()
        return self._load_table(path, file_format=file_format)

    def _load_figure_metadata(self, file_format: str) -> pd.DataFrame:
        artifact_paths = ensure_artifact_dirs(self.config)
        path = artifact_paths.figures / "reporting" / f"figure_metadata.{file_format}"
        if not path.exists():
            return pd.DataFrame()
        return self._load_table(path, file_format=file_format)

    def _load_governance_log(self, file_format: str) -> pd.DataFrame:
        artifact_paths = ensure_artifact_dirs(self.config)
        path = artifact_paths.logs / f"governance_log.{file_format}"
        if not path.exists():
            return pd.DataFrame()
        return self._load_table(path, file_format=file_format)

    def _load_table(self, path: Path, *, file_format: str) -> pd.DataFrame:
        return pd.read_csv(path)
