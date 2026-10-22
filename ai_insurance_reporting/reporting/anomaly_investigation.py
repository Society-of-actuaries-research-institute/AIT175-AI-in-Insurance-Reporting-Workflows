"""First-pass anomaly investigation support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import json

import pandas as pd

from ai_insurance_reporting.config.loader import AppConfig, load_config
from ai_insurance_reporting.utils.artifacts import ensure_artifact_dirs


@dataclass(slots=True)
class AnomalyInvestigationResult:
    """Structured anomaly investigation outputs."""

    anomaly_investigation: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {"anomaly_investigation": self.anomaly_investigation}


class AnomalyInvestigator:
    """Build deterministic first-pass explanations for validation anomalies."""

    TARGET_MAP = {
        "reserve_reconciliation_diff": "reserve_movement",
        "csm_reconciliation_diff": "csm_movement",
        "capital_difference": "capital_ratio",
        "required_columns": "",
        "non_negative_columns": "",
    }

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.settings = self.config.reporting.anomaly_investigation

    def generate(
        self,
        *,
        anomaly_table: pd.DataFrame,
        curated_reporting_dataset: pd.DataFrame,
        insight_summary: pd.DataFrame | None = None,
        shap_global_importance: pd.DataFrame | None = None,
        scenario_impact_summary: pd.DataFrame | None = None,
    ) -> AnomalyInvestigationResult:
        insight_summary = insight_summary if insight_summary is not None else pd.DataFrame()
        shap_global_importance = shap_global_importance if shap_global_importance is not None else pd.DataFrame()
        scenario_impact_summary = scenario_impact_summary if scenario_impact_summary is not None else pd.DataFrame()
        rows: list[dict[str, object]] = []
        history_window = int(self.settings.historical_window)

        for index, anomaly in enumerate(anomaly_table.itertuples(index=False), start=1):
            segment = curated_reporting_dataset.loc[
                (curated_reporting_dataset["quarter"] == anomaly.quarter)
                & (curated_reporting_dataset["product"] == anomaly.product)
                & (curated_reporting_dataset["region"] == anomaly.region)
            ]
            segment_row = segment.iloc[0] if not segment.empty else None
            history = curated_reporting_dataset.loc[
                (curated_reporting_dataset["product"] == anomaly.product)
                & (curated_reporting_dataset["region"] == anomaly.region)
            ].sort_values("quarter").tail(history_window)
            related_metrics = {
                "premium_income": float(segment_row["premium_income"]) if segment_row is not None else 0.0,
                "total_claims": float(segment_row["total_claims"]) if segment_row is not None else 0.0,
                "reserves": float(segment_row["reserves"]) if segment_row is not None else 0.0,
                "csm_closing": float(segment_row["csm_closing"]) if segment_row is not None else 0.0,
                "capital_proxy": float(segment_row["capital_proxy"]) if segment_row is not None else 0.0,
            }
            target_name = self.TARGET_MAP.get(str(anomaly.metric_name), "")
            matching_insights = insight_summary.loc[
                (insight_summary["product"] == anomaly.product)
                & (insight_summary["region"] == anomaly.region)
                & (insight_summary["metric"] == target_name)
            ]
            top_drivers = self._top_drivers(shap_global_importance, target_name)
            scenario_hits = scenario_impact_summary.loc[
                scenario_impact_summary.get("metric_name", pd.Series(dtype=str)).astype(str) == target_name
            ] if not scenario_impact_summary.empty else pd.DataFrame()
            likely_drivers = self._likely_drivers(
                rule_name=str(anomaly.rule_name),
                top_drivers=top_drivers,
                has_scenario_context=not scenario_hits.empty,
                has_matching_insight=not matching_insights.empty,
            )
            support_score = self._support_score(
                observed_value=float(anomaly.observed_value) if pd.notna(anomaly.observed_value) else 0.0,
                has_scenario_context=not scenario_hits.empty,
                has_matching_insight=not matching_insights.empty,
                top_driver_count=len(top_drivers),
            )
            rows.append(
                {
                    "anomaly_id": f"ANOM-{index:04d}",
                    "anomaly_type": str(anomaly.rule_name),
                    "product": str(anomaly.product),
                    "region": str(anomaly.region),
                    "quarter": str(anomaly.quarter),
                    "supporting_variables": json.dumps(
                        {
                            "related_metrics": related_metrics,
                            "recent_history": history[["quarter", "premium_income", "total_claims", "reserves", "csm_closing", "capital_proxy"]]
                            .to_dict(orient="records"),
                            "explainability_drivers": top_drivers,
                        },
                        ensure_ascii=True,
                    ),
                    "likely_drivers": " | ".join(likely_drivers),
                    "explanation_text": self._explanation_text(
                        anomaly_type=str(anomaly.rule_name),
                        product=str(anomaly.product),
                        region=str(anomaly.region),
                        quarter=str(anomaly.quarter),
                        likely_drivers=likely_drivers,
                        has_scenario_context=not scenario_hits.empty,
                        has_matching_insight=not matching_insights.empty,
                    ),
                    "support_score": round(support_score, 4),
                    "reviewer_status": "pending_review",
                }
            )

        return AnomalyInvestigationResult(anomaly_investigation=pd.DataFrame(rows))

    def write(
        self,
        result: AnomalyInvestigationResult,
        *,
        file_format: str = "csv",
    ) -> dict[str, Path]:
        artifact_paths = ensure_artifact_dirs(self.config)
        output_dir = artifact_paths.reports / "reporting"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths: dict[str, Path] = {}
        for name, frame in result.as_dict().items():
            destination = output_dir / f"{name}.{file_format}"
            frame.to_csv(destination, index=False)
            output_paths[name] = destination
        return output_paths

    def run(
        self,
        *,
        anomaly_table: pd.DataFrame,
        curated_reporting_dataset: pd.DataFrame,
        insight_summary: pd.DataFrame | None = None,
        shap_global_importance: pd.DataFrame | None = None,
        scenario_impact_summary: pd.DataFrame | None = None,
        file_format: str = "csv",
    ) -> tuple[AnomalyInvestigationResult, dict[str, Path]]:
        result = self.generate(
            anomaly_table=anomaly_table,
            curated_reporting_dataset=curated_reporting_dataset,
            insight_summary=insight_summary,
            shap_global_importance=shap_global_importance,
            scenario_impact_summary=scenario_impact_summary,
        )
        return result, self.write(result, file_format=file_format)

    def _top_drivers(self, shap_global_importance: pd.DataFrame, target_name: str) -> list[str]:
        if shap_global_importance.empty or not target_name:
            return []
        return (
            shap_global_importance.loc[shap_global_importance["target_name"] == target_name]
            .sort_values("mean_abs_shap", ascending=False)
            .head(3)["feature_name"]
            .astype(str)
            .tolist()
        )

    def _likely_drivers(
        self,
        *,
        rule_name: str,
        top_drivers: list[str],
        has_scenario_context: bool,
        has_matching_insight: bool,
    ) -> list[str]:
        drivers: list[str] = []
        if rule_name == "missing_values":
            drivers.append("missing required reporting inputs")
        elif rule_name == "negative_values":
            drivers.append("non-permitted negative balance or flow")
        elif rule_name == "reserve_reconciliation":
            drivers.append("inconsistency between liability components")
        elif rule_name == "csm_reconciliation":
            drivers.append("CSM roll-forward mismatch")
        elif rule_name == "capital_consistency":
            drivers.append("capital proxy differs from expected balance relationship")
        if has_matching_insight:
            drivers.append("flagged movement is outside the recent historical pattern")
        if has_scenario_context:
            drivers.append("scenario-driven stress effect may be contributing")
        if top_drivers and bool(self.settings.include_explainability):
            drivers.append(f"supporting forecast drivers include {', '.join(top_drivers)}")
        return drivers or ["requires analyst review"]

    def _support_score(
        self,
        *,
        observed_value: float,
        has_scenario_context: bool,
        has_matching_insight: bool,
        top_driver_count: int,
    ) -> float:
        score = 0.35
        if abs(observed_value) > 0:
            score += 0.2
        if has_matching_insight:
            score += 0.2
        if has_scenario_context:
            score += 0.15
        if top_driver_count > 0:
            score += 0.1
        return min(score, 1.0)

    def _explanation_text(
        self,
        *,
        anomaly_type: str,
        product: str,
        region: str,
        quarter: str,
        likely_drivers: list[str],
        has_scenario_context: bool,
        has_matching_insight: bool,
    ) -> str:
        context_text = ""
        if has_scenario_context:
            context_text = " Scenario outputs also show a related stressed movement."
        elif has_matching_insight:
            context_text = " Forecast insight screening shows a related material movement in the same segment."
        return (
            f"{anomaly_type.replace('_', ' ').title()} was flagged for {product} in {region} during {quarter}. "
            f"Likely drivers include {', '.join(likely_drivers)}.{context_text}"
        )
