"""Narrative quality checks for generated reporting commentary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from ai_insurance_reporting.config.loader import AppConfig, load_config
from ai_insurance_reporting.utils.artifacts import ensure_artifact_dirs


@dataclass(slots=True)
class NarrativeQualityResult:
    """Structured narrative review outputs."""

    narrative_quality_check: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {"narrative_quality_check": self.narrative_quality_check}


class NarrativeQualityChecker:
    """Check narrative statements against underlying data and drivers."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.settings = self.config.reporting.narrative_quality

    def generate(
        self,
        *,
        narrative_statements: pd.DataFrame,
        curated_reporting_dataset: pd.DataFrame,
        forecast_output_table: pd.DataFrame,
        shap_global_importance: pd.DataFrame | None = None,
        scenario_impact_summary: pd.DataFrame | None = None,
    ) -> NarrativeQualityResult:
        shap_global_importance = shap_global_importance if shap_global_importance is not None else pd.DataFrame()
        scenario_impact_summary = scenario_impact_summary if scenario_impact_summary is not None else pd.DataFrame()
        latest_quarter = str(curated_reporting_dataset["quarter"].max()) if not curated_reporting_dataset.empty else ""
        latest = curated_reporting_dataset.loc[curated_reporting_dataset["quarter"] == latest_quarter].copy()
        previous_quarter = self._previous_quarter(latest_quarter) if latest_quarter else ""
        previous = curated_reporting_dataset.loc[curated_reporting_dataset["quarter"] == previous_quarter].copy()
        rows: list[dict[str, object]] = []

        for statement in narrative_statements.itertuples(index=False):
            linked_metric = self._infer_metric(str(statement.source_columns), str(statement.source_filters), str(statement.section))
            current_value, comparison_value = self._supporting_values(
                linked_metric=linked_metric,
                latest=latest,
                previous=previous,
                forecast_output_table=forecast_output_table,
                scenario_impact_summary=scenario_impact_summary,
            )
            direction_ok = self._direction_consistent(str(statement.statement_text), current_value, comparison_value)
            severity_ok = self._severity_consistent(str(statement.statement_text), current_value, comparison_value)
            driver_ok = self._driver_consistent(str(statement.statement_text), linked_metric, shap_global_importance)
            warning_flag = not (direction_ok and severity_ok and driver_ok)
            rows.append(
                {
                    "statement_id": str(statement.statement_id),
                    "statement_text": str(statement.statement_text),
                    "linked_metric": linked_metric,
                    "supporting_values": f"current={current_value:.6f};comparison={comparison_value:.6f}",
                    "consistency_result": "pass" if not warning_flag else "review",
                    "warning_flag": warning_flag,
                    "suggestion_for_revised_wording": self._suggestion(
                        statement_text=str(statement.statement_text),
                        direction_ok=direction_ok,
                        severity_ok=severity_ok,
                        driver_ok=driver_ok,
                    ),
                }
            )
        return NarrativeQualityResult(narrative_quality_check=pd.DataFrame(rows))

    def write(
        self,
        result: NarrativeQualityResult,
        *,
        file_format: str = "csv",
    ) -> dict[str, Path]:
        artifact_paths = ensure_artifact_dirs(self.config)
        output_dir = artifact_paths.reports / "narrative"
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
        narrative_statements: pd.DataFrame,
        curated_reporting_dataset: pd.DataFrame,
        forecast_output_table: pd.DataFrame,
        shap_global_importance: pd.DataFrame | None = None,
        scenario_impact_summary: pd.DataFrame | None = None,
        file_format: str = "csv",
    ) -> tuple[NarrativeQualityResult, dict[str, Path]]:
        result = self.generate(
            narrative_statements=narrative_statements,
            curated_reporting_dataset=curated_reporting_dataset,
            forecast_output_table=forecast_output_table,
            shap_global_importance=shap_global_importance,
            scenario_impact_summary=scenario_impact_summary,
        )
        return result, self.write(result, file_format=file_format)

    def _infer_metric(self, source_columns: str, source_filters: str, section: str) -> str:
        source_lower = source_columns.lower()
        if "premium_income" in source_lower:
            return "premium"
        if "total_claims" in source_lower or "claim_count" in source_lower:
            return "claims"
        if "reserves" in source_lower:
            return "reserve_movement"
        if "csm" in source_lower:
            return "csm_movement"
        if "capital" in source_lower:
            return "capital_ratio"
        filters_lower = source_filters.lower()
        if "metric=reserves" in filters_lower:
            return "reserve_movement"
        if "metric=claims" in filters_lower:
            return "claims"
        if "metric=premium_income" in filters_lower:
            return "premium"
        if "metric=csm_closing" in filters_lower:
            return "csm_movement"
        if "metric=capital_proxy" in filters_lower:
            return "capital_ratio"
        if "capital" in section.lower():
            return "capital_ratio"
        return ""

    def _supporting_values(
        self,
        *,
        linked_metric: str,
        latest: pd.DataFrame,
        previous: pd.DataFrame,
        forecast_output_table: pd.DataFrame,
        scenario_impact_summary: pd.DataFrame,
    ) -> tuple[float, float]:
        if linked_metric == "premium":
            return float(latest["premium_income"].sum()), float(previous["premium_income"].sum()) if not previous.empty else 0.0
        if linked_metric == "claims":
            return float(latest["total_claims"].sum()), float(previous["total_claims"].sum()) if not previous.empty else 0.0
        if linked_metric == "reserve_movement":
            current = float(latest["reserve_movement"].sum()) if "reserve_movement" in latest.columns else 0.0
            comparison = float(previous["reserve_movement"].sum()) if not previous.empty and "reserve_movement" in previous.columns else 0.0
            return current, comparison
        if linked_metric == "csm_movement":
            return float(latest["csm_movement"].sum()), float(previous["csm_movement"].sum()) if not previous.empty else 0.0
        if linked_metric == "capital_ratio":
            current = float(latest["capital_proxy"].sum()) / max(float(latest["liability_balance"].sum()), 1.0)
            capital_forecast = forecast_output_table.loc[forecast_output_table["target_name"] == "capital_ratio"].copy()
            sort_columns = [column for column in ["forecast_horizon", "forecast_quarter"] if column in capital_forecast.columns]
            if sort_columns:
                capital_forecast = capital_forecast.sort_values(sort_columns)
            comparison = float(capital_forecast["forecast_value"].iloc[0]) if not capital_forecast.empty else current
            return current, comparison
        if not scenario_impact_summary.empty:
            return float(scenario_impact_summary["scenario_value"].iloc[0]), float(scenario_impact_summary["baseline_value"].iloc[0])
        return 0.0, 0.0

    def _direction_consistent(self, statement_text: str, current_value: float, comparison_value: float) -> bool:
        text = statement_text.lower()
        delta = current_value - comparison_value
        tolerance = float(self.settings.direction_tolerance)
        if any(word in text for word in {"increase", "higher", "up", "rise"}):
            return delta >= -tolerance
        if any(word in text for word in {"decrease", "lower", "down", "fall"}):
            return delta <= tolerance
        return True

    def _severity_consistent(self, statement_text: str, current_value: float, comparison_value: float) -> bool:
        text = statement_text.lower()
        pct_change = abs((current_value - comparison_value) / (abs(comparison_value) + 1.0))
        if any(word in text for word in {"material", "significant", "sharp", "critical"}):
            return pct_change >= float(self.settings.severity_material_pct)
        return True

    def _driver_consistent(self, statement_text: str, linked_metric: str, shap_global_importance: pd.DataFrame) -> bool:
        if shap_global_importance.empty or not linked_metric:
            return True
        top_features = (
            shap_global_importance.loc[shap_global_importance["target_name"] == linked_metric]
            .sort_values("mean_abs_shap", ascending=False)
            .head(int(self.settings.driver_match_top_n))["feature_name"]
            .astype(str)
            .str.lower()
            .tolist()
        )
        if not top_features:
            return True
        text = statement_text.lower()
        mentioned = [feature for feature in top_features if feature.replace("_", " ") in text or feature in text]
        if "driver" in text or "driven" in text:
            return bool(mentioned)
        return True

    def _suggestion(self, *, statement_text: str, direction_ok: bool, severity_ok: bool, driver_ok: bool) -> str:
        if direction_ok and severity_ok and driver_ok:
            return ""
        if not direction_ok:
            return "Check whether the stated direction of change matches the supporting values."
        if not severity_ok:
            return "Consider softening the severity wording or linking it to a larger movement."
        if not driver_ok:
            return "Align the stated driver with the top explainability drivers or remove the driver wording."
        return f"Review statement: {statement_text}"

    def _previous_quarter(self, quarter_label: str) -> str:
        year = int(quarter_label[:4])
        quarter = int(quarter_label[-1])
        if quarter == 1:
            return f"{year - 1}Q4"
        return f"{year}Q{quarter - 1}"
