"""Insight detection over forecasted reporting movements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ai_insurance_reporting.config.loader import AppConfig, load_config
from ai_insurance_reporting.utils.artifacts import ensure_artifact_dirs


@dataclass(slots=True)
class InsightDetectionResult:
    """Structured forecast insight outputs."""

    insight_summary: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {"insight_summary": self.insight_summary}


class InsightDetector:
    """Detect material projected movements from latest actuals and forecasts."""

    TARGET_COLUMNS = {
        "claims": "total_claims",
        "premium": "premium_income",
        "reserve_movement": "reserve_movement",
        "csm_movement": "csm_movement",
        "capital_ratio": "capital_ratio",
    }

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.settings = self.config.reporting.insight_detection

    def generate(
        self,
        *,
        curated_reporting_dataset: pd.DataFrame,
        forecast_output_table: pd.DataFrame,
    ) -> InsightDetectionResult:
        history = self._prepare_history(curated_reporting_dataset)
        latest_quarter = str(history["quarter"].max()) if not history.empty else ""
        latest_history = history.loc[history["quarter"] == latest_quarter].copy()
        rows: list[dict[str, object]] = []
        sort_columns = [column for column in ["target_name", "product", "region", "forecast_horizon", "forecast_quarter"] if column in forecast_output_table.columns]
        ordered_forecasts = forecast_output_table.sort_values(sort_columns) if sort_columns else forecast_output_table.copy()

        for _, segment_forecasts in ordered_forecasts.groupby(["target_name", "product", "region"], sort=False):
            segment_forecasts = segment_forecasts.copy()
            target_name = str(segment_forecasts["target_name"].iloc[0])
            product = str(segment_forecasts["product"].iloc[0])
            region = str(segment_forecasts["region"].iloc[0])
            segment_history = history.loc[
                (history["target_name"] == target_name)
                & (history["product"] == product)
                & (history["region"] == region)
            ].sort_values("quarter_index")
            latest_segment = latest_history.loc[
                (latest_history["target_name"] == target_name)
                & (latest_history["product"] == product)
                & (latest_history["region"] == region)
            ]
            latest_actual = (
                float(latest_segment["actual_value"].iloc[0])
                if not latest_segment.empty
                else float(segment_history["actual_value"].iloc[-1]) if not segment_history.empty else 0.0
            )
            history_window = segment_history["actual_value"].tail(int(self.settings.historical_window)).to_numpy()
            volatility = float(np.std(history_window, ddof=0)) if len(history_window) > 1 else 0.0
            comparison_value = latest_actual
            comparison_basis = "latest_actual"

            for forecast_row in segment_forecasts.itertuples(index=False):
                forecast_value = float(forecast_row.forecast_value)
                absolute_change = forecast_value - comparison_value
                denominator = abs(comparison_value) + float(self.settings.volatility_epsilon)
                percentage_change = absolute_change / denominator
                z_score = absolute_change / (volatility + float(self.settings.volatility_epsilon))
                severity = self._classify_severity(z_score, percentage_change)
                direction = "increase" if absolute_change >= 0 else "decrease"
                rows.append(
                    {
                        "quarter": str(forecast_row.forecast_quarter),
                        "forecast_horizon": int(getattr(forecast_row, "forecast_horizon", 1)),
                        "metric": target_name,
                        "product": product,
                        "region": region,
                        "latest_actual": round(latest_actual, 6),
                        "comparison_value": round(comparison_value, 6),
                        "comparison_basis": comparison_basis,
                        "forecast": round(forecast_value, 6),
                        "absolute_change": round(absolute_change, 6),
                        "percentage_change": round(percentage_change, 6),
                        "historical_volatility": round(volatility, 6),
                        "z_score": round(z_score, 6),
                        "severity_classification": severity,
                        "insight_title": f"{severity.title()} projected {direction} in {target_name.replace('_', ' ')}",
                        "short_explanation_seed": self._explanation_seed(
                            target_name=target_name,
                            product=product,
                            region=region,
                            direction=direction,
                            percentage_change=percentage_change,
                            z_score=z_score,
                            comparison_basis=comparison_basis,
                        ),
                    }
                )
                comparison_value = forecast_value
                comparison_basis = f"prior_forecast:{forecast_row.forecast_quarter}"

        insight_summary = pd.DataFrame(rows)
        if not insight_summary.empty:
            insight_summary = insight_summary.sort_values(
                ["severity_classification", "metric", "product", "region", "z_score"],
                ascending=[True, True, True, True, False],
            ).reset_index(drop=True)
        return InsightDetectionResult(insight_summary=insight_summary)

    def write(
        self,
        result: InsightDetectionResult,
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
        json_path = output_dir / "insight_summary.json"
        json_path.write_text(result.insight_summary.to_json(orient="records", indent=2), encoding="utf-8")
        output_paths["insight_summary_json"] = json_path
        return output_paths

    def run(
        self,
        *,
        curated_reporting_dataset: pd.DataFrame,
        forecast_output_table: pd.DataFrame,
        file_format: str = "csv",
    ) -> tuple[InsightDetectionResult, dict[str, Path]]:
        result = self.generate(
            curated_reporting_dataset=curated_reporting_dataset,
            forecast_output_table=forecast_output_table,
        )
        return result, self.write(result, file_format=file_format)

    def _prepare_history(self, curated_reporting_dataset: pd.DataFrame) -> pd.DataFrame:
        frame = curated_reporting_dataset.copy()
        frame["quarter_index"] = frame["quarter"].map(self._quarter_to_index)
        frame = frame.sort_values(["product", "region", "quarter_index"]).reset_index(drop=True)
        frame["reserve_movement"] = frame.groupby(["product", "region"])["reserves"].diff().fillna(0.0)
        frame["capital_ratio"] = np.where(
            frame["liability_balance"] > 0,
            frame["capital_proxy"] / frame["liability_balance"],
            0.0,
        )
        rows: list[pd.DataFrame] = []
        for target_name, column_name in self.TARGET_COLUMNS.items():
            target_frame = frame[["quarter", "quarter_index", "product", "region", column_name]].copy()
            target_frame = target_frame.rename(columns={column_name: "actual_value"})
            target_frame["target_name"] = target_name
            rows.append(target_frame)
        return pd.concat(rows, ignore_index=True)

    def _classify_severity(self, z_score: float, percentage_change: float) -> str:
        abs_z = abs(z_score)
        abs_pct = abs(percentage_change)
        if abs_z >= float(self.settings.critical_zscore) or abs_pct >= float(self.settings.critical_pct_change):
            return "critical"
        if abs_z >= float(self.settings.material_zscore) or abs_pct >= float(self.settings.material_pct_change):
            return "material"
        if abs_z >= float(self.settings.moderate_zscore) or abs_pct >= float(self.settings.moderate_pct_change):
            return "moderate"
        return "normal"

    def _explanation_seed(
        self,
        *,
        target_name: str,
        product: str,
        region: str,
        direction: str,
        percentage_change: float,
        z_score: float,
        comparison_basis: str,
    ) -> str:
        comparison_label = "the latest actual" if comparison_basis == "latest_actual" else comparison_basis.replace("prior_forecast:", "the prior forecast for ")
        return (
            f"Projected {direction} for {target_name.replace('_', ' ')} in {product} / {region} is "
            f"{percentage_change:.2%} versus {comparison_label}, with standardized movement score {z_score:.2f}."
        )

    def _quarter_to_index(self, quarter_label: str) -> int:
        year = int(quarter_label[:4])
        quarter = int(quarter_label[-1])
        return year * 4 + quarter
