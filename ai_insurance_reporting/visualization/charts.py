"""Visualization generation for reporting outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from ai_insurance_reporting.config.loader import AppConfig, load_config
from ai_insurance_reporting.utils.artifacts import ensure_artifact_dirs


@dataclass(slots=True)
class VisualizationResult:
    """Generated visualization metadata and saved figure paths."""

    figure_metadata: pd.DataFrame
    figure_paths: dict[str, Path]

    def as_dict(self) -> dict[str, pd.DataFrame]:
        """Return persisted tabular outputs."""

        return {"figure_metadata": self.figure_metadata}


class VisualizationGenerator:
    """Generate reporting figures and metadata."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()

    def run(
        self,
        *,
        file_format: str = "csv",
    ) -> tuple[VisualizationResult, dict[str, Path]]:
        """Load upstream outputs, generate figures, and persist metadata."""

        curated = self._load_processed_table("curated_reporting_dataset", file_format=file_format)
        validation_summary = self._load_processed_table("quarterly_validation_summary", file_format=file_format)
        backtest_predictions = self._load_model_table("backtest_predictions", file_format=file_format)
        forecast_output_table = self._load_model_table("forecast_output_table", file_format=file_format)
        shap_global_importance = self._load_report_table(
            "explainability",
            "shap_global_importance",
            file_format=file_format,
        )
        movement_bridge_summary = self._load_report_table(
            "reporting",
            "movement_bridge_summary",
            file_format=file_format,
        )

        result = self.generate(
            curated_reporting_dataset=curated,
            quarterly_validation_summary=validation_summary,
            backtest_predictions=backtest_predictions,
            forecast_output_table=forecast_output_table,
            shap_global_importance=shap_global_importance,
            movement_bridge_summary=movement_bridge_summary,
        )
        output_paths = self.write(result, file_format=file_format)
        return result, output_paths

    def generate(
        self,
        *,
        curated_reporting_dataset: pd.DataFrame,
        quarterly_validation_summary: pd.DataFrame,
        backtest_predictions: pd.DataFrame,
        forecast_output_table: pd.DataFrame,
        shap_global_importance: pd.DataFrame,
        movement_bridge_summary: pd.DataFrame | None = None,
    ) -> VisualizationResult:
        """Create required charts and metadata."""

        artifact_paths = ensure_artifact_dirs(self.config)
        output_dir = artifact_paths.figures / "reporting"
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata_rows: list[dict[str, object]] = []
        figure_paths: dict[str, Path] = {}

        figure_paths["actual_vs_forecast"] = self._plot_actual_vs_forecast(
            curated_reporting_dataset,
            forecast_output_table,
            output_dir / "actual_vs_forecast.png",
        )
        metadata_rows.append(
            self._metadata_row(
                chart_name="actual_vs_forecast",
                chart_title="Actual vs Forecast Premium",
                file_path=figure_paths["actual_vs_forecast"],
                source_datasets="curated_reporting_dataset,forecast_output_table",
                source_columns="quarter,premium_income,forecast_quarter,target_name,forecast_value",
            )
        )

        figure_paths["reserve_movement"] = self._plot_reserve_movement(
            curated_reporting_dataset,
            output_dir / "reserve_movement.png",
        )
        metadata_rows.append(
            self._metadata_row(
                chart_name="reserve_movement",
                chart_title="Quarterly Reserve Movement",
                file_path=figure_paths["reserve_movement"],
                source_datasets="curated_reporting_dataset",
                source_columns="quarter,reserves",
            )
        )

        figure_paths["capital_ratio"] = self._plot_capital_ratio(
            curated_reporting_dataset,
            forecast_output_table,
            output_dir / "capital_ratio.png",
        )
        metadata_rows.append(
            self._metadata_row(
                chart_name="capital_ratio",
                chart_title="Actual and Forecast Synthetic Capital-to-Liability Proxy Ratio",
                file_path=figure_paths["capital_ratio"],
                source_datasets="curated_reporting_dataset,forecast_output_table",
                source_columns="quarter,capital_proxy,liability_balance,forecast_quarter,target_name,forecast_value",
            )
        )

        figure_paths["validation_summaries"] = self._plot_validation_summaries(
            quarterly_validation_summary,
            output_dir / "validation_summaries.png",
        )
        metadata_rows.append(
            self._metadata_row(
                chart_name="validation_summaries",
                chart_title="Quarterly Validation Summary",
                file_path=figure_paths["validation_summaries"],
                source_datasets="quarterly_validation_summary",
                source_columns="quarter,records_with_issues,anomaly_count,validation_pass_rate",
            )
        )

        figure_paths["feature_importance"] = self._plot_feature_importance(
            shap_global_importance,
            output_dir / "feature_importance.png",
        )
        metadata_rows.append(
            self._metadata_row(
                chart_name="feature_importance",
                chart_title="Top SHAP Feature Importance",
                file_path=figure_paths["feature_importance"],
                source_datasets="shap_global_importance",
                source_columns="target_name,feature_name,mean_abs_shap",
            )
        )

        if movement_bridge_summary is not None and not movement_bridge_summary.empty:
            figure_paths["movement_bridge"] = self._plot_movement_bridge(
                movement_bridge_summary,
                output_dir / "movement_bridge.png",
            )
            metadata_rows.append(
                self._metadata_row(
                    chart_name="movement_bridge",
                    chart_title="Top Movement Drivers",
                    file_path=figure_paths["movement_bridge"],
                    source_datasets="movement_bridge_summary",
                    source_columns="quarter,metric,product,region,net_change,dominant_step",
                )
            )

        return VisualizationResult(
            figure_metadata=pd.DataFrame(metadata_rows),
            figure_paths=figure_paths,
        )

    def write(
        self,
        result: VisualizationResult,
        *,
        file_format: str = "csv",
    ) -> dict[str, Path]:
        """Persist metadata and return figure paths."""

        artifact_paths = ensure_artifact_dirs(self.config)
        output_dir = artifact_paths.figures / "reporting"
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = output_dir / f"figure_metadata.{file_format}"
        result.figure_metadata.to_csv(metadata_path, index=False)

        output_paths = dict(result.figure_paths)
        output_paths["figure_metadata"] = metadata_path
        return output_paths

    def _plot_actual_vs_forecast(
        self,
        curated_reporting_dataset: pd.DataFrame,
        forecast_output_table: pd.DataFrame,
        destination: Path,
    ) -> Path:
        actual = (
            curated_reporting_dataset.groupby("quarter", as_index=False)["premium_income"]
            .sum()
            .sort_values("quarter")
        )
        forecast = (
            forecast_output_table.loc[forecast_output_table["target_name"] == "premium"]
            .groupby("forecast_quarter", as_index=False)["forecast_value"]
            .sum()
            .rename(columns={"forecast_quarter": "quarter", "forecast_value": "premium_income"})
            .sort_values("quarter")
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(actual["quarter"], actual["premium_income"], marker="o", label="Actual premium")
        if not forecast.empty:
            ax.plot(forecast["quarter"], forecast["premium_income"], marker="o", linestyle="--", label="Forecast premium")
        ax.set_title("Actual vs Forecast Premium")
        ax.set_xlabel("Quarter")
        ax.set_ylabel("Premium income")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(destination, dpi=150)
        plt.close(fig)
        return destination

    def _plot_reserve_movement(self, curated_reporting_dataset: pd.DataFrame, destination: Path) -> Path:
        reserve_series = (
            curated_reporting_dataset.groupby("quarter", as_index=False)["reserves"]
            .sum()
            .sort_values("quarter")
        )
        reserve_series["reserve_movement"] = reserve_series["reserves"].diff().fillna(0.0)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(reserve_series["quarter"], reserve_series["reserve_movement"], color="#2f6f8f")
        ax.set_title("Quarterly Reserve Movement")
        ax.set_xlabel("Quarter")
        ax.set_ylabel("Reserve movement")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(destination, dpi=150)
        plt.close(fig)
        return destination

    def _plot_capital_ratio(
        self,
        curated_reporting_dataset: pd.DataFrame,
        forecast_output_table: pd.DataFrame,
        destination: Path,
    ) -> Path:
        actual = (
            curated_reporting_dataset.groupby("quarter", as_index=False)[["capital_proxy", "liability_balance"]]
            .sum()
            .sort_values("quarter")
        )
        actual["capital_ratio"] = actual["capital_proxy"] / actual["liability_balance"].where(
            actual["liability_balance"] != 0, 1.0
        )
        forecast = (
            forecast_output_table.loc[forecast_output_table["target_name"] == "capital_ratio"]
            .groupby("forecast_quarter", as_index=False)["forecast_value"]
            .mean()
            .rename(columns={"forecast_quarter": "quarter", "forecast_value": "capital_ratio"})
            .sort_values("quarter")
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            actual["quarter"],
            actual["capital_ratio"],
            marker="o",
            label="Actual synthetic capital-to-liability proxy ratio",
        )
        if not forecast.empty:
            ax.plot(
                forecast["quarter"],
                forecast["capital_ratio"],
                marker="o",
                linestyle="--",
                label="Forecast synthetic capital-to-liability proxy ratio",
            )
        ax.set_title("Actual and Forecast Synthetic Capital-to-Liability Proxy Ratio")
        ax.set_xlabel("Quarter")
        ax.set_ylabel("Synthetic capital-to-liability proxy ratio")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(destination, dpi=150)
        plt.close(fig)
        return destination

    def _plot_validation_summaries(self, quarterly_validation_summary: pd.DataFrame, destination: Path) -> Path:
        summary = quarterly_validation_summary.sort_values("quarter")

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.bar(summary["quarter"], summary["anomaly_count"], color="#cc5a3f", label="Anomaly count")
        ax1.bar(summary["quarter"], summary["records_with_issues"], color="#f1b24a", alpha=0.7, label="Records with issues")
        ax1.set_xlabel("Quarter")
        ax1.set_ylabel("Count")

        ax2 = ax1.twinx()
        ax2.plot(summary["quarter"], summary["validation_pass_rate"], color="#1d7f55", marker="o", label="Validation pass rate")
        ax2.set_ylabel("Pass rate")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        ax1.set_title("Quarterly Validation Summary")
        fig.tight_layout()
        fig.savefig(destination, dpi=150)
        plt.close(fig)
        return destination

    def _plot_feature_importance(self, shap_global_importance: pd.DataFrame, destination: Path) -> Path:
        top_features = (
            shap_global_importance.groupby("feature_name", as_index=False)["mean_abs_shap"]
            .mean()
            .sort_values("mean_abs_shap", ascending=False)
            .head(10)
            .sort_values("mean_abs_shap", ascending=True)
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_features["feature_name"], top_features["mean_abs_shap"], color="#6a8caf")
        ax.set_title("Top SHAP Feature Importance")
        ax.set_xlabel("Mean absolute SHAP value")
        ax.set_ylabel("Feature")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(destination, dpi=150)
        plt.close(fig)
        return destination

    def _plot_movement_bridge(self, movement_bridge_summary: pd.DataFrame, destination: Path) -> Path:
        latest_quarter = movement_bridge_summary["quarter"].max()
        latest = movement_bridge_summary.loc[movement_bridge_summary["quarter"] == latest_quarter].copy()
        latest["label"] = latest["metric"] + " | " + latest["product"] + " | " + latest["region"]
        latest = latest.assign(abs_change=latest["net_change"].abs()).sort_values("abs_change", ascending=False).head(8)

        fig, ax = plt.subplots(figsize=(11, 6))
        colors = ["#0d6c74" if value >= 0 else "#b45f2f" for value in latest["net_change"]]
        ax.barh(latest["label"], latest["net_change"], color=colors)
        ax.set_title("Top Movement Drivers")
        ax.set_xlabel("Net change from opening to closing")
        ax.set_ylabel("Metric / Product / Region")
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(destination, dpi=150)
        plt.close(fig)
        return destination

    def _metadata_row(
        self,
        *,
        chart_name: str,
        chart_title: str,
        file_path: Path,
        source_datasets: str,
        source_columns: str,
    ) -> dict[str, object]:
        return {
            "chart_name": chart_name,
            "chart_title": chart_title,
            "file_path": str(file_path),
            "source_datasets": source_datasets,
            "source_columns": source_columns,
            "file_format": file_path.suffix.lstrip("."),
        }

    def _load_processed_table(self, name: str, *, file_format: str) -> pd.DataFrame:
        artifact_paths = ensure_artifact_dirs(self.config)
        path = artifact_paths.data_processed / f"{name}.{file_format}"
        if not path.exists():
            raise FileNotFoundError(f"Processed dataset not found: {path}")
        return pd.read_csv(path)

    def _load_model_table(self, name: str, *, file_format: str) -> pd.DataFrame:
        artifact_paths = ensure_artifact_dirs(self.config)
        path = artifact_paths.models / f"{name}.{file_format}"
        if not path.exists():
            raise FileNotFoundError(f"Model output not found: {path}")
        return pd.read_csv(path)

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
            raise FileNotFoundError(f"Report output not found: {path}")
        return pd.read_csv(path)
