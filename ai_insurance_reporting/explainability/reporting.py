"""Explainability workflows for forecasting models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import lime.lime_tabular
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import partial_dependence
from sklearn.pipeline import Pipeline

from ai_insurance_reporting.config.loader import AppConfig, load_config
from ai_insurance_reporting.models.forecasting import ForecastingPipeline, ForecastingResult
from ai_insurance_reporting.utils.artifacts import ensure_artifact_dirs


@dataclass(slots=True)
class ExplainabilityResult:
    """Generated explainability outputs."""

    shap_global_importance: pd.DataFrame
    shap_local_explanations: pd.DataFrame
    lime_explanations: pd.DataFrame
    pdp_ice_table: pd.DataFrame
    explanation_report_path: Path | None = None

    def as_dict(self) -> dict[str, pd.DataFrame]:
        """Return tabular explainability outputs keyed by file stem."""

        return {
            "shap_global_importance": self.shap_global_importance,
            "shap_local_explanations": self.shap_local_explanations,
            "lime_explanations": self.lime_explanations,
            "pdp_ice_table": self.pdp_ice_table,
        }


class ExplainabilityPipeline:
    """Generate SHAP, LIME, PDP, and ICE outputs for forecasting models."""

    EXPLAINABLE_MODEL_PRIORITY = ("gradient_boosting", "time_series")

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.forecasting_pipeline = ForecastingPipeline(self.config)

    def run(
        self,
        *,
        file_format: str = "csv",
        sample_size: int = 24,
    ) -> tuple[ExplainabilityResult, dict[str, Path]]:
        """Generate explainability outputs from the curated dataset."""

        curated = self.forecasting_pipeline.load_curated_dataset(file_format=file_format)
        training_frame = self.forecasting_pipeline.prepare_training_frame(curated)
        forecasting_result = self.forecasting_pipeline.train(training_frame)
        result = self.generate(training_frame, forecasting_result, sample_size=sample_size)
        output_paths = self.write(result, file_format=file_format)
        return result, output_paths

    def generate(
        self,
        training_frame: pd.DataFrame,
        forecasting_result: ForecastingResult,
        *,
        sample_size: int = 24,
    ) -> ExplainabilityResult:
        """Create explainability outputs for the selected model per target."""

        shap_global_rows: list[dict[str, object]] = []
        shap_local_rows: list[dict[str, object]] = []
        lime_rows: list[dict[str, object]] = []
        pdp_ice_rows: list[dict[str, object]] = []
        report_sections: list[str] = ["# Forecast Explainability Report", ""]

        for target_name in self.forecasting_pipeline.TARGET_COLUMNS:
            target_frame = training_frame.loc[training_frame["target_name"] == target_name].copy()
            selected_family = self._select_explainable_model_family(
                forecasting_result.evaluation_table,
                target_name=target_name,
            )
            pipeline = self._fit_selected_model(target_frame, selected_family)
            X_train = target_frame[self.forecasting_pipeline.FEATURE_COLUMNS].copy()
            transformed = pipeline.named_steps["preprocessor"].transform(X_train)
            feature_names = self._get_feature_names(pipeline)

            sample_count = min(sample_size, len(target_frame))
            sample_indices = np.arange(sample_count)
            transformed_sample = transformed[sample_indices]
            raw_sample = X_train.iloc[sample_indices].reset_index(drop=True)
            row_labels = self._build_row_labels(target_frame.iloc[sample_indices])

            shap_values = self._compute_shap_values(pipeline, transformed, transformed_sample, selected_family)
            shap_global_rows.extend(
                self._build_shap_global_rows(
                    target_name=target_name,
                    model_family=selected_family,
                    feature_names=feature_names,
                    shap_values=shap_values,
                )
            )
            shap_local_rows.extend(
                self._build_shap_local_rows(
                    target_name=target_name,
                    model_family=selected_family,
                    feature_names=feature_names,
                    shap_values=shap_values,
                    row_labels=row_labels,
                )
            )

            lime_rows.extend(
                self._build_lime_rows(
                    target_name=target_name,
                    model_family=selected_family,
                    transformed_train=transformed,
                    transformed_sample=transformed_sample,
                    feature_names=feature_names,
                    row_labels=row_labels,
                    regressor=pipeline.named_steps["regressor"],
                )
            )

            pdp_ice_rows.extend(
                self._build_pdp_ice_rows(
                    target_name=target_name,
                    model_family=selected_family,
                    pipeline=pipeline,
                    X_train=X_train,
                    row_labels=row_labels,
                )
            )

            report_sections.extend(
                self._build_report_section(
                    target_name=target_name,
                    model_family=selected_family,
                    global_rows=shap_global_rows,
                    local_rows=shap_local_rows,
                    lime_rows=lime_rows,
                )
            )

        result = ExplainabilityResult(
            shap_global_importance=pd.DataFrame(shap_global_rows).sort_values(
                ["target_name", "mean_abs_shap"], ascending=[True, False]
            ).reset_index(drop=True),
            shap_local_explanations=pd.DataFrame(shap_local_rows).sort_values(
                ["target_name", "row_label", "abs_shap_value"], ascending=[True, True, False]
            ).reset_index(drop=True),
            lime_explanations=pd.DataFrame(lime_rows).sort_values(
                ["target_name", "row_label", "abs_weight"], ascending=[True, True, False]
            ).reset_index(drop=True),
            pdp_ice_table=pd.DataFrame(pdp_ice_rows).sort_values(
                ["target_name", "feature_name", "grid_value", "row_label"]
            ).reset_index(drop=True),
        )
        result.explanation_report_path = self._write_report("\n".join(report_sections))
        return result

    def write(
        self,
        result: ExplainabilityResult,
        *,
        file_format: str = "csv",
    ) -> dict[str, Path]:
        """Persist explainability outputs."""

        artifact_paths = ensure_artifact_dirs(self.config)
        output_dir = artifact_paths.reports / "explainability"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths: dict[str, Path] = {}
        for name, frame in result.as_dict().items():
            destination = output_dir / f"{name}.{file_format}"
            frame.to_csv(destination, index=False)
            output_paths[name] = destination

        if result.explanation_report_path is not None:
            output_paths["explanation_report"] = result.explanation_report_path

        return output_paths

    def _select_explainable_model_family(
        self,
        evaluation_table: pd.DataFrame,
        *,
        target_name: str,
    ) -> str:
        subset = evaluation_table.loc[evaluation_table["target_name"] == target_name].copy()
        for family in self.EXPLAINABLE_MODEL_PRIORITY:
            family_rows = subset.loc[subset["model_family"] == family]
            if not family_rows.empty:
                return family_rows.sort_values("mae").iloc[0]["model_family"]
        raise ValueError(f"No explainable model family found for target {target_name}")

    def _fit_selected_model(self, target_frame: pd.DataFrame, model_family: str) -> Pipeline:
        if model_family == "gradient_boosting":
            pipeline = self.forecasting_pipeline._build_gradient_boosting_model()
        elif model_family == "time_series":
            pipeline = self.forecasting_pipeline._build_time_series_model()
        else:
            raise ValueError(f"Unsupported explainability model family: {model_family}")

        pipeline.fit(
            target_frame[self.forecasting_pipeline.FEATURE_COLUMNS],
            target_frame["target_value"],
        )
        return pipeline

    def _compute_shap_values(
        self,
        pipeline: Pipeline,
        transformed_train: np.ndarray,
        transformed_sample: np.ndarray,
        model_family: str,
    ) -> np.ndarray:
        regressor = pipeline.named_steps["regressor"]
        if model_family == "gradient_boosting":
            explainer = shap.TreeExplainer(regressor)
            shap_values = explainer.shap_values(transformed_sample)
        else:
            background = transformed_train[: min(50, len(transformed_train))]
            explainer = shap.Explainer(regressor.predict, background)
            shap_values = explainer(transformed_sample).values
        return np.asarray(shap_values)

    def _build_shap_global_rows(
        self,
        *,
        target_name: str,
        model_family: str,
        feature_names: list[str],
        shap_values: np.ndarray,
    ) -> list[dict[str, object]]:
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        rows: list[dict[str, object]] = []
        for feature_name, importance in zip(feature_names, mean_abs, strict=False):
            rows.append(
                {
                    "target_name": target_name,
                    "model_family": model_family,
                    "feature_name": feature_name,
                    "mean_abs_shap": round(float(importance), 6),
                }
            )
        return rows

    def _build_shap_local_rows(
        self,
        *,
        target_name: str,
        model_family: str,
        feature_names: list[str],
        shap_values: np.ndarray,
        row_labels: list[str],
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for idx, row_label in enumerate(row_labels):
            for feature_name, shap_value in zip(feature_names, shap_values[idx], strict=False):
                rows.append(
                    {
                        "target_name": target_name,
                        "model_family": model_family,
                        "row_label": row_label,
                        "feature_name": feature_name,
                        "shap_value": round(float(shap_value), 6),
                        "abs_shap_value": round(abs(float(shap_value)), 6),
                    }
                )
        return rows

    def _build_lime_rows(
        self,
        *,
        target_name: str,
        model_family: str,
        transformed_train: np.ndarray,
        transformed_sample: np.ndarray,
        feature_names: list[str],
        row_labels: list[str],
        regressor: object,
    ) -> list[dict[str, object]]:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.asarray(transformed_train),
            feature_names=feature_names,
            mode="regression",
            discretize_continuous=False,
        )

        rows: list[dict[str, object]] = []
        for idx, row_label in enumerate(row_labels):
            explanation = explainer.explain_instance(
                np.asarray(transformed_sample[idx]),
                regressor.predict,
                num_features=min(8, len(feature_names)),
            )
            for feature_name, weight in explanation.as_list():
                rows.append(
                    {
                        "target_name": target_name,
                        "model_family": model_family,
                        "row_label": row_label,
                        "feature_name": feature_name,
                        "lime_weight": round(float(weight), 6),
                        "abs_weight": round(abs(float(weight)), 6),
                    }
                )
        return rows

    def _build_pdp_ice_rows(
        self,
        *,
        target_name: str,
        model_family: str,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        row_labels: list[str],
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        X_numeric = X_train.copy()
        for feature_name in ["quarter_index", "lag_1", "lag_2"]:
            X_numeric[feature_name] = X_numeric[feature_name].astype(float)

        for feature_name in ["quarter_index", "lag_1", "lag_2"]:
            unique_values = X_numeric[feature_name].nunique()
            if unique_values < 2:
                continue
            try:
                pd_result = partial_dependence(
                    pipeline,
                    X_numeric,
                    features=[feature_name],
                    kind="both",
                    grid_resolution=min(10, max(3, unique_values)),
                )
            except ValueError:
                continue
            grid_values = np.asarray(pd_result["grid_values"][0])
            average_values = np.asarray(pd_result["average"][0])
            individual_values = np.asarray(pd_result["individual"][0])

            for grid_idx, grid_value in enumerate(grid_values):
                rows.append(
                    {
                        "target_name": target_name,
                        "model_family": model_family,
                        "feature_name": feature_name,
                        "curve_type": "pdp",
                        "row_label": "aggregate",
                        "grid_value": round(float(grid_value), 6),
                        "effect_value": round(float(average_values[grid_idx]), 6),
                    }
                )

                for row_idx, row_label in enumerate(row_labels):
                    rows.append(
                        {
                            "target_name": target_name,
                            "model_family": model_family,
                            "feature_name": feature_name,
                            "curve_type": "ice",
                            "row_label": row_label,
                            "grid_value": round(float(grid_value), 6),
                            "effect_value": round(float(individual_values[row_idx, grid_idx]), 6),
                        }
                    )
        return rows

    def _build_row_labels(self, sample_frame: pd.DataFrame) -> list[str]:
        labels: list[str] = []
        for row in sample_frame.itertuples(index=False):
            labels.append(f"{row.quarter}|{row.product}|{row.region}")
        return labels

    def _get_feature_names(self, pipeline: Pipeline) -> list[str]:
        preprocessor = pipeline.named_steps["preprocessor"]
        names = preprocessor.get_feature_names_out(self.forecasting_pipeline.FEATURE_COLUMNS)
        return [str(name) for name in names]

    def _build_report_section(
        self,
        *,
        target_name: str,
        model_family: str,
        global_rows: list[dict[str, object]],
        local_rows: list[dict[str, object]],
        lime_rows: list[dict[str, object]],
    ) -> list[str]:
        section = [f"## {target_name.replace('_', ' ').title()}", ""]
        section.append(f"- Selected model family: `{model_family}`")

        target_global = [row for row in global_rows if row["target_name"] == target_name]
        top_global = sorted(target_global, key=lambda row: row["mean_abs_shap"], reverse=True)[:3]
        global_summary = ", ".join(
            f"{row['feature_name']} ({row['mean_abs_shap']:.4f})" for row in top_global
        )
        section.append(f"- Top SHAP global drivers: {global_summary}")

        target_local = [row for row in local_rows if row["target_name"] == target_name]
        first_row_label = target_local[0]["row_label"] if target_local else "n/a"
        top_local = sorted(
            [row for row in target_local if row["row_label"] == first_row_label],
            key=lambda row: row["abs_shap_value"],
            reverse=True,
        )[:3]
        local_summary = ", ".join(
            f"{row['feature_name']} ({row['shap_value']:.4f})" for row in top_local
        )
        section.append(f"- Example SHAP local explanation for `{first_row_label}`: {local_summary}")

        target_lime = [row for row in lime_rows if row["target_name"] == target_name]
        top_lime = sorted(target_lime, key=lambda row: row["abs_weight"], reverse=True)[:3]
        lime_summary = ", ".join(
            f"{row['feature_name']} ({row['lime_weight']:.4f})" for row in top_lime
        )
        section.append(f"- Top LIME contributors: {lime_summary}")
        section.append("")
        return section

    def _write_report(self, report_text: str) -> Path:
        artifact_paths = ensure_artifact_dirs(self.config)
        output_dir = artifact_paths.reports / "explainability"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "explanation_report.md"
        report_path.write_text(report_text, encoding="utf-8")
        return report_path
