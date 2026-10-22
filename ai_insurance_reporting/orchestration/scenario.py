"""Scenario workflow execution for what-if and stress testing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ai_insurance_reporting.config.loader import AppConfig, load_config
from ai_insurance_reporting.models.forecasting import ForecastingPipeline
from ai_insurance_reporting.orchestration.workflow import WorkflowOrchestrator
from ai_insurance_reporting.reporting.scenario_reporting import ScenarioReportingBuilder
from ai_insurance_reporting.utils.artifacts import ensure_artifact_dirs


@dataclass(slots=True)
class ScenarioExecutionResult:
    """Summary of a scenario workflow rerun."""

    scenario_name: str
    scenario_parameters: dict[str, float]
    scenario_artifact_root: Path
    output_paths: dict[str, Path]
    summary: dict[str, Any]
    comparison: dict[str, Any]
    metadata_path: Path | None = None


class ScenarioWorkflowRunner:
    """Run the reporting workflow under scenario assumptions in isolated paths."""

    DEFAULT_STRESS_PARAMETERS = {
        "premium_multiplier": 0.97,
        "claims_multiplier": 1.15,
        "reserve_multiplier": 1.10,
        "csm_multiplier": 0.95,
        "asset_return_shift": -0.01,
        "capital_multiplier": 0.90,
    }

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()

    def run_scenario(
        self,
        *,
        scenario_name: str,
        scenario_parameters: dict[str, float],
        file_format: str = "csv",
    ) -> ScenarioExecutionResult:
        scenario_parameters = self._normalize_parameters(scenario_parameters)
        self._ensure_baseline(file_format=file_format)
        scenario_config = self._build_scenario_config(scenario_name)
        scenario_paths = ensure_artifact_dirs(scenario_config)

        workflow_result = WorkflowOrchestrator(config=scenario_config).run(
            file_format=file_format,
            initial_context={
                "scenario_name": scenario_name,
                "scenario_parameters": scenario_parameters,
            },
        )

        baseline_curated = ForecastingPipeline(self.config).load_curated_dataset(file_format=file_format)
        scenario_curated = ForecastingPipeline(scenario_config).load_curated_dataset(file_format=file_format)
        baseline_forecast = self._load_table(self.config, "models", "forecast_output_table", file_format=file_format)
        scenario_forecast = self._load_table(scenario_config, "models", "forecast_output_table", file_format=file_format)

        summary = self._build_summary(scenario_curated, scenario_forecast)
        comparison = self._build_comparison(
            baseline_curated=baseline_curated,
            scenario_curated=scenario_curated,
            baseline_forecast=baseline_forecast,
            scenario_forecast=scenario_forecast,
        )
        scenario_reporting = ScenarioReportingBuilder(self.config).generate(
            scenario_name=scenario_name,
            scenario_parameters=scenario_parameters,
            baseline_curated=baseline_curated,
            scenario_curated=scenario_curated,
            baseline_forecast=baseline_forecast,
            scenario_forecast=scenario_forecast,
        )
        scenario_reporting_paths = ScenarioReportingBuilder(self.config).write(
            scenario_reporting,
            output_root=scenario_paths.root,
            file_format=file_format,
        )
        comparison["scenario_reporting_summary"] = scenario_reporting.scenario_narrative_summary
        metadata_path = self._write_metadata(
            scenario_paths.root,
            scenario_name=scenario_name,
            scenario_parameters=scenario_parameters,
            summary=summary,
            comparison=comparison,
        )

        output_paths = workflow_result.output_paths()
        output_paths.update(scenario_reporting_paths)

        return ScenarioExecutionResult(
            scenario_name=scenario_name,
            scenario_parameters=scenario_parameters,
            scenario_artifact_root=scenario_paths.root,
            output_paths=output_paths,
            summary=summary,
            comparison=comparison,
            metadata_path=metadata_path,
        )

    def infer_parameters(self, query: str) -> tuple[str, dict[str, float]]:
        """Infer a scenario name and parameter set from a natural-language question."""

        query_lower = query.lower()
        scenario_name = self._slugify(query)[:60] or "scenario"
        params = self._normalize_parameters({})

        if any(token in query_lower for token in {"stress", "stressed", "shock", "downside"}):
            params.update(self.DEFAULT_STRESS_PARAMETERS)

        claims_multiplier = self._extract_multiplier(query_lower, ("claim", "claims", "severity"), increase_default=True)
        if claims_multiplier is not None:
            params["claims_multiplier"] = claims_multiplier

        reserve_multiplier = self._extract_multiplier(query_lower, ("reserve", "reserves"), increase_default=True)
        if reserve_multiplier is not None:
            params["reserve_multiplier"] = reserve_multiplier

        premium_multiplier = self._extract_multiplier(query_lower, ("premium", "premiums"), increase_default=False)
        if premium_multiplier is not None:
            params["premium_multiplier"] = premium_multiplier

        csm_multiplier = self._extract_multiplier(query_lower, ("csm",), increase_default=False)
        if csm_multiplier is not None:
            params["csm_multiplier"] = csm_multiplier

        capital_multiplier = self._extract_multiplier(query_lower, ("capital",), increase_default=False)
        if capital_multiplier is not None:
            params["capital_multiplier"] = capital_multiplier

        asset_shift = self._extract_shift(query_lower, ("asset return", "asset returns", "return", "returns"))
        if asset_shift is not None:
            params["asset_return_shift"] = asset_shift

        return scenario_name, params

    def _build_scenario_config(self, scenario_name: str) -> AppConfig:
        root = Path(self.config.paths.artifacts_dir)
        scenario_root = root / "scenarios" / scenario_name
        return self.config.model_copy(
            update={
                "paths": self.config.paths.model_copy(
                    update={
                        "artifacts_dir": str(scenario_root),
                        "data_input_dir": str(scenario_root / "data" / "raw"),
                        "data_processed_dir": str(scenario_root / "data" / "processed"),
                        "reports_dir": str(scenario_root / "reports"),
                        "figures_dir": str(scenario_root / "figures"),
                        "models_dir": str(scenario_root / "models"),
                        "logs_dir": str(scenario_root / "logs"),
                    }
                )
            }
        )

    def _build_summary(self, curated_dataset: pd.DataFrame, forecast_output_table: pd.DataFrame) -> dict[str, Any]:
        latest_quarter = curated_dataset["quarter"].max() if not curated_dataset.empty else ""
        latest = curated_dataset.loc[curated_dataset["quarter"] == latest_quarter]
        forecast_preview = pd.DataFrame()
        available_forecast_quarters: list[str] = []
        forecast_horizon_count = 0
        next_forecast_quarter = ""
        if not forecast_output_table.empty:
            preview = forecast_output_table.copy()
            if "forecast_horizon" in preview.columns:
                preview = (
                    preview.groupby(["forecast_horizon", "forecast_quarter", "target_name"], as_index=False)["forecast_value"]
                    .mean()
                    .sort_values(["forecast_horizon", "target_name"])
                )
                forecast_horizon_count = int(preview["forecast_horizon"].max())
            else:
                preview = (
                    preview.groupby(["forecast_quarter", "target_name"], as_index=False)["forecast_value"]
                    .mean()
                    .sort_values(["forecast_quarter", "target_name"])
                )
                preview["forecast_horizon"] = range(1, len(preview) + 1)
                forecast_horizon_count = int(preview["forecast_horizon"].max()) if not preview.empty else 0
            available_forecast_quarters = preview["forecast_quarter"].drop_duplicates().tolist()
            next_forecast_quarter = available_forecast_quarters[0] if available_forecast_quarters else ""
            forecast_preview = preview.head(15)
        return {
            "latest_quarter": latest_quarter,
            "premium_income": round(float(latest["premium_income"].sum()), 2) if not latest.empty else 0.0,
            "total_claims": round(float(latest["total_claims"].sum()), 2) if not latest.empty else 0.0,
            "reserves": round(float(latest["reserves"].sum()), 2) if not latest.empty else 0.0,
            "csm_closing": round(float(latest["csm_closing"].sum()), 2) if not latest.empty else 0.0,
            "investment_income": round(float(latest["asset_investment_income"].sum()), 2) if not latest.empty else 0.0,
            "capital_ratio": round(self._capital_ratio(latest), 6) if not latest.empty else 0.0,
            "next_forecast_quarter": next_forecast_quarter,
            "available_forecast_quarters": available_forecast_quarters,
            "forecast_horizon_count": forecast_horizon_count,
            "forecast_preview": forecast_preview.to_dict(orient="records"),
        }

    def _build_comparison(
        self,
        *,
        baseline_curated: pd.DataFrame,
        scenario_curated: pd.DataFrame,
        baseline_forecast: pd.DataFrame,
        scenario_forecast: pd.DataFrame,
    ) -> dict[str, Any]:
        latest_baseline = baseline_curated.loc[baseline_curated["quarter"] == baseline_curated["quarter"].max()].copy()
        latest_scenario = scenario_curated.loc[scenario_curated["quarter"] == scenario_curated["quarter"].max()].copy()

        comparison_metrics = []
        for metric_name in ("premium_income", "total_claims", "reserves", "csm_closing", "asset_investment_income"):
            baseline_value = float(latest_baseline[metric_name].sum()) if not latest_baseline.empty else 0.0
            scenario_value = float(latest_scenario[metric_name].sum()) if not latest_scenario.empty else 0.0
            comparison_metrics.append(self._metric_delta(metric_name, baseline_value, scenario_value))

        comparison_metrics.append(
            self._metric_delta(
                "capital_ratio",
                self._capital_ratio(latest_baseline),
                self._capital_ratio(latest_scenario),
            )
        )

        forecast_comparison = []
        forecast_group_columns = [column for column in ["forecast_horizon", "forecast_quarter", "target_name"] if column in baseline_forecast.columns or column in scenario_forecast.columns]
        if forecast_group_columns:
            baseline_forecast_grouped = (
                baseline_forecast.groupby(forecast_group_columns, as_index=False)["forecast_value"]
                .mean()
                .rename(columns={"forecast_value": "baseline_value"})
            ) if not baseline_forecast.empty else pd.DataFrame(columns=[*forecast_group_columns, "baseline_value"])
            scenario_forecast_grouped = (
                scenario_forecast.groupby(forecast_group_columns, as_index=False)["forecast_value"]
                .mean()
                .rename(columns={"forecast_value": "scenario_value"})
            ) if not scenario_forecast.empty else pd.DataFrame(columns=[*forecast_group_columns, "scenario_value"])
            joined_forecasts = baseline_forecast_grouped.merge(
                scenario_forecast_grouped,
                on=forecast_group_columns,
                how="outer",
            ).fillna({"baseline_value": 0.0, "scenario_value": 0.0})
            for row in joined_forecasts.itertuples(index=False):
                delta = self._metric_delta(
                    str(row.target_name),
                    float(row.baseline_value),
                    float(row.scenario_value),
                )
                delta["forecast_quarter"] = str(getattr(row, "forecast_quarter", ""))
                delta["forecast_horizon"] = int(getattr(row, "forecast_horizon", 1))
                forecast_comparison.append(delta)

        return {
            "baseline_latest_quarter": latest_baseline["quarter"].iloc[0] if not latest_baseline.empty else "",
            "scenario_latest_quarter": latest_scenario["quarter"].iloc[0] if not latest_scenario.empty else "",
            "comparison_metrics": comparison_metrics,
            "forecast_comparison": forecast_comparison,
        }

    def _load_table(
        self,
        config: AppConfig,
        area: str,
        stem: str,
        *,
        file_format: str,
    ) -> pd.DataFrame:
        paths = ensure_artifact_dirs(config)
        area_map = {
            "models": paths.models,
            "processed": paths.data_processed,
            "reports": paths.reports,
            "logs": paths.logs,
        }
        base = area_map[area]
        path = base / f"{stem}.{file_format}"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def _ensure_baseline(self, *, file_format: str) -> None:
        paths = ensure_artifact_dirs(self.config)
        baseline_path = paths.data_processed / f"curated_reporting_dataset.{file_format}"
        if baseline_path.exists():
            return
        WorkflowOrchestrator(config=self.config).run(file_format=file_format)

    def _normalize_parameters(self, params: dict[str, float]) -> dict[str, float]:
        normalized = {
            "premium_multiplier": 1.0,
            "claims_multiplier": 1.0,
            "reserve_multiplier": 1.0,
            "csm_multiplier": 1.0,
            "asset_return_shift": 0.0,
            "capital_multiplier": 1.0,
        }
        normalized.update(params)
        return normalized

    def _extract_contexts(self, query: str, anchors: tuple[str, ...]) -> list[str]:
        contexts: list[str] = []
        for anchor in anchors:
            for match in re.finditer(re.escape(anchor), query):
                start = max(0, match.start() - 12)
                end = min(len(query), match.end() + 36)
                contexts.append(query[start:end])
        return contexts

    def _extract_percentage(self, text: str) -> float | None:
        match = re.search(r"(\d+(?:\.\d+)?)\s*%?", text)
        if match:
            return float(match.group(1)) / 100.0
        return None

    def _extract_multiplier(self, query: str, anchors: tuple[str, ...], *, increase_default: bool) -> float | None:
        for context in self._extract_contexts(query, anchors):
            if re.search(r"\b(unchanged|no change|flat|same)\b", context):
                return 1.0
            pct_value = self._extract_percentage(context)
            if pct_value is None:
                continue
            return self._apply_direction(context, pct_value, increase_default=increase_default)
        return None

    def _extract_shift(self, query: str, anchors: tuple[str, ...]) -> float | None:
        for context in self._extract_contexts(query, anchors):
            if re.search(r"\b(unchanged|no change|flat|same)\b", context):
                return 0.0
            pct_value = self._extract_percentage(context)
            if pct_value is None:
                continue
            if any(word in context for word in ("lower", "down", "drop", "fall", "decrease", "reduction")):
                return -pct_value
            if any(word in context for word in ("increase", "rise", "up", "higher", "grow")):
                return pct_value
        return None

    def _apply_direction(self, query: str, pct_value: float, *, increase_default: bool) -> float:
        decrease_words = ("lower", "down", "drop", "fall", "decrease", "reduction")
        increase_words = ("increase", "rise", "up", "higher", "grow")
        if any(word in query for word in decrease_words):
            return max(0.1, 1.0 - pct_value)
        if any(word in query for word in increase_words):
            return 1.0 + pct_value
        if increase_default:
            return 1.0 + pct_value
        return max(0.1, 1.0 - pct_value)

    def _metric_delta(self, metric_name: str, baseline_value: float, scenario_value: float) -> dict[str, float | str]:
        change = scenario_value - baseline_value
        change_pct = (change / baseline_value) if abs(baseline_value) > 1e-9 else 0.0
        return {
            "metric_name": metric_name,
            "baseline_value": round(baseline_value, 6),
            "scenario_value": round(scenario_value, 6),
            "change": round(change, 6),
            "change_pct": round(change_pct, 6),
        }

    def _capital_ratio(self, frame: pd.DataFrame) -> float:
        liability = float(frame["liability_balance"].sum()) if not frame.empty else 0.0
        capital = float(frame["capital_proxy"].sum()) if not frame.empty else 0.0
        if liability <= 0:
            return 0.0
        return capital / liability

    def _slugify(self, text: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        return slug or "scenario"

    def _write_metadata(
        self,
        scenario_root: Path,
        *,
        scenario_name: str,
        scenario_parameters: dict[str, float],
        summary: dict[str, Any],
        comparison: dict[str, Any],
    ) -> Path:
        metadata_path = scenario_root / "logs" / "scenario_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "scenario_name": scenario_name,
            "scenario_parameters": scenario_parameters,
            "summary": summary,
            "comparison": comparison,
        }
        metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return metadata_path
