"""Scenario reporting summaries for management analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import json

import pandas as pd

from ai_insurance_reporting.config.loader import AppConfig, load_config
from ai_insurance_reporting.utils.artifacts import ensure_artifact_dirs


@dataclass(slots=True)
class ScenarioReportingResult:
    """Structured scenario reporting outputs."""

    scenario_impact_summary: pd.DataFrame
    scenario_top_impacts: pd.DataFrame
    scenario_narrative_summary: dict[str, Any]

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "scenario_impact_summary": self.scenario_impact_summary,
            "scenario_top_impacts": self.scenario_top_impacts,
        }


class ScenarioReportingBuilder:
    """Build reporting-friendly scenario comparison artifacts."""

    METRIC_COLUMNS = [
        "premium_income",
        "total_claims",
        "reserves",
        "csm_closing",
        "asset_investment_income",
        "capital_ratio",
    ]

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.settings = self.config.reporting.scenario_reporting

    def generate(
        self,
        *,
        scenario_name: str,
        scenario_parameters: dict[str, float],
        baseline_curated: pd.DataFrame,
        scenario_curated: pd.DataFrame,
        baseline_forecast: pd.DataFrame,
        scenario_forecast: pd.DataFrame,
    ) -> ScenarioReportingResult:
        latest_baseline = baseline_curated.loc[baseline_curated["quarter"] == baseline_curated["quarter"].max()].copy()
        latest_scenario = scenario_curated.loc[scenario_curated["quarter"] == scenario_curated["quarter"].max()].copy()
        impact_rows: list[dict[str, object]] = []
        for metric_name in self.METRIC_COLUMNS:
            baseline_value = self._metric_value(latest_baseline, metric_name)
            scenario_value = self._metric_value(latest_scenario, metric_name)
            impact_rows.append(self._metric_delta("metric", metric_name, "", "", baseline_value, scenario_value))
        scenario_impact_summary = pd.DataFrame(impact_rows)

        top_impact_rows: list[dict[str, object]] = []
        merged = latest_baseline.merge(
            latest_scenario,
            on=["quarter", "product", "region"],
            suffixes=("_baseline", "_scenario"),
        )
        for row in merged.itertuples(index=False):
            for metric_name in ["premium_income", "total_claims", "reserves", "csm_closing"]:
                baseline_value = float(getattr(row, f"{metric_name}_baseline"))
                scenario_value = float(getattr(row, f"{metric_name}_scenario"))
                top_impact_rows.append(
                    self._metric_delta(
                        "segment",
                        metric_name,
                        str(row.product),
                        str(row.region),
                        baseline_value,
                        scenario_value,
                    )
                )
        if not baseline_forecast.empty and not scenario_forecast.empty:
            joined = baseline_forecast.merge(
                scenario_forecast,
                on=["forecast_quarter", "product", "region", "target_name"],
                suffixes=("_baseline", "_scenario"),
            )
            for row in joined.itertuples(index=False):
                top_impact_rows.append(
                    self._metric_delta(
                        "forecast_segment",
                        str(row.target_name),
                        str(row.product),
                        str(row.region),
                        float(row.forecast_value_baseline),
                        float(row.forecast_value_scenario),
                    )
                )
        scenario_top_impacts = pd.DataFrame(top_impact_rows)
        if not scenario_top_impacts.empty:
            scenario_top_impacts = scenario_top_impacts.assign(abs_change_pct=lambda frame: frame["change_pct"].abs())
            scenario_top_impacts = scenario_top_impacts.sort_values("abs_change_pct", ascending=False).head(
                int(self.settings.top_segments_count)
            ).drop(columns=["abs_change_pct"]).reset_index(drop=True)

        top_metrics = scenario_impact_summary.assign(abs_change_pct=lambda frame: frame["change_pct"].abs()).sort_values(
            "abs_change_pct", ascending=False
        ).head(int(self.settings.top_metrics_count))
        narrative_summary = {
            "scenario_name": scenario_name,
            "scenario_parameters": scenario_parameters,
            "summary_text": self._summary_text(scenario_name, top_metrics, scenario_top_impacts),
            "top_impacted_metrics": top_metrics.drop(columns=["abs_change_pct"]).to_dict(orient="records") if not top_metrics.empty else [],
            "top_impacted_segments": scenario_top_impacts.head(int(self.settings.top_segments_count)).to_dict(orient="records") if not scenario_top_impacts.empty else [],
        }
        return ScenarioReportingResult(
            scenario_impact_summary=scenario_impact_summary,
            scenario_top_impacts=scenario_top_impacts,
            scenario_narrative_summary=narrative_summary,
        )

    def write(
        self,
        result: ScenarioReportingResult,
        *,
        output_root: Path,
        file_format: str = "csv",
    ) -> dict[str, Path]:
        output_dir = output_root / "reports" / "scenario"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths: dict[str, Path] = {}
        for name, frame in result.as_dict().items():
            destination = output_dir / f"{name}.{file_format}"
            frame.to_csv(destination, index=False)
            output_paths[name] = destination
        narrative_path = output_dir / "scenario_narrative_summary.json"
        narrative_path.write_text(json.dumps(result.scenario_narrative_summary, indent=2, ensure_ascii=True), encoding="utf-8")
        output_paths["scenario_narrative_summary"] = narrative_path
        return output_paths

    def _metric_value(self, frame: pd.DataFrame, metric_name: str) -> float:
        if frame.empty:
            return 0.0
        if metric_name == "capital_ratio":
            liability = float(frame["liability_balance"].sum())
            capital = float(frame["capital_proxy"].sum())
            return capital / liability if abs(liability) > 1e-9 else 0.0
        return float(frame[metric_name].sum())

    def _metric_delta(
        self,
        impact_type: str,
        metric_name: str,
        product: str,
        region: str,
        baseline_value: float,
        scenario_value: float,
    ) -> dict[str, object]:
        epsilon = float(self.settings.impact_epsilon)
        change = scenario_value - baseline_value
        change_pct = change / (abs(baseline_value) + epsilon)
        return {
            "impact_type": impact_type,
            "metric_name": metric_name,
            "product": product,
            "region": region,
            "baseline_value": round(baseline_value, 6),
            "scenario_value": round(scenario_value, 6),
            "change": round(change, 6),
            "change_pct": round(change_pct, 6),
            "materiality": "material" if abs(change_pct) >= float(self.settings.materiality_pct) else "normal",
        }

    def _summary_text(
        self,
        scenario_name: str,
        top_metrics: pd.DataFrame,
        top_segments: pd.DataFrame,
    ) -> str:
        if top_metrics.empty:
            return f"Scenario {scenario_name} did not produce material impacts versus baseline."
        lead_metric = top_metrics.iloc[0]
        sentence = (
            f"Scenario {scenario_name} shows the largest metric impact in {lead_metric['metric_name'].replace('_', ' ')}, "
            f"with change of {lead_metric['change_pct']:.2%} versus baseline."
        )
        if not top_segments.empty:
            lead_segment = top_segments.iloc[0]
            sentence += (
                f" The most affected segment is {lead_segment['product']} / {lead_segment['region']} for "
                f"{lead_segment['metric_name'].replace('_', ' ')} at {lead_segment['change_pct']:.2%}."
            )
        return sentence
