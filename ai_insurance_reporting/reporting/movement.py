"""Movement analysis and bridge reporting for management reporting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from ai_insurance_reporting.chatbot.llm_client import get_default_llm_client
from ai_insurance_reporting.config.loader import AppConfig, load_config
from ai_insurance_reporting.utils.artifacts import ensure_artifact_dirs


@dataclass(slots=True)
class MovementAnalysisResult:
    """Structured movement analysis outputs."""

    movement_analysis: pd.DataFrame
    movement_bridge_summary: pd.DataFrame
    movement_llm_summary: str = ""
    movement_llm_summary_path: Path | None = None

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "movement_analysis": self.movement_analysis,
            "movement_bridge_summary": self.movement_bridge_summary,
        }


class MovementAnalysisBuilder:
    """Build beginning-to-end movement bridges for key reporting metrics."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.settings = self.config.reporting.movement_analysis

    def run(
        self,
        *,
        curated_reporting_dataset: pd.DataFrame,
        forecast_output_table: pd.DataFrame,
        file_format: str = "csv",
    ) -> tuple[MovementAnalysisResult, dict[str, Path]]:
        result = self.generate(
            curated_reporting_dataset=curated_reporting_dataset,
            forecast_output_table=forecast_output_table,
        )
        return result, self.write(result, file_format=file_format)

    def generate(
        self,
        *,
        curated_reporting_dataset: pd.DataFrame,
        forecast_output_table: pd.DataFrame,
    ) -> MovementAnalysisResult:
        frame = curated_reporting_dataset.copy().sort_values(["product", "region", "quarter"]).reset_index(drop=True)
        movement_rows: list[dict[str, object]] = []

        for (product, region), segment in frame.groupby(["product", "region"], sort=False):
            segment = segment.copy().sort_values("quarter")
            previous_row: pd.Series | None = None
            for row in segment.to_dict(orient="records"):
                movement_rows.extend(self._segment_metric_rows(product, region, row, previous_row))
                previous_row = pd.Series(row)

        movement_analysis = pd.DataFrame(movement_rows)
        if movement_analysis.empty:
            movement_bridge_summary = pd.DataFrame()
        else:
            movement_bridge_summary = self._build_bridge_summary(movement_analysis, forecast_output_table)
        movement_llm_summary = self._build_llm_summary(movement_bridge_summary)
        return MovementAnalysisResult(
            movement_analysis=movement_analysis,
            movement_bridge_summary=movement_bridge_summary,
            movement_llm_summary=movement_llm_summary,
        )

    def write(
        self,
        result: MovementAnalysisResult,
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
        json_path = output_dir / "movement_bridge_summary.json"
        json_path.write_text(result.movement_bridge_summary.to_json(orient="records", indent=2), encoding="utf-8")
        output_paths["movement_bridge_summary_json"] = json_path
        summary_path = output_dir / "movement_llm_summary.md"
        summary_path.write_text(result.movement_llm_summary, encoding="utf-8")
        result.movement_llm_summary_path = summary_path
        output_paths["movement_llm_summary"] = summary_path
        return output_paths

    def _segment_metric_rows(
        self,
        product: str,
        region: str,
        current: dict[str, object],
        previous: pd.Series | None,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        quarter = str(current["quarter"])
        prev = previous if previous is not None else pd.Series(dtype=float)

        rows.extend(
            self._build_bridge_rows(
                quarter=quarter,
                metric="premium_income",
                product=product,
                region=region,
                opening_value=float(prev.get("premium_income", 0.0)),
                closing_value=float(current.get("premium_income", 0.0)),
                steps=[
                    ("new_business", float(current.get("new_business_count", 0.0)) * self.settings.new_business_premium_weight),
                    ("retention", float(prev.get("premium_income", 0.0)) * (float(prev.get("average_lapse_rate", 0.0)) - float(current.get("average_lapse_rate", 0.0))) * self.settings.retention_weight),
                    ("rate_and_mix", float(current.get("policies_inforce", 0.0)) * (float(current.get("average_premium_per_policy", 0.0)) - float(prev.get("average_premium_per_policy", current.get("average_premium_per_policy", 0.0)))) * self.settings.rate_change_weight),
                ],
            )
        )
        current_avg_severity = float(current.get("total_claims", 0.0)) / max(float(current.get("claim_count", 0.0)), 1.0)
        previous_avg_severity = float(prev.get("total_claims", 0.0)) / max(float(prev.get("claim_count", 0.0)), 1.0)
        rows.extend(
            self._build_bridge_rows(
                quarter=quarter,
                metric="claims",
                product=product,
                region=region,
                opening_value=float(prev.get("total_claims", 0.0)),
                closing_value=float(current.get("total_claims", 0.0)),
                steps=[
                    ("frequency_effect", (float(current.get("claim_count", 0.0)) - float(prev.get("claim_count", 0.0))) * max(previous_avg_severity, 0.0) * self.settings.claims_frequency_weight),
                    ("severity_effect", max(float(current.get("claim_count", 0.0)), 1.0) * (current_avg_severity - previous_avg_severity) * self.settings.claims_severity_weight),
                    ("mortality_and_lapse", float(current.get("premium_income", 0.0)) * ((float(current.get("average_mortality_rate", 0.0)) - float(prev.get("average_mortality_rate", current.get("average_mortality_rate", 0.0)))) + (float(current.get("average_lapse_rate", 0.0)) - float(prev.get("average_lapse_rate", current.get("average_lapse_rate", 0.0))))) * self.settings.claims_assumption_weight),
                ],
            )
        )
        rows.extend(
            self._build_bridge_rows(
                quarter=quarter,
                metric="reserves",
                product=product,
                region=region,
                opening_value=float(prev.get("reserves", 0.0)),
                closing_value=float(current.get("reserves", 0.0)),
                steps=[
                    ("new_business", float(current.get("new_business_count", 0.0)) * self.settings.reserve_new_business_weight),
                    ("claims_experience", (float(current.get("case_reserves", 0.0)) - float(prev.get("case_reserves", 0.0))) * self.settings.reserve_claims_weight),
                    ("economic_effect", (float(current.get("average_discount_rate", 0.0)) - float(prev.get("average_discount_rate", current.get("average_discount_rate", 0.0)))) * max(float(prev.get("reserves", 0.0)), 0.0) * self.settings.reserve_economic_weight),
                    ("runoff_and_release", -float(current.get("total_claims_paid", 0.0)) * self.settings.reserve_runoff_weight),
                ],
            )
        )
        rows.extend(
            self._build_bridge_rows(
                quarter=quarter,
                metric="csm_closing",
                product=product,
                region=region,
                opening_value=float(current.get("csm_opening", 0.0)),
                closing_value=float(current.get("csm_closing", 0.0)),
                steps=[
                    ("new_business", float(current.get("csm_new_business", 0.0))),
                    ("interest_accretion", float(current.get("csm_interest_accretion", 0.0))),
                    ("release", -float(current.get("csm_release", 0.0))),
                    ("assumption_effect", (float(current.get("average_discount_rate", 0.0)) - float(prev.get("average_discount_rate", current.get("average_discount_rate", 0.0)))) * max(float(current.get("csm_opening", 0.0)), 0.0) * self.settings.csm_assumption_weight),
                ],
            )
        )
        reserve_movement = float(current.get("reserves", 0.0)) - float(prev.get("reserves", 0.0))
        csm_movement = float(current.get("csm_closing", 0.0)) - float(prev.get("csm_closing", 0.0))
        rows.extend(
            self._build_bridge_rows(
                quarter=quarter,
                metric="capital_proxy",
                product=product,
                region=region,
                opening_value=float(prev.get("capital_proxy", 0.0)),
                closing_value=float(current.get("capital_proxy", 0.0)),
                steps=[
                    ("underwriting_effect", (float(current.get("premium_income", 0.0)) - float(current.get("total_claims", 0.0))) * self.settings.capital_underwriting_weight),
                    ("investment_income", float(current.get("asset_investment_income", 0.0)) * self.settings.capital_investment_weight),
                    ("reserve_strain", -reserve_movement * self.settings.capital_reserve_weight),
                    ("csm_support", csm_movement * self.settings.capital_csm_weight),
                ],
            )
        )
        return rows

    def _build_bridge_rows(
        self,
        *,
        quarter: str,
        metric: str,
        product: str,
        region: str,
        opening_value: float,
        closing_value: float,
        steps: list[tuple[str, float]],
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        accumulated = 0.0
        for step_order, (step_name, amount) in enumerate(steps, start=1):
            accumulated += amount
            rows.append(
                {
                    "quarter": quarter,
                    "metric": metric,
                    "product": product,
                    "region": region,
                    "opening_value": round(opening_value, 6),
                    "step_order": step_order,
                    "movement_step": step_name,
                    "movement_amount": round(amount, 6),
                    "projected_position": round(opening_value + accumulated, 6),
                    "closing_value": round(closing_value, 6),
                }
            )
        residual = closing_value - opening_value - accumulated
        rows.append(
            {
                "quarter": quarter,
                "metric": metric,
                "product": product,
                "region": region,
                "opening_value": round(opening_value, 6),
                "step_order": len(steps) + 1,
                "movement_step": "residual",
                "movement_amount": round(residual, 6),
                "projected_position": round(closing_value, 6),
                "closing_value": round(closing_value, 6),
            }
        )
        return rows

    def _build_bridge_summary(
        self,
        movement_analysis: pd.DataFrame,
        forecast_output_table: pd.DataFrame,
    ) -> pd.DataFrame:
        summary_rows: list[dict[str, object]] = []
        for (quarter, metric, product, region), segment in movement_analysis.groupby(["quarter", "metric", "product", "region"], sort=False):
            opening_value = float(segment["opening_value"].iloc[0])
            closing_value = float(segment["closing_value"].iloc[0])
            net_change = closing_value - opening_value
            top_steps = segment.loc[segment["movement_step"] != "residual"].copy()
            top_steps["abs_amount"] = top_steps["movement_amount"].abs()
            top_steps = top_steps.sort_values("abs_amount", ascending=False).head(int(self.settings.top_steps_per_metric))
            top_step_text = ", ".join(
                f"{row.movement_step} ({row.movement_amount:,.2f})" for row in top_steps.itertuples(index=False)
            )
            forecast_match = self._lookup_forecast(forecast_output_table, metric, product, region)
            summary_rows.append(
                {
                    "quarter": quarter,
                    "metric": metric,
                    "product": product,
                    "region": region,
                    "opening_value": round(opening_value, 6),
                    "closing_value": round(closing_value, 6),
                    "net_change": round(net_change, 6),
                    "net_change_pct": round(net_change / (abs(opening_value) + float(self.settings.epsilon)), 6),
                    "top_movement_steps": top_step_text,
                    "dominant_step": str(top_steps["movement_step"].iloc[0]) if not top_steps.empty else "residual",
                    "forecast_value": round(float(forecast_match["forecast_value"]), 6) if forecast_match is not None else None,
                    "forecast_quarter": str(forecast_match["forecast_quarter"]) if forecast_match is not None else "",
                    "forecast_horizon": int(forecast_match["forecast_horizon"]) if forecast_match is not None and forecast_match.get("forecast_horizon") is not None else 1 if forecast_match is not None else None,
                }
            )
        return pd.DataFrame(summary_rows).sort_values(["quarter", "metric", "product", "region"]).reset_index(drop=True)

    def _lookup_forecast(
        self,
        forecast_output_table: pd.DataFrame,
        metric: str,
        product: str,
        region: str,
    ) -> dict[str, object] | None:
        target_map = {
            "premium_income": "premium",
            "claims": "claims",
            "reserves": "reserve_movement",
            "csm_closing": "csm_movement",
            "capital_proxy": "capital_ratio",
        }
        target_name = target_map.get(metric)
        if target_name is None or forecast_output_table.empty:
            return None
        subset = forecast_output_table.loc[
            (forecast_output_table["target_name"] == target_name)
            & (forecast_output_table["product"] == product)
            & (forecast_output_table["region"] == region)
        ].copy()
        if subset.empty:
            return None
        sort_columns = [column for column in ["forecast_horizon", "forecast_quarter"] if column in subset.columns]
        if sort_columns:
            subset = subset.sort_values(sort_columns)
        first_row = subset.iloc[0]
        return {
            "forecast_value": float(first_row["forecast_value"]),
            "forecast_quarter": str(first_row["forecast_quarter"]) if "forecast_quarter" in subset.columns else "",
            "forecast_horizon": int(first_row["forecast_horizon"]) if "forecast_horizon" in subset.columns else 1,
        }

    def _build_llm_summary(self, movement_bridge_summary: pd.DataFrame) -> str:
        if movement_bridge_summary.empty:
            return "# Movement Analysis Summary\n\nNo movement analysis was available."
        latest_quarter = str(movement_bridge_summary["quarter"].max())
        latest = movement_bridge_summary.loc[movement_bridge_summary["quarter"] == latest_quarter].copy()
        ranked = latest.assign(abs_change=latest["net_change"].abs()).sort_values("abs_change", ascending=False).head(int(self.settings.summary_top_n))
        payload = {
            "latest_quarter": latest_quarter,
            "movement_rows": ranked[["metric", "product", "region", "opening_value", "closing_value", "net_change", "net_change_pct", "dominant_step", "top_movement_steps", "forecast_value", "forecast_quarter", "forecast_horizon"]].to_dict(orient="records"),
        }
        prompt = (
            "You are an insurance management reporting assistant.\n"
            "Use only the provided context and tool outputs to answer the question.\n\n"
            "Question:\nSummarize the latest movement analysis for management reporting.\n\n"
            "Tool outputs:\n"
            + json.dumps({"MovementAnalysisTool": payload}, indent=2)
            + "\n\nRetrieved context:\n"
            + "\n\nRequirements:\n- answer clearly and concisely\n- explain relevant drivers where applicable\n- reference source artifacts\n- do not invent facts"
        )
        answer = get_default_llm_client().generate(prompt).strip()
        if not answer:
            answer = "No LLM movement summary was generated."
        return "# Movement Analysis Summary\n\n" + answer
