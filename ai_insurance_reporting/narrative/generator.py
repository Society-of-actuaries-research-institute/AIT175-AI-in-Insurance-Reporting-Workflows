"""Narrative generation for management reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from ai_insurance_reporting.chatbot.llm_client import MockLLMClient, get_default_llm_client
from ai_insurance_reporting.config.loader import AppConfig, load_config
from ai_insurance_reporting.utils.artifacts import ensure_artifact_dirs


@dataclass(slots=True)
class NarrativeResult:
    """Narrative outputs with traceability metadata."""

    narrative_statements: pd.DataFrame
    report_path: Path | None = None
    llm_draft_path: Path | None = None

    def as_dict(self) -> dict[str, pd.DataFrame]:
        """Return persisted tabular narrative outputs."""

        return {"narrative_statements": self.narrative_statements}


class NarrativeGenerator:
    """Generate traceable management reporting commentary."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()

    def run(
        self,
        *,
        file_format: str = "csv",
    ) -> tuple[NarrativeResult, dict[str, Path]]:
        """Generate narratives from processed reporting outputs and persist them."""

        curated = self._load_processed_table("curated_reporting_dataset", file_format=file_format)
        validation_summary = self._load_processed_table("quarterly_validation_summary", file_format=file_format)
        forecasts = self._load_model_table("forecast_output_table", file_format=file_format)

        result = self.generate(
            curated_reporting_dataset=curated,
            quarterly_validation_summary=validation_summary,
            forecast_output_table=forecasts,
        )
        output_paths = self.write(result, file_format=file_format)
        return result, output_paths

    def generate(
        self,
        *,
        curated_reporting_dataset: pd.DataFrame,
        quarterly_validation_summary: pd.DataFrame,
        forecast_output_table: pd.DataFrame,
        movement_bridge_summary: pd.DataFrame | None = None,
    ) -> NarrativeResult:
        """Create sectioned narrative statements with source traceability."""

        latest_quarter = str(curated_reporting_dataset["quarter"].max())
        current = curated_reporting_dataset.loc[curated_reporting_dataset["quarter"] == latest_quarter].copy()
        previous_quarter = self._previous_quarter(latest_quarter)
        previous = curated_reporting_dataset.loc[curated_reporting_dataset["quarter"] == previous_quarter].copy()
        validation_row = quarterly_validation_summary.loc[
            quarterly_validation_summary["quarter"] == latest_quarter
        ]

        statement_rows: list[dict[str, object]] = []
        statement_rows.extend(self._build_quarterly_summary(current, previous, validation_row, latest_quarter))
        statement_rows.extend(self._build_claims_analysis(current, previous, latest_quarter))
        statement_rows.extend(self._build_reserve_csm_commentary(current, previous, latest_quarter))
        statement_rows.extend(self._build_capital_outlook(current, forecast_output_table, latest_quarter))
        if movement_bridge_summary is not None and not movement_bridge_summary.empty:
            statement_rows.extend(self._build_movement_analysis(movement_bridge_summary, latest_quarter))

        statements = pd.DataFrame(statement_rows).sort_values(["section_order", "statement_order"]).reset_index(drop=True)
        report_path = self._write_report(statements, latest_quarter)
        llm_draft_path = self._write_llm_draft(statements, latest_quarter)
        return NarrativeResult(narrative_statements=statements, report_path=report_path, llm_draft_path=llm_draft_path)

    def write(
        self,
        result: NarrativeResult,
        *,
        file_format: str = "csv",
    ) -> dict[str, Path]:
        """Persist narrative outputs."""

        artifact_paths = ensure_artifact_dirs(self.config)
        output_dir = artifact_paths.reports / "narrative"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths: dict[str, Path] = {}
        for name, frame in result.as_dict().items():
            destination = output_dir / f"{name}.{file_format}"
            frame.to_csv(destination, index=False)
            output_paths[name] = destination

        if result.report_path is not None:
            output_paths["management_report"] = result.report_path
        if result.llm_draft_path is not None:
            output_paths["management_report_llm_draft"] = result.llm_draft_path
        return output_paths

    def _build_quarterly_summary(
        self,
        current: pd.DataFrame,
        previous: pd.DataFrame,
        validation_row: pd.DataFrame,
        latest_quarter: str,
    ) -> list[dict[str, object]]:
        premium = float(current["premium_income"].sum())
        claims = float(current["total_claims"].sum())
        combined = float(current["combined_ratio"].mean())
        prev_premium = float(previous["premium_income"].sum()) if not previous.empty else premium
        premium_change_pct = self._pct_change(premium, prev_premium)

        validation_pass_rate = (
            float(validation_row["validation_pass_rate"].iloc[0]) if not validation_row.empty else 1.0
        )
        records_with_issues = int(validation_row["records_with_issues"].iloc[0]) if not validation_row.empty else 0

        return [
            self._statement(
                section="quarterly_summary",
                section_order=1,
                statement_order=1,
                latest_quarter=latest_quarter,
                statement=(
                    f"{latest_quarter} premium income was {self._fmt_currency(premium)}, "
                    f"{self._fmt_pct(premium_change_pct)} versus {self._label_previous(latest_quarter)}."
                ),
                source_dataset="curated_reporting_dataset",
                source_columns="quarter,premium_income",
                source_filters=f"quarter={latest_quarter};comparison_quarter={self._label_previous(latest_quarter)}",
                source_value=f"current={premium:.2f};previous={prev_premium:.2f}",
            ),
            self._statement(
                section="quarterly_summary",
                section_order=1,
                statement_order=2,
                latest_quarter=latest_quarter,
                statement=(
                    f"Total claims for {latest_quarter} were {self._fmt_currency(claims)} "
                    f"with an average combined ratio of {combined:.2f}."
                ),
                source_dataset="curated_reporting_dataset",
                source_columns="quarter,total_claims,combined_ratio",
                source_filters=f"quarter={latest_quarter}",
                source_value=f"claims={claims:.2f};combined_ratio={combined:.4f}",
            ),
            self._statement(
                section="quarterly_summary",
                section_order=1,
                statement_order=3,
                latest_quarter=latest_quarter,
                statement=(
                    f"Data quality checks for {latest_quarter} show a validation pass rate of "
                    f"{validation_pass_rate:.2%} with {records_with_issues} records flagged for review."
                ),
                source_dataset="quarterly_validation_summary",
                source_columns="quarter,validation_pass_rate,records_with_issues",
                source_filters=f"quarter={latest_quarter}",
                source_value=f"validation_pass_rate={validation_pass_rate:.4f};records_with_issues={records_with_issues}",
            ),
        ]

    def _build_claims_analysis(
        self,
        current: pd.DataFrame,
        previous: pd.DataFrame,
        latest_quarter: str,
    ) -> list[dict[str, object]]:
        claims_by_product = current.groupby("product", as_index=False)["total_claims"].sum()
        top_claim_product = claims_by_product.sort_values("total_claims", ascending=False).iloc[0]

        loss_ratio_by_region = current.groupby("region", as_index=False)["loss_ratio"].mean()
        top_loss_region = loss_ratio_by_region.sort_values("loss_ratio", ascending=False).iloc[0]

        current_claim_count = float(current["claim_count"].sum())
        previous_claim_count = float(previous["claim_count"].sum()) if not previous.empty else current_claim_count
        claim_count_change = self._pct_change(current_claim_count, previous_claim_count)

        return [
            self._statement(
                section="claims_analysis",
                section_order=2,
                statement_order=1,
                latest_quarter=latest_quarter,
                statement=(
                    f"{top_claim_product['product']} generated the highest claim volume in {latest_quarter} "
                    f"at {self._fmt_currency(float(top_claim_product['total_claims']))}."
                ),
                source_dataset="curated_reporting_dataset",
                source_columns="quarter,product,total_claims",
                source_filters=f"quarter={latest_quarter}",
                source_value=f"product={top_claim_product['product']};total_claims={float(top_claim_product['total_claims']):.2f}",
            ),
            self._statement(
                section="claims_analysis",
                section_order=2,
                statement_order=2,
                latest_quarter=latest_quarter,
                statement=(
                    f"{top_loss_region['region']} recorded the highest average loss ratio at "
                    f"{float(top_loss_region['loss_ratio']):.2f} in {latest_quarter}."
                ),
                source_dataset="curated_reporting_dataset",
                source_columns="quarter,region,loss_ratio",
                source_filters=f"quarter={latest_quarter}",
                source_value=f"region={top_loss_region['region']};loss_ratio={float(top_loss_region['loss_ratio']):.4f}",
            ),
            self._statement(
                section="claims_analysis",
                section_order=2,
                statement_order=3,
                latest_quarter=latest_quarter,
                statement=(
                    f"Reported claim count changed by {self._fmt_pct(claim_count_change)} "
                    f"from {self._label_previous(latest_quarter)} to {latest_quarter}."
                ),
                source_dataset="curated_reporting_dataset",
                source_columns="quarter,claim_count",
                source_filters=f"quarter={latest_quarter};comparison_quarter={self._label_previous(latest_quarter)}",
                source_value=f"current={current_claim_count:.0f};previous={previous_claim_count:.0f}",
            ),
        ]

    def _build_reserve_csm_commentary(
        self,
        current: pd.DataFrame,
        previous: pd.DataFrame,
        latest_quarter: str,
    ) -> list[dict[str, object]]:
        reserves = float(current["reserves"].sum())
        prev_reserves = float(previous["reserves"].sum()) if not previous.empty else reserves
        reserve_change = reserves - prev_reserves

        csm_closing = float(current["csm_closing"].sum())
        csm_movement = float(current["csm_movement"].sum())
        reserve_recon = float(current["reserve_reconciliation_diff"].abs().sum())
        csm_recon = float(current["csm_reconciliation_diff"].abs().sum())

        return [
            self._statement(
                section="reserve_and_csm_movements",
                section_order=3,
                statement_order=1,
                latest_quarter=latest_quarter,
                statement=(
                    f"Ending reserves for {latest_quarter} were {self._fmt_currency(reserves)}, "
                    f"a movement of {self._fmt_currency(reserve_change)} versus {self._label_previous(latest_quarter)}."
                ),
                source_dataset="curated_reporting_dataset",
                source_columns="quarter,reserves",
                source_filters=f"quarter={latest_quarter};comparison_quarter={self._label_previous(latest_quarter)}",
                source_value=f"current={reserves:.2f};previous={prev_reserves:.2f};movement={reserve_change:.2f}",
            ),
            self._statement(
                section="reserve_and_csm_movements",
                section_order=3,
                statement_order=2,
                latest_quarter=latest_quarter,
                statement=(
                    f"Closing CSM totaled {self._fmt_currency(csm_closing)} with net quarterly CSM movement of "
                    f"{self._fmt_currency(csm_movement)}."
                ),
                source_dataset="curated_reporting_dataset",
                source_columns="quarter,csm_closing,csm_movement",
                source_filters=f"quarter={latest_quarter}",
                source_value=f"csm_closing={csm_closing:.2f};csm_movement={csm_movement:.2f}",
            ),
            self._statement(
                section="reserve_and_csm_movements",
                section_order=3,
                statement_order=3,
                latest_quarter=latest_quarter,
                statement=(
                    f"Reserve and CSM reconciliations remained stable in {latest_quarter}, "
                    f"with aggregate absolute differences of {self._fmt_currency(reserve_recon)} and "
                    f"{self._fmt_currency(csm_recon)} respectively."
                ),
                source_dataset="curated_reporting_dataset",
                source_columns="quarter,reserve_reconciliation_diff,csm_reconciliation_diff",
                source_filters=f"quarter={latest_quarter}",
                source_value=f"reserve_abs_diff={reserve_recon:.2f};csm_abs_diff={csm_recon:.2f}",
            ),
        ]

    def _build_capital_outlook(
        self,
        current: pd.DataFrame,
        forecast_output_table: pd.DataFrame,
        latest_quarter: str,
    ) -> list[dict[str, object]]:
        capital_ratio = (
            float(current["capital_proxy"].sum()) / float(current["liability_balance"].sum())
            if float(current["liability_balance"].sum()) > 0
            else 0.0
        )
        capital_forecast = forecast_output_table.loc[
            forecast_output_table["target_name"] == "capital_ratio"
        ].copy()
        if not capital_forecast.empty:
            sort_columns = [column for column in ["forecast_horizon", "forecast_quarter"] if column in capital_forecast.columns]
            if sort_columns:
                capital_forecast = capital_forecast.sort_values(sort_columns)
            next_quarter = str(capital_forecast["forecast_quarter"].iloc[0])
            next_capital_ratio = float(
                capital_forecast.loc[capital_forecast["forecast_quarter"] == next_quarter, "forecast_value"].mean()
            )
        else:
            next_quarter = self._next_quarter(latest_quarter)
            next_capital_ratio = capital_ratio

        premium_forecast = forecast_output_table.loc[
            (forecast_output_table["target_name"] == "premium")
            & (forecast_output_table["forecast_quarter"] == next_quarter),
            "forecast_value",
        ]
        claims_forecast = forecast_output_table.loc[
            (forecast_output_table["target_name"] == "claims")
            & (forecast_output_table["forecast_quarter"] == next_quarter),
            "forecast_value",
        ]
        forecast_margin = float(premium_forecast.sum() - claims_forecast.sum()) if not premium_forecast.empty and not claims_forecast.empty else 0.0

        return [
            self._statement(
                section="capital_outlook",
                section_order=4,
                statement_order=1,
                latest_quarter=latest_quarter,
                statement=(
                    f"The implied synthetic capital-to-liability proxy ratio for {latest_quarter} was {capital_ratio:.2%}, "
                    f"with the current balance sheet supported by {self._fmt_currency(float(current['capital_proxy'].sum()))} of capital proxy."
                ),
                source_dataset="curated_reporting_dataset",
                source_columns="quarter,capital_proxy,liability_balance",
                source_filters=f"quarter={latest_quarter}",
                source_value=f"capital_ratio={capital_ratio:.6f};capital_proxy={float(current['capital_proxy'].sum()):.2f}",
            ),
            self._statement(
                section="capital_outlook",
                section_order=4,
                statement_order=2,
                latest_quarter=latest_quarter,
                statement=(
                    f"Modelled outlook for {next_quarter} indicates an average synthetic capital-to-liability proxy ratio of "
                    f"{next_capital_ratio:.2%} across product-region segments."
                ),
                source_dataset="forecast_output_table",
                source_columns="forecast_quarter,target_name,forecast_value",
                source_filters=f"target_name=capital_ratio;forecast_quarter={next_quarter}",
                source_value=f"forecast_capital_ratio_mean={next_capital_ratio:.6f}",
            ),
            self._statement(
                section="capital_outlook",
                section_order=4,
                statement_order=3,
                latest_quarter=latest_quarter,
                statement=(
                    f"Forward-looking premium less claims forecasts imply a margin of "
                    f"{self._fmt_currency(forecast_margin)} for {next_quarter}."
                ),
                source_dataset="forecast_output_table",
                source_columns="forecast_quarter,target_name,forecast_value",
                source_filters=f"forecast_quarter={next_quarter};target_name in (premium,claims)",
                source_value=f"forecast_margin={forecast_margin:.2f}",
            ),
        ]

    def _build_movement_analysis(
        self,
        movement_bridge_summary: pd.DataFrame,
        latest_quarter: str,
    ) -> list[dict[str, object]]:
        latest = movement_bridge_summary.loc[movement_bridge_summary["quarter"] == latest_quarter].copy()
        if latest.empty:
            return []
        ranked = latest.assign(abs_change=latest["net_change"].abs()).sort_values("abs_change", ascending=False).head(3)
        rows: list[dict[str, object]] = []
        for idx, row in enumerate(ranked.itertuples(index=False), start=1):
            rows.append(
                self._statement(
                    section="movement_analysis",
                    section_order=5,
                    statement_order=idx,
                    latest_quarter=latest_quarter,
                    statement=(
                        f"Movement analysis shows {row.metric.replace('_', ' ')} for {row.product} in {row.region} moved "
                        f"from {self._fmt_currency(float(row.opening_value))} to {self._fmt_currency(float(row.closing_value))}, "
                        f"with {row.dominant_step.replace('_', ' ')} as the dominant driver."
                    ),
                    source_dataset="movement_bridge_summary",
                    source_columns="quarter,metric,product,region,opening_value,closing_value,dominant_step,top_movement_steps",
                    source_filters=f"quarter={latest_quarter};metric={row.metric};product={row.product};region={row.region}",
                    source_value=f"opening={float(row.opening_value):.2f};closing={float(row.closing_value):.2f};dominant_step={row.dominant_step};top_steps={row.top_movement_steps}",
                )
            )
        return rows

    def _statement(
        self,
        *,
        section: str,
        section_order: int,
        statement_order: int,
        latest_quarter: str,
        statement: str,
        source_dataset: str,
        source_columns: str,
        source_filters: str,
        source_value: str,
    ) -> dict[str, object]:
        statement_id = f"{section[:3].upper()}-{section_order:02d}-{statement_order:02d}"
        return {
            "statement_id": statement_id,
            "reporting_quarter": latest_quarter,
            "section": section,
            "section_order": section_order,
            "statement_order": statement_order,
            "statement_text": statement,
            "source_dataset": source_dataset,
            "source_columns": source_columns,
            "source_filters": source_filters,
            "source_value": source_value,
        }

    def _write_report(self, statements: pd.DataFrame, latest_quarter: str) -> Path:
        artifact_paths = ensure_artifact_dirs(self.config)
        output_dir = artifact_paths.reports / "narrative"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"management_report_{latest_quarter}.md"

        lines = [f"# Management Reporting Commentary: {latest_quarter}", ""]
        for section_name in [
            "quarterly_summary",
            "claims_analysis",
            "reserve_and_csm_movements",
            "capital_outlook",
            "movement_analysis",
        ]:
            title = section_name.replace("_", " ").title()
            lines.append(f"## {title}")
            lines.append("")
            subset = statements.loc[statements["section"] == section_name]
            for row in subset.itertuples(index=False):
                lines.append(f"- {row.statement_text}")
                lines.append(
                    f"  Traceability: dataset={row.source_dataset}; columns={row.source_columns}; "
                    f"filters={row.source_filters}; values={row.source_value}"
                )
            lines.append("")

        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    def _write_llm_draft(self, statements: pd.DataFrame, latest_quarter: str) -> Path | None:
        artifact_paths = ensure_artifact_dirs(self.config)
        output_dir = artifact_paths.reports / "narrative"
        output_dir.mkdir(parents=True, exist_ok=True)
        draft_path = output_dir / f"management_report_llm_{latest_quarter}.md"

        ordered = statements.sort_values(["section_order", "statement_order"])
        prompt_lines = [
            f"Create a concise management reporting draft for {latest_quarter}.",
            "Use only the factual statements below and do not invent new figures.",
            "",
        ]
        for row in ordered.itertuples(index=False):
            prompt_lines.append(f"[{row.section}] {row.statement_text}")

        llm_client = get_default_llm_client()
        if isinstance(llm_client, MockLLMClient):
            lines = [f"# LLM-Assisted Management Reporting Draft: {latest_quarter}", ""]
            current_section = None
            for row in ordered.itertuples(index=False):
                if row.section != current_section:
                    current_section = row.section
                    lines.append(f"## {str(row.section).replace('_', ' ').title()}")
                    lines.append("")
                lines.append(f"- {row.statement_text}")
            draft_text = "\n".join(lines)
        else:
            generated = llm_client.generate("\n".join(prompt_lines)).strip()
            draft_text = generated or f"# LLM-Assisted Management Reporting Draft: {latest_quarter}\n\nNo draft content was generated."

        draft_path.write_text(draft_text, encoding="utf-8")
        return draft_path

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

    def _fmt_currency(self, value: float) -> str:
        return f"${value:,.2f}"

    def _fmt_pct(self, value: float) -> str:
        return f"{value:.2%}"

    def _pct_change(self, current: float, previous: float) -> float:
        if abs(previous) < 1e-9:
            return 0.0
        return (current - previous) / previous

    def _previous_quarter(self, quarter_label: str) -> str:
        year = int(quarter_label[:4])
        quarter = int(quarter_label[-1])
        if quarter == 1:
            return f"{year - 1}Q4"
        return f"{year}Q{quarter - 1}"

    def _next_quarter(self, quarter_label: str) -> str:
        year = int(quarter_label[:4])
        quarter = int(quarter_label[-1])
        if quarter == 4:
            return f"{year + 1}Q1"
        return f"{year}Q{quarter + 1}"

    def _label_previous(self, quarter_label: str) -> str:
        return self._previous_quarter(quarter_label)
