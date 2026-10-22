"""Full report assembly for the management reporting workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from ai_insurance_reporting.chatbot.llm_client import MockLLMClient, get_default_llm_client
from ai_insurance_reporting.config.loader import AppConfig, load_config
from ai_insurance_reporting.utils.artifacts import ensure_artifact_dirs


@dataclass(slots=True)
class FullReportResult:
    """Structured full-report outputs."""

    report_sections: pd.DataFrame
    markdown_path: Path | None = None
    html_path: Path | None = None
    manifest_path: Path | None = None
    llm_draft_path: Path | None = None

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {"management_report_sections": self.report_sections}


class FullReportBuilder:
    """Assemble a report package from existing workflow artifacts."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.settings = self.config.reporting.full_report

    def run(
        self,
        *,
        latest_quarter: str,
        quarterly_validation_summary: pd.DataFrame,
        insight_summary: pd.DataFrame,
        anomaly_investigation: pd.DataFrame,
        movement_bridge_summary: pd.DataFrame,
        narrative_statements: pd.DataFrame,
        narrative_quality_check: pd.DataFrame,
        figure_metadata: pd.DataFrame,
        file_format: str = "csv",
    ) -> tuple[FullReportResult, dict[str, Path]]:
        result = self.generate(
            latest_quarter=latest_quarter,
            quarterly_validation_summary=quarterly_validation_summary,
            insight_summary=insight_summary,
            anomaly_investigation=anomaly_investigation,
            movement_bridge_summary=movement_bridge_summary,
            narrative_statements=narrative_statements,
            narrative_quality_check=narrative_quality_check,
            figure_metadata=figure_metadata,
        )
        return result, self.write(result, latest_quarter=latest_quarter, file_format=file_format)

    def generate(
        self,
        *,
        latest_quarter: str,
        quarterly_validation_summary: pd.DataFrame,
        insight_summary: pd.DataFrame,
        anomaly_investigation: pd.DataFrame,
        movement_bridge_summary: pd.DataFrame,
        narrative_statements: pd.DataFrame,
        narrative_quality_check: pd.DataFrame,
        figure_metadata: pd.DataFrame,
    ) -> FullReportResult:
        sections: list[dict[str, object]] = []
        latest_validation = quarterly_validation_summary.loc[quarterly_validation_summary["quarter"] == latest_quarter]
        top_insights = self._top_insights(insight_summary, latest_quarter)
        top_anomalies = anomaly_investigation.sort_values("support_score", ascending=False).head(int(self.settings.max_anomalies)) if not anomaly_investigation.empty else pd.DataFrame()
        top_movements = movement_bridge_summary.loc[movement_bridge_summary["quarter"] == latest_quarter].assign(abs_change=lambda frame: frame["net_change"].abs()).sort_values("abs_change", ascending=False).head(int(self.settings.max_movement_rows)) if not movement_bridge_summary.empty else pd.DataFrame()
        warnings = narrative_quality_check.loc[narrative_quality_check["warning_flag"] == True].head(int(self.settings.max_quality_warnings)) if not narrative_quality_check.empty else pd.DataFrame()  # noqa: E712

        sections.append(self._section_row(1, "Executive Summary", self._build_executive_summary(latest_quarter, latest_validation, top_insights)))
        sections.append(self._section_row(2, "Material Insights", self._build_insight_section(top_insights)))
        sections.append(self._section_row(3, "Anomaly Investigation", self._build_anomaly_section(top_anomalies)))
        sections.append(self._section_row(4, "Movement Analysis", self._build_movement_section(top_movements)))
        sections.append(self._section_row(5, "Draft Commentary", self._build_narrative_section(narrative_statements)))
        sections.append(self._section_row(6, "Narrative Quality Review", self._build_quality_section(warnings)))
        sections.append(self._section_row(7, "Figures", self._build_figure_section(figure_metadata)))

        return FullReportResult(report_sections=pd.DataFrame(sections))

    def write(
        self,
        result: FullReportResult,
        *,
        latest_quarter: str,
        file_format: str = "csv",
    ) -> dict[str, Path]:
        artifact_paths = ensure_artifact_dirs(self.config)
        output_dir = artifact_paths.reports / "final"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths: dict[str, Path] = {}
        for name, frame in result.as_dict().items():
            destination = output_dir / f"{name}.{file_format}"
            frame.to_csv(destination, index=False)
            output_paths[name] = destination

        markdown_path = output_dir / f"management_report_full_{latest_quarter}.md"
        markdown_path.write_text(self._to_markdown(result.report_sections, latest_quarter), encoding="utf-8")
        result.markdown_path = markdown_path
        output_paths["management_report_markdown"] = markdown_path

        html_path = output_dir / f"management_report_full_{latest_quarter}.html"
        html_path.write_text(self._to_html(result.report_sections, latest_quarter), encoding="utf-8")
        result.html_path = html_path
        output_paths["management_report_html"] = html_path

        llm_draft_path = self._write_llm_draft(result.report_sections, latest_quarter)
        result.llm_draft_path = llm_draft_path
        if llm_draft_path is not None:
            output_paths["management_report_llm_markdown"] = llm_draft_path

        manifest = {
            "latest_quarter": latest_quarter,
            "section_titles": result.report_sections["section_title"].tolist(),
            "markdown_path": str(markdown_path),
            "html_path": str(html_path),
        }
        manifest_path = output_dir / "management_report_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
        result.manifest_path = manifest_path
        output_paths["management_report_manifest"] = manifest_path
        return output_paths

    def _section_row(self, section_order: int, section_title: str, section_text: str) -> dict[str, object]:
        return {
            "section_order": section_order,
            "section_title": section_title,
            "section_text": section_text,
        }

    def _build_executive_summary(self, latest_quarter: str, latest_validation: pd.DataFrame, top_insights: pd.DataFrame) -> str:
        pass_rate = float(latest_validation["validation_pass_rate"].iloc[0]) if not latest_validation.empty else 1.0
        issues = int(latest_validation["records_with_issues"].iloc[0]) if not latest_validation.empty else 0
        insight_count = int(len(top_insights))
        return (
            f"The {latest_quarter} reporting package combines validation, forecast, explainability, movement analysis, and draft commentary outputs. "
            f"Validation pass rate was {pass_rate:.2%} with {issues} records flagged for review, and {insight_count} material insight items were highlighted for management attention."
        )

    def _build_insight_section(self, top_insights: pd.DataFrame) -> str:
        if top_insights.empty:
            return "No material insight items were identified in the latest cycle."
        lines = []
        for row in top_insights.itertuples(index=False):
            lines.append(
                f"- {row.insight_title}: {row.metric} for {row.product} / {row.region} changes by {float(row.percentage_change):.2%} with z-score {float(row.z_score):.2f}."
            )
        return "\n".join(lines)

    def _build_anomaly_section(self, top_anomalies: pd.DataFrame) -> str:
        if top_anomalies.empty:
            return "No anomaly investigation items were available."
        lines = []
        for row in top_anomalies.itertuples(index=False):
            lines.append(
                f"- {row.anomaly_type} in {row.product} / {row.region} ({row.quarter}): {row.explanation_text} Support score {float(row.support_score):.2f}."
            )
        return "\n".join(lines)

    def _build_movement_section(self, top_movements: pd.DataFrame) -> str:
        if top_movements.empty:
            return "No movement bridge rows were available."
        lines = []
        for row in top_movements.itertuples(index=False):
            lines.append(
                f"- {row.metric} for {row.product} / {row.region} moved from {float(row.opening_value):,.2f} to {float(row.closing_value):,.2f}. Dominant step: {row.dominant_step}. Top steps: {row.top_movement_steps}."
            )
        return "\n".join(lines)

    def _build_narrative_section(self, narrative_statements: pd.DataFrame) -> str:
        if narrative_statements.empty:
            return "No narrative statements were available."
        subset = narrative_statements.head(int(self.settings.max_narrative_statements))
        return "\n".join(f"- {row.statement_text}" for row in subset.itertuples(index=False))

    def _build_quality_section(self, warnings: pd.DataFrame) -> str:
        if warnings.empty:
            return "No narrative quality warnings were raised."
        return "\n".join(
            f"- Statement {row.statement_id}: {row.consistency_result}. Suggested wording: {row.suggestion_for_revised_wording}" for row in warnings.itertuples(index=False)
        )

    def _build_figure_section(self, figure_metadata: pd.DataFrame) -> str:
        if figure_metadata.empty:
            return "No figures were available."
        subset = figure_metadata.head(int(self.settings.max_figures))
        return "\n".join(f"- {row.chart_title} ({row.file_path})" for row in subset.itertuples(index=False))

    def _top_insights(self, insight_summary: pd.DataFrame, latest_quarter: str) -> pd.DataFrame:
        if insight_summary.empty:
            return pd.DataFrame()
        ranking = {"critical": 4, "material": 3, "moderate": 2, "normal": 1}
        subset = insight_summary.loc[insight_summary["quarter"] == latest_quarter].copy()
        if subset.empty:
            subset = insight_summary.copy()
        subset["severity_order"] = subset["severity_classification"].map(ranking).fillna(0)
        return subset.sort_values(["severity_order", "z_score"], ascending=[False, False]).head(int(self.settings.max_insights))

    def _write_llm_draft(self, sections: pd.DataFrame, latest_quarter: str) -> Path | None:
        artifact_paths = ensure_artifact_dirs(self.config)
        output_dir = artifact_paths.reports / "final"
        output_dir.mkdir(parents=True, exist_ok=True)
        draft_path = output_dir / f"management_report_llm_{latest_quarter}.md"

        ordered = sections.sort_values("section_order")
        llm_client = get_default_llm_client()
        if isinstance(llm_client, MockLLMClient):
            lines = [f"# LLM-Assisted Full Management Report: {latest_quarter}", ""]
            for row in ordered.itertuples(index=False):
                lines.append(f"## {row.section_title}")
                lines.append("")
                lines.append(str(row.section_text))
                lines.append("")
            draft_text = "\n".join(lines)
        else:
            prompt_lines = [
                f"Draft a concise management report for {latest_quarter} using only the section content below.",
                "Do not invent figures or unsupported claims.",
                "",
            ]
            for row in ordered.itertuples(index=False):
                prompt_lines.append(f"[{row.section_title}] {row.section_text}")
            generated = llm_client.generate("\n".join(prompt_lines)).strip()
            draft_text = generated or f"# LLM-Assisted Full Management Report: {latest_quarter}\n\nNo draft content was generated."

        draft_path.write_text(draft_text, encoding="utf-8")
        return draft_path

    def _to_markdown(self, sections: pd.DataFrame, latest_quarter: str) -> str:
        lines = [f"# Full Management Report: {latest_quarter}", ""]
        for row in sections.sort_values("section_order").itertuples(index=False):
            lines.append(f"## {row.section_title}")
            lines.append("")
            lines.append(str(row.section_text))
            lines.append("")
        return "\n".join(lines)

    def _to_html(self, sections: pd.DataFrame, latest_quarter: str) -> str:
        parts = [
            "<html><head><meta charset='utf-8'><title>Management Report</title></head><body>",
            f"<h1>Full Management Report: {latest_quarter}</h1>",
        ]
        for row in sections.sort_values("section_order").itertuples(index=False):
            parts.append(f"<h2>{row.section_title}</h2>")
            for line in str(row.section_text).splitlines():
                if line.startswith("- "):
                    parts.append(f"<p>{line}</p>")
                else:
                    parts.append(f"<p>{line}</p>")
        parts.append("</body></html>")
        return "".join(parts)
