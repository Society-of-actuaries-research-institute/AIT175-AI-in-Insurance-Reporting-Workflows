"""Unified analyst review queue for AI-assisted reporting outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ai_insurance_reporting.config.loader import AppConfig, load_config
from ai_insurance_reporting.utils.artifacts import ensure_artifact_dirs


@dataclass(slots=True)
class AnalystReviewQueueResult:
    """Structured review queue outputs for analyst triage."""

    analyst_review_queue: pd.DataFrame
    analyst_review_summary: pd.DataFrame
    analyst_review_overview: dict[str, Any]

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "analyst_review_queue": self.analyst_review_queue,
            "analyst_review_summary": self.analyst_review_summary,
        }


class AnalystReviewQueueBuilder:
    """Consolidate AI-generated review items into a single analyst queue.

    This builder is the handoff point between automated reporting support and
    human review. It gathers the highest-signal findings from the insight,
    anomaly, narrative-quality, and LLM-evaluation layers, assigns a comparable
    priority score, and writes a queue that analysts can triage in the UI.
    """

    INSIGHT_SEVERITY_SCORES = {
        "normal": 0.25,
        "moderate": 0.5,
        "material": 0.75,
        "critical": 1.0,
    }

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.settings = self.config.reporting.analyst_review

    def run(
        self,
        *,
        insight_summary: pd.DataFrame,
        anomaly_investigation: pd.DataFrame,
        narrative_quality_check: pd.DataFrame,
        llm_eval_results: pd.DataFrame,
        file_format: str = "csv",
    ) -> tuple[AnalystReviewQueueResult, dict[str, Path]]:
        result = self.generate(
            insight_summary=insight_summary,
            anomaly_investigation=anomaly_investigation,
            narrative_quality_check=narrative_quality_check,
            llm_eval_results=llm_eval_results,
        )
        return result, self.write(result, file_format=file_format)

    def generate(
        self,
        *,
        insight_summary: pd.DataFrame,
        anomaly_investigation: pd.DataFrame,
        narrative_quality_check: pd.DataFrame,
        llm_eval_results: pd.DataFrame,
    ) -> AnalystReviewQueueResult:
        rows: list[dict[str, Any]] = []
        rows.extend(self._build_insight_rows(insight_summary))
        rows.extend(self._build_anomaly_rows(anomaly_investigation))
        rows.extend(self._build_narrative_rows(narrative_quality_check))
        rows.extend(self._build_llm_rows(llm_eval_results))

        queue = pd.DataFrame(rows)
        if queue.empty:
            summary = pd.DataFrame(
                columns=[
                    "review_source",
                    "item_count",
                    "critical_count",
                    "high_count",
                    "medium_count",
                    "monitor_count",
                    "average_priority",
                ]
            )
            overview = {
                "total_items": 0,
                "pending_review": 0,
                "critical_items": 0,
                "high_priority_items": 0,
                "sources_with_items": 0,
            }
            return AnalystReviewQueueResult(queue, summary, overview)

        queue = queue.sort_values(
            ["priority_score", "review_source", "quarter"],
            ascending=[False, True, False],
            na_position="last",
        ).reset_index(drop=True)
        queue["queue_rank"] = range(1, len(queue) + 1)
        queue = queue[
            [
                "queue_rank",
                "review_item_id",
                "review_source",
                "review_type",
                "quarter",
                "metric",
                "product",
                "region",
                "priority_label",
                "priority_score",
                "issue_summary",
                "recommended_action",
                "reviewer_status",
                "source_artifact",
                "source_record_id",
                "support_reference",
            ]
        ]

        summary = (
            queue.groupby("review_source", as_index=False)
            .agg(
                item_count=("review_item_id", "count"),
                critical_count=("priority_label", lambda series: int((series == "critical").sum())),
                high_count=("priority_label", lambda series: int((series == "high").sum())),
                medium_count=("priority_label", lambda series: int((series == "medium").sum())),
                monitor_count=("priority_label", lambda series: int((series == "monitor").sum())),
                average_priority=("priority_score", "mean"),
            )
            .sort_values(["item_count", "average_priority"], ascending=[False, False])
            .reset_index(drop=True)
        )
        summary["average_priority"] = summary["average_priority"].round(4)

        overview = {
            "total_items": int(len(queue)),
            "pending_review": int((queue["reviewer_status"] == "pending_review").sum()),
            "critical_items": int((queue["priority_label"] == "critical").sum()),
            "high_priority_items": int(queue["priority_label"].isin(["critical", "high"]).sum()),
            "sources_with_items": int(queue["review_source"].nunique()),
        }
        return AnalystReviewQueueResult(queue, summary, overview)

    def write(
        self,
        result: AnalystReviewQueueResult,
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
        overview_path = output_dir / "analyst_review_overview.json"
        overview_path.write_text(json.dumps(result.analyst_review_overview, indent=2, ensure_ascii=True), encoding="utf-8")
        output_paths["analyst_review_overview"] = overview_path
        return output_paths

    def _build_insight_rows(self, insight_summary: pd.DataFrame) -> list[dict[str, Any]]:
        if insight_summary.empty:
            return []
        frame = insight_summary.copy()
        if not bool(self.settings.include_normal_insights):
            frame = frame.loc[frame["severity_classification"] != "normal"]
        frame = self._limit_per_source(
            frame.sort_values(["z_score", "percentage_change"], ascending=[False, False]),
        )
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(frame.itertuples(index=False), start=1):
            score = self.INSIGHT_SEVERITY_SCORES.get(str(row.severity_classification), 0.25)
            rows.append(
                {
                    "review_item_id": f"INS-{index:03d}",
                    "review_source": "insight_detection",
                    "review_type": "projected_movement",
                    "quarter": str(getattr(row, "quarter", "")),
                    "metric": str(getattr(row, "metric", "")),
                    "product": str(getattr(row, "product", "")),
                    "region": str(getattr(row, "region", "")),
                    "priority_label": self._priority_label(score),
                    "priority_score": round(score, 4),
                    "issue_summary": str(getattr(row, "insight_title", "Material projected movement identified")),
                    "recommended_action": self._insight_action(str(getattr(row, "metric", ""))),
                    "reviewer_status": "pending_review",
                    "source_artifact": "insight_summary",
                    "source_record_id": f"{getattr(row, 'metric', '')}|{getattr(row, 'product', '')}|{getattr(row, 'region', '')}|{getattr(row, 'forecast_horizon', 1)}",
                    "support_reference": str(getattr(row, "short_explanation_seed", "")),
                }
            )
        return rows

    def _build_anomaly_rows(self, anomaly_investigation: pd.DataFrame) -> list[dict[str, Any]]:
        if anomaly_investigation.empty:
            return []
        frame = anomaly_investigation.loc[
            anomaly_investigation["support_score"] >= float(self.settings.min_anomaly_support_score)
        ].copy()
        frame = self._limit_per_source(frame.sort_values("support_score", ascending=False))
        rows: list[dict[str, Any]] = []
        for row in frame.itertuples(index=False):
            score = min(max(float(getattr(row, "support_score", 0.0)), 0.0), 1.0)
            rows.append(
                {
                    "review_item_id": str(getattr(row, "anomaly_id", "anomaly")),
                    "review_source": "anomaly_investigation",
                    "review_type": str(getattr(row, "anomaly_type", "validation_exception")),
                    "quarter": str(getattr(row, "quarter", "")),
                    "metric": str(getattr(row, "anomaly_type", "")),
                    "product": str(getattr(row, "product", "")),
                    "region": str(getattr(row, "region", "")),
                    "priority_label": self._priority_label(score),
                    "priority_score": round(score, 4),
                    "issue_summary": str(getattr(row, "explanation_text", "Validation exception requires review")),
                    "recommended_action": "Investigate the flagged validation exception and reconcile supporting balances.",
                    "reviewer_status": str(getattr(row, "reviewer_status", "pending_review")),
                    "source_artifact": "anomaly_investigation",
                    "source_record_id": str(getattr(row, "anomaly_id", "")),
                    "support_reference": str(getattr(row, "likely_drivers", "")),
                }
            )
        return rows

    def _build_narrative_rows(self, narrative_quality_check: pd.DataFrame) -> list[dict[str, Any]]:
        if narrative_quality_check.empty:
            return []
        frame = narrative_quality_check.loc[narrative_quality_check["warning_flag"] == True].copy()  # noqa: E712
        frame = self._limit_per_source(frame.sort_values("statement_id"))
        rows: list[dict[str, Any]] = []
        for row in frame.itertuples(index=False):
            score = float(self.settings.narrative_warning_priority)
            rows.append(
                {
                    "review_item_id": str(getattr(row, "statement_id", "statement")),
                    "review_source": "narrative_quality",
                    "review_type": "narrative_warning",
                    "quarter": "",
                    "metric": str(getattr(row, "linked_metric", "")),
                    "product": "",
                    "region": "",
                    "priority_label": self._priority_label(score),
                    "priority_score": round(score, 4),
                    "issue_summary": str(getattr(row, "consistency_result", "Narrative statement needs review")),
                    "recommended_action": "Review the draft wording against the supporting data and revise the commentary if needed.",
                    "reviewer_status": "pending_review",
                    "source_artifact": "narrative_quality_check",
                    "source_record_id": str(getattr(row, "statement_id", "")),
                    "support_reference": str(getattr(row, "suggestion_for_revised_wording", "")),
                }
            )
        return rows

    def _build_llm_rows(self, llm_eval_results: pd.DataFrame) -> list[dict[str, Any]]:
        if llm_eval_results.empty:
            return []
        frame = llm_eval_results.loc[llm_eval_results["evaluation_label"] == "review"].copy()
        frame["score_gap"] = 1.0 - frame["overall_score"].astype(float)
        frame = frame.loc[frame["score_gap"] >= float(self.settings.min_llm_review_gap)]
        frame = self._limit_per_source(frame.sort_values("overall_score"))
        rows: list[dict[str, Any]] = []
        for row in frame.itertuples(index=False):
            score = min(max(float(getattr(row, "score_gap", 0.0)), 0.0), 1.0)
            rows.append(
                {
                    "review_item_id": str(getattr(row, "query_id", "llm-review")),
                    "review_source": "llm_evaluation",
                    "review_type": str(getattr(row, "category", "benchmark_question")),
                    "quarter": "",
                    "metric": "",
                    "product": "",
                    "region": "",
                    "priority_label": self._priority_label(score),
                    "priority_score": round(score, 4),
                    "issue_summary": f"Benchmark question requires review: {getattr(row, 'question', '')}",
                    "recommended_action": self._llm_action(row),
                    "reviewer_status": "pending_review",
                    "source_artifact": "llm_eval_results",
                    "source_record_id": str(getattr(row, "query_id", "")),
                    "support_reference": f"overall_score={float(getattr(row, 'overall_score', 0.0)):.2f}; expected_artifacts={getattr(row, 'expected_artifacts', '')}",
                }
            )
        return rows

    def _limit_per_source(self, frame: pd.DataFrame) -> pd.DataFrame:
        limit = int(self.settings.max_items_per_source)
        return frame.head(limit).copy() if limit > 0 else frame.copy()

    def _priority_label(self, score: float) -> str:
        if score >= float(self.settings.critical_priority_score):
            return "critical"
        if score >= float(self.settings.high_priority_score):
            return "high"
        if score >= float(self.settings.medium_priority_score):
            return "medium"
        return "monitor"

    def _insight_action(self, metric: str) -> str:
        metric_lower = metric.lower()
        if "claim" in metric_lower:
            return "Review the forecast drivers and segment sensitivity for claims movement."
        if "reserve" in metric_lower or "csm" in metric_lower:
            return "Inspect the movement bridge and supporting narrative for balance sheet changes."
        if "capital" in metric_lower:
            return "Review capital outlook drivers and compare with scenario sensitivity outputs."
        return "Review the supporting forecast and explanation artifacts for this projected movement."

    def _llm_action(self, row: Any) -> str:
        grounded_score = float(getattr(row, "grounded_score", 0.0))
        artifact_score = float(getattr(row, "artifact_match_score", 0.0))
        tool_score = float(getattr(row, "tool_match_score", 0.0))
        citation_score = float(getattr(row, "citation_score", 0.0))
        used_fallback = bool(getattr(row, "used_fallback", False))
        if used_fallback and grounded_score == 0.0:
            return "Review retrieval coverage and evidence availability for this benchmark question."
        if grounded_score < 1.0:
            return "Tighten retrieval thresholds or improve grounding for this answer path."
        if artifact_score < 0.5:
            return "Improve artifact indexing or evidence mapping for this reporting question."
        if tool_score < 0.5:
            return "Refine planner routing so the appropriate structured tools are used."
        if citation_score < 1.0:
            return "Improve answer formatting so supporting citations are preserved consistently."
        return "Review the management-answer prompt and synthesis logic for this benchmark item."

def load_review_status_updates(config: AppConfig | None = None) -> pd.DataFrame:
    """Load persisted analyst decisions for previously reviewed queue items."""
    cfg = config or load_config()
    artifact_paths = ensure_artifact_dirs(cfg)
    status_path = artifact_paths.reports / "reporting" / "analyst_review_status_log.jsonl"
    if not status_path.exists():
        return pd.DataFrame(columns=["timestamp", "review_item_id", "reviewer_status", "comment", "review_owner"])
    rows = [json.loads(line) for line in status_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["timestamp", "review_item_id", "reviewer_status", "comment", "review_owner"])


def apply_review_status_updates(queue: pd.DataFrame, updates: pd.DataFrame | None) -> pd.DataFrame:
    """Overlay the latest saved analyst status onto the freshly generated queue.

    The queue itself is rebuilt on each workflow run. Status decisions are kept
    separately so analysts do not lose their review trail when the underlying
    artifacts are regenerated.
    """
    if queue.empty or updates is None or updates.empty:
        return queue.copy()
    latest = updates.sort_values("timestamp").drop_duplicates(subset=["review_item_id"], keep="last")
    merged = queue.merge(
        latest[["review_item_id", "reviewer_status", "comment", "review_owner", "timestamp"]],
        on="review_item_id",
        how="left",
        suffixes=("", "_override"),
    )
    merged["reviewer_status"] = merged["reviewer_status_override"].fillna(merged["reviewer_status"])
    merged["review_comment"] = merged["comment"].fillna("")
    merged["review_owner"] = merged["review_owner"].fillna("")
    merged["status_updated_at"] = merged["timestamp"].fillna("")
    return merged.drop(columns=["reviewer_status_override", "comment", "timestamp"])


def record_review_status(
    *,
    config: AppConfig | None = None,
    review_item_id: str,
    reviewer_status: str,
    comment: str = "",
    review_owner: str = "",
) -> Path:
    """Append one analyst decision to the persistent review-status log."""
    cfg = config or load_config()
    artifact_paths = ensure_artifact_dirs(cfg)
    output_dir = artifact_paths.reports / "reporting"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "analyst_review_status_log.jsonl"
    record = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "review_item_id": review_item_id,
        "reviewer_status": reviewer_status,
        "comment": comment,
        "review_owner": review_owner,
    }
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return output_path

