"""Validation engine for cleaned reporting data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ai_insurance_reporting.config.loader import AppConfig, load_config
from ai_insurance_reporting.utils.artifacts import ensure_artifact_dirs


@dataclass(slots=True)
class ValidationResult:
    """Validation outputs for cleaned reporting data."""

    validation_flags: pd.DataFrame
    quarterly_validation_summary: pd.DataFrame
    anomaly_table: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        """Return validation outputs keyed by file stem."""

        return {
            "validation_flags": self.validation_flags,
            "quarterly_validation_summary": self.quarterly_validation_summary,
            "anomaly_table": self.anomaly_table,
        }


class ReportingValidationEngine:
    """Apply rule-based validation to the cleaned reporting dataset."""

    REQUIRED_COLUMNS = [
        "quarter",
        "product",
        "region",
        "premium_income",
        "total_claims",
        "reserves",
        "liability_balance",
        "csm_opening",
        "csm_new_business",
        "csm_interest_accretion",
        "csm_release",
        "csm_closing",
        "capital_proxy",
    ]
    NON_NEGATIVE_COLUMNS = [
        "premium_income",
        "total_claims",
        "total_claims_paid",
        "reserves",
        "case_reserves",
        "liability_balance",
        "csm_opening",
        "csm_new_business",
        "csm_interest_accretion",
        "csm_release",
        "csm_closing",
        "capital_proxy",
        "average_claim_size",
        "max_claim_size",
    ]

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        reserve_tolerance: float | None = None,
        csm_tolerance: float | None = None,
        capital_tolerance: float | None = None,
    ) -> None:
        self.config = config or load_config()
        settings = self.config.validation
        self.reserve_tolerance = settings.reserve_tolerance_pct if reserve_tolerance is None else reserve_tolerance
        self.csm_tolerance = settings.csm_tolerance_pct if csm_tolerance is None else csm_tolerance
        self.capital_tolerance = settings.capital_tolerance_pct if capital_tolerance is None else capital_tolerance
        self.reserve_min_tolerance = settings.reserve_min_tolerance
        self.csm_min_tolerance = settings.csm_min_tolerance
        self.capital_min_tolerance = settings.capital_min_tolerance

    def validate(self, curated_reporting_dataset: pd.DataFrame) -> ValidationResult:
        """Run validation checks and return detailed outputs."""

        flags = curated_reporting_dataset[["quarter", "product", "region"]].copy()
        flags["missing_values_flag"] = curated_reporting_dataset[self.REQUIRED_COLUMNS].isna().any(axis=1)
        flags["negative_values_flag"] = (
            curated_reporting_dataset[self.NON_NEGATIVE_COLUMNS].lt(0).any(axis=1)
        )

        reserve_reference = curated_reporting_dataset[["liability_balance", "reserve_reconciliation_amount", "reserves"]].abs().max(axis=1)
        csm_reference = curated_reporting_dataset[["csm_closing", "csm_reconciliation_amount", "csm_opening"]].abs().max(axis=1)
        capital_reference = curated_reporting_dataset[["capital_proxy", "capital_expected", "liability_balance"]].abs().max(axis=1)

        flags["reserve_tolerance_used"] = self._scaled_tolerance(
            reserve_reference,
            self.reserve_tolerance,
            self.reserve_min_tolerance,
        ).round(2)
        flags["csm_tolerance_used"] = self._scaled_tolerance(
            csm_reference,
            self.csm_tolerance,
            self.csm_min_tolerance,
        ).round(2)
        flags["capital_tolerance_used"] = self._scaled_tolerance(
            capital_reference,
            self.capital_tolerance,
            self.capital_min_tolerance,
        ).round(2)

        flags["reserve_reconciliation_flag"] = (
            curated_reporting_dataset["reserve_reconciliation_diff"].abs() > flags["reserve_tolerance_used"]
        )
        flags["csm_reconciliation_flag"] = (
            curated_reporting_dataset["csm_reconciliation_diff"].abs() > flags["csm_tolerance_used"]
        )
        flags["capital_consistency_flag"] = (
            curated_reporting_dataset["capital_difference"].abs() > flags["capital_tolerance_used"]
        )

        rule_columns = [
            "missing_values_flag",
            "negative_values_flag",
            "reserve_reconciliation_flag",
            "csm_reconciliation_flag",
            "capital_consistency_flag",
        ]
        flags["validation_issue_count"] = flags[rule_columns].sum(axis=1)
        flags["has_validation_issue"] = flags["validation_issue_count"] > 0
        flags["validation_status"] = np.where(flags["has_validation_issue"], "fail", "pass")

        anomaly_records: list[dict[str, object]] = []
        anomaly_records.extend(
            self._collect_missing_value_anomalies(curated_reporting_dataset, flags["missing_values_flag"])
        )
        anomaly_records.extend(
            self._collect_negative_value_anomalies(curated_reporting_dataset, flags["negative_values_flag"])
        )
        anomaly_records.extend(
            self._collect_reconciliation_anomalies(
                curated_reporting_dataset,
                flags["reserve_reconciliation_flag"],
                flags["reserve_tolerance_used"],
                rule_name="reserve_reconciliation",
                metric_name="reserve_reconciliation_diff",
            )
        )
        anomaly_records.extend(
            self._collect_reconciliation_anomalies(
                curated_reporting_dataset,
                flags["csm_reconciliation_flag"],
                flags["csm_tolerance_used"],
                rule_name="csm_reconciliation",
                metric_name="csm_reconciliation_diff",
            )
        )
        anomaly_records.extend(
            self._collect_reconciliation_anomalies(
                curated_reporting_dataset,
                flags["capital_consistency_flag"],
                flags["capital_tolerance_used"],
                rule_name="capital_consistency",
                metric_name="capital_difference",
            )
        )

        anomaly_table = pd.DataFrame(
            anomaly_records,
            columns=["quarter", "product", "region", "rule_name", "metric_name", "observed_value", "details"],
        )
        quarterly_summary = self._build_quarterly_summary(flags, anomaly_table)

        return ValidationResult(
            validation_flags=flags,
            quarterly_validation_summary=quarterly_summary,
            anomaly_table=anomaly_table,
        )

    def write(
        self,
        validation_result: ValidationResult,
        *,
        file_format: str = "csv",
    ) -> dict[str, Path]:
        """Persist validation outputs to the processed data directory."""

        artifact_paths = ensure_artifact_dirs(self.config)
        output_dir = artifact_paths.data_processed
        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths: dict[str, Path] = {}
        for name, frame in validation_result.as_dict().items():
            destination = output_dir / f"{name}.{file_format}"
            frame.to_csv(destination, index=False)
            output_paths[name] = destination

        return output_paths

    def run(
        self,
        curated_reporting_dataset: pd.DataFrame,
        *,
        file_format: str = "csv",
    ) -> tuple[ValidationResult, dict[str, Path]]:
        """Validate and persist validation outputs."""

        validation_result = self.validate(curated_reporting_dataset)
        output_paths = self.write(validation_result, file_format=file_format)
        return validation_result, output_paths

    def _scaled_tolerance(
        self,
        reference_values: pd.Series,
        tolerance_pct: float,
        min_tolerance: float,
    ) -> pd.Series:
        reference = pd.to_numeric(reference_values, errors="coerce").abs().fillna(0.0)
        return pd.Series(np.maximum(reference * tolerance_pct, min_tolerance), index=reference.index)

    def _collect_missing_value_anomalies(
        self,
        curated_reporting_dataset: pd.DataFrame,
        failing_rows: pd.Series,
    ) -> list[dict[str, object]]:
        anomalies: list[dict[str, object]] = []
        subset = curated_reporting_dataset.loc[failing_rows]
        for row in subset.itertuples(index=False):
            missing_columns = [column for column in self.REQUIRED_COLUMNS if pd.isna(getattr(row, column))]
            anomalies.append(
                {
                    "quarter": row.quarter,
                    "product": row.product,
                    "region": row.region,
                    "rule_name": "missing_values",
                    "metric_name": "required_columns",
                    "observed_value": len(missing_columns),
                    "details": ", ".join(missing_columns),
                }
            )
        return anomalies

    def _collect_negative_value_anomalies(
        self,
        curated_reporting_dataset: pd.DataFrame,
        failing_rows: pd.Series,
    ) -> list[dict[str, object]]:
        anomalies: list[dict[str, object]] = []
        subset = curated_reporting_dataset.loc[failing_rows]
        for row in subset.itertuples(index=False):
            negative_columns = [
                column for column in self.NON_NEGATIVE_COLUMNS if pd.notna(getattr(row, column)) and getattr(row, column) < 0
            ]
            anomalies.append(
                {
                    "quarter": row.quarter,
                    "product": row.product,
                    "region": row.region,
                    "rule_name": "negative_values",
                    "metric_name": "non_negative_columns",
                    "observed_value": len(negative_columns),
                    "details": ", ".join(negative_columns),
                }
            )
        return anomalies

    def _collect_reconciliation_anomalies(
        self,
        curated_reporting_dataset: pd.DataFrame,
        failing_rows: pd.Series,
        tolerance_used: pd.Series,
        *,
        rule_name: str,
        metric_name: str,
    ) -> list[dict[str, object]]:
        anomalies: list[dict[str, object]] = []
        subset = curated_reporting_dataset.loc[failing_rows, ["quarter", "product", "region", metric_name]].copy()
        subset["tolerance_used"] = tolerance_used.loc[failing_rows].values
        for row in subset.itertuples(index=False):
            anomalies.append(
                {
                    "quarter": row.quarter,
                    "product": row.product,
                    "region": row.region,
                    "rule_name": rule_name,
                    "metric_name": metric_name,
                    "observed_value": round(float(getattr(row, metric_name)), 4),
                    "details": f"Tolerance exceeded for {metric_name}; tolerance={float(row.tolerance_used):.2f}",
                }
            )
        return anomalies

    def _build_quarterly_summary(
        self,
        validation_flags: pd.DataFrame,
        anomaly_table: pd.DataFrame,
    ) -> pd.DataFrame:
        summary = (
            validation_flags.groupby("quarter", as_index=False)
            .agg(
                records_validated=("quarter", "size"),
                records_with_issues=("has_validation_issue", "sum"),
                missing_values_issues=("missing_values_flag", "sum"),
                negative_values_issues=("negative_values_flag", "sum"),
                reserve_reconciliation_issues=("reserve_reconciliation_flag", "sum"),
                csm_reconciliation_issues=("csm_reconciliation_flag", "sum"),
                capital_consistency_issues=("capital_consistency_flag", "sum"),
            )
        )

        if anomaly_table.empty:
            summary["anomaly_count"] = 0
        else:
            anomaly_counts = (
                anomaly_table.groupby("quarter", as_index=False)
                .agg(anomaly_count=("rule_name", "size"))
            )
            summary = summary.merge(anomaly_counts, on="quarter", how="left")
            summary["anomaly_count"] = summary["anomaly_count"].fillna(0).astype(int)

        summary["validation_pass_rate"] = (
            (summary["records_validated"] - summary["records_with_issues"]) / summary["records_validated"]
        ).round(4)
        return summary.sort_values("quarter").reset_index(drop=True)
