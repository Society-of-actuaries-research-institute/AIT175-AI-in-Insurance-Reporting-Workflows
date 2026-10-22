"""ETL pipeline for synthetic insurance reporting data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from ai_insurance_reporting.config.loader import AppConfig, load_config
from ai_insurance_reporting.utils.artifacts import ensure_artifact_dirs


@dataclass(slots=True)
class RawDataBundle:
    """Container for raw input datasets."""

    policy_data: pd.DataFrame
    claims_data: pd.DataFrame
    asset_data: pd.DataFrame
    financial_balances: pd.DataFrame
    reporting_metrics: pd.DataFrame


class InsuranceETLPipeline:
    """Build a cleaned quarterly reporting dataset from raw synthetic data."""

    RAW_DATASETS = (
        "policy_data",
        "claims_data",
        "asset_data",
        "financial_balances",
        "reporting_metrics",
    )

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()

    def extract(self, *, file_format: str = "csv") -> RawDataBundle:
        """Load raw synthetic datasets from the configured input directory."""

        artifact_paths = ensure_artifact_dirs(self.config)
        input_dir = artifact_paths.data_input

        datasets: dict[str, pd.DataFrame] = {}
        for dataset_name in self.RAW_DATASETS:
            source = input_dir / f"{dataset_name}.{file_format}"
            if not source.exists():
                raise FileNotFoundError(f"Raw dataset not found: {source}")

            datasets[dataset_name] = pd.read_csv(source)

        return RawDataBundle(**datasets)

    def transform(self, raw_data: RawDataBundle) -> pd.DataFrame:
        """Create the cleaned quarterly reporting dataset."""

        policy_summary = (
            raw_data.policy_data.groupby(["quarter", "product", "region"], as_index=False)
            .agg(
                policies_total=("policy_id", "count"),
                active_policies=("policy_status", lambda s: int((s == "Inforce").sum())),
                claimed_policies=("policy_status", lambda s: int((s == "Claimed").sum())),
                lapsed_policies=("policy_status", lambda s: int((s == "Lapsed").sum())),
                new_business_count=("new_business_indicator", "sum"),
                new_business_rate=("new_business_indicator", "mean"),
                average_face_amount=("face_amount", "mean"),
                average_annual_premium=("annual_premium", "mean"),
                average_lapse_rate=("lapse_rate", "mean"),
                average_mortality_rate=("mortality_rate", "mean"),
                average_discount_rate=("discount_rate", "mean"),
                average_expense_inflation=("expense_inflation", "mean"),
            )
        )

        if raw_data.claims_data.empty:
            claims_summary = pd.DataFrame(
                columns=[
                    "quarter",
                    "product",
                    "region",
                    "claim_count",
                    "average_claim_size",
                    "max_claim_size",
                ]
            )
        else:
            claims_summary = (
                raw_data.claims_data.groupby(["quarter", "product", "region"], as_index=False)
                .agg(
                    claim_count=("claim_id", "count"),
                    average_claim_size=("incurred_amount", "mean"),
                    max_claim_size=("incurred_amount", "max"),
                )
            )

        asset_summary = (
            raw_data.asset_data.groupby(["quarter", "region"], as_index=False)
            .agg(
                asset_market_value=("market_value", "sum"),
                asset_investment_income=("investment_income", "sum"),
                asset_return_mean=("asset_return", "mean"),
                asset_duration_mean=("duration_years", "mean"),
            )
        )

        curated = raw_data.financial_balances.merge(
            raw_data.reporting_metrics,
            on=["quarter", "product", "region"],
            suffixes=("_balance", "_metric"),
        )
        curated = curated.merge(policy_summary, on=["quarter", "product", "region"], how="left")
        curated = curated.merge(claims_summary, on=["quarter", "product", "region"], how="left")
        curated = curated.merge(asset_summary, on=["quarter", "region"], how="left")

        balance_columns = {
            "premium_income_balance": "premium_income",
            "total_claims_balance": "total_claims",
            "reserves_balance": "reserves",
            "csm_opening_balance": "csm_opening",
            "csm_new_business_balance": "csm_new_business",
            "csm_interest_accretion_balance": "csm_interest_accretion",
            "csm_release_balance": "csm_release",
            "csm_closing_balance": "csm_closing",
            "capital_proxy_balance": "capital_proxy",
            "average_asset_return_balance": "average_asset_return",
        }
        curated = curated.rename(columns=balance_columns)

        curated["claim_count"] = pd.to_numeric(curated["claim_count"], errors="coerce").fillna(0).astype(int)
        for column in [
            "average_claim_size",
            "max_claim_size",
            "average_face_amount",
            "average_annual_premium",
            "average_lapse_rate",
            "average_mortality_rate",
            "average_discount_rate",
            "average_expense_inflation",
            "new_business_rate",
        ]:
            curated[column] = pd.to_numeric(curated[column], errors="coerce").fillna(0.0)
        curated["new_business_count"] = pd.to_numeric(curated["new_business_count"], errors="coerce").fillna(0).astype(int)
        curated["lapse_rate"] = curated["average_lapse_rate"].round(4)
        curated["mortality_rate"] = curated["average_mortality_rate"].round(6)
        curated["discount_rate"] = curated["average_discount_rate"].round(4)
        curated["expense_inflation"] = curated["average_expense_inflation"].round(4)

        curated["reserve_reconciliation_amount"] = (
            curated["reserves"] + curated["case_reserves"] + curated["csm_closing"]
        ).round(2)
        curated["reserve_reconciliation_diff"] = (
            curated["liability_balance"] - curated["reserve_reconciliation_amount"]
        ).round(2)
        curated["csm_reconciliation_amount"] = (
            curated["csm_opening"]
            + curated["csm_new_business"]
            + curated["csm_interest_accretion"]
            - curated["csm_release"]
        ).round(2)
        curated["csm_reconciliation_diff"] = (
            curated["csm_closing"] - curated["csm_reconciliation_amount"]
        ).round(2)
        quarter_index = curated["quarter"].str[:4].astype(int) * 4 + curated["quarter"].str[-1].astype(int)
        curated = curated.assign(quarter_index_internal=quarter_index)
        formula_capital_ratio = (
            0.22
            + curated["product"].map(
                {
                    "Term Life": -0.01,
                    "Whole Life": 0.015,
                    "Universal Life": 0.005,
                    "Annuity": 0.02,
                }
            ).fillna(0.0)
            + curated["region"].map(
                {
                    "North": -0.004,
                    "South": 0.004,
                    "East": 0.0,
                    "West": 0.006,
                }
            ).fillna(0.0)
            + curated["average_asset_return"] * 1.9
            - curated["mortality_rate"] * 9.5
            - curated["lapse_rate"] * 0.35
            + curated["discount_rate"] * 0.65
        ).clip(lower=0.12, upper=0.42)
        curated["formula_capital_ratio_internal"] = formula_capital_ratio
        capital_expected_parts: list[pd.DataFrame] = []
        for _, segment_frame in curated.sort_values(["product", "region", "quarter_index_internal"]).groupby(
            ["product", "region"], sort=False
        ):
            previous_ratio: float | None = None
            segment = segment_frame.copy()
            smoothed_ratios: list[float] = []
            for row in segment.itertuples(index=False):
                formula_ratio = float(row.formula_capital_ratio_internal)
                applied_ratio = formula_ratio if previous_ratio is None else 0.72 * previous_ratio + 0.28 * formula_ratio
                smoothed_ratios.append(applied_ratio)
                previous_ratio = applied_ratio
            segment["capital_expected"] = (segment["liability_balance"] * pd.Series(smoothed_ratios, index=segment.index)).round(2)
            capital_expected_parts.append(segment)
        curated = pd.concat(capital_expected_parts, ignore_index=True)
        curated["capital_difference"] = (curated["capital_proxy"] - curated["capital_expected"]).round(2)
        curated = curated.drop(columns=["quarter_index_internal", "formula_capital_ratio_internal"])

        preferred_order = [
            "quarter",
            "product",
            "region",
            "policies_total",
            "active_policies",
            "claimed_policies",
            "lapsed_policies",
            "claim_count",
            "new_business_count",
            "new_business_rate",
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
            "capital_expected",
            "capital_difference",
            "average_asset_return",
            "loss_ratio",
            "expense_ratio",
            "combined_ratio",
            "reserve_coverage_ratio",
            "csm_movement",
            "capital_intensity",
            "return_on_assets",
            "mortality_rate",
            "lapse_rate",
            "discount_rate",
            "expense_inflation",
            "average_face_amount",
            "average_annual_premium",
            "average_lapse_rate",
            "average_mortality_rate",
            "average_discount_rate",
            "average_expense_inflation",
            "average_claim_size",
            "max_claim_size",
            "asset_market_value",
            "asset_investment_income",
            "asset_return_mean",
            "asset_duration_mean",
            "reserve_reconciliation_amount",
            "reserve_reconciliation_diff",
            "csm_reconciliation_amount",
            "csm_reconciliation_diff",
        ]

        return curated[preferred_order].sort_values(["quarter", "product", "region"]).reset_index(drop=True)

    def load(
        self,
        curated_reporting_dataset: pd.DataFrame,
        *,
        file_format: str = "csv",
    ) -> Path:
        """Persist the cleaned reporting dataset to the processed data directory."""

        artifact_paths = ensure_artifact_dirs(self.config)
        output_path = artifact_paths.data_processed / f"curated_reporting_dataset.{file_format}"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        curated_reporting_dataset.to_csv(output_path, index=False)

        return output_path

    def run(self, *, file_format: str = "csv") -> tuple[pd.DataFrame, Path]:
        """Execute the ETL flow end to end."""

        raw_data = self.extract(file_format=file_format)
        curated = self.transform(raw_data)
        output_path = self.load(curated, file_format=file_format)
        return curated, output_path
