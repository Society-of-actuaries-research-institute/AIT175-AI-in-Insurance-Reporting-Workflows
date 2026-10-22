"""Synthetic life insurer data generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ai_insurance_reporting.config.loader import AppConfig, load_config
from ai_insurance_reporting.utils.artifacts import ensure_artifact_dirs


@dataclass(slots=True)
class SyntheticDataBundle:
    """Container for generated synthetic datasets."""

    policy_data: pd.DataFrame
    claims_data: pd.DataFrame
    asset_data: pd.DataFrame
    financial_balances: pd.DataFrame
    reporting_metrics: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        """Return bundle contents keyed by output file stem."""

        return {
            "policy_data": self.policy_data,
            "claims_data": self.claims_data,
            "asset_data": self.asset_data,
            "financial_balances": self.financial_balances,
            "reporting_metrics": self.reporting_metrics,
        }


class SyntheticDataGenerator:
    """Generate the raw datasets used throughout the reporting case study.

    The generator creates a quarterly life insurer data set across products and
    regions. Outputs are intended to support downstream ETL, validation,
    forecasting, explainability, narrative, and scenario workflows.

    The values are synthetic and calibrated for demonstration purposes rather
    than regulatory or actuarial use.
    """

    INVESTMENT_INCOME_CAPITAL_FACTOR = 0.0025

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        random_seed: int = 42,
        policies_per_segment: int = 40,
        products: list[str] | None = None,
        regions: list[str] | None = None,
        quarters: list[str] | None = None,
        premium_multiplier: float = 1.0,
        claims_multiplier: float = 1.0,
        reserve_multiplier: float = 1.0,
        csm_multiplier: float = 1.0,
        asset_return_shift: float = 0.0,
        capital_multiplier: float = 1.0,
        include_validation_examples: bool = True,
    ) -> None:
        self.config = config or load_config()
        self.rng = np.random.default_rng(random_seed)
        self.policies_per_segment = policies_per_segment
        self.products = products or ["Term Life", "Whole Life", "Universal Life", "Annuity"]
        self.regions = regions or ["North", "South", "East", "West"]
        self.quarters = quarters or [
            "2024Q1",
            "2024Q2",
            "2024Q3",
            "2024Q4",
            "2025Q1",
            "2025Q2",
            "2025Q3",
            "2025Q4",
        ]
        self.premium_multiplier = premium_multiplier
        self.claims_multiplier = claims_multiplier
        self.reserve_multiplier = reserve_multiplier
        self.csm_multiplier = csm_multiplier
        self.asset_return_shift = asset_return_shift
        self.capital_multiplier = capital_multiplier
        self.include_validation_examples = include_validation_examples

    def generate(self) -> SyntheticDataBundle:
        """Generate the full raw data bundle in memory.

        Returns policy-level data, claims, asset records, aggregated financial
        balances, and derived reporting metrics. Optional validation examples are
        injected after the core data is built so the validation pipeline has
        meaningful exceptions to surface.
        """

        policy_data = self._generate_policy_data()
        claims_data = self._generate_claims_data(policy_data)
        asset_data = self._generate_asset_data()
        financial_balances = self._generate_financial_balances(policy_data, claims_data, asset_data)
        reporting_metrics = self._generate_reporting_metrics(financial_balances)
        if self.include_validation_examples:
            financial_balances, reporting_metrics = self._inject_validation_examples(
                financial_balances,
                reporting_metrics,
            )

        return SyntheticDataBundle(
            policy_data=policy_data,
            claims_data=claims_data,
            asset_data=asset_data,
            financial_balances=financial_balances,
            reporting_metrics=reporting_metrics,
        )

    def write(
        self,
        bundle: SyntheticDataBundle | None = None,
        *,
        file_format: str = "csv",
    ) -> dict[str, Path]:
        """Persist the generated raw datasets to the configured input directory."""

        bundle = bundle or self.generate()
        artifact_paths = ensure_artifact_dirs(self.config)
        output_dir = artifact_paths.data_input
        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths: dict[str, Path] = {}
        for name, frame in bundle.as_dict().items():
            destination = output_dir / f"{name}.{file_format}"
            frame.to_csv(destination, index=False)
            output_paths[name] = destination

        return output_paths

    def _generate_policy_data(self) -> pd.DataFrame:
        """Create policy-level quarterly records."""

        rows: list[dict[str, object]] = []
        policy_counter = 1

        product_base_premium = {
            "Term Life": 1_200,
            "Whole Life": 2_400,
            "Universal Life": 2_000,
            "Annuity": 3_000,
        }
        product_face_multiplier = {
            "Term Life": 110,
            "Whole Life": 95,
            "Universal Life": 100,
            "Annuity": 70,
        }

        quarter_count = len(self.quarters)
        for quarter_idx, quarter in enumerate(self.quarters):
            seasonal_factor = [0.96, 1.0, 1.05, 1.02][quarter_idx % 4]
            trend_factor = 1.0 + (quarter_idx / max(quarter_count - 1, 1)) * 0.12
            for product in self.products:
                product_factor = {
                    "Term Life": 0.94,
                    "Whole Life": 1.05,
                    "Universal Life": 1.02,
                    "Annuity": 1.08,
                }[product]
                for region in self.regions:
                    region_factor = 1.0 + (self.regions.index(region) * 0.025)
                    for _ in range(self.policies_per_segment):
                        discount_rate = float(
                            np.clip(
                                0.024
                                + quarter_idx * 0.0012
                                + (0.0015 if product in {"Whole Life", "Annuity"} else 0.0)
                                + (0.0005 if region in {"South", "West"} else 0.0)
                                + self.rng.normal(0.0, 0.0008),
                                0.015,
                                0.055,
                            )
                        )
                        expense_inflation = float(
                            np.clip(
                                0.018
                                + quarter_idx * 0.0015
                                + (0.001 if region in {"East", "West"} else 0.0)
                                + self.rng.normal(0.0, 0.0015),
                                0.01,
                                0.06,
                            )
                        )
                        mortality_rate = float(
                            np.clip(
                                0.0012
                                + (0.001 if product == "Term Life" else 0.0)
                                + (0.0003 if region in {"South", "West"} else 0.0)
                                + quarter_idx * 0.00008
                                + self.rng.normal(0.0, 0.0002),
                                0.0005,
                                0.0045,
                            )
                        )
                        lapse_rate = float(
                            np.clip(
                                0.02
                                + (0.01 if product == "Universal Life" else 0.0)
                                + (0.006 if product == "Term Life" else 0.0)
                                + max(discount_rate - 0.03, 0.0) * 2.5
                                + self.rng.normal(0.0, 0.006),
                                0.005,
                                0.12,
                            )
                        )
                        new_business_probability = float(
                            np.clip(
                                0.18
                                + (0.06 if quarter_idx % 4 == 0 else 0.0)
                                + (0.05 if product in {"Annuity", "Universal Life"} else 0.0)
                                - lapse_rate * 0.4
                                + self.rng.normal(0.0, 0.015),
                                0.05,
                                0.45,
                            )
                        )
                        new_business_indicator = int(self.rng.random() < new_business_probability)
                        premium_mean = (
                            product_base_premium[product]
                            * seasonal_factor
                            * trend_factor
                            * product_factor
                            * region_factor
                            * (1.0 + new_business_indicator * 0.06)
                            * (1.0 + expense_inflation * 0.4)
                            * (1.0 - lapse_rate * 0.3)
                        )
                        annual_premium = float(
                            self.rng.normal(premium_mean, premium_mean * 0.08)
                        )
                        annual_premium = max(annual_premium * self.premium_multiplier, 300.0)
                        periodic_premium = round(annual_premium / 4, 2)
                        face_amount = round(
                            annual_premium * product_face_multiplier[product] * self.rng.uniform(18, 38),
                            2,
                        )
                        reserve_ratio = np.clip(
                            0.48
                            + (quarter_idx * 0.01)
                            + (0.03 if product == "Whole Life" else 0.0)
                            + mortality_rate * 18.0
                            - discount_rate * 1.8
                            - lapse_rate * 0.45
                            + self.rng.normal(0.0, 0.03),
                            0.42,
                            0.92,
                        )
                        reserve_balance = round(
                            periodic_premium * reserve_ratio * self.rng.uniform(5.1, 6.3) * self.reserve_multiplier,
                            2,
                        )
                        csm_opening = round(periodic_premium * self.rng.uniform(1.4, 3.5) * self.csm_multiplier, 2)
                        csm_new_business = round(
                            periodic_premium
                            * self.rng.uniform(0.2, 0.7)
                            * self.csm_multiplier
                            * (1.0 + new_business_indicator * 0.85),
                            2,
                        )
                        csm_interest_accretion = round(csm_opening * self.rng.uniform(0.008, 0.018), 2)
                        csm_release = round(
                            min(
                                csm_opening + csm_new_business + csm_interest_accretion,
                                periodic_premium * self.rng.uniform(0.3, 0.9) * (1.0 + lapse_rate * 1.5),
                            ),
                            2,
                        )
                        csm_closing = round(
                            csm_opening + csm_new_business + csm_interest_accretion - csm_release,
                            2,
                        )
                        expected_claim_ratio = np.clip(
                            0.22
                            + (0.02 if product == "Term Life" else 0.0)
                            + (0.015 if region in {"South", "West"} else 0.0)
                            + (quarter_idx % 4) * 0.01
                            + mortality_rate * 22.0
                            + lapse_rate * 0.08
                            + self.rng.normal(0.0, 0.015),
                            0.16,
                            0.58,
                        )
                        incurred_claims = round(periodic_premium * expected_claim_ratio * self.claims_multiplier, 2)
                        paid_claims = round(incurred_claims * self.rng.uniform(0.7, 1.0), 2)
                        rows.append(
                            {
                                "policy_id": f"POL-{policy_counter:06d}",
                                "quarter": quarter,
                                "product": product,
                                "region": region,
                                "policy_status": self.rng.choice(
                                    ["Inforce", "Lapsed", "Claimed"], p=[0.9, 0.06, 0.04]
                                ),
                                "face_amount": face_amount,
                                "annual_premium": round(annual_premium, 2),
                                "periodic_premium": periodic_premium,
                                "reserve_balance": reserve_balance,
                                "csm_opening": csm_opening,
                                "csm_new_business": csm_new_business,
                                "csm_interest_accretion": csm_interest_accretion,
                                "csm_release": csm_release,
                                "csm_closing": csm_closing,
                                "incurred_claims": incurred_claims,
                                "paid_claims": paid_claims,
                                "lapse_rate": round(lapse_rate, 4),
                                "mortality_rate": round(mortality_rate, 6),
                                "discount_rate": round(discount_rate, 4),
                                "expense_inflation": round(expense_inflation, 4),
                                "new_business_indicator": new_business_indicator,
                            }
                        )
                        policy_counter += 1

        policy_data = pd.DataFrame(rows)
        return policy_data.sort_values(["quarter", "product", "region", "policy_id"]).reset_index(drop=True)

    def _generate_claims_data(self, policy_data: pd.DataFrame) -> pd.DataFrame:
        """Create claim-level records sampled from policy experience."""

        claim_rows: list[dict[str, object]] = []
        claim_counter = 1

        for policy in policy_data.itertuples(index=False):
            product_claim_factor = {
                "Term Life": 0.09,
                "Whole Life": 0.07,
                "Universal Life": 0.06,
                "Annuity": 0.05,
            }
            quarter_factor = 1.0 + ((int(str(policy.quarter)[-1]) - 1) * 0.08)
            region_factor = 1.05 if policy.region in {"South", "West"} else 0.95
            base_probability = product_claim_factor.get(policy.product, 0.06) * quarter_factor * region_factor
            if policy.policy_status == "Claimed":
                base_probability += 0.35
            elif policy.policy_status == "Inforce":
                base_probability += 0.03

            claim_count = int(self.rng.poisson(base_probability))
            if policy.policy_status == "Claimed":
                claim_count = max(claim_count, 1)
            claim_count = min(claim_count, 3)

            for _ in range(claim_count):
                severity = float(self.rng.uniform(0.35, 0.75))
                incurred_amount = round(
                    max(float(policy.incurred_claims) * severity / max(claim_count, 1), 0.0),
                    2,
                )
                paid_ratio = self.rng.uniform(0.55, 0.98)
                paid_amount = round(min(incurred_amount, incurred_amount * paid_ratio), 2)

                claim_rows.append(
                    {
                        "claim_id": f"CLM-{claim_counter:07d}",
                        "policy_id": policy.policy_id,
                        "quarter": policy.quarter,
                        "product": policy.product,
                        "region": policy.region,
                        "claim_type": self.rng.choice(
                            ["Death", "Maturity", "Disability", "Surrender"], p=[0.42, 0.24, 0.16, 0.18]
                        ),
                        "claim_status": self.rng.choice(
                            ["Reported", "Approved", "Paid"], p=[0.2, 0.35, 0.45]
                        ),
                        "incurred_amount": incurred_amount,
                        "paid_amount": paid_amount,
                        "case_reserve": round(max(incurred_amount - paid_amount, 0.0), 2),
                    }
                )
                claim_counter += 1

        columns = [
            "claim_id",
            "policy_id",
            "quarter",
            "product",
            "region",
            "claim_type",
            "claim_status",
            "incurred_amount",
            "paid_amount",
            "case_reserve",
        ]
        return pd.DataFrame(claim_rows, columns=columns)

    def _generate_asset_data(self) -> pd.DataFrame:
        """Create quarterly investment portfolio records."""

        rows: list[dict[str, object]] = []
        asset_counter = 1
        asset_classes = {
            "Corporate Bonds": (0.008, 0.015),
            "Government Bonds": (0.005, 0.009),
            "Equities": (0.012, 0.045),
            "Real Estate": (0.01, 0.02),
        }

        for quarter in self.quarters:
            for region in self.regions:
                regional_scale = 1.0 + (self.regions.index(region) * 0.06)
                for asset_class, (mean_return, return_vol) in asset_classes.items():
                    market_value = round(
                        float(self.rng.uniform(20_000_000, 90_000_000) * regional_scale),
                        2,
                    )
                    asset_return = round(float(self.rng.normal(mean_return, return_vol) + self.asset_return_shift), 4)
                    investment_income = round(market_value * asset_return, 2)
                    duration_years = round(float(self.rng.uniform(2.0, 11.0)), 2)

                    rows.append(
                        {
                            "asset_id": f"AST-{asset_counter:06d}",
                            "quarter": quarter,
                            "region": region,
                            "asset_class": asset_class,
                            "market_value": market_value,
                            "asset_return": asset_return,
                            "investment_income": investment_income,
                            "duration_years": duration_years,
                        }
                    )
                    asset_counter += 1

        return pd.DataFrame(rows).sort_values(["quarter", "region", "asset_class"]).reset_index(drop=True)

    def _generate_financial_balances(
        self,
        policy_data: pd.DataFrame,
        claims_data: pd.DataFrame,
        asset_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate financial balances by quarter, product, and region."""

        policy_grouped = (
            policy_data.groupby(["quarter", "product", "region"], as_index=False)
            .agg(
                premium_income=("periodic_premium", "sum"),
                reserves=("reserve_balance", "sum"),
                csm_opening=("csm_opening", "sum"),
                csm_new_business=("csm_new_business", "sum"),
                csm_interest_accretion=("csm_interest_accretion", "sum"),
                csm_release=("csm_release", "sum"),
                csm_closing=("csm_closing", "sum"),
                policy_claims_incurred=("incurred_claims", "sum"),
                policy_claims_paid=("paid_claims", "sum"),
                policies_inforce=("policy_id", "count"),
                new_business_count=("new_business_indicator", "sum"),
                average_lapse_rate=("lapse_rate", "mean"),
                average_mortality_rate=("mortality_rate", "mean"),
                average_discount_rate=("discount_rate", "mean"),
                average_expense_inflation=("expense_inflation", "mean"),
            )
        )

        if claims_data.empty:
            claims_grouped = pd.DataFrame(
                columns=["quarter", "product", "region", "claim_incurred_amount", "claim_paid_amount", "case_reserves"]
            )
        else:
            claims_grouped = (
                claims_data.groupby(["quarter", "product", "region"], as_index=False)
                .agg(
                    claim_incurred_amount=("incurred_amount", "sum"),
                    claim_paid_amount=("paid_amount", "sum"),
                    case_reserves=("case_reserve", "sum"),
                )
            )

        asset_grouped = (
            asset_data.groupby(["quarter", "region"], as_index=False)
            .agg(
                invested_assets=("market_value", "sum"),
                investment_income=("investment_income", "sum"),
                average_asset_return=("asset_return", "mean"),
            )
        )

        financial_balances = policy_grouped.merge(
            claims_grouped,
            on=["quarter", "product", "region"],
            how="left",
        ).merge(
            asset_grouped,
            on=["quarter", "region"],
            how="left",
        )

        claim_columns = ["claim_incurred_amount", "claim_paid_amount", "case_reserves"]
        for column in claim_columns:
            financial_balances[column] = pd.to_numeric(financial_balances[column], errors="coerce").fillna(0.0)

        financial_balances = self._stabilize_segment_balances(financial_balances)

        ordered_columns = [
            "quarter",
            "product",
            "region",
            "policies_inforce",
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
            "invested_assets",
            "investment_income",
            "average_asset_return",
            "capital_proxy",
            "equity_proxy",
        ]
        return financial_balances[ordered_columns].sort_values(
            ["quarter", "product", "region"]
        ).reset_index(drop=True)

    def _generate_reporting_metrics(self, financial_balances: pd.DataFrame) -> pd.DataFrame:
        """Derive reporting metrics for management reporting."""

        metrics = financial_balances.copy()
        metrics["loss_ratio"] = np.where(
            metrics["premium_income"] > 0,
            metrics["total_claims"] / metrics["premium_income"],
            0.0,
        )
        metrics["expense_ratio"] = 0.08 + metrics["policies_inforce"] / (metrics["policies_inforce"] + 500.0) * 0.03
        metrics["combined_ratio"] = metrics["loss_ratio"] + metrics["expense_ratio"]
        metrics["reserve_coverage_ratio"] = np.where(
            metrics["total_claims"] > 0,
            metrics["liability_balance"] / metrics["total_claims"],
            0.0,
        )
        metrics["csm_movement"] = (
            metrics["csm_new_business"] + metrics["csm_interest_accretion"] - metrics["csm_release"]
        )
        metrics["capital_intensity"] = np.where(
            metrics["premium_income"] > 0,
            metrics["capital_proxy"] / metrics["premium_income"],
            0.0,
        )
        metrics["return_on_assets"] = np.where(
            metrics["invested_assets"] > 0,
            metrics["investment_income"] / metrics["invested_assets"],
            0.0,
        )

        ordered_columns = [
            "quarter",
            "product",
            "region",
            "premium_income",
            "total_claims",
            "reserves",
            "csm_opening",
            "csm_new_business",
            "csm_interest_accretion",
            "csm_release",
            "csm_closing",
            "capital_proxy",
            "average_asset_return",
            "loss_ratio",
            "expense_ratio",
            "combined_ratio",
            "reserve_coverage_ratio",
            "csm_movement",
            "capital_intensity",
            "return_on_assets",
        ]
        return metrics[ordered_columns].round(4)

    def _stabilize_segment_balances(self, financial_balances: pd.DataFrame) -> pd.DataFrame:
        """Smooth segment balances so quarterly targets have consistent drivers."""

        balances = financial_balances.copy()
        balances["quarter_index"] = balances["quarter"].map(self._quarter_to_index)
        balances = balances.sort_values(["product", "region", "quarter_index"]).reset_index(drop=True)

        claim_product_offset = {
            "Term Life": 0.035,
            "Whole Life": 0.018,
            "Universal Life": 0.022,
            "Annuity": 0.012,
        }
        reserve_product_offset = {
            "Term Life": 0.10,
            "Whole Life": 0.28,
            "Universal Life": 0.20,
            "Annuity": 0.24,
        }
        capital_product_offset = {
            "Term Life": -0.01,
            "Whole Life": 0.015,
            "Universal Life": 0.005,
            "Annuity": 0.02,
        }
        region_offset = {
            "North": -0.004,
            "South": 0.004,
            "East": 0.0,
            "West": 0.006,
        }

        updated_rows: list[dict[str, object]] = []
        for _, segment_frame in balances.groupby(["product", "region"], sort=False):
            previous_reserves: float | None = None
            previous_csm: float | None = None
            previous_capital_ratio: float | None = None

            for row in segment_frame.itertuples(index=False):
                quarter_of_year = int(str(row.quarter)[-1])
                seasonal_claim_factor = [0.97, 1.0, 1.04, 1.02][quarter_of_year - 1]
                premium_income = float(row.premium_income)
                new_business_rate = float(row.new_business_count) / max(float(row.policies_inforce), 1.0)
                expected_claim_ratio = float(
                    np.clip(
                        0.17
                        + claim_product_offset.get(str(row.product), 0.02)
                        + region_offset.get(str(row.region), 0.0)
                        + float(row.average_mortality_rate) * 22.0
                        + float(row.average_lapse_rate) * 0.24
                        + (quarter_of_year - 1) * 0.006
                        + (float(row.average_expense_inflation) - 0.02) * 0.5,
                        0.16,
                        0.46,
                    )
                )
                total_claims = (
                    0.7 * float(row.policy_claims_incurred)
                    + 0.3 * (premium_income * expected_claim_ratio * seasonal_claim_factor)
                    + 0.65 * float(row.claim_incurred_amount)
                )
                total_claims_paid = min(
                    total_claims,
                    0.78 * total_claims + 0.22 * (float(row.policy_claims_paid) + float(row.claim_paid_amount)),
                )
                case_reserves = max(total_claims - total_claims_paid, 0.0)

                if previous_reserves is None:
                    reserves = premium_income * (
                        4.5
                        + reserve_product_offset.get(str(row.product), 0.18)
                        + float(row.average_mortality_rate) * 280.0
                        - float(row.average_discount_rate) * 20.0
                        + new_business_rate * 1.2
                    )
                else:
                    reserves = (
                        previous_reserves * 0.91
                        + premium_income * (0.88 + reserve_product_offset.get(str(row.product), 0.18))
                        + case_reserves * 0.55
                        + float(row.new_business_count) * 24.0
                        - float(row.average_discount_rate) * 4500.0
                    )
                reserves = max(reserves * self.reserve_multiplier, premium_income * 2.2)

                csm_opening = previous_csm if previous_csm is not None else premium_income * (1.25 + new_business_rate * 1.4)
                csm_new_business = premium_income * (
                    0.08 + new_business_rate * 0.72 + reserve_product_offset.get(str(row.product), 0.18) * 0.08
                ) * self.csm_multiplier
                csm_interest_accretion = csm_opening * max(float(row.average_discount_rate) * 0.35, 0.006)
                csm_release = min(
                    csm_opening + csm_new_business + csm_interest_accretion,
                    premium_income * (0.12 + float(row.average_lapse_rate) * 0.9 + seasonal_claim_factor * 0.015),
                )
                csm_closing = max(csm_opening + csm_new_business + csm_interest_accretion - csm_release, 0.0)

                liability_balance = reserves + csm_closing + case_reserves
                formula_capital_ratio = float(
                    np.clip(
                        0.22
                        + capital_product_offset.get(str(row.product), 0.0)
                        + region_offset.get(str(row.region), 0.0)
                        + float(row.average_asset_return) * 1.9
                        - float(row.average_mortality_rate) * 9.5
                        - float(row.average_lapse_rate) * 0.35
                        + float(row.average_discount_rate) * 0.65,
                        0.12,
                        0.42,
                    )
                )
                capital_ratio_target = (
                    formula_capital_ratio
                    if previous_capital_ratio is None
                    else 0.72 * previous_capital_ratio + 0.28 * formula_capital_ratio
                )
                capital_proxy = liability_balance * capital_ratio_target * self.capital_multiplier
                equity_proxy = float(row.invested_assets) - liability_balance + float(row.investment_income)

                updated_rows.append(
                    {
                        **row._asdict(),
                        "reserves": round(reserves, 2),
                        "csm_opening": round(csm_opening, 2),
                        "csm_new_business": round(csm_new_business, 2),
                        "csm_interest_accretion": round(csm_interest_accretion, 2),
                        "csm_release": round(csm_release, 2),
                        "csm_closing": round(csm_closing, 2),
                        "total_claims": round(total_claims, 2),
                        "total_claims_paid": round(total_claims_paid, 2),
                        "case_reserves": round(case_reserves, 2),
                        "liability_balance": round(liability_balance, 2),
                        "capital_proxy": round(capital_proxy, 2),
                        "equity_proxy": round(equity_proxy, 2),
                    }
                )
                previous_reserves = reserves
                previous_csm = csm_closing
                previous_capital_ratio = capital_ratio_target

        stabilized = pd.DataFrame(updated_rows)
        return stabilized.drop(columns=["quarter_index"])

    def _inject_validation_examples(
        self,
        financial_balances: pd.DataFrame,
        reporting_metrics: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Inject a small, reviewable set of quality issues for demonstration."""

        balances = financial_balances.copy()
        metrics = reporting_metrics.copy()
        if balances.empty:
            return balances, metrics

        issue_count = max(0, min(len(balances), int(self.config.validation.synthetic_issue_count)))
        if issue_count == 0:
            return balances, metrics

        issue_indices = sorted(set(np.linspace(0, len(balances) - 1, num=issue_count, dtype=int).tolist()))

        if len(issue_indices) >= 1:
            balances.loc[issue_indices[0], "total_claims_paid"] = round(
                -max(abs(float(balances.loc[issue_indices[0], "total_claims_paid"])) * 0.15, 1_500.0),
                2,
            )

        if len(issue_indices) >= 2:
            balances.loc[issue_indices[1], "csm_opening"] = np.nan

        if len(issue_indices) >= 3:
            balances.loc[issue_indices[2], "liability_balance"] = round(
                float(balances.loc[issue_indices[2], "liability_balance"]) * 0.6,
                2,
            )

        if len(issue_indices) >= 4:
            balances.loc[issue_indices[3], "capital_proxy"] = round(
                float(balances.loc[issue_indices[3], "capital_proxy"]) * 1.2 + 20_000.0,
                2,
            )

        merge_keys = ["quarter", "product", "region"]
        metric_columns = [column for column in metrics.columns if column not in merge_keys]
        metrics = metrics.drop(columns=metric_columns).merge(
            balances,
            on=merge_keys,
            how="left",
        )

        metrics["loss_ratio"] = np.where(
            metrics["premium_income"] > 0,
            metrics["total_claims"] / metrics["premium_income"],
            0.0,
        )
        metrics["expense_ratio"] = 0.08 + metrics["policies_inforce"] / (metrics["policies_inforce"] + 500.0) * 0.03
        metrics["combined_ratio"] = metrics["loss_ratio"] + metrics["expense_ratio"]
        metrics["reserve_coverage_ratio"] = np.where(
            metrics["total_claims"] > 0,
            metrics["liability_balance"] / metrics["total_claims"],
            0.0,
        )
        metrics["csm_movement"] = (
            metrics["csm_new_business"] + metrics["csm_interest_accretion"] - metrics["csm_release"]
        )
        metrics["capital_intensity"] = np.where(
            metrics["premium_income"] > 0,
            metrics["capital_proxy"] / metrics["premium_income"],
            0.0,
        )
        metrics["return_on_assets"] = np.where(
            metrics["invested_assets"] > 0,
            metrics["investment_income"] / metrics["invested_assets"],
            0.0,
        )

        ordered_columns = [
            "quarter",
            "product",
            "region",
            "premium_income",
            "total_claims",
            "reserves",
            "csm_opening",
            "csm_new_business",
            "csm_interest_accretion",
            "csm_release",
            "csm_closing",
            "capital_proxy",
            "average_asset_return",
            "loss_ratio",
            "expense_ratio",
            "combined_ratio",
            "reserve_coverage_ratio",
            "csm_movement",
            "capital_intensity",
            "return_on_assets",
        ]
        return balances, metrics[ordered_columns].round(4)

    def _quarter_to_index(self, quarter_label: str) -> int:
        year = int(quarter_label[:4])
        quarter = int(quarter_label[-1])
        return year * 4 + quarter
