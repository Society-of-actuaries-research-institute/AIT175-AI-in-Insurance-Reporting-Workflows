"""Forecasting models for insurance reporting targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ai_insurance_reporting.config.loader import AppConfig, load_config
from ai_insurance_reporting.utils.artifacts import ensure_artifact_dirs


@dataclass(slots=True)
class ForecastingResult:
    """Forecasting outputs and trained model registry."""

    evaluation_table: pd.DataFrame
    backtest_predictions: pd.DataFrame
    forecast_output_table: pd.DataFrame
    trained_models: dict[tuple[str, str], object]

    def as_dict(self) -> dict[str, pd.DataFrame]:
        """Return persisted output tables."""

        return {
            "model_evaluation": self.evaluation_table,
            "backtest_predictions": self.backtest_predictions,
            "forecast_output_table": self.forecast_output_table,
        }


class ForecastingPipeline:
    """Train and evaluate forecasting models for management reporting targets.

    The pipeline converts the cleaned quarterly reporting dataset into a
    supervised learning frame, runs multiple model families for each target, and
    produces both evaluation tables and next-period forecasts.

    The intent is to provide a compact comparison framework that is easy to
    inspect in the dashboard and chatbot rather than a production forecasting
    platform.
    """

    TARGET_COLUMNS = {
        "claims": "total_claims",
        "premium": "premium_income",
        "reserve_movement": "reserve_movement",
        "csm_movement": "csm_movement",
        "capital_ratio": "capital_ratio",
    }
    CATEGORY_FEATURES = ["product", "region"]
    NUMERIC_FEATURES = [
        "quarter_index",
        "quarter_of_year",
        "lag_1",
        "lag_2",
        "lag_4",
        "level_lag_1",
        "level_lag_2",
        "mortality_rate",
        "lapse_rate",
        "discount_rate",
        "expense_inflation",
        "new_business_count",
        "new_business_rate",
    ]
    FEATURE_COLUMNS = CATEGORY_FEATURES + NUMERIC_FEATURES
    MIN_TRAIN_QUARTERS = 2
    SELECTION_METRICS = {"mae", "rmse", "mape"}
    TARGET_LEVEL_COLUMNS = {
        "claims": "total_claims",
        "premium": "premium_income",
        "reserve_movement": "reserves",
        "csm_movement": "csm_closing",
        "capital_ratio": "capital_ratio",
    }

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        selection_metric: str = "mae",
        error_tolerance_pct: float = 0.25,
        gb_max_depth: int = 3,
        gb_n_estimators: int = 100,
        gb_learning_rate: float = 0.1,
        forecast_horizon_quarters: int = 1,
    ) -> None:
        self.config = config or load_config()
        if selection_metric not in self.SELECTION_METRICS:
            raise ValueError(f"Unsupported selection metric: {selection_metric}")
        self.selection_metric = selection_metric
        self.error_tolerance_pct = error_tolerance_pct
        self.gb_max_depth = gb_max_depth
        self.gb_n_estimators = gb_n_estimators
        self.gb_learning_rate = gb_learning_rate
        self.forecast_horizon_quarters = max(int(forecast_horizon_quarters), 1)

    def load_curated_dataset(self, *, file_format: str = "csv") -> pd.DataFrame:
        """Load the cleaned reporting dataset from the processed data directory."""

        artifact_paths = ensure_artifact_dirs(self.config)
        source = artifact_paths.data_processed / f"curated_reporting_dataset.{file_format}"
        if not source.exists():
            raise FileNotFoundError(f"Cleaned reporting dataset not found: {source}")

        return pd.read_csv(source)

    def prepare_training_frame(self, curated_reporting_dataset: pd.DataFrame) -> pd.DataFrame:
        """Create a target-by-target training frame with lagged features.

        Each target is expanded into its own supervised series so the same model
        families can be compared consistently across claims, premium, reserve
        movement, CSM movement, and the synthetic capital proxy ratio.
        """

        frame = curated_reporting_dataset.copy()
        frame["quarter_index"] = frame["quarter"].map(self._quarter_to_index)
        frame["quarter_of_year"] = frame["quarter"].str[-1].astype(int)
        frame = frame.sort_values(["product", "region", "quarter_index"]).reset_index(drop=True)

        frame["reserve_movement"] = (
            frame.groupby(["product", "region"])["reserves"].diff().fillna(0.0).round(4)
        )
        frame["capital_ratio"] = np.where(
            frame["liability_balance"] > 0,
            frame["capital_proxy"] / frame["liability_balance"],
            0.0,
        ).round(6)
        for column in [
            "mortality_rate",
            "lapse_rate",
            "discount_rate",
            "expense_inflation",
            "new_business_count",
            "new_business_rate",
        ]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

        training_frames: list[pd.DataFrame] = []
        for target_name, source_column in self.TARGET_COLUMNS.items():
            level_column = self.TARGET_LEVEL_COLUMNS[target_name]
            target_frame = frame[
                [
                    "quarter",
                    "quarter_index",
                    "quarter_of_year",
                    "product",
                    "region",
                    "mortality_rate",
                    "lapse_rate",
                    "discount_rate",
                    "expense_inflation",
                    "new_business_count",
                    "new_business_rate",
                ]
            ].copy()
            target_frame["target_value"] = frame[source_column].to_numpy()
            target_frame["level_value"] = frame[level_column].to_numpy()
            target_frame["target_name"] = target_name
            target_frame["lag_1"] = (
                target_frame.groupby(["product", "region"])["target_value"].shift(1)
            )
            target_frame["lag_2"] = (
                target_frame.groupby(["product", "region"])["target_value"].shift(2)
            )
            target_frame["lag_4"] = (
                target_frame.groupby(["product", "region"])["target_value"].shift(4)
            )
            target_frame["level_lag_1"] = (
                target_frame.groupby(["product", "region"])["level_value"].shift(1)
            )
            target_frame["level_lag_2"] = (
                target_frame.groupby(["product", "region"])["level_value"].shift(2)
            )
            training_frames.append(target_frame)

        supervised = pd.concat(training_frames, ignore_index=True)
        supervised["lag_1"] = supervised["lag_1"].fillna(supervised["target_value"])
        supervised["lag_2"] = supervised["lag_2"].fillna(supervised["lag_1"])
        supervised["lag_4"] = supervised["lag_4"].fillna(supervised["lag_2"])
        supervised["level_lag_1"] = supervised["level_lag_1"].fillna(supervised["level_value"])
        supervised["level_lag_2"] = supervised["level_lag_2"].fillna(supervised["level_lag_1"])
        return supervised.sort_values(["target_name", "product", "region", "quarter_index"]).reset_index(drop=True)

    def train(self, training_frame: pd.DataFrame) -> ForecastingResult:
        """Run rolling backtests, select models, and generate forward forecasts.

        Evaluation metrics are computed from repeated holdout quarters rather
        than a single end-period split. The returned tables are written directly
        to the model artifact area and used by the dashboard, explainability
        layer, and chatbot tools.
        """

        evaluation_rows: list[dict[str, object]] = []
        backtest_rows: list[dict[str, object]] = []
        forecast_rows: list[dict[str, object]] = []
        trained_models: dict[tuple[str, str], object] = {}

        for target_name in self.TARGET_COLUMNS:
            target_frame = training_frame.loc[training_frame["target_name"] == target_name].copy()
            split_quarters = sorted(target_frame["quarter_index"].unique().tolist())
            evaluation_quarters = split_quarters[self.MIN_TRAIN_QUARTERS :]
            if not evaluation_quarters:
                raise ValueError(f"Insufficient training history for target: {target_name}")
            target_backtest_rows: list[dict[str, object]] = []

            for holdout_quarter in evaluation_quarters:
                train_frame = target_frame.loc[target_frame["quarter_index"] < holdout_quarter].copy()
                test_frame = target_frame.loc[target_frame["quarter_index"] == holdout_quarter].copy()
                if train_frame.empty or test_frame.empty:
                    continue

                family_predictions: dict[str, np.ndarray] = {}

                family_predictions["baseline_actuarial"] = self._predict_baseline(
                    target_name=target_name,
                    train_frame=train_frame,
                    test_frame=test_frame,
                )

                family_predictions["time_series"] = self._predict_time_series(
                    target_name=target_name,
                    train_frame=train_frame,
                    test_frame=test_frame,
                )

                gb_model = self._build_gradient_boosting_model()
                gb_model.fit(train_frame[self.FEATURE_COLUMNS], train_frame["target_value"])
                family_predictions["gradient_boosting"] = gb_model.predict(test_frame[self.FEATURE_COLUMNS])

                for family_name, predictions in family_predictions.items():
                    target_backtest_rows.extend(
                        self._build_backtest_rows(
                            target_name=target_name,
                            model_family=family_name,
                            test_frame=test_frame,
                            predictions=predictions,
                            train_observations=len(train_frame),
                        )
                    )

            target_backtest_frame = pd.DataFrame(target_backtest_rows)
            backtest_rows.extend(target_backtest_rows)
            target_evaluations = self._summarize_backtest_results(target_name, target_backtest_frame)
            evaluation_rows.extend(target_evaluations)

            best_model = min(target_evaluations, key=lambda row: row[self.selection_metric])["model_family"]

            full_train_frame = target_frame.copy()
            ts_model = self._build_time_series_model()
            ts_model.fit(full_train_frame[self.FEATURE_COLUMNS], full_train_frame["target_value"])
            trained_models[(target_name, "time_series")] = ts_model

            gb_model = self._build_gradient_boosting_model()
            gb_model.fit(full_train_frame[self.FEATURE_COLUMNS], full_train_frame["target_value"])
            trained_models[(target_name, "gradient_boosting")] = gb_model

            forecast_rows.extend(
                self._forecast_iteratively(
                    target_name=target_name,
                    model_family=best_model,
                    train_frame=target_frame,
                )
            )

        evaluation_table = pd.DataFrame(evaluation_rows).sort_values(["target_name", "mae", "rmse"]).reset_index(drop=True)
        backtest_predictions = pd.DataFrame(backtest_rows).sort_values(
            ["target_name", "model_family", "quarter", "product", "region"]
        ).reset_index(drop=True)
        forecast_output_table = pd.DataFrame(forecast_rows).sort_values(
            ["target_name", "forecast_horizon", "forecast_quarter", "product", "region"]
        ).reset_index(drop=True)

        return ForecastingResult(
            evaluation_table=evaluation_table,
            backtest_predictions=backtest_predictions,
            forecast_output_table=forecast_output_table,
            trained_models=trained_models,
        )

    def write(
        self,
        result: ForecastingResult,
        *,
        file_format: str = "csv",
    ) -> dict[str, Path]:
        """Persist forecast output tables to the models artifact directory."""

        artifact_paths = ensure_artifact_dirs(self.config)
        output_dir = artifact_paths.models
        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths: dict[str, Path] = {}
        for name, frame in result.as_dict().items():
            destination = output_dir / f"{name}.{file_format}"
            frame.to_csv(destination, index=False)
            output_paths[name] = destination

        return output_paths

    def run(
        self,
        *,
        file_format: str = "csv",
    ) -> tuple[ForecastingResult, dict[str, Path]]:
        """Train forecasts from the cleaned dataset and persist the outputs."""

        curated_reporting_dataset = self.load_curated_dataset(file_format=file_format)
        training_frame = self.prepare_training_frame(curated_reporting_dataset)
        result = self.train(training_frame)
        output_paths = self.write(result, file_format=file_format)
        return result, output_paths

    def _build_time_series_model(self) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.CATEGORY_FEATURES,
                ),
                ("numeric", "passthrough", self.NUMERIC_FEATURES),
            ]
        )
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", LinearRegression()),
            ]
        )

    def _build_gradient_boosting_model(self) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.CATEGORY_FEATURES,
                ),
                ("numeric", "passthrough", self.NUMERIC_FEATURES),
            ]
        )
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    GradientBoostingRegressor(
                        random_state=42,
                        max_depth=self.gb_max_depth,
                        n_estimators=self.gb_n_estimators,
                        learning_rate=self.gb_learning_rate,
                    ),
                ),
            ]
        )

    def _predict_baseline(
        self,
        *,
        target_name: str,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
    ) -> np.ndarray:
        if target_name == "csm_movement":
            predictions = []
            for row in test_frame.itertuples(index=False):
                csm_driver = (
                    float(row.level_lag_1)
                    * (
                        float(row.new_business_rate) * 0.09
                        + float(row.discount_rate) * 0.04
                        - float(row.lapse_rate) * 0.06
                        - 0.012
                    )
                )
                predictions.append(0.5 * float(row.lag_1) + csm_driver)
            return np.asarray(predictions)

        if target_name in {"reserve_movement", "csm_movement"}:
            growth_lookup = (
                train_frame.assign(
                    level_growth_factor=np.where(
                        train_frame["level_lag_1"].abs() > 1e-9,
                        train_frame["level_value"] / train_frame["level_lag_1"],
                        1.0,
                    )
                )
                .groupby(["product", "region"], as_index=False)["level_growth_factor"]
                .mean()
            )
            lookup = {
                (row.product, row.region): float(np.clip(row.level_growth_factor, 0.92, 1.12))
                for row in growth_lookup.itertuples(index=False)
            }
            predictions = []
            for row in test_frame.itertuples(index=False):
                factor = lookup.get((row.product, row.region), 1.0)
                predicted_level = float(row.level_lag_1) * factor
                predictions.append(predicted_level - float(row.level_lag_1))
            return np.asarray(predictions)

        growth_lookup = (
            train_frame.assign(
                development_factor=np.where(
                    train_frame["lag_1"].abs() > 1e-9,
                    train_frame["target_value"] / train_frame["lag_1"],
                    1.0,
                )
            )
            .groupby(["product", "region"], as_index=False)["development_factor"]
            .mean()
        )
        lookup = {
            (row.product, row.region): float(row.development_factor)
            for row in growth_lookup.itertuples(index=False)
        }

        predictions = []
        for row in test_frame.itertuples(index=False):
            factor = lookup.get((row.product, row.region), 1.0)
            predictions.append(float(row.lag_1) * factor)
        return np.asarray(predictions)

    def _predict_time_series(
        self,
        *,
        target_name: str,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
    ) -> np.ndarray:
        if target_name in {"reserve_movement", "csm_movement"}:
            if target_name == "csm_movement":
                predictions = []
                for row in test_frame.itertuples(index=False):
                    csm_driver = (
                        float(row.level_lag_1)
                        * (
                            float(row.new_business_rate) * 0.12
                            + float(row.discount_rate) * 0.05
                            - float(row.lapse_rate) * 0.08
                            - 0.018
                        )
                    )
                    seasonal_change = float(row.lag_4) if abs(float(row.lag_4)) > 1e-9 else 0.0
                    predictions.append(0.45 * float(row.lag_1) + 0.25 * seasonal_change + csm_driver)
                return np.asarray(predictions)

            predictions = []
            for row in test_frame.itertuples(index=False):
                previous_level_change = float(row.level_lag_1) - float(row.level_lag_2)
                seasonal_change = float(row.lag_4) if abs(float(row.lag_4)) > 1e-9 else float(row.lag_1)
                predictions.append(
                    0.5 * float(row.lag_1)
                    + 0.35 * previous_level_change
                    + 0.15 * seasonal_change
                )
            return np.asarray(predictions)

        model = self._build_time_series_model()
        model.fit(train_frame[self.FEATURE_COLUMNS], train_frame["target_value"])
        return model.predict(test_frame[self.FEATURE_COLUMNS])

    def _predict_future(
        self,
        *,
        target_name: str,
        model_family: str,
        train_frame: pd.DataFrame,
        future_frame: pd.DataFrame,
    ) -> np.ndarray:
        if model_family == "baseline_actuarial":
            return self._predict_baseline(
                target_name=target_name,
                train_frame=train_frame,
                test_frame=future_frame,
            )

        if model_family == "time_series":
            return self._predict_time_series(
                target_name=target_name,
                train_frame=train_frame,
                test_frame=future_frame,
            )
        elif model_family == "gradient_boosting":
            model = self._build_gradient_boosting_model()
        else:
            raise ValueError(f"Unsupported model family: {model_family}")

        model.fit(train_frame[self.FEATURE_COLUMNS], train_frame["target_value"])
        return model.predict(future_frame[self.FEATURE_COLUMNS])

    def _forecast_iteratively(
        self,
        *,
        target_name: str,
        model_family: str,
        train_frame: pd.DataFrame,
    ) -> list[dict[str, object]]:
        driver_columns = [
            "mortality_rate",
            "lapse_rate",
            "discount_rate",
            "expense_inflation",
            "new_business_count",
            "new_business_rate",
        ]
        states: dict[tuple[str, str], dict[str, object]] = {}
        for (product, region), segment in train_frame.groupby(["product", "region"], sort=False):
            ordered = segment.sort_values("quarter_index").reset_index(drop=True)
            seasonal_lookup = {
                column: {
                    int(quarter): float(value)
                    for quarter, value in ordered.groupby("quarter_of_year")[column].mean().items()
                }
                for column in driver_columns
            }
            states[(str(product), str(region))] = {
                "product": str(product),
                "region": str(region),
                "last_quarter_index": int(ordered["quarter_index"].iloc[-1]),
                "target_history": ordered["target_value"].astype(float).tolist(),
                "level_history": ordered["level_value"].astype(float).tolist(),
                "latest_driver_values": {column: float(ordered[column].iloc[-1]) for column in driver_columns},
                "seasonal_driver_lookup": seasonal_lookup,
            }

        forecast_rows: list[dict[str, object]] = []
        for horizon in range(1, self.forecast_horizon_quarters + 1):
            step_rows: list[dict[str, object]] = []
            state_keys: list[tuple[str, str]] = []
            for state_key, state in states.items():
                next_quarter_index = int(state["last_quarter_index"]) + 1
                forecast_quarter = self._index_to_quarter_label(next_quarter_index)
                quarter_of_year = int(forecast_quarter[-1])
                target_history = state["target_history"]
                level_history = state["level_history"]
                latest_driver_values = state["latest_driver_values"]
                seasonal_driver_lookup = state["seasonal_driver_lookup"]
                step_rows.append(
                    {
                        "quarter": forecast_quarter,
                        "forecast_quarter": forecast_quarter,
                        "quarter_index": next_quarter_index,
                        "quarter_of_year": quarter_of_year,
                        "product": state["product"],
                        "region": state["region"],
                        "lag_1": float(target_history[-1]),
                        "lag_2": float(target_history[-2] if len(target_history) > 1 else target_history[-1]),
                        "lag_4": float(target_history[-4] if len(target_history) > 3 else target_history[-1]),
                        "level_lag_1": float(level_history[-1]),
                        "level_lag_2": float(level_history[-2] if len(level_history) > 1 else level_history[-1]),
                        "mortality_rate": float(seasonal_driver_lookup["mortality_rate"].get(quarter_of_year, latest_driver_values["mortality_rate"])),
                        "lapse_rate": float(seasonal_driver_lookup["lapse_rate"].get(quarter_of_year, latest_driver_values["lapse_rate"])),
                        "discount_rate": float(seasonal_driver_lookup["discount_rate"].get(quarter_of_year, latest_driver_values["discount_rate"])),
                        "expense_inflation": float(seasonal_driver_lookup["expense_inflation"].get(quarter_of_year, latest_driver_values["expense_inflation"])),
                        "new_business_count": float(seasonal_driver_lookup["new_business_count"].get(quarter_of_year, latest_driver_values["new_business_count"])),
                        "new_business_rate": float(seasonal_driver_lookup["new_business_rate"].get(quarter_of_year, latest_driver_values["new_business_rate"])),
                    }
                )
                state_keys.append(state_key)

            future_frame = pd.DataFrame(step_rows)
            predictions = self._predict_future(
                target_name=target_name,
                model_family=model_family,
                train_frame=train_frame,
                future_frame=future_frame,
            )
            forecast_rows.extend(
                self._build_forecast_rows(
                    target_name=target_name,
                    model_family=model_family,
                    future_frame=future_frame,
                    predictions=predictions,
                    forecast_horizon=horizon,
                )
            )
            for state_key, prediction in zip(state_keys, predictions, strict=False):
                state = states[state_key]
                state["last_quarter_index"] = int(state["last_quarter_index"]) + 1
                state["target_history"].append(float(prediction))
                if target_name in {"reserve_movement", "csm_movement"}:
                    next_level = float(state["level_history"][-1]) + float(prediction)
                else:
                    next_level = float(prediction)
                state["level_history"].append(next_level)

        return forecast_rows

    def _evaluate_predictions(
        self,
        *,
        target_name: str,
        model_family: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        train_observations: int,
    ) -> dict[str, object]:
        safe_r2 = 0.0
        if len(y_true) > 1 and np.unique(y_true).size > 1:
            safe_r2 = max(float(r2_score(y_true, y_pred)), 0.0)

        return {
            "target_name": target_name,
            "model_family": model_family,
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "mape": round(float(self._mean_absolute_percentage_error(y_true, y_pred)), 4),
            "r2": round(safe_r2, 4),
            "train_observations": int(train_observations),
        }

    def _build_backtest_rows(
        self,
        *,
        target_name: str,
        model_family: str,
        test_frame: pd.DataFrame,
        predictions: np.ndarray,
        train_observations: int,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for row, prediction in zip(test_frame.itertuples(index=False), predictions, strict=False):
            rows.append(
                {
                    "quarter": row.quarter,
                    "product": row.product,
                    "region": row.region,
                    "target_name": target_name,
                    "model_family": model_family,
                    "actual_value": round(float(row.target_value), 4),
                    "predicted_value": round(float(prediction), 4),
                    "absolute_error": round(abs(float(row.target_value) - float(prediction)), 4),
                    "train_observations": int(train_observations),
                }
            )
        return rows

    def _summarize_backtest_results(self, target_name: str, backtest_frame: pd.DataFrame) -> list[dict[str, object]]:
        evaluation_rows: list[dict[str, object]] = []
        for model_family, subset in backtest_frame.groupby("model_family"):
            y_true = subset["actual_value"].to_numpy()
            y_pred = subset["predicted_value"].to_numpy()
            evaluation = self._evaluate_predictions(
                target_name=target_name,
                model_family=str(model_family),
                y_true=y_true,
                y_pred=y_pred,
                train_observations=int(subset["train_observations"].max()),
            )
            evaluation["evaluation_quarters"] = int(subset["quarter"].nunique())
            evaluation["backtest_rows"] = int(len(subset))
            evaluation["quality_flag"] = self._quality_flag(evaluation)
            evaluation["selection_metric"] = self.selection_metric
            evaluation["error_tolerance_pct"] = round(float(self.error_tolerance_pct), 4)
            evaluation["gb_max_depth"] = int(self.gb_max_depth)
            evaluation["gb_n_estimators"] = int(self.gb_n_estimators)
            evaluation["gb_learning_rate"] = round(float(self.gb_learning_rate), 4)
            evaluation["forecast_horizon_quarters"] = int(self.forecast_horizon_quarters)
            evaluation_rows.append(evaluation)
        return evaluation_rows

    def _quality_flag(self, evaluation: dict[str, object]) -> str:
        r2 = float(evaluation["r2"])
        mape = float(evaluation["mape"])
        tolerance_pct = max(float(self.error_tolerance_pct), 0.01) * 100.0
        if r2 >= 0.75 and mape <= 12.0:
            return "strong"
        if r2 >= 0.4 and mape <= max(25.0, tolerance_pct):
            return "acceptable"
        return "review"

    def _build_forecast_rows(
        self,
        *,
        target_name: str,
        model_family: str,
        future_frame: pd.DataFrame,
        predictions: np.ndarray,
        forecast_horizon: int,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for row, prediction in zip(future_frame.itertuples(index=False), predictions, strict=False):
            rows.append(
                {
                    "forecast_quarter": row.forecast_quarter,
                    "product": row.product,
                    "region": row.region,
                    "target_name": target_name,
                    "selected_model_family": model_family,
                    "forecast_horizon": int(forecast_horizon),
                    "forecast_value": round(float(prediction), 4),
                }
            )
        return rows

    def _mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        denominator = np.where(np.abs(y_true) > 1e-9, np.abs(y_true), 1.0)
        return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)

    def _quarter_to_index(self, quarter_label: str) -> int:
        year = int(quarter_label[:4])
        quarter = int(quarter_label[-1])
        return year * 4 + quarter

    def _next_quarter_label(self, quarter_label: str) -> str:
        year = int(quarter_label[:4])
        quarter = int(quarter_label[-1])
        if quarter == 4:
            return f"{year + 1}Q1"
        return f"{year}Q{quarter + 1}"

    def _index_to_quarter_label(self, quarter_index: int) -> str:
        year = quarter_index // 4
        quarter = quarter_index % 4
        if quarter == 0:
            year -= 1
            quarter = 4
        return f"{year}Q{quarter}"
