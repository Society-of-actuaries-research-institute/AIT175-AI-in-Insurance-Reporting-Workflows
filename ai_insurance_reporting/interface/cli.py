"""CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

import typer

from ai_insurance_reporting.chatbot.agent import ReportingAssistantAgent
from ai_insurance_reporting.chatbot.retrieval import ReportingChatbot
from ai_insurance_reporting.config.loader import load_config
from ai_insurance_reporting.data.etl import InsuranceETLPipeline
from ai_insurance_reporting.data.synthetic import SyntheticDataGenerator
from ai_insurance_reporting.data.validation import ReportingValidationEngine
from ai_insurance_reporting.explainability.reporting import ExplainabilityPipeline
from ai_insurance_reporting.models.forecasting import ForecastingPipeline
from ai_insurance_reporting.narrative.generator import NarrativeGenerator
from ai_insurance_reporting.orchestration.workflow import WorkflowOrchestrator
from ai_insurance_reporting.reporting.llm_evaluation import LLMEvaluator
from ai_insurance_reporting.utils.file_formats import normalize_tabular_output_format
from ai_insurance_reporting.utils.logging_utils import get_logger, setup_logging
from ai_insurance_reporting.visualization.charts import VisualizationGenerator

app = typer.Typer(help="AI insurance reporting case study CLI.")


def _setup() -> tuple[object, object]:
    config = load_config()
    setup_logging(config)
    return config, get_logger(__name__)


def _build_chatbot_state(
    *,
    file_format: str,
    premium_multiplier: float,
    claims_multiplier: float,
    reserve_multiplier: float,
    csm_multiplier: float,
    asset_return_shift: float,
    capital_multiplier: float,
    reserve_tolerance: float,
    csm_tolerance: float,
    capital_tolerance: float,
    forecast_selection_metric: str,
    forecast_error_tolerance_pct: float,
    forecast_gb_max_depth: int,
    forecast_gb_n_estimators: int,
    forecast_gb_learning_rate: float,
    forecast_horizon_quarters: int,
    policy_data_path: str | None,
    claims_data_path: str | None,
    asset_data_path: str | None,
    financial_balances_path: str | None,
    reporting_metrics_path: str | None,
) -> dict[str, object]:
    workflow_assumption_overrides = {
        key: value
        for key, value in {
            "premium_multiplier": premium_multiplier,
            "claims_multiplier": claims_multiplier,
            "reserve_multiplier": reserve_multiplier,
            "csm_multiplier": csm_multiplier,
            "asset_return_shift": asset_return_shift,
            "capital_multiplier": capital_multiplier,
        }.items()
        if abs(value - (0.0 if key == "asset_return_shift" else 1.0)) > 1e-9
    }
    validation_override_params = {
        key: value
        for key, value in {
            "reserve_tolerance": reserve_tolerance,
            "csm_tolerance": csm_tolerance,
            "capital_tolerance": capital_tolerance,
        }.items()
        if abs(value - 0.05) > 1e-9
    }
    forecast_override_params = {
        key: value
        for key, value in {
            "selection_metric": forecast_selection_metric,
            "error_tolerance_pct": forecast_error_tolerance_pct,
            "gb_max_depth": forecast_gb_max_depth,
            "gb_n_estimators": forecast_gb_n_estimators,
            "gb_learning_rate": forecast_gb_learning_rate,
            "forecast_horizon_quarters": forecast_horizon_quarters,
        }.items()
        if (
            (key == "selection_metric" and value != "mae")
            or (key == "error_tolerance_pct" and abs(float(value) - 0.25) > 1e-9)
            or (key == "gb_max_depth" and int(value) != 3)
            or (key == "gb_n_estimators" and int(value) != 100)
            or (key == "gb_learning_rate" and abs(float(value) - 0.10) > 1e-9)
            or (key == "forecast_horizon_quarters" and int(value) != 1)
        )
    }
    uploaded_raw_paths = {
        key: str(Path(path))
        for key, path in {
            "policy_data": policy_data_path,
            "claims_data": claims_data_path,
            "asset_data": asset_data_path,
            "financial_balances": financial_balances_path,
            "reporting_metrics": reporting_metrics_path,
        }.items()
        if path
    }
    return {
        "file_format": file_format,
        "workflow_assumption_overrides": workflow_assumption_overrides,
        "validation_override_params": validation_override_params,
        "forecast_override_params": forecast_override_params,
        "uploaded_raw_paths": uploaded_raw_paths,
    }


@app.command("run")
def run_case_study(file_format: str = "csv") -> None:
    """Run the full case study workflow."""

    file_format = normalize_tabular_output_format(file_format)
    config, logger = _setup()
    result = WorkflowOrchestrator(config=config).run(file_format=file_format)
    logger.info("Completed workflow with %s agents", len(result.execution_log))
    typer.echo(f"Completed full workflow with {len(result.execution_log)} agents.")


@app.command("generate-data")
def generate_data(file_format: str = "csv") -> None:
    """Generate synthetic raw data and the cleaned reporting dataset."""

    file_format = normalize_tabular_output_format(file_format)
    config, logger = _setup()
    raw_paths = SyntheticDataGenerator(config=config).write(file_format=file_format)
    _, curated_path = InsuranceETLPipeline(config=config).run(file_format=file_format)
    logger.info("Generated %s raw datasets and cleaned dataset", len(raw_paths))
    typer.echo(f"Generated data and cleaned dataset at: {curated_path}")


@app.command("validate")
def validate(file_format: str = "csv") -> None:
    """Run validation on the cleaned reporting dataset."""

    file_format = normalize_tabular_output_format(file_format)
    config, logger = _setup()
    curated_dataset = ForecastingPipeline(config=config).load_curated_dataset(file_format=file_format)
    result, _ = ReportingValidationEngine(config=config).run(curated_dataset, file_format=file_format)
    issues = int(result.validation_flags["has_validation_issue"].sum())
    logger.info("Validation completed with %s flagged records", issues)
    typer.echo(f"Validation completed. Flagged records: {issues}")


@app.command("forecast")
def forecast(file_format: str = "csv", forecast_horizon_quarters: int = 1) -> None:
    """Train forecasts and persist output tables."""

    file_format = normalize_tabular_output_format(file_format)
    config, logger = _setup()
    result, _ = ForecastingPipeline(config=config, forecast_horizon_quarters=forecast_horizon_quarters).run(file_format=file_format)
    logger.info("Forecasting completed for targets: %s", ", ".join(sorted(result.evaluation_table["target_name"].unique())))
    typer.echo("Forecasting outputs generated.")


@app.command("explain")
def explain(file_format: str = "csv") -> None:
    """Generate explainability outputs."""

    file_format = normalize_tabular_output_format(file_format)
    config, logger = _setup()
    result, _ = ExplainabilityPipeline(config=config).run(file_format=file_format)
    logger.info("Explainability completed with %s SHAP global rows", len(result.shap_global_importance))
    typer.echo("Explainability outputs generated.")


@app.command("narrate")
def narrate(file_format: str = "csv") -> None:
    """Generate management narrative outputs."""

    file_format = normalize_tabular_output_format(file_format)
    config, logger = _setup()
    result, _ = NarrativeGenerator(config=config).run(file_format=file_format)
    logger.info("Narrative generation completed with %s statements", len(result.narrative_statements))
    typer.echo("Narrative outputs generated.")


@app.command("visualize")
def visualize(file_format: str = "csv") -> None:
    """Generate charts and figure metadata."""

    file_format = normalize_tabular_output_format(file_format)
    config, logger = _setup()
    result, _ = VisualizationGenerator(config=config).run(file_format=file_format)
    logger.info("Visualization completed with %s charts", len(result.figure_paths))
    typer.echo("Visualization outputs generated.")


@app.command("chatbot-index")
def chatbot_index(file_format: str = "csv") -> None:
    """Build the chatbot retrieval and semantic index."""

    file_format = normalize_tabular_output_format(file_format)
    config, logger = _setup()
    result, _ = ReportingChatbot(config=config).build_index(file_format=file_format)
    logger.info("Chatbot index built with %s documents", len(result.chatbot_index))
    typer.echo("Chatbot index generated.")


@app.command("evaluate-llm")
def evaluate_llm(file_format: str = "csv") -> None:
    """Run deterministic benchmark evaluation over grounded LLM outputs."""

    file_format = normalize_tabular_output_format(file_format)
    config, logger = _setup()
    result, _ = LLMEvaluator(config=config).run(file_format=file_format)
    logger.info("LLM evaluation completed with %s benchmark queries", len(result.llm_eval_results))
    typer.echo(
        f"LLM evaluation completed. Benchmark queries: {len(result.llm_eval_results)} | "
        f"Pass rate: {result.llm_eval_summary.get('pass_rate', 0.0):.2%}"
    )


@app.command("chatbot-ask")
def chatbot_ask(
    question: str,
    file_format: str = "csv",
    top_k: int = 5,
    show_tools: bool = False,
    premium_multiplier: float = 1.0,
    claims_multiplier: float = 1.0,
    reserve_multiplier: float = 1.0,
    csm_multiplier: float = 1.0,
    asset_return_shift: float = 0.0,
    capital_multiplier: float = 1.0,
    reserve_tolerance: float = 0.05,
    csm_tolerance: float = 0.05,
    capital_tolerance: float = 0.05,
    forecast_selection_metric: str = "mae",
    forecast_error_tolerance_pct: float = 0.25,
    forecast_gb_max_depth: int = 3,
    forecast_gb_n_estimators: int = 100,
    forecast_gb_learning_rate: float = 0.10,
    forecast_horizon_quarters: int = 1,
    policy_data_path: str | None = None,
    claims_data_path: str | None = None,
    asset_data_path: str | None = None,
    financial_balances_path: str | None = None,
    reporting_metrics_path: str | None = None,
) -> None:
    """Ask the agentic reporting assistant over generated artifacts."""

    file_format = normalize_tabular_output_format(file_format)
    config, logger = _setup()
    state = _build_chatbot_state(
        file_format=file_format,
        premium_multiplier=premium_multiplier,
        claims_multiplier=claims_multiplier,
        reserve_multiplier=reserve_multiplier,
        csm_multiplier=csm_multiplier,
        asset_return_shift=asset_return_shift,
        capital_multiplier=capital_multiplier,
        reserve_tolerance=reserve_tolerance,
        csm_tolerance=csm_tolerance,
        capital_tolerance=capital_tolerance,
        forecast_selection_metric=forecast_selection_metric,
        forecast_error_tolerance_pct=forecast_error_tolerance_pct,
        forecast_gb_max_depth=forecast_gb_max_depth,
        forecast_gb_n_estimators=forecast_gb_n_estimators,
        forecast_gb_learning_rate=forecast_gb_learning_rate,
        forecast_horizon_quarters=forecast_horizon_quarters,
        policy_data_path=policy_data_path,
        claims_data_path=claims_data_path,
        asset_data_path=asset_data_path,
        financial_balances_path=financial_balances_path,
        reporting_metrics_path=reporting_metrics_path,
    )
    response = ReportingAssistantAgent(config=config).answer(
        question,
        state=state,
        file_format=file_format,
        top_k=top_k,
        prompt_mode="management_qa",
    )
    logger.info("Answered chatbot question with %s sources and %s tools", len(response.sources), len(response.tools_used))
    typer.echo(f"Answer: {response.answer}")
    if response.citations:
        typer.echo(f"Citations: {' '.join(f'[{citation}]' for citation in response.citations)}")
    if response.sources:
        typer.echo("\nSources:")
        for source in response.sources:
            typer.echo(
                f"- {source['document']} ({source['source_dataset']}) "
                f"[score={source['score']}]"
            )
    if response.tools_used:
        typer.echo("\nTools used:")
        for tool_name in response.tools_used:
            typer.echo(f"- {tool_name}")
    if show_tools and response.tool_outputs:
        typer.echo("\nTool outputs:")
        for tool_name, payload in response.tool_outputs.items():
            typer.echo(f"{tool_name}: {payload}")


if __name__ == "__main__":
    app()
