"""Application configuration loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class AppSettings(BaseModel):
    """Application identity and runtime environment."""

    name: str = "ai-insurance-reporting"
    env: str = "dev"


class LoggingSettings(BaseModel):
    """Logging defaults."""

    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


class PathSettings(BaseModel):
    """Filesystem paths used by the project."""

    artifacts_dir: str = "artifacts"
    data_input_dir: str = "artifacts/data/raw"
    data_processed_dir: str = "artifacts/data/processed"
    reports_dir: str = "artifacts/reports"
    figures_dir: str = "artifacts/figures"
    models_dir: str = "artifacts/models"
    logs_dir: str = "artifacts/logs"


class InterfaceSettings(BaseModel):
    """Interface defaults."""

    default: str = "cli"


class ValidationSettings(BaseModel):
    """Controls for validation tolerances and synthetic issue volume."""

    reserve_tolerance_pct: float = 0.03
    reserve_min_tolerance: float = 25_000.0
    csm_tolerance_pct: float = 0.03
    csm_min_tolerance: float = 5_000.0
    capital_tolerance_pct: float = 0.04
    capital_min_tolerance: float = 10_000.0
    synthetic_issue_count: int = 4


class InsightDetectionSettings(BaseModel):
    """Controls for forecast insight detection."""

    historical_window: int = 4
    volatility_epsilon: float = 1.0
    moderate_zscore: float = 0.75
    material_zscore: float = 1.5
    critical_zscore: float = 2.25
    moderate_pct_change: float = 0.05
    material_pct_change: float = 0.12
    critical_pct_change: float = 0.2
    top_n: int = 25


class AnomalyInvestigationSettings(BaseModel):
    """Controls for anomaly investigation heuristics."""

    historical_window: int = 4
    support_score_high: float = 0.75
    support_score_medium: float = 0.5
    scenario_context_lookback: int = 1
    include_explainability: bool = True


class NarrativeQualitySettings(BaseModel):
    """Controls for narrative consistency checks."""

    direction_tolerance: float = 0.01
    severity_material_pct: float = 0.12
    severity_critical_pct: float = 0.2
    driver_match_top_n: int = 5


class ScenarioReportingSettings(BaseModel):
    """Controls for scenario reporting summaries."""

    impact_epsilon: float = 1.0
    materiality_pct: float = 0.05
    top_metrics_count: int = 5
    top_segments_count: int = 10


class MovementAnalysisSettings(BaseModel):
    """Controls for movement bridge calculations and reporting."""

    epsilon: float = 1.0
    top_steps_per_metric: int = 3
    summary_top_n: int = 8
    new_business_premium_weight: float = 180.0
    retention_weight: float = 0.9
    rate_change_weight: float = 0.8
    claims_frequency_weight: float = 1.0
    claims_severity_weight: float = 1.0
    claims_assumption_weight: float = 950.0
    reserve_new_business_weight: float = 32.0
    reserve_claims_weight: float = 0.75
    reserve_economic_weight: float = 1.8
    reserve_runoff_weight: float = 0.18
    csm_assumption_weight: float = 0.55
    capital_underwriting_weight: float = 0.08
    capital_investment_weight: float = 0.6
    capital_reserve_weight: float = 0.04
    capital_csm_weight: float = 0.05


class FullReportSettings(BaseModel):
    """Controls for full management report assembly."""

    max_insights: int = 5
    max_anomalies: int = 5
    max_movement_rows: int = 6
    max_narrative_statements: int = 8
    max_quality_warnings: int = 5
    max_figures: int = 8


class LLMDraftingSettings(BaseModel):
    """Controls for optional external-LLM drafting of management reports."""

    enabled: bool = True
    skip_mock_provider: bool = True
    max_input_sections: int = 12
    max_input_characters: int = 12000


class ChatbotReportingSettings(BaseModel):
    """Controls for chatbot reporting demonstrations and retrieval."""

    demo_query_limit: int = 12
    retrieval_top_k: int = 5


class LLMEvaluationSettings(BaseModel):
    """Controls for deterministic LLM-output evaluation and reviewer feedback."""

    benchmark_query_limit: int = 12
    passing_score: float = 0.7
    grounded_weight: float = 0.4
    artifact_match_weight: float = 0.35
    tool_match_weight: float = 0.15
    citation_weight: float = 0.1
    reviewer_default_status: str = "pending_review"


class AnalystReviewSettings(BaseModel):
    """Controls for the unified analyst review queue."""

    max_items_per_source: int = 20
    include_normal_insights: bool = False
    min_anomaly_support_score: float = 0.5
    narrative_warning_priority: float = 0.7
    min_llm_review_gap: float = 0.15
    medium_priority_score: float = 0.4
    high_priority_score: float = 0.65
    critical_priority_score: float = 0.85


class ReportingSettings(BaseModel):
    """Configuration for reporting-support AI layers."""

    insight_detection: InsightDetectionSettings = InsightDetectionSettings()
    anomaly_investigation: AnomalyInvestigationSettings = AnomalyInvestigationSettings()
    narrative_quality: NarrativeQualitySettings = NarrativeQualitySettings()
    scenario_reporting: ScenarioReportingSettings = ScenarioReportingSettings()
    movement_analysis: MovementAnalysisSettings = MovementAnalysisSettings()
    full_report: FullReportSettings = FullReportSettings()
    llm_drafting: LLMDraftingSettings = LLMDraftingSettings()
    chatbot: ChatbotReportingSettings = ChatbotReportingSettings()
    llm_evaluation: LLMEvaluationSettings = LLMEvaluationSettings()
    analyst_review: AnalystReviewSettings = AnalystReviewSettings()


class AppConfig(BaseModel):
    """Top-level application configuration."""

    model_config = ConfigDict(extra="ignore")

    app: AppSettings = AppSettings()
    logging: LoggingSettings = LoggingSettings()
    paths: PathSettings = PathSettings()
    interfaces: InterfaceSettings = InterfaceSettings()
    validation: ValidationSettings = ValidationSettings()
    reporting: ReportingSettings = ReportingSettings()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file into a dictionary."""

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must contain a mapping: {path}")

    return data


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Apply a small set of explicit environment variable overrides."""

    env_map = {
        "AIR_ENV": ("app", "env"),
        "AIR_LOG_LEVEL": ("logging", "level"),
        "AIR_ARTIFACTS_DIR": ("paths", "artifacts_dir"),
        "AIR_DATA_INPUT_DIR": ("paths", "data_input_dir"),
        "AIR_DATA_PROCESSED_DIR": ("paths", "data_processed_dir"),
        "AIR_INTERFACE_DEFAULT": ("interfaces", "default"),
    }

    for env_name, path_parts in env_map.items():
        env_value = os.getenv(env_name)
        if env_value is None:
            continue

        target = config
        for part in path_parts[:-1]:
            target = target.setdefault(part, {})
        target[path_parts[-1]] = env_value

    # If the artifact root is redirected, keep the other artifact subpaths aligned
    # unless they were explicitly overridden elsewhere.
    artifact_root = os.getenv("AIR_ARTIFACTS_DIR")
    if artifact_root:
        paths = config.setdefault("paths", {})
        if os.getenv("AIR_REPORTS_DIR") is None:
            paths["reports_dir"] = str(Path(artifact_root) / "reports")
        if os.getenv("AIR_FIGURES_DIR") is None:
            paths["figures_dir"] = str(Path(artifact_root) / "figures")
        if os.getenv("AIR_MODELS_DIR") is None:
            paths["models_dir"] = str(Path(artifact_root) / "models")
        if os.getenv("AIR_LOGS_DIR") is None:
            paths["logs_dir"] = str(Path(artifact_root) / "logs")

    return config


def get_project_root() -> Path:
    """Return the repository root from the package location."""

    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configuration from the default YAML file and optional overrides."""

    root = get_project_root()
    default_path = root / "configs" / "default.yaml"

    config_data = _load_yaml(default_path)
    if config_path is not None:
        override_data = _load_yaml(Path(config_path))
        config_data = _deep_merge(config_data, override_data)

    config_data = _apply_env_overrides(config_data)
    return AppConfig.model_validate(config_data)


