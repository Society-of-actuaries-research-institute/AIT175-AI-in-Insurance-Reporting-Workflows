"""Artifact path helpers."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from ai_insurance_reporting.config.loader import AppConfig, get_project_root


class ArtifactPaths(BaseModel):
    """Resolved artifact directories."""

    root: Path
    data_input: Path
    data_processed: Path
    reports: Path
    figures: Path
    models: Path
    logs: Path


def resolve_artifact_paths(config: AppConfig) -> ArtifactPaths:
    """Resolve configured artifact paths relative to the project root."""

    project_root = get_project_root()

    def _resolve(path_value: str) -> Path:
        path = Path(path_value)
        return path if path.is_absolute() else project_root / path

    return ArtifactPaths(
        root=_resolve(config.paths.artifacts_dir),
        data_input=_resolve(config.paths.data_input_dir),
        data_processed=_resolve(config.paths.data_processed_dir),
        reports=_resolve(config.paths.reports_dir),
        figures=_resolve(config.paths.figures_dir),
        models=_resolve(config.paths.models_dir),
        logs=_resolve(config.paths.logs_dir),
    )


def ensure_artifact_dirs(config: AppConfig) -> ArtifactPaths:
    """Create configured artifact directories if they do not already exist."""

    paths = resolve_artifact_paths(config)
    for path in paths.model_dump().values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return paths
