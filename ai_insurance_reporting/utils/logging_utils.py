"""Logging utilities."""

from __future__ import annotations

import logging

from ai_insurance_reporting.config.loader import AppConfig


def setup_logging(config: AppConfig) -> None:
    """Configure application logging from the loaded config."""

    level_name = config.logging.level.upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format=config.logging.format,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""

    return logging.getLogger(name)
