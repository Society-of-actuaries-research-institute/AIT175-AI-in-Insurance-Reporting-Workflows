"""Helpers for normalizing supported tabular output formats."""

from __future__ import annotations

from typing import Literal

TabularOutputFormat = Literal["csv"]

def normalize_tabular_output_format(file_format: str = "csv") -> TabularOutputFormat:
    """Return the supported tabular output format for official workflow runs.

    Parquet output is intentionally retired in favour of CSV so that generated
    artifacts remain easy to inspect and compatible with the dashboard and
    report documentation.
    """

    return "csv"
