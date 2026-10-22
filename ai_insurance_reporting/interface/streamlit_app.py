"""Streamlit dashboard for the case study outputs."""

from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from ai_insurance_reporting.chatbot.agent import ReportingAssistantAgent
from ai_insurance_reporting.chatbot.llm_client import GeminiLLMClient, MockLLMClient, OpenAILLMClient
from ai_insurance_reporting.chatbot.rag_pipeline import RAGPipeline
from ai_insurance_reporting.config.loader import load_config
from ai_insurance_reporting.reporting.llm_evaluation import LLMEvaluator
from ai_insurance_reporting.reporting.review_queue import apply_review_status_updates, load_review_status_updates, record_review_status
from ai_insurance_reporting.utils.artifacts import ensure_artifact_dirs


PAGES = [
    "Data overview",
    "Validation results",
    "Forecast results",
    "Explainability",
    "Narrative reporting",
    "Visualizations",
    "Scenario comparison",
    "Analyst Review",
    "Chatbot Q&A",
    "LLM Evaluation",
]

THEMES: dict[str, dict[str, str]] = {
    "Executive Light": {
        "bg": "#f4efe6",
        "panel": "rgba(255, 252, 247, 0.86)",
        "panel_strong": "rgba(255, 248, 239, 0.96)",
        "text": "#1f2933",
        "muted": "#566372",
        "accent": "#0d6c74",
        "accent_2": "#b45f2f",
        "border": "rgba(31, 41, 51, 0.10)",
        "shadow": "0 18px 40px rgba(30, 41, 59, 0.10)",
        "hero": "linear-gradient(135deg, rgba(13,108,116,0.16), rgba(180,95,47,0.18))",
    },
    "Executive Dark": {
        "bg": "#0f1720",
        "panel": "rgba(19, 30, 40, 0.86)",
        "panel_strong": "rgba(24, 38, 51, 0.96)",
        "text": "#e8edf2",
        "muted": "#9db0bd",
        "accent": "#52c7cf",
        "accent_2": "#f2a65a",
        "border": "rgba(232, 237, 242, 0.12)",
        "shadow": "0 18px 40px rgba(0, 0, 0, 0.35)",
        "hero": "linear-gradient(135deg, rgba(82,199,207,0.18), rgba(242,166,90,0.12))",
    },
}

LLM_PROVIDERS = ("mock", "openai", "gemini")
DEFAULT_PROVIDER_MODELS = {
    "mock": "",
    "openai": "gpt-5.1",
    "gemini": "gemini-3-flash-preview",
}
PROVIDER_MODEL_OPTIONS = {
    "mock": ["mock"],
    "openai": [
        "gpt-5.1",
        "gpt-5-mini",
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
        "gpt-4.1",
    ],
    "gemini": [
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-flash-latest",
    ],
}


def load_table(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return None


def find_existing_path(directory: Path, stem: str) -> Path | None:
    for extension in (".csv",):
        candidate = directory / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def list_scenario_roots(artifacts_root: Path) -> list[Path]:
    scenarios_root = artifacts_root / "scenarios"
    if not scenarios_root.exists():
        return []
    return sorted([path for path in scenarios_root.iterdir() if path.is_dir()], key=lambda item: item.name)


def file_bytes(path: Path) -> bytes:
    return path.read_bytes()


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_sidebar_open_on_first_load() -> None:
    """Reopen the sidebar once per session if the browser restored it collapsed."""

    if st.session_state.get("_sidebar_default_open_applied", False):
        return

    components.html(
        """
        <script>
        (function() {
          const selectors = [
            '[data-testid="stSidebarCollapsedControl"] button',
            '[data-testid="stSidebarCollapsedControl"]',
            'button[aria-label*="sidebar"]',
            'button[title*="sidebar"]'
          ];
          const expandedSidebar = document.querySelector('[data-testid="stSidebar"][aria-expanded="true"]');
          if (expandedSidebar) {
            return;
          }
          for (const selector of selectors) {
            const control = window.parent.document.querySelector(selector);
            if (control) {
              control.click();
              break;
            }
          }
        })();
        </script>
        """,
        height=0,
        width=0,
    )
    st.session_state["_sidebar_default_open_applied"] = True


def format_scenario_change(label: str, value: float, *, is_shift: bool = False) -> str | None:
    if is_shift:
        if abs(value) < 1e-12:
            return None
        return f"{label} {value:+.0%}"

    if abs(value - 1.0) < 1e-12:
        return None
    change_pct = (value - 1.0) * 100.0
    direction = "up" if change_pct > 0 else "down"
    return f"{label} {direction} {abs(change_pct):.0f}%"


def format_identifier_label(value: str) -> str:
    """Convert stored slug-like identifiers into a cleaner display label."""

    return str(value or "").replace("-", " ").replace("_", " ").strip().title()


def format_metric_label(value: str) -> str:
    """Convert internal metric and target names into cleaner dashboard labels."""

    mapping = {
        "capital_ratio": "Synthetic Capital Proxy Ratio",
        "premium": "Premium",
        "premium_income": "Premium Income",
        "claims": "Claims",
        "total_claims": "Total Claims",
        "reserve_movement": "Reserve Movement",
        "reserves": "Reserves",
        "csm_movement": "CSM Movement",
        "csm_closing": "Closing CSM",
        "asset_investment_income": "Investment Income",
        "average_asset_return": "Average Asset Return",
    }
    key = str(value or "").strip()
    return mapping.get(key, format_identifier_label(key))


def format_display_frame(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    """Return a UI-friendly copy of a table without changing stored artifacts."""

    if frame is None or frame.empty:
        return frame
    display = frame.copy()
    metric_columns = ["target_name", "metric_name", "metric", "linked_metric"]
    for column in metric_columns:
        if column in display.columns:
            display[column] = display[column].apply(lambda value: format_metric_label(str(value)))
    for column in ["priority_label", "review_source", "reviewer_status", "model_family"]:
        if column in display.columns:
            display[column] = display[column].apply(lambda value: format_identifier_label(str(value)))
    return display


def strip_traceability_lines(markdown_text: str) -> str:
    """Return a clean reading view of a report by hiding inline traceability lines."""

    lines = []
    for line in str(markdown_text or "").splitlines():
        if line.strip().startswith("Traceability:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def format_scenario_label(scenario_name: str, scenario_metadata: dict[str, Any] | None) -> str:
    if scenario_metadata is None:
        return format_identifier_label(scenario_name)

    params = scenario_metadata.get("scenario_parameters", {})
    changes = [
        format_scenario_change("Claims", float(params.get("claims_multiplier", 1.0))),
        format_scenario_change("Premium", float(params.get("premium_multiplier", 1.0))),
        format_scenario_change("Reserves", float(params.get("reserve_multiplier", 1.0))),
        format_scenario_change("CSM", float(params.get("csm_multiplier", 1.0))),
        format_scenario_change("Capital", float(params.get("capital_multiplier", 1.0))),
        format_scenario_change("Asset returns", float(params.get("asset_return_shift", 0.0)), is_shift=True),
    ]
    summary_parts = [change for change in changes if change]
    if summary_parts:
        return ", ".join(summary_parts)
    return "No material scenario overrides"


def save_uploaded_raw_files(paths: Any, uploads: dict[str, Any]) -> dict[str, str]:
    uploaded_dir = paths.root / "chatbot" / "uploaded_raw"
    uploaded_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, str] = {}
    for dataset_name, uploaded_file in uploads.items():
        if uploaded_file is None:
            continue
        suffix = Path(uploaded_file.name).suffix or ".csv"
        destination = uploaded_dir / f"{dataset_name}{suffix}"
        destination.write_bytes(uploaded_file.getbuffer())
        saved_paths[dataset_name] = str(destination)
    return saved_paths


def build_streamlit_llm_client(provider: str, model_name: str, api_key: str | None) -> Any:
    provider = provider.lower()
    if provider == "mock":
        return MockLLMClient()

    clean_model_name = model_name.strip() or DEFAULT_PROVIDER_MODELS[provider]
    if provider == "openai":
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        return OpenAILLMClient(model=clean_model_name)
    if provider == "gemini":
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
        return GeminiLLMClient(model=clean_model_name)
    raise ValueError(f"Unsupported LLM provider: {provider}")


def build_streamlit_assistant(config: Any, provider: str, model_name: str, api_key: str | None) -> ReportingAssistantAgent:
    llm_client = build_streamlit_llm_client(provider, model_name, api_key)
    rag_pipeline = RAGPipeline(config=config, llm_client=llm_client)
    return ReportingAssistantAgent(config=config, rag_pipeline=rag_pipeline)


def apply_theme(theme_name: str) -> dict[str, str]:
    theme = THEMES[theme_name]
    css = f"""
    <style>
      .stApp {{
        background:
          radial-gradient(circle at top left, rgba(255,255,255,0.06), transparent 26%),
          radial-gradient(circle at 85% 15%, rgba(255,255,255,0.05), transparent 22%),
          {theme["bg"]};
        color: {theme["text"]};
      }}
      header[data-testid="stHeader"] {{
        background: transparent !important;
      }}
      [data-testid="stDecoration"],
      [data-testid="stDeployButton"] {{
        display: none !important;
      }}
      [data-testid="stToolbar"] * {{
        color: {theme["text"]} !important;
        fill: {theme["text"]} !important;
      }}
      [data-testid="stToolbar"] {{
        opacity: 1 !important;
        visibility: visible !important;
        pointer-events: auto !important;
      }}
      [data-testid="stSidebarCollapsedControl"] {{
        position: fixed !important;
        top: 0.45rem !important;
        left: 0.65rem !important;
        z-index: 1002 !important;
      }}
      [data-testid="stToolbar"] button,
      [data-testid="stToolbar"] a,
      [data-testid="stDecoration"] {{
        color: {theme["text"]} !important;
      }}
      [data-testid="stSidebarCollapsedControl"] button,
      [data-testid="stSidebarCollapsedControl"] [role="button"] {{
        background: {theme["panel"]} !important;
        border: 1px solid {theme["border"]} !important;
        border-radius: 999px !important;
        min-width: 2.5rem !important;
        min-height: 2.5rem !important;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.18) !important;
      }}
      [data-testid="stSidebarCollapsedControl"] button:hover,
      [data-testid="stSidebarCollapsedControl"] [role="button"]:hover {{
        border-color: {theme["accent"]} !important;
        background: {theme["panel_strong"]} !important;
      }}
      [data-testid="stStatusWidget"] *,
      [data-testid="stMainMenu"] *,
      [data-testid="stDeployButton"] * {{
        color: {theme["text"]} !important;
        fill: {theme["text"]} !important;
      }}
      [data-testid="stHeaderActionElements"] * {{
        color: {theme["text"]} !important;
        fill: {theme["text"]} !important;
      }}
      [data-testid="stHeaderActionElements"] button,
      [data-testid="stHeaderActionElements"] a,
      [data-testid="stHeaderActionElements"] > div,
      [data-testid="stDeployButton"],
      [data-testid="stDeployButton"] > div,
      [data-testid="stDeployButton"] button,
      [data-testid="stDeployButton"] a,
      [data-testid="stDeployButton"] [role="button"] {{
        background: {theme["panel"]} !important;
        border: 1px solid {theme["border"]} !important;
        color: {theme["text"]} !important;
        border-radius: 12px !important;
        box-shadow: none !important;
      }}
      [data-testid="stHeaderActionElements"] button:hover,
      [data-testid="stHeaderActionElements"] a:hover,
      [data-testid="stHeaderActionElements"] > div:hover,
      [data-testid="stDeployButton"]:hover,
      [data-testid="stDeployButton"] > div:hover,
      [data-testid="stDeployButton"] button:hover,
      [data-testid="stDeployButton"] a:hover,
      [data-testid="stDeployButton"] [role="button"]:hover {{
        border-color: {theme["accent"]} !important;
        background: {theme["panel_strong"]} !important;
      }}
      [data-testid="stHeaderActionElements"] svg,
      [data-testid="stDeployButton"] svg,
      [data-testid="stDeployButton"] *[data-testid="stIconMaterial"] {{
        fill: {theme["text"]} !important;
        color: {theme["text"]} !important;
      }}
      .block-container {{
        padding-top: 0.75rem;
        padding-bottom: 2rem;
        max-width: 1420px;
      }}
      h1, h2, h3 {{
        color: {theme["text"]};
        letter-spacing: -0.02em;
      }}
      p, label, span, div {{
        color: {theme["text"]};
      }}
      .stCaption, .stMarkdown, .stText, small {{
        color: {theme["muted"]};
      }}
      .air-hero {{
        background: {theme["hero"]};
        border: 1px solid {theme["border"]};
        border-radius: 24px;
        padding: 1.35rem 1.5rem;
        box-shadow: {theme["shadow"]};
        margin-bottom: 1rem;
      }}
      .air-hero-title {{
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
      }}
      .air-hero-sub {{
        color: {theme["muted"]};
        font-size: 1rem;
      }}
      .air-card {{
        background: {theme["panel"]};
        border: 1px solid {theme["border"]};
        border-radius: 20px;
        padding: 1rem 1.1rem;
        box-shadow: {theme["shadow"]};
        backdrop-filter: blur(10px);
        min-height: 126px;
        margin-bottom: 0.75rem;
      }}
      .air-card strong {{
        color: {theme["text"]};
      }}
      .air-label {{
        color: {theme["muted"]};
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }}
      .air-value {{
        color: {theme["text"]};
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 0.2rem;
      }}
      .air-note {{
        color: {theme["muted"]};
        font-size: 0.92rem;
        margin-top: 0.35rem;
      }}
      .air-section {{
        background: {theme["panel_strong"]};
        border: 1px solid {theme["border"]};
        border-radius: 22px;
        padding: 1rem 1.2rem 1.2rem 1.2rem;
        box-shadow: {theme["shadow"]};
        margin-bottom: 1rem;
      }}
      .air-chip {{
        display: inline-block;
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        background: rgba(82, 199, 207, 0.10);
        border: 1px solid {theme["border"]};
        color: {theme["text"]};
        font-size: 0.82rem;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
      }}
      .air-answer {{
        background: {theme["panel_strong"]};
        border-left: 4px solid {theme["accent"]};
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: {theme["shadow"]};
      }}
      .air-figure {{
        background: {theme["panel"]};
        border: 1px solid {theme["border"]};
        border-radius: 18px;
        padding: 0.85rem 0.95rem 0.3rem 0.95rem;
        margin-bottom: 1rem;
        box-shadow: {theme["shadow"]};
      }}
      .air-kicker {{
        color: {theme["accent"]};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.74rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
      }}
      .air-muted {{
        color: {theme["muted"]};
      }}
      [data-testid="stSidebar"] {{
        background: {theme["panel_strong"]};
        border-right: 1px solid {theme["border"]};
      }}
      [data-testid="stSidebar"] * {{
        color: {theme["text"]} !important;
      }}
      [data-testid="stSidebar"] .stCaption {{
        color: {theme["muted"]} !important;
      }}
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] [role="radiogroup"] label,
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: {theme["text"]} !important;
      }}
      div[data-testid="stMetric"] {{
        background: transparent;
      }}
      div[data-testid="stButton"] > button,
      div[data-testid="stDownloadButton"] > button {{
        background: {theme["panel"]} !important;
        color: {theme["text"]} !important;
        border: 1px solid {theme["border"]} !important;
        border-radius: 14px !important;
      }}
      div[data-testid="stButton"] > button:hover,
      div[data-testid="stDownloadButton"] > button:hover {{
        border-color: {theme["accent"]} !important;
        box-shadow: 0 0 0 1px {theme["accent"]} inset !important;
      }}
      div[data-testid="stButton"] > button[kind="primary"] {{
        background: {theme["accent"]} !important;
        color: {theme["bg"]} !important;
        border-color: {theme["accent"]} !important;
      }}
      div[data-testid="stButton"] > button p,
      div[data-testid="stDownloadButton"] > button p,
      div[data-testid="stButton"] > button span,
      div[data-testid="stDownloadButton"] > button span {{
        color: inherit !important;
      }}
      div[data-testid="stExpander"] details summary p,
      div[data-testid="stExpander"] details summary span {{
        color: {theme["text"]} !important;
      }}
      div[data-baseweb="select"] > div,
      div[data-baseweb="base-input"] > div,
      textarea,
      input {{
        background: {theme["panel"]} !important;
        color: {theme["text"]} !important;
      }}
      div[role="listbox"] {{
        background: {theme["panel_strong"]} !important;
        border: 1px solid {theme["border"]} !important;
        box-shadow: {theme["shadow"]} !important;
      }}
      [data-testid="stPopover"],
      [data-testid="stPopover"] > div,
      [data-baseweb="popover"],
      [data-baseweb="popover"] > div,
      [data-baseweb="menu"],
      [data-baseweb="menu"] > div,
      div[role="menu"],
      div[role="dialog"] {{
        background: {theme["panel_strong"]} !important;
        color: {theme["text"]} !important;
        border: 1px solid {theme["border"]} !important;
        box-shadow: {theme["shadow"]} !important;
      }}
      div[role="menuitem"],
      div[role="menuitem"] > div,
      [data-baseweb="menu"] li,
      [data-baseweb="menu"] button,
      [data-baseweb="menu"] span,
      [data-baseweb="menu"] p {{
        background: transparent !important;
        color: {theme["text"]} !important;
      }}
      div[role="menuitem"]:hover,
      div[role="menuitem"]:focus,
      [data-baseweb="menu"] li:hover,
      [data-baseweb="menu"] button:hover {{
        background: rgba(82, 199, 207, 0.12) !important;
        color: {theme["text"]} !important;
      }}
      [data-baseweb="menu"] hr,
      div[role="separator"] {{
        border-color: {theme["border"]} !important;
        background: {theme["border"]} !important;
      }}
      [data-baseweb="menu"] input,
      [data-baseweb="menu"] label,
      [data-baseweb="menu"] svg,
      div[role="menu"] input,
      div[role="menu"] label,
      div[role="menu"] svg {{
        color: {theme["text"]} !important;
        fill: {theme["text"]} !important;
      }}
      [data-baseweb="menu"] [aria-checked="true"],
      [data-baseweb="menu"] [data-checked="true"] {{
        background: rgba(82, 199, 207, 0.18) !important;
      }}
      div[role="option"] {{
        background: {theme["panel"]} !important;
        color: {theme["text"]} !important;
      }}
      div[role="listbox"] *,
      ul[role="listbox"] *,
      li[role="option"] *,
      div[role="option"] * {{
        color: {theme["text"]} !important;
      }}
      ul[role="listbox"],
      li[role="option"] {{
        background: {theme["panel_strong"]} !important;
        color: {theme["text"]} !important;
      }}
      div[role="option"][aria-selected="true"] {{
        background: rgba(82, 199, 207, 0.18) !important;
        color: {theme["text"]} !important;
      }}
      div[role="option"]:hover {{
        background: rgba(82, 199, 207, 0.12) !important;
        color: {theme["text"]} !important;
      }}
      [data-baseweb="popover"] *,
      [data-baseweb="menu"] *,
      [role="presentation"] [role="listbox"] * {{
        color: {theme["text"]} !important;
      }}
      div[data-baseweb="slider"] * {{
        color: {theme["text"]} !important;
      }}
      div[data-testid="stWidgetLabel"] * {{
        color: {theme["text"]} !important;
      }}
      div[data-testid="stRadio"] label *,
      div[data-testid="stSelectbox"] label *,
      div[data-testid="stTextArea"] label *,
      div[data-testid="stSlider"] label *,
      div[data-testid="stToggle"] label * {{
        color: {theme["text"]} !important;
      }}
      code {{
        background: rgba(255,255,255,0.14);
        color: {theme["text"]};
        border: 1px solid {theme["border"]};
      }}
      @media (max-width: 900px) {{
        .block-container {{
          padding-left: 1rem;
          padding-right: 1rem;
        }}
        .air-hero-title {{
          font-size: 1.45rem;
        }}
        .air-card {{
          min-height: 108px;
        }}
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    return theme


def render_page_header(title: str, subtitle: str, *, chips: list[str] | None = None) -> None:
    st.markdown(
        f"""
        <div class="air-hero">
          <div class="air-hero-title">{title}</div>
          <div class="air-hero-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if chips:
        st.markdown("".join(f'<span class="air-chip">{chip}</span>' for chip in chips), unsafe_allow_html=True)


def render_page_explainer(title: str, items: list[tuple[str, str]]) -> None:
    """Render a compact glossary-style explainer for a dashboard page."""

    with st.expander(title, expanded=False):
        for label, explanation in items:
            st.markdown(f"**{label}:** {explanation}")


def render_metric_cards(cards: list[dict[str, str]]) -> None:
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards, strict=False):
        with column:
            st.markdown(
                f"""
                <div class="air-card">
                  <div class="air-label">{card["label"]}</div>
                  <div class="air-value">{card["value"]}</div>
                  <div class="air-note">{card["note"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def fmt_currency(value: float) -> str:
    return f"${value:,.0f}"


def fmt_pct(value: float) -> str:
    return f"{value:.2%}"


def split_feature_label(raw_feature_name: str) -> tuple[str, str]:
    """Split stored feature labels into a cleaner type and display name for the UI."""

    value = str(raw_feature_name or "")
    if "__" in value:
        feature_type, feature_name = value.split("__", 1)
    else:
        feature_type, feature_name = "feature", value
    feature_type = feature_type.replace("_", " ").strip().title()
    feature_name = feature_name.replace("_", " ").strip().title()
    return feature_type or "Feature", feature_name or value


def format_feature_frame(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    """Return a display-friendly explainability table without changing saved artifacts."""

    if frame is None or frame.empty or "feature_name" not in frame.columns:
        return frame
    display = frame.copy()
    display[["feature_type", "feature_label"]] = display["feature_name"].apply(
        lambda value: pd.Series(split_feature_label(str(value)))
    )
    columns = list(display.columns)
    feature_index = columns.index("feature_name")
    columns = [column for column in columns if column not in {"feature_type", "feature_label"}]
    columns[feature_index:feature_index + 1] = ["feature_type", "feature_label"]
    return display.loc[:, columns]


def infer_llm_improvement_area(row: Any) -> str:
    grounded_score = float(getattr(row, "grounded_score", 0.0))
    artifact_score = float(getattr(row, "artifact_match_score", 0.0))
    tool_score = float(getattr(row, "tool_match_score", 0.0))
    citation_score = float(getattr(row, "citation_score", 0.0))
    used_fallback = bool(getattr(row, "used_fallback", False))

    if used_fallback and grounded_score == 0.0:
        return "Retrieval coverage"
    if grounded_score < 1.0:
        return "Grounding threshold or retrieval"
    if artifact_score < 0.5:
        return "Artifact indexing or evidence mapping"
    if tool_score < 0.5:
        return "Planner or tool routing"
    if citation_score < 1.0:
        return "Answer formatting and citation handling"
    return "Prompt quality or answer synthesis"


def render_table_section(title: str, frame: pd.DataFrame | None, *, caption: str | None = None, rows: int | None = None) -> None:
    st.markdown('<div class="air-section">', unsafe_allow_html=True)
    st.subheader(title)
    if caption:
        st.caption(caption)
    if frame is None:
        st.info("No artifact available for this view yet.")
    else:
        st.dataframe(frame.head(rows) if rows is not None else frame, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_download_buttons(title: str, items: list[tuple[str, Path]]) -> None:
    available = [(label, path) for label, path in items if label and path is not None and path.exists() and path.is_file()]
    if not available:
        return
    st.markdown('<div class="air-section">', unsafe_allow_html=True)
    st.subheader(title)
    columns = st.columns(min(3, len(available)))
    for index, (label, path) in enumerate(available):
        with columns[index % len(columns)]:
            st.download_button(
                label=label,
                data=file_bytes(path),
                file_name=path.name,
                mime="application/octet-stream",
                use_container_width=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def metric_from_curated(frame: pd.DataFrame, metric_name: str) -> float:
    if frame.empty:
        return 0.0
    latest_quarter = frame["quarter"].max()
    latest = frame.loc[frame["quarter"] == latest_quarter]
    if metric_name == "capital_ratio":
        liability = float(latest["liability_balance"].sum())
        capital = float(latest["capital_proxy"].sum())
        return (capital / liability) if liability > 0 else 0.0
    if metric_name == "average_asset_return":
        invested_assets = float(latest["asset_market_value"].sum())
        investment_income = float(latest["asset_investment_income"].sum())
        return (investment_income / invested_assets) if invested_assets > 0 else 0.0
    return float(latest[metric_name].sum())


def build_scenario_comparison_table(baseline_curated: pd.DataFrame, scenario_curated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric_name in (
        "premium_income",
        "total_claims",
        "reserves",
        "csm_closing",
        "asset_investment_income",
        "average_asset_return",
        "capital_ratio",
    ):
        baseline_value = metric_from_curated(baseline_curated, metric_name)
        scenario_value = metric_from_curated(scenario_curated, metric_name)
        change = scenario_value - baseline_value
        change_pct = (change / baseline_value) if abs(baseline_value) > 1e-9 else 0.0
        rows.append(
            {
                "metric_name": metric_name,
                "baseline_value": baseline_value,
                "scenario_value": scenario_value,
                "change": change,
                "change_pct": change_pct,
            }
        )
    return pd.DataFrame(rows)


def build_scenario_trend_frame(baseline_curated: pd.DataFrame, scenario_curated: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    if metric_name == "capital_ratio":
        baseline = (
            baseline_curated.groupby("quarter", as_index=False)[["capital_proxy", "liability_balance"]]
            .sum()
            .assign(metric_value=lambda frame: frame["capital_proxy"] / frame["liability_balance"].where(frame["liability_balance"] != 0, 1.0))
            [["quarter", "metric_value"]]
        )
        scenario = (
            scenario_curated.groupby("quarter", as_index=False)[["capital_proxy", "liability_balance"]]
            .sum()
            .assign(metric_value=lambda frame: frame["capital_proxy"] / frame["liability_balance"].where(frame["liability_balance"] != 0, 1.0))
            [["quarter", "metric_value"]]
        )
    elif metric_name == "average_asset_return":
        baseline = (
            baseline_curated.groupby("quarter", as_index=False)[["asset_market_value", "asset_investment_income"]]
            .sum()
            .assign(
                metric_value=lambda frame: frame["asset_investment_income"]
                / frame["asset_market_value"].where(frame["asset_market_value"] != 0, 1.0)
            )[["quarter", "metric_value"]]
        )
        scenario = (
            scenario_curated.groupby("quarter", as_index=False)[["asset_market_value", "asset_investment_income"]]
            .sum()
            .assign(
                metric_value=lambda frame: frame["asset_investment_income"]
                / frame["asset_market_value"].where(frame["asset_market_value"] != 0, 1.0)
            )[["quarter", "metric_value"]]
        )
    else:
        baseline = baseline_curated.groupby("quarter", as_index=False)[metric_name].sum().rename(columns={metric_name: "metric_value"})
        scenario = scenario_curated.groupby("quarter", as_index=False)[metric_name].sum().rename(columns={metric_name: "metric_value"})

    baseline["series"] = "Baseline"
    scenario["series"] = "Scenario"
    return pd.concat([baseline, scenario], ignore_index=True)


def build_actual_forecast_frame(curated: pd.DataFrame, forecasts: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    if curated.empty or forecasts.empty:
        return pd.DataFrame()

    if metric_name == "premium":
        actual = curated.groupby("quarter", as_index=False)["premium_income"].sum().rename(columns={"premium_income": "metric_value"})
    elif metric_name == "capital_ratio":
        actual = (
            curated.groupby("quarter", as_index=False)[["capital_proxy", "liability_balance"]]
            .sum()
            .assign(metric_value=lambda frame: frame["capital_proxy"] / frame["liability_balance"].where(frame["liability_balance"] != 0, 1.0))
            [["quarter", "metric_value"]]
        )
    else:
        return pd.DataFrame()

    forecast = (
        forecasts.loc[forecasts["target_name"] == metric_name]
        .groupby("forecast_quarter", as_index=False)["forecast_value"]
        .agg("sum" if metric_name == "premium" else "mean")
        .rename(columns={"forecast_quarter": "quarter", "forecast_value": "metric_value"})
    )

    if actual.empty or forecast.empty:
        return pd.DataFrame()

    actual["series"] = "Actual"
    forecast["series"] = "Forecast"
    return pd.concat([actual, forecast], ignore_index=True)


def render_plotly_line_chart(frame: pd.DataFrame, *, x_col: str, y_col: str, series_col: str, title: str, y_title: str) -> None:
    figure = go.Figure()
    colors = {"Baseline": "#0d6c74", "Scenario": "#b45f2f"}
    for series_name, subset in frame.groupby(series_col):
        figure.add_trace(
            go.Scatter(
                x=subset[x_col],
                y=subset[y_col],
                mode="lines+markers",
                name=str(series_name),
                line={"width": 3, "color": colors.get(str(series_name), None)},
            )
        )
    figure.update_layout(
        title=title,
        xaxis_title="Quarter",
        yaxis_title=y_title,
        height=380,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    st.plotly_chart(figure, use_container_width=True)


def render_plotly_bar_chart(frame: pd.DataFrame, *, x_col: str, y_col: str, title: str, y_title: str) -> None:
    chart_frame = frame[[x_col, y_col]].copy()
    chart_frame[y_col] = pd.to_numeric(chart_frame[y_col], errors="coerce")
    chart_frame = chart_frame.dropna(subset=[y_col])
    if chart_frame.empty:
        st.info(f"No plotted values are available for {title.lower()}.")
        return

    if chart_frame[y_col].abs().max() < 1e-12:
        st.info(f"{title} has no material movement for the selected scenario.")
        return

    figure = go.Figure(
        data=[
            go.Bar(
                x=chart_frame[x_col],
                y=chart_frame[y_col],
                marker_color=["#b45f2f" if value < 0 else "#0d6c74" for value in chart_frame[y_col]],
                marker_line_width=1,
                marker_line_color="rgba(255,255,255,0.25)",
            )
        ]
    )
    figure.update_layout(
        title=title,
        xaxis_title="Metric",
        yaxis_title=y_title,
        height=360,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    figure.update_yaxes(zeroline=True, zerolinewidth=1.5, zerolinecolor="#7a8794")
    st.plotly_chart(figure, use_container_width=True)


def summarize_run_status(paths: Any) -> dict[str, str]:
    governance_path = find_existing_path(paths.logs, "governance_log")
    governance = load_table(governance_path)
    chat_history_path = paths.root / "chatbot" / "chat_history.jsonl"
    figures_path = find_existing_path(paths.figures / "reporting", "figure_metadata")
    figures = load_table(figures_path)
    scenario_roots = list_scenario_roots(paths.root)
    scenario_name = ""
    latest_artifact_path = governance_path or figures_path

    if governance is None or governance.empty:
        latest_scenario_path: Path | None = None
        latest_mtime = -1.0
        for scenario_root in scenario_roots:
            scenario_governance_path = find_existing_path(scenario_root / "logs", "governance_log")
            if scenario_governance_path is None or not scenario_governance_path.exists():
                continue
            mtime = scenario_governance_path.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_scenario_path = scenario_governance_path
                scenario_name = scenario_root.name

        if latest_scenario_path is not None:
            governance_path = latest_scenario_path
            governance = load_table(latest_scenario_path)
            figures_path = find_existing_path(latest_scenario_path.parent.parent / "figures" / "reporting", "figure_metadata")
            figures = load_table(figures_path)
            latest_artifact_path = latest_scenario_path

    if governance is None or governance.empty:
        return {
            "status": "No run detected",
            "latest_stage": "n/a",
            "stage_count": "0",
            "chat_count": "0",
            "figure_count": "0",
            "run_scope": "none",
            "last_artifact_update": "n/a",
        }

    if figures_path is not None and figures_path.exists():
        if latest_artifact_path is None or figures_path.stat().st_mtime > latest_artifact_path.stat().st_mtime:
            latest_artifact_path = figures_path

    if chat_history_path.exists():
        if latest_artifact_path is None or chat_history_path.stat().st_mtime > latest_artifact_path.stat().st_mtime:
            latest_artifact_path = chat_history_path

    chat_count = 0
    if chat_history_path.exists():
        chat_count = len(chat_history_path.read_text(encoding="utf-8").splitlines())

    last_artifact_update = "n/a"
    if latest_artifact_path is not None and latest_artifact_path.exists():
        last_artifact_update = datetime.fromtimestamp(latest_artifact_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    return {
        "status": "Ready",
        "latest_stage": str(governance.iloc[-1]["agent_name"]),
        "stage_count": str(len(governance)),
        "chat_count": str(chat_count),
        "figure_count": str(0 if figures is None else len(figures)),
        "run_scope": f"Scenario: {format_identifier_label(scenario_name)}" if scenario_name else "Baseline",
        "last_artifact_update": last_artifact_update,
    }


def build_scenario_question(claims_shift: int, asset_shift: int, reserve_shift: int, stress_mode: bool) -> str:
    parts = []
    if claims_shift != 0:
        direction = "increase" if claims_shift > 0 else "decrease"
        parts.append(f"claims {direction} {abs(claims_shift)}%")
    if asset_shift != 0:
        direction = "rise" if asset_shift > 0 else "fall"
        parts.append(f"asset returns {direction} {abs(asset_shift)}%")
    if reserve_shift != 0:
        direction = "increase" if reserve_shift > 0 else "decrease"
        parts.append(f"reserves {direction} {abs(reserve_shift)}%")

    if not parts:
        return "Run a stress scenario with lower asset returns."
    prefix = "What if " if not stress_mode else "Run a stress scenario where "
    connector = " and ".join(parts)
    suffix = "?" if not stress_mode else "."
    return f"{prefix}{connector}{suffix}"


def render_data_overview(paths: Any) -> None:
    curated_path = find_existing_path(paths.data_processed, "curated_reporting_dataset")
    training_path = find_existing_path(paths.data_processed, "forecast_training_frame")
    curated = load_table(curated_path)
    training = load_table(training_path)
    if curated is None:
        render_page_header(
            "Data Overview",
            "Cleaned exposure, claims, reserves, and capital proxy inputs for the reporting workflow.",
        )
        st.info("Cleaned reporting dataset not found. Run `case-study generate-data` or `case-study run` first.")
        return

    latest_quarter = str(curated["quarter"].max())
    latest = curated.loc[curated["quarter"] == latest_quarter]
    render_page_header(
        "Data Overview",
        "Synthetic life insurer data cleaned into a management reporting dataset.",
        chips=[latest_quarter, f"{curated['product'].nunique()} products", f"{curated['region'].nunique()} regions"],
    )
    render_metric_cards(
        [
            {"label": "Rows", "value": f"{len(curated):,}", "note": "Cleaned product-region-quarter observations"},
            {"label": "Premium", "value": fmt_currency(float(latest["premium_income"].sum())), "note": f"Latest quarter {latest_quarter}"},
            {"label": "Claims", "value": fmt_currency(float(latest["total_claims"].sum())), "note": "Aggregate incurred claims"},
            {"label": "Reserves", "value": fmt_currency(float(latest["reserves"].sum())), "note": "Ending reserves in latest quarter"},
        ]
    )
    render_page_explainer(
        "How to read this page",
        [
            ("Rows", "Count of cleaned product-region-quarter observations available to downstream workflow stages."),
            ("Segment Snapshot", "Latest-quarter product summary for a fast business mix check before deeper analysis."),
            ("Cleaned Dataset", "Analysis-ready quarterly dataset used by validation, forecasting, narratives, and the chatbot."),
            ("Training Frame", "Forecasting feature set after lag, seasonality, and driver fields have been prepared."),
        ],
    )

    tab_summary, tab_curated, tab_training = st.tabs(["Segment Snapshot", "Cleaned Dataset", "Training Frame"])
    with tab_summary:
        by_product = (
            latest.groupby("product", as_index=False)[["premium_income", "total_claims", "reserves"]]
            .sum()
            .sort_values("premium_income", ascending=False)
        )
        render_table_section("Latest Quarter by Product", by_product, caption="Premium, claims, and reserve mix by product line.")
    with tab_curated:
        render_table_section("Cleaned Reporting Dataset", curated, rows=75)
    with tab_training:
        if training is None:
            st.info("Forecast training frame not found. Run `case-study forecast` or `case-study run` first.")
        else:
            render_table_section("Forecast Training Frame", training, rows=75)
    render_download_buttons(
        "Exports",
        [
            ("Download cleaned dataset", curated_path),
            ("Download training frame", training_path),
        ],
    )


def render_validation_results(paths: Any) -> None:
    summary_path = find_existing_path(paths.data_processed, "quarterly_validation_summary")
    anomalies_path = find_existing_path(paths.data_processed, "anomaly_table")
    anomaly_investigation_path = find_existing_path(paths.reports / "reporting", "anomaly_investigation")
    summary = load_table(summary_path)
    anomalies = load_table(anomalies_path)
    anomaly_investigation = load_table(anomaly_investigation_path)
    render_page_header(
        "Validation Results",
        "Quarterly quality checks, reconciliation controls, and anomaly monitoring.",
    )
    if summary is None:
        st.info("Validation outputs not found. Run `case-study validate` or `case-study run` first.")
        return

    latest = summary.sort_values("quarter").iloc[-1]
    total_flagged_records = int(summary["records_with_issues"].sum())
    total_anomalies = int(summary["anomaly_count"].sum())
    worst_quarter = summary.sort_values(["anomaly_count", "records_with_issues"], ascending=[False, False]).iloc[0]
    render_metric_cards(
        [
            {"label": "Latest Quarter", "value": str(latest["quarter"]), "note": "Most recent validation period"},
            {"label": "Pass Rate", "value": fmt_pct(float(latest["validation_pass_rate"])), "note": "Records passing configured checks"},
            {"label": "Flagged Records", "value": f"{total_flagged_records:,}", "note": f"Cumulative flagged records across all quarters; latest quarter = {int(latest['records_with_issues']):,}"},
            {"label": "Anomalies", "value": f"{total_anomalies:,}", "note": f"Cumulative anomalies across all quarters; worst quarter = {worst_quarter['quarter']}"},
        ]
    )
    render_page_explainer(
        "How to read this page",
        [
            ("Pass Rate", "Share of records passing configured data-quality and reconciliation checks in the latest quarter."),
            ("Flagged Records", "Rows with at least one validation issue across the saved reporting history."),
            ("Anomalies", "Exceptions detected by rule-based checks and anomaly logic; these may come from earlier quarters too."),
            ("Anomaly Review", "Detailed exception list for follow-up, not necessarily limited to the latest quarter."),
            ("Investigation", "Structured first-pass explanation layer that links anomalies to related metrics, recent history, and supporting drivers."),
        ],
    )
    st.caption(
        f"Latest quarter {latest['quarter']} has {int(latest['records_with_issues'])} flagged records and "
        f"{int(latest['anomaly_count'])} anomalies. The anomaly table below may include earlier quarters."
    )

    tab_summary, tab_anomalies, tab_investigation = st.tabs(
        ["Quarterly Summary", "Anomaly Review", "Anomaly Investigation"]
    )
    with tab_summary:
        render_table_section("Quarterly Validation Summary", summary)
    with tab_anomalies:
        render_table_section("Anomaly Table", anomalies, rows=150)
    with tab_investigation:
        if anomaly_investigation is None or anomaly_investigation.empty:
            st.info("Anomaly investigation output not found. Run `case-study run` to generate first-pass explanations.")
        else:
            investigation_columns = [
                "anomaly_id",
                "anomaly_type",
                "product",
                "region",
                "quarter",
                "likely_drivers",
                "explanation_text",
                "support_score",
                "reviewer_status",
            ]
            available_investigation_columns = [
                column for column in investigation_columns if column in anomaly_investigation.columns
            ]
            render_table_section(
                "Anomaly Investigation",
                format_display_frame(anomaly_investigation.loc[:, available_investigation_columns]),
                rows=150,
                caption=(
                    "First-pass explanation output assembled from validation anomalies, recent segment history, "
                    "and related analytical artifacts."
                ),
            )
    render_download_buttons(
        "Exports",
        [
            ("Download validation summary", summary_path),
            ("Download anomaly table", anomalies_path),
            ("Download anomaly investigation", anomaly_investigation_path),
        ],
    )


def render_forecast_results(paths: Any) -> None:
    evaluation_path = find_existing_path(paths.models, "model_evaluation")
    forecasts_path = find_existing_path(paths.models, "forecast_output_table")
    backtests_path = find_existing_path(paths.models, "backtest_predictions")
    evaluation = load_table(evaluation_path)
    forecasts = load_table(forecasts_path)
    backtests = load_table(backtests_path)
    render_page_header(
        "Forecast Results",
        "Model evaluation, backtesting, and forward quarter projections across reporting targets.",
    )
    if evaluation is None:
        st.info("Forecast outputs not found. Run `case-study forecast` or `case-study run` first.")
        return

    best_models = evaluation.sort_values(["target_name", "mae"]).groupby("target_name", as_index=False).first()
    strong_models = int((evaluation.get("quality_flag", pd.Series(dtype=str)) == "strong").sum())
    forecast_horizon = (
        int(evaluation["forecast_horizon_quarters"].max())
        if "forecast_horizon_quarters" in evaluation.columns and not evaluation.empty
        else 1
    )
    rolling_quarters = (
        int(evaluation["evaluation_quarters"].max())
        if "evaluation_quarters" in evaluation.columns and not evaluation.empty
        else 1
    )
    render_metric_cards(
        [
            {"label": "Targets", "value": f"{evaluation['target_name'].nunique()}", "note": "Claims, premium, reserves, CSM, and capital proxy ratio"},
            {"label": "Model Families", "value": f"{evaluation['model_family'].nunique()}", "note": "Baseline, time-series, and boosting"},
            {"label": "Rolling Backtest Quarters", "value": f"{rolling_quarters}", "note": "Sequential holdout quarters used in evaluation"},
            {"label": "Forecast Horizon", "value": f"{forecast_horizon}", "note": "Forward quarters generated per target and segment"},
            {"label": "Strong Models", "value": f"{strong_models}", "note": "Models meeting the strongest current quality thresholds"},
        ]
    )
    st.caption(
        "Evaluation now uses rolling backtests across multiple holdout quarters. "
        "Use MAE/RMSE and `quality_flag` as the primary signals; some targets may still show weak R2 if their synthetic series are low-variance or noisy."
    )
    render_page_explainer(
        "How to read this page",
        [
            ("Rolling Backtest Quarters", "Number of sequential holdout quarters used to evaluate forecast quality rather than relying on a single split."),
            ("Forecast Horizon", "Forward quarters generated per target and segment in the latest forecasting run."),
            ("Strong Models", "Models currently meeting the strongest configured quality thresholds; weaker models remain visible for review."),
            ("Evaluation vs Forecast Output", "Evaluation shows model quality and benchmark comparisons, while Forecast Output stores the projected future values themselves."),
        ],
    )

    tab_eval, tab_forecasts, tab_backtests = st.tabs(["Evaluation", "Forecast Output", "Backtests"])
    with tab_eval:
        render_table_section("Model Evaluation", format_display_frame(evaluation))
    with tab_forecasts:
        render_table_section("Forecast Output Table", format_display_frame(forecasts))
    with tab_backtests:
        render_table_section("Backtest Predictions", format_display_frame(backtests), rows=150)
    render_download_buttons(
        "Exports",
        [
            ("Download model evaluation", evaluation_path),
            ("Download forecast output", forecasts_path),
            ("Download backtests", backtests_path),
        ],
    )


def render_explainability(paths: Any) -> None:
    base_dir = paths.reports / "explainability"
    shap_global_path = find_existing_path(base_dir, "shap_global_importance")
    shap_local_path = find_existing_path(base_dir, "shap_local_explanations")
    lime_path = find_existing_path(base_dir, "lime_explanations")
    pdp_ice_path = find_existing_path(base_dir, "pdp_ice_table")
    shap_global = load_table(shap_global_path)
    shap_local = load_table(shap_local_path)
    lime = load_table(lime_path)
    pdp_ice = load_table(pdp_ice_path)
    report_path = base_dir / "explanation_report.md"
    render_page_header(
        "Explainability",
        "Model interpretation across global importance, local explanations, and response diagnostics.",
    )
    if shap_global is None:
        st.info("Explainability outputs not found. Run `case-study explain` or `case-study run` first.")
        return

    top_driver = shap_global.sort_values("mean_abs_shap", ascending=False).iloc[0]
    top_driver_type, top_driver_name = split_feature_label(str(top_driver["feature_name"]))
    shap_global_display = format_feature_frame(shap_global)
    shap_local_display = format_feature_frame(shap_local)
    lime_display = format_feature_frame(lime)
    pdp_ice_display = format_feature_frame(pdp_ice)
    render_metric_cards(
        [
            {"label": "Targets Explained", "value": f"{shap_global['target_name'].nunique()}", "note": "Coverage across forecast targets"},
            {"label": "Top Driver", "value": top_driver_name, "note": f"{top_driver_type} feature with highest mean SHAP across targets"},
            {"label": "Local Rows", "value": f"{0 if shap_local is None else len(shap_local):,}", "note": "Stored local explanation records"},
            {"label": "PDP / ICE Rows", "value": f"{0 if pdp_ice is None else len(pdp_ice):,}", "note": "Partial dependence and ICE metadata"},
        ]
    )
    render_page_explainer(
        "How to read this page",
        [
            ("Top Driver", "Feature with the highest average SHAP importance across the saved explainability outputs."),
            ("Global", "Portfolio-level driver view showing which variables matter most on average."),
            ("Local", "Record-level explanations showing why a specific prediction moved the way it did."),
            ("PDP / ICE", "Response diagnostics that show how model predictions change as one driver changes while others are held constant or traced individually."),
        ],
    )

    tab_global, tab_local, tab_report = st.tabs(["Global", "Local", "Report"])
    with tab_global:
        render_table_section("SHAP Global Importance", shap_global_display, rows=120)
        render_table_section("LIME Explanations", lime_display, rows=100)
    with tab_local:
        render_table_section("SHAP Local Explanations", shap_local_display, rows=120)
        render_table_section("PDP and ICE", format_display_frame(pdp_ice_display), rows=120)
    with tab_report:
        st.markdown('<div class="air-section">', unsafe_allow_html=True)
        st.subheader("Explainability Report")
        if report_path.exists():
            st.markdown(report_path.read_text(encoding="utf-8"))
        else:
            st.info("Explainability report markdown is not available.")
        st.markdown("</div>", unsafe_allow_html=True)
    render_download_buttons(
        "Exports",
        [
            ("Download SHAP global", shap_global_path),
            ("Download SHAP local", shap_local_path),
            ("Download LIME explanations", lime_path),
            ("Download PDP and ICE", pdp_ice_path),
            ("Download explanation report", report_path),
        ],
    )


def render_narrative_reporting(paths: Any) -> None:
    base_dir = paths.reports / "narrative"
    final_report_dir = paths.reports / "final"
    statements_path = find_existing_path(base_dir, "narrative_statements")
    statements = load_table(statements_path)
    markdown_reports = sorted(
        path for path in base_dir.glob("management_report_*.md") if "_llm_" not in path.stem
    )
    llm_markdown_reports = sorted(base_dir.glob("management_report_llm_*.md"))
    full_report_markdowns = sorted(
        path for path in final_report_dir.glob("management_report_full_*.md") if "_llm_" not in path.stem
    )
    full_report_llm_markdowns = sorted(final_report_dir.glob("management_report_llm_*.md"))
    full_report_sections_path = find_existing_path(final_report_dir, "management_report_sections")
    full_report_sections = load_table(full_report_sections_path)
    render_page_header(
        "Narrative Reporting",
        "Traceable management commentary generated from the reporting workflow.",
    )
    if statements is None:
        st.info("Narrative outputs not found. Run `case-study narrate` or `case-study run` first.")
        return

    latest_quarter = str(statements["reporting_quarter"].max())
    render_metric_cards(
        [
            {"label": "Statements", "value": f"{len(statements):,}", "note": "Traceable commentary lines"},
            {"label": "Sections", "value": f"{statements['section'].nunique()}", "note": "Quarterly, claims, reserves/CSM, capital"},
            {"label": "Latest Quarter", "value": latest_quarter, "note": "Most recent narrative pack"},
            {"label": "Report Files", "value": f"{len(markdown_reports) + len(llm_markdown_reports) + len(full_report_markdowns) + len(full_report_llm_markdowns)}", "note": "Deterministic and LLM-assisted report markdown files"},
        ]
    )
    render_page_explainer(
        "How to read this page",
        [
            ("Statements", "Individual traceable commentary lines linked back to supporting data or model outputs."),
            ("Sections", "Narrative groupings used to assemble the management report."),
            ("Statement Library", "Granular commentary view for reviewers who want to inspect traceability line by line."),
            ("Management Report", "Concise management-facing commentary focused on the quarter narrative itself."),
            ("Full Review Pack", "Broader assembled output that adds insights, anomaly investigation, movement analysis, quality review, and figures."),
        ],
    )

    tab_statements, tab_report, tab_trace, tab_full = st.tabs(["Statement Library", "Management Report", "Traceability View", "Full Review Pack"])
    with tab_statements:
        render_table_section("Narrative Statements", format_display_frame(statements), rows=150)
    with tab_report:
        deterministic_tab, llm_tab = st.tabs(["Deterministic View", "LLM-Assisted Draft"])
        with deterministic_tab:
            st.subheader("Management Report")
            if markdown_reports:
                raw_report = markdown_reports[-1].read_text(encoding="utf-8")
                st.markdown(strip_traceability_lines(raw_report))
                st.caption("This reading view hides inline traceability details so the report reads like management commentary. The separate Traceability View keeps the full saved markdown available for review.")
            else:
                st.info("No markdown management report is available.")
        with llm_tab:
            st.subheader("LLM-Assisted Draft")
            if llm_markdown_reports:
                st.markdown(llm_markdown_reports[-1].read_text(encoding="utf-8"))
                st.caption(
                    "This draft may remain very close to the deterministic version when the local mock client is "
                    "active. With a real external LLM, the wording should usually diverge more."
                )
            else:
                st.info("No LLM-assisted narrative draft is available. Configure an external LLM provider and rerun the workflow to generate one.")
    with tab_trace:
        st.subheader("Report Traceability View")
        if statements is not None and not statements.empty:
            st.caption(
                "This view separates each narrative statement from the source dataset, columns, filters, "
                "and values used to support it."
            )
            traceability_columns = [
                "statement_id",
                "section",
                "statement_text",
                "source_dataset",
                "source_columns",
                "source_filters",
                "source_value",
            ]
            available_traceability_columns = [
                column for column in traceability_columns if column in statements.columns
            ]
            traceability_frame = statements.loc[:, available_traceability_columns].copy()
            st.markdown("**Statement Evidence Map**")
            st.caption(
                "Each row records the data source and extraction logic behind a management commentary statement."
            )
            st.dataframe(
                format_display_frame(traceability_frame).head(150),
                use_container_width=True,
            )
            if markdown_reports:
                with st.expander("Raw markdown with inline traceability", expanded=False):
                    st.markdown(markdown_reports[-1].read_text(encoding="utf-8"))
        elif markdown_reports:
            st.warning(
                "Structured statement traceability is unavailable. Showing the saved markdown report instead."
            )
            st.markdown(markdown_reports[-1].read_text(encoding="utf-8"))
        else:
            st.info("No traceability output is available.")
    with tab_full:
        full_report_markdown = full_report_markdowns[-1] if full_report_markdowns else None
        full_report_llm_markdown = full_report_llm_markdowns[-1] if full_report_llm_markdowns else None
        full_report_text = full_report_markdown.read_text(encoding="utf-8") if full_report_markdown is not None else ""
        full_tab_report, full_tab_llm, full_tab_sections = st.tabs(["Reading View", "LLM-Assisted Draft", "Section View"])
        with full_tab_report:
            st.subheader("Full Review Pack")
            if full_report_markdown is not None:
                st.markdown(full_report_text)
                st.caption(
                    "This view is intentionally broader than the management report. It combines executive summary, "
                    "material insights, anomaly investigation, movement analysis, draft commentary, quality review, and figures."
                )
            else:
                st.info("No full management report markdown is available.")
        with full_tab_llm:
            st.subheader("LLM-Assisted Full Review Pack Draft")
            if full_report_llm_markdown is not None:
                st.markdown(full_report_llm_markdown.read_text(encoding="utf-8"))
                st.caption(
                    "This draft may remain close to the assembled review pack when the local mock client is active. "
                    "A real external LLM should usually produce more varied wording."
                )
            else:
                st.info("No LLM-assisted full report draft is available. Configure an external LLM provider and rerun the workflow to generate one.")
        with full_tab_sections:
            render_table_section(
                "Full Report Sections",
                full_report_sections,
                caption="Structured section outputs used to assemble the full review pack.",
            )
    render_download_buttons(
        "Exports",
        [
            ("Download narrative statements", statements_path),
            ("Download management report", markdown_reports[-1] if markdown_reports else None),
            ("Download LLM-assisted narrative draft", llm_markdown_reports[-1] if llm_markdown_reports else None),
            ("Download full management report", full_report_markdowns[-1] if full_report_markdowns else None),
            ("Download LLM-assisted full report", full_report_llm_markdowns[-1] if full_report_llm_markdowns else None),
            ("Download full report sections", full_report_sections_path),
        ],
    )


def render_visualizations(paths: Any) -> None:
    figure_dir = paths.figures / "reporting"
    metadata_path = find_existing_path(figure_dir, "figure_metadata")
    metadata = load_table(metadata_path)
    curated_path = find_existing_path(paths.data_processed, "curated_reporting_dataset")
    forecasts_path = find_existing_path(paths.models, "forecast_output_table")
    curated = load_table(curated_path)
    forecasts = load_table(forecasts_path)
    render_page_header(
        "Visualizations",
        "Saved reporting figures with metadata, captions, and source traceability.",
    )
    if metadata is None and (curated is None or forecasts is None):
        st.info("Visualization outputs not found. Run `case-study visualize` or `case-study run` first.")
        return

    figure_count = 0 if metadata is None else len(metadata)
    forecast_horizon = 0
    if forecasts is not None and not forecasts.empty and "forecast_horizon" in forecasts.columns:
        forecast_horizon = int(forecasts["forecast_horizon"].max())
    render_metric_cards(
        [
            {"label": "Figures", "value": f"{figure_count:,}", "note": "Saved charts in the reporting pack"},
            {"label": "Artifact Folder", "value": "reporting", "note": "Figure output subdirectory"},
            {"label": "Forecast Horizon", "value": str(forecast_horizon or 1), "note": "Live charts use the latest forecast table"},
            {"label": "Feature Importance", "value": "Available" if metadata is not None else "Pending", "note": "Top SHAP feature visualization"},
        ]
    )
    render_page_explainer(
        "How to read this page",
        [
            ("Live Forecast Charts", "Charts built directly from the latest saved datasets, so they update as soon as forecast artifacts change."),
            ("Figure Metadata", "Catalog of saved figures, their titles, and the source artifacts used to build them."),
            ("Forecast Horizon", "Forward quarters currently reflected in live forecast charts."),
            ("Exports", "Saved image files and metadata intended for the report pack or appendix."),
        ],
    )

    if curated is not None and forecasts is not None and not curated.empty and not forecasts.empty:
        st.markdown('<div class="air-section">', unsafe_allow_html=True)
        st.subheader("Live Forecast Charts")
        st.caption("These charts read directly from the latest cleaned dataset and forecast table, so they update as soon as forecasting artifacts change.")
        live_left, live_right = st.columns(2)
        with live_left:
            premium_frame = build_actual_forecast_frame(curated, forecasts, "premium")
            render_plotly_line_chart(
                premium_frame,
                x_col="quarter",
                y_col="metric_value",
                series_col="series",
                title="Actual vs Forecast Premium",
                y_title="Premium income",
            )
        with live_right:
            capital_frame = build_actual_forecast_frame(curated, forecasts, "capital_ratio")
            render_plotly_line_chart(
                capital_frame,
                x_col="quarter",
                y_col="metric_value",
                series_col="series",
                title="Actual and Forecast Synthetic Capital-to-Liability Proxy Ratio",
                y_title="Synthetic capital-to-liability proxy ratio",
            )
        st.markdown("</div>", unsafe_allow_html=True)

    if metadata is not None:
        render_table_section("Figure Metadata", metadata)
        figure_rows = [row for row in metadata.itertuples(index=False)]
        for row_index in range(0, len(figure_rows), 2):
            left_col, right_col = st.columns(2)
            for column, row in zip((left_col, right_col), figure_rows[row_index : row_index + 2]):
                image_path = Path(row.file_path)
                with column:
                    st.markdown('<div class="air-figure">', unsafe_allow_html=True)
                    st.markdown('<div class="air-kicker">Figure</div>', unsafe_allow_html=True)
                    st.subheader(str(row.chart_title))
                    st.caption(f"Sources: {row.source_datasets}")
                    if image_path.exists():
                        st.image(str(image_path), use_container_width=True)
                    else:
                        st.info(f"Figure file not found: {image_path}")
                    st.markdown("</div>", unsafe_allow_html=True)
    render_download_buttons(
        "Exports",
        ([("Download figure metadata", metadata_path)] if metadata is not None else [])
        + ([(f"Download {Path(row.file_path).name}", Path(row.file_path)) for row in metadata.itertuples(index=False)] if metadata is not None else []),
    )


def render_scenario_comparison(paths: Any) -> None:
    scenario_roots = list_scenario_roots(paths.root)
    render_page_header(
        "Scenario Comparison",
        "Compare baseline reporting outputs against isolated scenario reruns produced by the analytical assistant.",
        chips=[f"{len(scenario_roots)} scenarios found", "Baseline vs stressed view"],
    )
    if not scenario_roots:
        st.info("No scenario runs found yet. Use the Chatbot Q&A scenario builder or ask a what-if question first.")
        return

    scenario_names = [path.name for path in scenario_roots]
    selected_name = st.selectbox(
        "Scenario",
        options=scenario_names,
        index=len(scenario_names) - 1,
        format_func=lambda name: format_scenario_label(name, load_json(next(path for path in scenario_roots if path.name == name) / "logs" / "scenario_metadata.json")),
    )
    scenario_root = next(path for path in scenario_roots if path.name == selected_name)

    baseline_curated_path = find_existing_path(paths.data_processed, "curated_reporting_dataset")
    baseline_forecast_path = find_existing_path(paths.models, "forecast_output_table")
    scenario_curated_path = find_existing_path(scenario_root / "data" / "processed", "curated_reporting_dataset")
    scenario_forecast_path = find_existing_path(scenario_root / "models", "forecast_output_table")
    scenario_governance_path = find_existing_path(scenario_root / "logs", "governance_log")
    scenario_metadata_path = scenario_root / "logs" / "scenario_metadata.json"

    baseline_curated = load_table(baseline_curated_path)
    scenario_curated = load_table(scenario_curated_path)
    baseline_forecast = load_table(baseline_forecast_path)
    scenario_forecast = load_table(scenario_forecast_path)
    scenario_governance = load_table(scenario_governance_path)
    scenario_metadata = load_json(scenario_metadata_path)

    if baseline_curated is None or scenario_curated is None:
        st.info("Baseline or scenario cleaned dataset is missing for this comparison.")
        return

    comparison = build_scenario_comparison_table(baseline_curated, scenario_curated)
    latest_quarter = str(scenario_curated["quarter"].max())
    scenario_label = format_scenario_label(selected_name, scenario_metadata)
    render_metric_cards(
        [
            {"label": "Scenario", "value": scenario_label, "note": f"Scenario id: {format_identifier_label(selected_name)}"},
            {"label": "Latest Quarter", "value": latest_quarter, "note": "Scenario reporting period"},
            {"label": "Capital Delta", "value": fmt_pct(float(comparison.loc[comparison["metric_name"] == "capital_ratio", "change_pct"].iloc[0])), "note": "Scenario vs baseline synthetic capital proxy ratio"},
            {"label": "Claims Delta", "value": fmt_pct(float(comparison.loc[comparison["metric_name"] == "total_claims", "change_pct"].iloc[0])), "note": "Scenario vs baseline claims movement"},
            {"label": "Asset Income Delta", "value": fmt_pct(float(comparison.loc[comparison["metric_name"] == "asset_investment_income", "change_pct"].iloc[0])), "note": "Scenario vs baseline investment income"},
        ]
    )
    render_page_explainer(
        "How to read this page",
        [
            ("Scenario", "Human-readable summary of the saved scenario assumptions, with the raw scenario id retained for traceability."),
            ("Delta cards", "Latest-quarter scenario-versus-baseline change for selected headline metrics."),
            ("Metric Comparison", "Side-by-side actual comparison between baseline and scenario outputs for the latest quarter."),
            ("Forecast Comparison", "Forward-looking difference between baseline and scenario forecasts across future quarters and targets."),
        ],
    )
    if scenario_metadata is not None:
        params = scenario_metadata.get("scenario_parameters", {})
        param_chips = [
            f"claims x{params.get('claims_multiplier', 1.0):.2f}",
            f"premium x{params.get('premium_multiplier', 1.0):.2f}",
            f"reserves x{params.get('reserve_multiplier', 1.0):.2f}",
            f"asset shift {params.get('asset_return_shift', 0.0):+.2%}",
        ]
        st.caption("Applied scenario assumptions: " + " | ".join(param_chips))
    else:
        st.warning(
            "This scenario run does not include saved metadata. It may have been generated before the current "
            "scenario parser was introduced. Rerun the scenario from the chatbot for a fresh comparison."
        )

    tab_compare, tab_forecast, tab_governance = st.tabs(["Metric Comparison", "Forecast Comparison", "Scenario Governance"])
    with tab_compare:
        trend_metric = st.selectbox(
            "Trend metric",
            options=[
                ("premium_income", "Premium income"),
                ("total_claims", "Total claims"),
                ("reserves", "Reserves"),
                ("csm_closing", "Closing CSM"),
                ("asset_investment_income", "Investment income"),
                ("average_asset_return", "Average asset return"),
                ("capital_ratio", "Synthetic capital proxy ratio"),
            ],
            format_func=lambda option: option[1],
        )
        chart_left, chart_right = st.columns(2)
        with chart_left:
            trend_frame = build_scenario_trend_frame(baseline_curated, scenario_curated, trend_metric[0])
            render_plotly_line_chart(
                trend_frame,
                x_col="quarter",
                y_col="metric_value",
                series_col="series",
                title=f"Baseline vs Scenario Trend: {trend_metric[1]}",
                y_title=trend_metric[1],
            )
        with chart_right:
            delta_frame = comparison.copy()
            delta_frame["metric_label"] = delta_frame["metric_name"].replace(
                {
                    "premium_income": "Premium",
                    "total_claims": "Claims",
                    "reserves": "Reserves",
                    "csm_closing": "Closing CSM",
                    "asset_investment_income": "Investment income",
                    "average_asset_return": "Average asset return",
                    "capital_ratio": "Capital proxy ratio",
                }
            )
            render_plotly_bar_chart(
                delta_frame,
                x_col="metric_label",
                y_col="change_pct",
                title="Scenario Change vs Baseline",
                y_title="Change percentage",
            )
        render_table_section(
            "Baseline vs Scenario Metrics",
            format_display_frame(comparison),
            caption="Latest-quarter comparison across core reporting metrics.",
        )
        baseline_latest = baseline_curated.loc[baseline_curated["quarter"] == baseline_curated["quarter"].max()].copy()
        scenario_latest = scenario_curated.loc[scenario_curated["quarter"] == scenario_curated["quarter"].max()].copy()
        view = (
            baseline_latest.groupby("product", as_index=False)[["premium_income", "total_claims", "reserves"]].sum()
            .rename(
                columns={
                    "premium_income": "baseline_premium_income",
                    "total_claims": "baseline_total_claims",
                    "reserves": "baseline_reserves",
                }
            )
            .merge(
                scenario_latest.groupby("product", as_index=False)[["premium_income", "total_claims", "reserves"]].sum(),
                on="product",
                how="outer",
            )
            .rename(
                columns={
                    "premium_income": "scenario_premium_income",
                    "total_claims": "scenario_total_claims",
                    "reserves": "scenario_reserves",
                }
            )
        )
        render_table_section("Product-Level Comparison", view, rows=50)
    with tab_forecast:
        if baseline_forecast is None or scenario_forecast is None:
            st.info("Forecast outputs are not available for this scenario comparison.")
        else:
            baseline_view = (
                baseline_forecast.groupby(["forecast_quarter", "target_name"], as_index=False)["forecast_value"].mean()
                .rename(columns={"forecast_value": "baseline_forecast_value"})
            )
            scenario_view = (
                scenario_forecast.groupby(["forecast_quarter", "target_name"], as_index=False)["forecast_value"].mean()
                .rename(columns={"forecast_value": "scenario_forecast_value"})
            )
            joined = baseline_view.merge(
                scenario_view,
                on=["forecast_quarter", "target_name"],
                how="outer",
            )
            joined["change"] = joined["scenario_forecast_value"] - joined["baseline_forecast_value"]
            forecast_chart = joined.copy()
            forecast_chart["series_label"] = forecast_chart["target_name"] + " | " + forecast_chart["forecast_quarter"]
            render_plotly_bar_chart(
                forecast_chart,
                x_col="series_label",
                y_col="change",
                title="Forecast Delta by Target",
                y_title="Scenario minus baseline forecast",
            )
            render_table_section(
                "Baseline vs Scenario Forecasts",
                format_display_frame(joined.sort_values(["forecast_quarter", "target_name"])),
                caption="Average next-quarter forecast comparison by target.",
            )
    with tab_governance:
        render_table_section(
            "Scenario Governance Log",
            scenario_governance,
            caption="Workflow stages executed for the selected scenario.",
        )
        if scenario_metadata is not None:
            st.markdown('<div class="air-section">', unsafe_allow_html=True)
            st.subheader("Scenario Metadata")
            st.json(scenario_metadata)
            st.markdown("</div>", unsafe_allow_html=True)

    render_download_buttons(
        "Scenario Exports",
        [
            ("Download baseline cleaned dataset", baseline_curated_path),
            ("Download scenario cleaned dataset", scenario_curated_path),
            ("Download baseline forecast", baseline_forecast_path),
            ("Download scenario forecast", scenario_forecast_path),
            ("Download scenario governance log", scenario_governance_path),
            ("Download scenario metadata", scenario_metadata_path if scenario_metadata_path.exists() else None),
        ],
    )


def render_chatbot(paths: Any) -> None:
    render_page_header(
        "Chatbot Q&A",
        "Agentic RAG assistant grounded in indexed artifacts, with structured tools for reporting analysis.",
        chips=["Grounded answers only", "Scenario tools enabled", "Workflow reruns enabled", "Management Q&A ready"],
    )
    st.caption(
        "`capital_ratio` is presented as a synthetic capital-to-liability proxy ratio. "
        "The assistant should only answer from retrieved evidence and tool outputs."
    )
    render_page_explainer(
        "How to use this page",
        [
            ("Grounded answers", "The assistant should answer only from retrieved artifacts or structured tool outputs, not from unsupported free-form inference."),
            ("Tools Used", "Structured tools provide direct access to workflow artifacts such as validation, forecasts, explainability, scenarios, and analyst review items."),
            ("Supporting Sources", "Retrieved document-level evidence passed into the answer layer."),
            ("Reviewer Feedback", "Human feedback is logged so answer quality and grounding can be assessed over time."),
        ],
    )
    st.caption(
        "The assistant can also run controlled workflow actions such as rerunning validation, forecasting, "
        "visualizations, or the full workflow."
    )
    if st.session_state.get("open_chatbot_review_hint"):
        st.info(f"Evaluation follow-up: inspect this question in Chatbot Q&A. Likely improvement area: {st.session_state['open_chatbot_review_hint']}.")
        st.session_state.pop("open_chatbot_review_hint", None)

    suggestion_cols = st.columns(3)
    suggestions = [
        "What drove the reserve increase in Q4?",
        "Which model performed best for capital ratio forecasting?",
        "What if claims increase 20% and asset returns fall 5%?",
    ]
    if "chat_question" not in st.session_state:
        st.session_state["chat_question"] = (
            "What is the forecast outlook for the synthetic capital-to-liability proxy ratio?"
        )
    for column, suggestion in zip(suggestion_cols, suggestions, strict=False):
        with column:
            if st.button(suggestion, use_container_width=True):
                st.session_state["chat_question"] = suggestion

    workflow_cols = st.columns(3)
    workflow_suggestions = [
        "Rerun validation and forecasting.",
        "Refresh the narrative and charts.",
        "Rebuild the chatbot index.",
    ]
    for column, suggestion in zip(workflow_cols, workflow_suggestions, strict=False):
        with column:
            if st.button(suggestion, use_container_width=True):
                st.session_state["chat_question"] = suggestion

    with st.expander("Scenario Builder", expanded=False):
        scenario_cols = st.columns(4)
        with scenario_cols[0]:
            claims_shift = st.slider("Claims shift %", min_value=-30, max_value=30, value=20, step=5)
        with scenario_cols[1]:
            asset_shift = st.slider("Asset return shift %", min_value=-10, max_value=10, value=-5, step=1)
        with scenario_cols[2]:
            reserve_shift = st.slider("Reserve shift %", min_value=-20, max_value=20, value=0, step=5)
        with scenario_cols[3]:
            stress_mode = st.toggle("Stress wording", value=True)
        scenario_question = build_scenario_question(claims_shift, asset_shift, reserve_shift, stress_mode)
        st.caption(f"Generated scenario question: {scenario_question}")
        if st.button("Use scenario question", use_container_width=True):
            st.session_state["chat_question"] = scenario_question

    with st.expander("Workflow Overrides", expanded=False):
        st.caption(
            "Use these overrides when you want a workflow rerun to regenerate synthetic inputs with new assumptions, "
            "apply different validation tolerances, or replace raw input tables with uploaded files."
        )
        assumption_cols = st.columns(3)
        with assumption_cols[0]:
            workflow_claims_multiplier = st.number_input(
                "Claims multiplier",
                min_value=0.10,
                max_value=3.00,
                value=1.00,
                step=0.05,
            )
            workflow_reserve_multiplier = st.number_input(
                "Reserve multiplier",
                min_value=0.10,
                max_value=3.00,
                value=1.00,
                step=0.05,
            )
        with assumption_cols[1]:
            workflow_premium_multiplier = st.number_input(
                "Premium multiplier",
                min_value=0.10,
                max_value=3.00,
                value=1.00,
                step=0.05,
            )
            workflow_csm_multiplier = st.number_input(
                "CSM multiplier",
                min_value=0.10,
                max_value=3.00,
                value=1.00,
                step=0.05,
            )
        with assumption_cols[2]:
            workflow_asset_return_shift = st.number_input(
                "Asset return shift",
                min_value=-0.20,
                max_value=0.20,
                value=0.00,
                step=0.01,
                format="%.2f",
            )
            workflow_capital_multiplier = st.number_input(
                "Capital multiplier",
                min_value=0.10,
                max_value=3.00,
                value=1.00,
                step=0.05,
            )

        tolerance_cols = st.columns(3)
        with tolerance_cols[0]:
            validation_reserve_tolerance = st.number_input(
                "Reserve tolerance",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
                format="%.2f",
            )
        with tolerance_cols[1]:
            validation_csm_tolerance = st.number_input(
                "CSM tolerance",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
                format="%.2f",
            )
        with tolerance_cols[2]:
            validation_capital_tolerance = st.number_input(
                "Capital tolerance",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
                format="%.2f",
            )

        forecast_cols = st.columns(3)
        with forecast_cols[0]:
            forecast_selection_metric = st.selectbox(
                "Forecast selection metric",
                options=["mae", "rmse", "mape"],
                index=0,
            )
            forecast_error_tolerance_pct = st.number_input(
                "Forecast error tolerance",
                min_value=0.01,
                max_value=1.00,
                value=0.25,
                step=0.01,
                format="%.2f",
            )
        with forecast_cols[1]:
            forecast_gb_max_depth = st.number_input(
                "Boosting tree depth",
                min_value=1,
                max_value=8,
                value=3,
                step=1,
            )
            forecast_gb_n_estimators = st.number_input(
                "Boosting estimators",
                min_value=20,
                max_value=500,
                value=100,
                step=10,
            )
        with forecast_cols[2]:
            forecast_gb_learning_rate = st.number_input(
                "Boosting learning rate",
                min_value=0.01,
                max_value=0.50,
                value=0.10,
                step=0.01,
                format="%.2f",
            )
            forecast_horizon_quarters = st.number_input(
                "Forecast horizon (quarters)",
                min_value=1,
                max_value=16,
                value=1,
                step=1,
            )

        upload_cols = st.columns(2)
        with upload_cols[0]:
            upload_policy_data = st.file_uploader("Upload policy_data.csv", type=["csv"], key="upload_policy_data")
            upload_claims_data = st.file_uploader("Upload claims_data.csv", type=["csv"], key="upload_claims_data")
            upload_asset_data = st.file_uploader("Upload asset_data.csv", type=["csv"], key="upload_asset_data")
        with upload_cols[1]:
            upload_financial_balances = st.file_uploader(
                "Upload financial_balances.csv",
                type=["csv"],
                key="upload_financial_balances",
            )
            upload_reporting_metrics = st.file_uploader(
                "Upload reporting_metrics.csv",
                type=["csv"],
                key="upload_reporting_metrics",
            )

    controls_col, meta_col = st.columns([1.6, 1.0])
    with controls_col:
        question = st.text_area(
            "Ask a question",
            key="chat_question",
            height=110,
        )
    with meta_col:
        top_k = st.slider("Top documents", min_value=1, max_value=10, value=5)
        show_tool_outputs = st.toggle("Show tool outputs", value=False)

    llm_col, credential_col = st.columns([1.0, 1.6])
    with llm_col:
        selected_provider = st.selectbox(
            "LLM provider",
            options=list(LLM_PROVIDERS),
            index=list(LLM_PROVIDERS).index(st.session_state.get("chat_llm_provider", "mock")),
            help="Choose the answer generator. Retrieval and tool usage remain grounded in local reporting artifacts.",
            key="chat_llm_provider",
        )
        default_model = DEFAULT_PROVIDER_MODELS[selected_provider]
        previous_provider = st.session_state.get("chat_llm_provider_previous")
        if previous_provider != selected_provider:
            st.session_state["chat_model_name"] = default_model
            st.session_state["chat_model_select"] = default_model if selected_provider != "mock" else "mock"
            st.session_state["chat_model_custom"] = False
            st.session_state["chat_llm_provider_previous"] = selected_provider
        model_options = PROVIDER_MODEL_OPTIONS[selected_provider]
        model_choice = st.selectbox(
            "Suggested models",
            options=model_options,
            index=model_options.index(st.session_state.get("chat_model_select", model_options[0]))
            if st.session_state.get("chat_model_select", model_options[0]) in model_options
            else 0,
            key="chat_model_select",
            help="Suggested provider models based on currently documented public options.",
            disabled=selected_provider == "mock",
        )
        use_custom_model = st.toggle(
            "Type custom model",
            key="chat_model_custom",
            value=st.session_state.get("chat_model_custom", False),
            disabled=selected_provider == "mock",
        )
        if use_custom_model and selected_provider != "mock":
            model_name = st.text_input(
                "Custom model name",
                key="chat_model_name",
                help="Use this when you want a model name that is not in the suggested list.",
                placeholder=default_model,
            )
        else:
            st.session_state["chat_model_name"] = model_choice
            model_name = model_choice
        if selected_provider == "mock":
            model_name = "mock"
    with credential_col:
        if selected_provider == "mock":
            st.info("Mock mode stays fully local. No external credentials are required.")
            api_key = ""
        else:
            credential_label = "OpenAI API key" if selected_provider == "openai" else "Gemini API key"
            env_hint = "OPENAI_API_KEY" if selected_provider == "openai" else "GEMINI_API_KEY or GOOGLE_API_KEY"
            api_key = st.text_input(
                credential_label,
                type="password",
                key=f"{selected_provider}_api_key_input",
                help=f"If left blank, the app will use {env_hint} from the environment.",
            )
            st.caption(
                "Credentials entered here are used for this local Streamlit session only. "
                "They are not written to project config files."
            )
            st.caption(
                "Suggested model names reflect currently documented provider options. "
                "If your account exposes a different model, enable 'Type custom model'."
            )

    if st.button("Run analysis", type="primary", use_container_width=True):
        try:
            config = load_config()
            uploaded_raw_paths = save_uploaded_raw_files(
                paths,
                {
                    "policy_data": upload_policy_data,
                    "claims_data": upload_claims_data,
                    "asset_data": upload_asset_data,
                    "financial_balances": upload_financial_balances,
                    "reporting_metrics": upload_reporting_metrics,
                },
            )
            workflow_assumption_overrides = {
                key: value
                for key, value in {
                    "premium_multiplier": float(workflow_premium_multiplier),
                    "claims_multiplier": float(workflow_claims_multiplier),
                    "reserve_multiplier": float(workflow_reserve_multiplier),
                    "csm_multiplier": float(workflow_csm_multiplier),
                    "asset_return_shift": float(workflow_asset_return_shift),
                    "capital_multiplier": float(workflow_capital_multiplier),
                }.items()
                if abs(value - (0.0 if key == "asset_return_shift" else 1.0)) > 1e-9
            }
            validation_override_params = {
                key: value
                for key, value in {
                    "reserve_tolerance": float(validation_reserve_tolerance),
                    "csm_tolerance": float(validation_csm_tolerance),
                    "capital_tolerance": float(validation_capital_tolerance),
                }.items()
                if abs(value - 0.05) > 1e-9
            }
            forecast_override_params = {
                key: value
                for key, value in {
                    "selection_metric": forecast_selection_metric,
                    "error_tolerance_pct": float(forecast_error_tolerance_pct),
                    "gb_max_depth": int(forecast_gb_max_depth),
                    "gb_n_estimators": int(forecast_gb_n_estimators),
                    "gb_learning_rate": float(forecast_gb_learning_rate),
                    "forecast_horizon_quarters": int(forecast_horizon_quarters),
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
            assistant_state = {
                "file_format": "csv",
                "uploaded_raw_paths": uploaded_raw_paths,
                "workflow_assumption_overrides": workflow_assumption_overrides,
                "validation_override_params": validation_override_params,
                "forecast_override_params": forecast_override_params,
            }
            assistant = build_streamlit_assistant(
                config,
                provider=selected_provider,
                model_name=model_name,
                api_key=api_key,
            )
            response = assistant.answer(
                question,
                state=assistant_state,
                top_k=top_k,
                file_format="csv",
                prompt_mode="management_qa",
            )
        except FileNotFoundError:
            st.info("Chatbot index not found. Run `case-study chatbot-index` or `case-study run` first.")
            return
        except ValueError as exc:
            st.error(str(exc))
            return

        answer_col, support_col = st.columns([1.35, 1.0])
        with answer_col:
            st.subheader("Answer")
            st.markdown(f'<div class="air-answer">{response.answer}</div>', unsafe_allow_html=True)
            if response.citations:
                st.caption(f"Citations: {' '.join(f'[{citation}]' for citation in response.citations)}")
        with support_col:
            st.subheader("Tools Used")
            if response.tools_used:
                st.markdown("".join(f'<span class="air-chip">{tool}</span>' for tool in response.tools_used), unsafe_allow_html=True)
            else:
                st.write("No structured tools were used.")
            st.caption(
                f"Provider: {selected_provider}"
                + (f" | Model: {model_name or DEFAULT_PROVIDER_MODELS[selected_provider]}" if selected_provider != "mock" else " | Model: mock")
            )

        source_tab, retrieved_tab, tool_tab = st.tabs(["Supporting Sources", "Retrieved Documents", "Tool Trace"])
        with source_tab:
            render_table_section(
                "Supporting Sources",
                pd.DataFrame(response.sources) if response.sources else None,
                caption="Artifact-level support passed to the answer layer.",
            )
        with retrieved_tab:
            rows = []
            for hit in response.sources:
                rows.append(
                    {
                        "document": hit.get("document"),
                        "dataset": hit.get("source_dataset"),
                        "filters": hit.get("source_filters"),
                        "score": hit.get("score"),
                    }
                )
            render_table_section(
                "Retrieved Documents",
                pd.DataFrame(rows) if rows else None,
                caption="Ranked retrieval context used for grounding.",
            )
        with tool_tab:
            st.markdown('<div class="air-section">', unsafe_allow_html=True)
            st.subheader("Tool Trace")
            st.write(", ".join(response.tools_used) if response.tools_used else "No structured tools were used.")
            if show_tool_outputs and response.tool_outputs:
                st.json(response.tool_outputs)
            elif response.tool_outputs:
                st.caption("Enable 'Show tool outputs' to inspect structured tool payloads.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="air-section">', unsafe_allow_html=True)
        st.subheader("Reviewer Feedback")
        feedback_cols = st.columns(3)
        with feedback_cols[0]:
            feedback_rating = st.selectbox("Accuracy", options=["accurate", "needs_review", "inaccurate"], key="feedback_rating")
        with feedback_cols[1]:
            feedback_grounded = st.selectbox("Grounding", options=["grounded", "partially_grounded", "weakly_grounded"], key="feedback_grounded")
        with feedback_cols[2]:
            feedback_helpful = st.selectbox("Helpfulness", options=["helpful", "mixed", "not_helpful"], key="feedback_helpful")
        feedback_comment = st.text_area("Reviewer comment", key="feedback_comment", height=80)
        if st.button("Save feedback", use_container_width=False):
            feedback_path = LLMEvaluator(config=config).record_feedback(
                question=question,
                answer=response.answer,
                rating=feedback_rating,
                grounded=feedback_grounded,
                helpful=feedback_helpful,
                comment=feedback_comment,
                metadata={
                    "provider": selected_provider,
                    "model_name": model_name,
                    "citations": response.citations,
                    "tools_used": response.tools_used,
                },
            )
            st.success(f"Feedback saved to {feedback_path.name}.")
        st.markdown("</div>", unsafe_allow_html=True)




def render_analyst_review(paths: Any) -> None:
    config = load_config()
    queue_path = find_existing_path(paths.reports / "reporting", "analyst_review_queue")
    summary_path = find_existing_path(paths.reports / "reporting", "analyst_review_summary")
    overview_path = paths.reports / "reporting" / "analyst_review_overview.json"
    status_log_path = paths.reports / "reporting" / "analyst_review_status_log.jsonl"
    queue = load_table(queue_path)
    summary = load_table(summary_path)
    overview = load_json(overview_path)
    status_updates = load_review_status_updates(config)
    if queue is not None and not queue.empty:
        queue = apply_review_status_updates(queue, status_updates)

    total_items = int(overview.get("total_items", 0)) if overview else int(len(queue)) if queue is not None else 0
    pending_review = int((queue["reviewer_status"] == "pending_review").sum()) if queue is not None and not queue.empty else int(overview.get("pending_review", 0)) if overview else 0
    critical_items = int((queue["priority_label"] == "critical").sum()) if queue is not None and not queue.empty else int(overview.get("critical_items", 0)) if overview else 0
    sources_with_items = int(queue["review_source"].nunique()) if queue is not None and not queue.empty else int(overview.get("sources_with_items", 0)) if overview else 0

    render_page_header(
        "Analyst Review",
        "Unified review queue across AI-detected insights, anomaly investigation, narrative quality checks, and LLM benchmark exceptions.",
        chips=["Analyst oversight", "Prioritized triage", "AI-assisted workflow"],
    )
    render_metric_cards(
        [
            {"label": "Queue Items", "value": f"{total_items:,}", "note": "Consolidated analyst review items"},
            {"label": "Pending Review", "value": f"{pending_review:,}", "note": "Items awaiting analyst action"},
            {"label": "Critical", "value": f"{critical_items:,}", "note": "Highest-priority review items"},
            {"label": "Sources", "value": f"{sources_with_items:,}", "note": "AI layers contributing to the queue"},
        ]
    )
    render_page_explainer(
        "How to read this page",
        [
            ("Queue Items", "Combined analyst review queue across insights, anomaly investigation, narrative quality, and LLM evaluation."),
            ("Priority", "Priority labels are derived from configured scoring thresholds so analysts can triage quickly."),
            ("Suggested Actions", "Structured next steps inferred from the queue contents rather than free-form advice."),
            ("Update Status", "Persist analyst decisions so review outcomes survive workflow reruns and can be referenced later."),
        ],
    )
    if queue is None or queue.empty:
        st.info("No analyst review queue artifact is available yet. Rerun the workflow to generate the consolidated review queue.")
        return

    queue_tab, summary_tab, action_tab, update_tab = st.tabs(["Review Queue", "By Source", "Suggested Actions", "Update Status"])
    with queue_tab:
        filter_cols = st.columns(3)
        with filter_cols[0]:
            source_options = ["All", *sorted(queue["review_source"].dropna().unique().tolist())]
            selected_source = st.selectbox("Review source", options=source_options, index=0)
        with filter_cols[1]:
            priority_options = ["All", "critical", "high", "medium", "monitor"]
            selected_priority = st.selectbox("Priority", options=priority_options, index=0)
        with filter_cols[2]:
            max_rows = min(100, len(queue))
            row_count = st.slider("Rows to show", min_value=1, max_value=max_rows, value=min(25, max_rows))
        queue_view = queue.copy()
        if selected_source != "All":
            queue_view = queue_view.loc[queue_view["review_source"] == selected_source]
        if selected_priority != "All":
            queue_view = queue_view.loc[queue_view["priority_label"] == selected_priority]
        render_table_section(
            "Analyst Review Queue",
            format_display_frame(queue_view),
            caption="Use this queue to triage AI-generated reporting items before final management review.",
            rows=row_count,
        )

    with summary_tab:
        render_table_section(
            "Queue Summary by Source",
            format_display_frame(summary),
            caption="Counts and average priority by AI layer.",
        )
        if summary is not None and not summary.empty:
            chart_frame = summary.copy()
            render_plotly_bar_chart(
                chart_frame,
                x_col="review_source",
                y_col="item_count",
                title="Review Items by Source",
                y_title="Item count",
            )

    with action_tab:
        action_view = (
            queue.groupby("recommended_action", as_index=False)
            .agg(item_count=("review_item_id", "count"), average_priority=("priority_score", "mean"))
            .sort_values(["item_count", "average_priority"], ascending=[False, False])
            .reset_index(drop=True)
        )
        if not action_view.empty:
            action_view["average_priority"] = action_view["average_priority"].round(4)
        render_table_section(
            "Suggested Analyst Actions",
            format_display_frame(action_view),
            caption="Structured next steps inferred from the current queue composition.",
        )

    with update_tab:
        st.markdown('<div class="air-section">', unsafe_allow_html=True)
        st.subheader("Update Review Status")
        st.caption("Persist analyst decisions for queue items so the review workflow survives reloads and can be referenced later.")
        selection_view = queue[["review_item_id", "review_source", "priority_label", "issue_summary", "reviewer_status"]].copy()
        selection_view["display_label"] = selection_view.apply(
            lambda row: (
                f"{row['review_item_id']} | {format_identifier_label(str(row['priority_label']))} | "
                f"{format_identifier_label(str(row['review_source']))} | {str(row['issue_summary'])[:80]}"
            ),
            axis=1,
        )
        selected_label = st.selectbox("Review item", options=selection_view["display_label"].tolist())
        selected_row = selection_view.loc[selection_view["display_label"] == selected_label].iloc[0]
        detail_row = queue.loc[queue["review_item_id"] == selected_row["review_item_id"]].iloc[0]
        st.markdown(f"**Issue:** {detail_row['issue_summary']}")
        st.markdown(f"**Recommended action:** {detail_row['recommended_action']}")
        status_cols = st.columns(2)
        with status_cols[0]:
            new_status = st.selectbox(
                "New status",
                options=["pending_review", "accepted", "resolved", "escalated"],
                index=["pending_review", "accepted", "resolved", "escalated"].index(str(detail_row["reviewer_status"])) if str(detail_row["reviewer_status"]) in ["pending_review", "accepted", "resolved", "escalated"] else 0,
            )
        with status_cols[1]:
            review_owner = st.text_input("Reviewer", value=str(detail_row.get("review_owner", "") or ""))
        review_comment = st.text_area("Comment", value=str(detail_row.get("review_comment", "") or ""), height=90)
        if st.button("Save review status", use_container_width=False):
            output_path = record_review_status(
                config=config,
                review_item_id=str(detail_row["review_item_id"]),
                reviewer_status=new_status,
                comment=review_comment,
                review_owner=review_owner,
            )
            st.success(f"Review status saved to {output_path.name}.")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    render_download_buttons(
        "Review Queue Exports",
        [
            ("Download analyst_review_queue", queue_path) if queue_path else ("", Path()),
            ("Download analyst_review_summary", summary_path) if summary_path else ("", Path()),
            ("Download analyst_review_overview", overview_path),
            ("Download analyst_review_status_log", status_log_path) if status_log_path.exists() else ("", Path()),
        ],
    )


def render_llm_evaluation(paths: Any) -> None:
    render_page_header(
        "LLM Evaluation",
        "Deterministic benchmark scoring and reviewer feedback over grounded LLM outputs.",
        chips=["Governed evaluation", "Benchmark queries", "Reviewer feedback"],
    )
    eval_results_path = find_existing_path(paths.reports / "reporting", "llm_eval_results")
    eval_feedback_path = find_existing_path(paths.reports / "reporting", "llm_feedback_summary")
    eval_results = load_table(eval_results_path)
    eval_feedback = load_table(eval_feedback_path)
    summary_path = paths.reports / "reporting" / "llm_eval_summary.json"
    summary_payload = load_json(summary_path)
    if eval_results is None or summary_payload is None:
        st.info("LLM evaluation artifacts not found. Run `case-study evaluate-llm` or `case-study run` first.")
        return

    render_metric_cards([
        {"label": "Benchmark Queries", "value": str(summary_payload.get("benchmark_queries", 0)), "note": "Deterministic evaluation set"},
        {"label": "Pass Rate", "value": fmt_pct(float(summary_payload.get("pass_rate", 0.0))), "note": "Queries above the configured passing score"},
        {"label": "Average Score", "value": f"{float(summary_payload.get('average_score', 0.0)):.2f}", "note": "Weighted grounded-answer score"},
        {"label": "Review Items", "value": str(summary_payload.get("review_items", 0)), "note": "Items flagged for analyst review"},
    ])

    with st.expander("How LLM evaluation scoring works", expanded=False):
        st.markdown(
            """
**Grounded**: The assistant returned a normal answer supported by retrieved evidence instead of falling back for lack of evidence.\n
**Artifact match**: The answer path touched the datasets or structured artifacts expected for the benchmark question, either through retrieval or tool outputs.\n
**Tool match**: The assistant used the structured reporting tools expected for that question type, such as validation, forecast, explainability, movement, or scenario tools.\n
**Citation**: The final answer preserved at least one source reference so a reviewer can trace it back to supporting artifacts.\n
**Used fallback**: The assistant returned the governed fallback response because the available artifacts were not strong enough to answer reliably.
            """
        )
        st.caption("The overall score is a weighted combination of these checks. A low score does not always mean the answer is wrong; it often points us to retrieval, artifact mapping, or tool-routing improvements.")

    benchmark_tab, review_tab, feedback_tab = st.tabs(["Benchmark Results", "Review Queue", "Reviewer Feedback"])
    with benchmark_tab:
        render_table_section(
            "Benchmark Results",
            format_display_frame(eval_results),
            caption="Per-query scoring across grounding, artifact coverage, tool coverage, and citations.",
        )

    with review_tab:
        review_items = eval_results.loc[eval_results["evaluation_label"] == "review"].copy()
        if review_items.empty:
            st.success("No benchmark items are currently flagged for review.")
        else:
            st.markdown('<div class="air-section">', unsafe_allow_html=True)
            st.subheader("Failed Examples")
            st.caption("These benchmark items scored below the configured threshold. Use the artifact and tool gaps below to refine prompts, retrieval, or tool routing.")
            max_examples = min(5, len(review_items))
            example_count = st.slider("Examples to inspect", min_value=1, max_value=max_examples, value=max_examples, key="llm_review_example_count")
            inspect_rows = review_items.sort_values(["overall_score", "grounded_score", "artifact_match_score"]).head(example_count)
            for row in inspect_rows.itertuples(index=False):
                improvement_area = infer_llm_improvement_area(row)
                with st.expander(f"{row.query_id} | {row.category} | score {float(row.overall_score):.2f}", expanded=False):
                    st.markdown(f"**Question:** {row.question}")
                    st.markdown(f"**Answer:** {row.answer}")
                    st.markdown(f"**Likely improvement area:** {improvement_area}")
                    action_cols = st.columns(3)
                    with action_cols[0]:
                        if st.button(f"Send to Chatbot Q&A", key=f"send_{row.query_id}", use_container_width=True):
                            st.session_state["chat_question"] = row.question
                            st.session_state["current_page"] = "Chatbot Q&A"
                            st.rerun()
                    with action_cols[1]:
                        if st.button(f"Use as Scenario Prompt", key=f"scenario_{row.query_id}", use_container_width=True):
                            st.session_state["chat_question"] = row.question
                            st.session_state["current_page"] = "Chatbot Q&A"
                            st.session_state["open_chatbot_review_hint"] = improvement_area
                            st.rerun()
                    with action_cols[2]:
                        st.caption(f"Category: {row.category}")
                    gap_left, gap_right = st.columns(2)
                    with gap_left:
                        st.markdown(f"**Expected artifacts:** {row.expected_artifacts or 'n/a'}")
                        matched_artifacts = row.artifact_matches if pd.notna(row.artifact_matches) and str(row.artifact_matches).strip() else "none"
                        st.markdown(f"**Matched artifacts:** {matched_artifacts}")
                        st.markdown(f"**Expected tools:** {row.expected_tools or 'n/a'}")
                        matched_tools = row.tool_matches if pd.notna(row.tool_matches) and str(row.tool_matches).strip() else "none"
                        st.markdown(f"**Matched tools:** {matched_tools}")
                    with gap_right:
                        st.markdown(f"**Tools used:** {row.tools_used or 'none'}")
                        st.markdown(f"**Citations:** {int(row.citation_count)}")
                        st.markdown(f"**Used fallback:** {'yes' if bool(row.used_fallback) else 'no'}")
                        st.markdown(f"**Sources returned:** {int(row.source_count)}")
                    score_frame = pd.DataFrame(
                        [
                            {"dimension": "grounded", "score": float(row.grounded_score), "meaning": "Answered from evidence instead of defaulting to the governed fallback."},
                            {"dimension": "artifact match", "score": float(row.artifact_match_score), "meaning": "Used the reporting artifacts expected for this benchmark question."},
                            {"dimension": "tool match", "score": float(row.tool_match_score), "meaning": "Used the structured tools expected for this question type."},
                            {"dimension": "citation", "score": float(row.citation_score), "meaning": "Preserved at least one source reference in the final answer."},
                            {"dimension": "overall", "score": float(row.overall_score), "meaning": "Weighted summary of the evaluation checks above."},
                        ]
                    )
                    st.dataframe(score_frame, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with feedback_tab:
        render_table_section(
            "Reviewer Feedback Summary",
            format_display_frame(eval_feedback),
            caption="Aggregated reviewer feedback captured from the chatbot page.",
        )

    render_download_buttons(
        "Evaluation Exports",
        [
            ("Download llm_eval_results", eval_results_path) if eval_results_path else ("", Path()),
            ("Download llm_feedback_summary", eval_feedback_path) if eval_feedback_path else ("", Path()),
            ("Download llm_eval_summary", summary_path),
        ],
    )


def render_sidebar(theme_name: str, paths: Any, config: Any) -> tuple[str, str]:
    run_status = summarize_run_status(paths)
    st.sidebar.markdown("## Reporting Workspace")
    st.sidebar.caption("Executive dashboard and analytical assistant")
    selected_theme = st.sidebar.selectbox("Theme", options=list(THEMES.keys()), index=list(THEMES.keys()).index(theme_name))
    if st.sidebar.button("Refresh artifacts", use_container_width=True):
        if hasattr(st, "cache_data"):
            st.cache_data.clear()
        if hasattr(st, "cache_resource"):
            st.cache_resource.clear()
        st.session_state["artifact_refresh_count"] = st.session_state.get("artifact_refresh_count", 0) + 1
        st.rerun()
    st.sidebar.caption("Use after rerunning the workflow to reload the latest files.")
    if "current_page" not in st.session_state or st.session_state["current_page"] not in PAGES:
        st.session_state["current_page"] = PAGES[0]
    page = st.sidebar.radio("Pages", PAGES, key="current_page")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Run status:** {run_status['status']}")
    st.sidebar.markdown(f"**Scope:** {run_status['run_scope']}")
    st.sidebar.markdown(f"**Latest stage:** {run_status['latest_stage']}")
    st.sidebar.markdown(f"**Last artifact update:** {run_status['last_artifact_update']}")
    status_cols = st.sidebar.columns(2)
    status_cols[0].metric("Stages", run_status["stage_count"])
    status_cols[1].metric("Figures", run_status["figure_count"])
    chat_cols = st.sidebar.columns(2)
    chat_cols[0].metric("Chats", run_status["chat_count"])
    chat_cols[1].metric("Theme", "Dark" if "Dark" in selected_theme else "Light")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Environment:** {config.app.env}")
    st.sidebar.markdown(f"**Artifacts:** {paths.root}")
    return selected_theme, page


def main() -> None:
    """Render the Streamlit dashboard."""

    st.set_page_config(
        page_title="AI Insurance Reporting Demo",
        page_icon=str(Path(__file__).with_name("dashboard_icon.png")),
        layout="wide",
        initial_sidebar_state="expanded",
    )
    ensure_sidebar_open_on_first_load()
    config = load_config()
    paths = ensure_artifact_dirs(config)
    default_theme = st.session_state.get("theme_name", "Executive Light")
    selected_theme, page = render_sidebar(default_theme, paths, config)
    st.session_state["theme_name"] = selected_theme
    apply_theme(selected_theme)

    if page == "Data overview":
        render_data_overview(paths)
    elif page == "Validation results":
        render_validation_results(paths)
    elif page == "Forecast results":
        render_forecast_results(paths)
    elif page == "Explainability":
        render_explainability(paths)
    elif page == "Narrative reporting":
        render_narrative_reporting(paths)
    elif page == "Visualizations":
        render_visualizations(paths)
    elif page == "Scenario comparison":
        render_scenario_comparison(paths)
    elif page == "Analyst Review":
        render_analyst_review(paths)
    elif page == "Chatbot Q&A":
        render_chatbot(paths)
    elif page == "LLM Evaluation":
        render_llm_evaluation(paths)


if __name__ == "__main__":
    main()
