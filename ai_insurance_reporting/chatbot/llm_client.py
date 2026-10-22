"""Pluggable LLM clients for the chatbot."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
import json
from typing import Callable, Protocol
from urllib import error, request


class LLMClient(Protocol):
    """Protocol for pluggable LLM generation."""

    def generate(self, prompt: str) -> str:
        """Generate an answer from a prompt."""


@dataclass(slots=True)
class MockLLMClient:
    """Deterministic local fallback for grounded answer generation."""

    def generate(self, prompt: str) -> str:
        question = self._extract_question(prompt)
        tool_output_text = self._extract_block(prompt, "Tool outputs:", "Retrieved context:")
        if tool_output_text and tool_output_text.strip() not in {"{}", ""}:
            tool_summary = self._summarize_tool_outputs(tool_output_text, question=question)
            if tool_summary:
                return tool_summary

        context_items = self._parse_context(prompt)
        if not context_items:
            return "The reporting artifacts do not contain enough information to answer this question."

        management_mode = "assisting senior management reviewing an insurance reporting package" in prompt.lower()
        primary = context_items[0]
        primary_content = primary["content"].rstrip(".")
        answer_parts: list[str] = []

        if management_mode:
            answer_parts.append(f"Management view: {primary_content}.")
        else:
            answer_parts.append(primary_content + ".")

        if len(context_items) > 1:
            secondary = context_items[1]["content"].rstrip(".")
            connector = self._choose_connector(primary_content, secondary, management_mode=management_mode)
            answer_parts.append(f"{connector} {secondary}.")

        if len(context_items) > 2:
            tertiary = context_items[2]["content"].rstrip(".")
            if management_mode:
                answer_parts.append(
                    f"From a management perspective, supporting evidence indicates that {self._lowercase_first(tertiary)}."
                )
            else:
                answer_parts.append(f"Supporting evidence also indicates that {self._lowercase_first(tertiary)}.")

        return " ".join(answer_parts)

    def _parse_context(self, prompt: str) -> list[dict[str, str]]:
        context_lines = [line.strip() for line in prompt.splitlines() if line.strip().startswith("[")]
        items: list[dict[str, str]] = []
        pattern = re.compile(
            r"^\[(?P<document_id>[^\]]+)\]\s+"
            r"(?P<title>.*?)\s+\|\s+dataset=(?P<dataset>.*?)\s+\|\s+"
            r"filters=(?P<filters>.*?)\s+\|\s+content=(?P<content>.*)$"
        )
        for line in context_lines:
            match = pattern.match(line)
            if not match:
                continue
            items.append(match.groupdict())
        return items

    def _extract_block(self, prompt: str, start_marker: str, end_marker: str) -> str:
        if start_marker not in prompt:
            return ""
        start = prompt.index(start_marker) + len(start_marker)
        end = prompt.index(end_marker) if end_marker in prompt[start:] else len(prompt)
        return prompt[start:end].strip()

    def _extract_question(self, prompt: str) -> str:
        block = self._extract_block(prompt, "Question:", "Tool outputs:")
        return block.strip()

    def _summarize_tool_outputs(self, tool_output_text: str, *, question: str) -> str:
        try:
            import json

            payload = json.loads(tool_output_text)
        except Exception:
            return ""

        if not payload:
            return ""

        question_lower = question.lower()
        parts: list[str] = []
        if "WorkflowExecutionTool" in payload:
            workflow = payload["WorkflowExecutionTool"]
            mode = workflow.get("mode", "workflow")
            requested = workflow.get("requested_stage_labels") or workflow.get("requested_agents") or []
            executed = workflow.get("executed_stage_labels") or workflow.get("executed_agents") or []
            if requested:
                if mode == "full":
                    parts.append("Management view: The full reporting workflow was rerun successfully.")
                else:
                    parts.append(
                        f"Management view: The requested workflow stages were rerun successfully for {', '.join(requested)}."
                    )
            if executed:
                parts.append(f"Executed stages included {', '.join(executed)}.")
            governance_log_path = workflow.get("governance_log_path", "")
            if governance_log_path:
                parts.append(f"Updated governance output was written to {governance_log_path}.")

        if "ScenarioRunTool" in payload:
            scenario = payload["ScenarioRunTool"]
            summary = scenario.get("summary") or {}
            comparison = scenario.get("comparison") or {}
            deltas = comparison.get("comparison_metrics") or []
            biggest_delta = max(deltas, key=lambda item: abs(item.get("change_pct", 0.0))) if deltas else {}
            if summary:
                parts.append(
                    f"Management view: Scenario {scenario.get('scenario_name', 'scenario')} reran successfully through "
                    f"{summary.get('latest_quarter', 'the latest quarter')} with a synthetic capital-to-liability proxy ratio of "
                    f"{summary.get('capital_ratio', 'n/a')}."
                )
            if biggest_delta:
                relative_change_text = self._format_relative_change(
                    biggest_delta.get("change_pct", 0.0),
                    baseline_value=biggest_delta.get("baseline_value", 0.0),
                    scenario_value=biggest_delta.get("scenario_value", 0.0),
                )
                parts.append(
                    f"The largest balance sheet movement versus baseline is in {biggest_delta.get('metric_name', 'the selected metric')}, "
                    f"changing by {self._format_numeric_value(biggest_delta.get('change', 'n/a'))}. {relative_change_text}"
                )
            forecast_deltas = comparison.get("forecast_comparison") or []
            if forecast_deltas:
                capital_outlook = next(
                    (item for item in forecast_deltas if item.get("metric_name") == "capital_ratio"),
                    forecast_deltas[0],
                )
                parts.append(
                    f"Scenario forecast comparisons indicate {self._metric_label(capital_outlook.get('metric_name', 'the outlook metric'))} "
                    f"moves from {capital_outlook.get('baseline_value', 'n/a')} to {capital_outlook.get('scenario_value', 'n/a')}."
                )

        if "ForecastComparisonTool" in payload:
            forecast = payload["ForecastComparisonTool"]
            best_model = forecast.get("best_model") or {}
            selected_target = forecast.get("selected_target") or "the selected metric"
            insight_candidates = forecast.get("insight_candidates") or []
            segment_movement_question = any(
                token in question_lower for token in {"largest", "increase", "increases", "increased", "decrease", "decreases", "decreased", "segment"}
            )
            if segment_movement_question and insight_candidates:
                primary = insight_candidates[0]
                metric_label = self._metric_label(primary.get("metric", selected_target))
                pct = primary.get("percentage_change", "n/a")
                z_score = primary.get("z_score", "n/a")
                change_value = float(primary.get("absolute_change", 0.0) or 0.0)
                increase_question = any(token in question_lower for token in {"increase", "increases", "increased", "rise", "rises"})
                decrease_question = any(token in question_lower for token in {"decrease", "decreases", "decreased", "fall", "falls", "decline"})
                if increase_question and change_value <= 0:
                    parts.append(
                        f"The saved insight summary for {primary.get('quarter', 'the latest forecast quarter')} does not show a projected increase in {metric_label}. "
                        f"Instead, the strongest projected movement is a decrease in {primary.get('product', 'the leading product')} / {primary.get('region', 'the leading region')}, "
                        f"moving from {primary.get('latest_actual', 'n/a')} to {primary.get('forecast', 'n/a')}."
                    )
                elif decrease_question and change_value >= 0:
                    parts.append(
                        f"The saved insight summary for {primary.get('quarter', 'the latest forecast quarter')} does not show a projected decrease in {metric_label}. "
                        f"Instead, the strongest projected movement is an increase in {primary.get('product', 'the leading product')} / {primary.get('region', 'the leading region')}, "
                        f"moving from {primary.get('latest_actual', 'n/a')} to {primary.get('forecast', 'n/a')}."
                    )
                else:
                    parts.append(
                        f"Based on the saved insight summary for {primary.get('quarter', 'the latest forecast quarter')}, "
                        f"{primary.get('product', 'the leading product')} / {primary.get('region', 'the leading region')} "
                        f"shows the largest projected change in {metric_label}, moving from {primary.get('latest_actual', 'n/a')} "
                        f"to {primary.get('forecast', 'n/a')}."
                    )
                parts.append(
                    f"The projected percentage change is {pct}, with a z score of {z_score} and a severity classification of "
                    f"{primary.get('severity_classification', 'n/a')}."
                )
                if len(insight_candidates) > 1:
                    secondary = insight_candidates[1]
                    parts.append(
                        f"Other notable movement is visible in {secondary.get('product', 'the next segment')} / "
                        f"{secondary.get('region', 'the next region')}."
                    )
            if best_model and not segment_movement_question:
                parts.append(
                    f"Management view: For {self._metric_label(selected_target)}, the best-performing model is "
                    f"{best_model.get('model_family', 'the available model')} with MAE {best_model.get('mae', 'n/a')} "
                    f"and RMSE {best_model.get('rmse', 'n/a')}."
                )
            forecast_rows = forecast.get("forecast_rows") or []
            if forecast_rows and not segment_movement_question:
                first = forecast_rows[0]
                parts.append(
                    f"The forward outlook indicates {self._metric_label(selected_target)} of {first.get('forecast_value', 'n/a')} "
                    f"for {first.get('forecast_quarter', 'the next quarter')}."
                )

        if "ValidationSummaryTool" in payload:
            validation = payload["ValidationSummaryTool"]
            latest = (validation.get("validation_summary") or [{}])[0]
            if latest:
                parts.append(
                    f"Control results for {latest.get('quarter', validation.get('latest_quarter', 'the latest quarter'))} "
                    f"show a validation pass rate of {latest.get('validation_pass_rate', 'n/a')} with "
                    f"{latest.get('records_with_issues', 'n/a')} records with issues and "
                    f"{latest.get('anomaly_count', validation.get('anomaly_count_total', 'n/a'))} anomalies."
                )
            material = validation.get("material_exceptions") or []
            if material:
                top_issue = material[0]
                issue_label = top_issue.get("anomaly_type") or top_issue.get("validation_issue") or "validation issue"
                explanation = top_issue.get("explanation_text") or top_issue.get("likely_drivers") or ""
                if explanation:
                    parts.append(
                        f"The most material exception appears to be {issue_label}, with supporting evidence indicating {self._lowercase_first(str(explanation).rstrip('.'))}."
                    )
                else:
                    parts.append(f"The most material open exception appears to be {issue_label}.")

        if "MovementAnalysisTool" in payload:
            movement = payload["MovementAnalysisTool"]
            selected_scope = movement.get("selected_scope", "segment")
            portfolio_summary = movement.get("portfolio_summary") or {}
            summary_rows = movement.get("movement_summary") or []
            if selected_scope == "portfolio" and portfolio_summary:
                top_segments = movement.get("top_segments") or summary_rows
                parts.extend(self._summarize_portfolio_movement(portfolio_summary, top_segments, question_lower=question_lower))
            elif summary_rows:
                primary = summary_rows[0]
                movement_steps = movement.get("movement_steps") or []
                parts.extend(self._summarize_movement(primary, movement_steps, question_lower=question_lower))

        if "ExplainabilityTool" in payload:
            explain = payload["ExplainabilityTool"]
            drivers = explain.get("shap_top_drivers") or []
            if drivers:
                top_driver = drivers[0]
                if any(token in question_lower for token in {"factor", "factors", "driver", "drivers", "drove", "driven", "why"}):
                    top_names = [str(row.get("feature_name", "")) for row in drivers[:4] if row.get("feature_name")]
                    if top_names:
                        metric_label = self._metric_label(top_driver.get("target_name", explain.get("selected_target", "the selected target")))
                        parts.append(
                            f"Supporting explainability signals for {metric_label} include {', '.join(top_names)}."
                        )
                else:
                    parts.append(
                        f"Key model drivers indicate that {top_driver.get('feature_name', 'the leading feature')} "
                        f"is the strongest SHAP contributor for "
                        f"{self._metric_label(top_driver.get('target_name', explain.get('selected_target', 'the selected target')))}."
                    )

        if "NarrativeLookupTool" in payload:
            narrative = payload["NarrativeLookupTool"]
            sections = narrative.get("narrative_sections") or []
            if sections:
                parts.append(sections[0].get("statement_text", ""))

        if "FigureLookupTool" in payload:
            figures = payload["FigureLookupTool"].get("figures") or []
            if figures:
                parts.append(f"Relevant visuals include {figures[0].get('chart_title', 'the available reporting figure')}.")

        if "RunSummaryTool" in payload:
            run_summary = payload["RunSummaryTool"]
            completed = run_summary.get("completed_stages") or []
            if completed:
                parts.append(f"Completed workflow stages include {', '.join(completed[:5])}.")

        return " ".join(part for part in parts if part).strip()

    def _summarize_movement(
        self,
        primary: dict[str, object],
        movement_steps: list[dict[str, object]],
        *,
        question_lower: str,
    ) -> list[str]:
        metric_label = self._metric_label(primary.get("metric", "the selected metric"))
        product = primary.get("product", "the selected product")
        region = primary.get("region", "the selected region")
        latest_quarter = primary.get("quarter", "the latest quarter")
        opening_value = primary.get("opening_value", "n/a")
        closing_value = primary.get("closing_value", "n/a")
        net_change = float(primary.get("net_change", 0.0) or 0.0)
        change_direction = "increase" if net_change >= 0 else "decrease"
        parts = [
            f"Management view: In {latest_quarter}, {metric_label} for {product} / {region} moved from {opening_value} to {closing_value}, representing a net {change_direction} of {abs(net_change):,.2f}."
        ]

        segment_steps = [
            row
            for row in movement_steps
            if str(row.get("metric", "")) == str(primary.get("metric", ""))
            and str(row.get("product", "")) == str(product)
            and str(row.get("region", "")) == str(region)
        ]
        non_residual = [row for row in segment_steps if str(row.get("movement_step", "")) != "residual"]
        positive_steps = sorted(
            [row for row in non_residual if float(row.get("movement_amount", 0.0) or 0.0) > 0],
            key=lambda row: float(row.get("movement_amount", 0.0) or 0.0),
            reverse=True,
        )
        negative_steps = sorted(
            [row for row in non_residual if float(row.get("movement_amount", 0.0) or 0.0) < 0],
            key=lambda row: float(row.get("movement_amount", 0.0) or 0.0),
        )

        if net_change >= 0:
            if positive_steps:
                parts.append(
                    f"The main upward contributors were {self._format_step_list(positive_steps[:2])}."
                )
            if negative_steps:
                parts.append(
                    f"These were partly offset by {self._format_step_list(negative_steps[:2], absolute_amounts=True)}."
                )
        else:
            if negative_steps:
                parts.append(
                    f"The main downward contributors were {self._format_step_list(negative_steps[:2], absolute_amounts=True)}."
                )
            if positive_steps:
                parts.append(
                    f"These were partly offset by {self._format_step_list(positive_steps[:2])}."
                )

        if "reserve" in question_lower and not positive_steps and negative_steps:
            parts.append(
                "The bridge therefore suggests that runoff and release pressures outweighed the offsetting positive effects in the saved quarter."
            )
        return parts

    def _format_step_list(self, steps: list[dict[str, object]], *, absolute_amounts: bool = False) -> str:
        formatted: list[str] = []
        for row in steps:
            raw_amount = float(row.get("movement_amount", 0.0) or 0.0)
            display_amount = abs(raw_amount) if absolute_amounts else raw_amount
            step_name = self._humanize_step_name(str(row.get("movement_step", "step")))
            formatted.append(f"{step_name} ({display_amount:,.2f})")
        return ", ".join(formatted)

    def _humanize_step_name(self, step_name: str) -> str:
        return step_name.replace("_", " ")

    def _summarize_portfolio_movement(
        self,
        portfolio_summary: dict[str, object],
        top_segments: list[dict[str, object]],
        *,
        question_lower: str,
    ) -> list[str]:
        metric_label = self._metric_label(portfolio_summary.get("metric", "the selected metric"))
        quarter = portfolio_summary.get("quarter", "the latest quarter")
        opening_value = portfolio_summary.get("opening_value", "n/a")
        closing_value = portfolio_summary.get("closing_value", "n/a")
        net_change = float(portfolio_summary.get("net_change", 0.0) or 0.0)
        direction = "increase" if net_change >= 0 else "decrease"
        parts = [
            f"Management view: At portfolio level in {quarter}, {metric_label} moved from {opening_value} to {closing_value}, representing a net {direction} of {abs(net_change):,.2f}."
        ]

        top_positive = portfolio_summary.get("top_positive_steps") or []
        top_negative = portfolio_summary.get("top_negative_steps") or []
        if net_change >= 0:
            if top_positive:
                parts.append(f"The main upward contributors were {self._format_step_list(top_positive[:2])}.")
            if top_negative:
                parts.append(f"These were partly offset by {self._format_step_list(top_negative[:2], absolute_amounts=True)}.")
        else:
            if top_negative:
                parts.append(f"The main downward contributors were {self._format_step_list(top_negative[:2], absolute_amounts=True)}.")
            if top_positive:
                parts.append(f"These were partly offset by {self._format_step_list(top_positive[:2])}.")

        if top_segments:
            lead = top_segments[0]
            lead_direction = "increase" if float(lead.get("net_change", 0.0) or 0.0) >= 0 else "decrease"
            parts.append(
                f"The largest segment-level movement was in {lead.get('product', 'the leading product')} / {lead.get('region', 'the leading region')}, where {metric_label} showed a net {lead_direction} of {abs(float(lead.get('net_change', 0.0) or 0.0)):,.2f}."
            )
        return parts

    def _choose_connector(self, primary: str, secondary: str, *, management_mode: bool) -> str:
        primary_lower = primary.lower()
        secondary_lower = secondary.lower()
        if management_mode:
            if "forecast" in primary_lower or "forecast" in secondary_lower:
                return "The forward outlook suggests"
            if "validation" in primary_lower or "validation" in secondary_lower:
                return "Control metrics indicate"
            if "shap" in primary_lower or "feature" in secondary_lower:
                return "Key model drivers indicate"
            if "reserve" in primary_lower or "csm" in secondary_lower:
                return "Balance sheet indicators show"
            return "Supporting management evidence shows"
        if "forecast" in primary_lower or "forecast" in secondary_lower:
            return "The forecast evidence further suggests"
        if "validation" in primary_lower or "validation" in secondary_lower:
            return "Validation results also show"
        if "shap" in primary_lower or "feature" in secondary_lower:
            return "Explainability results indicate"
        return "Additional evidence shows"

    def _lowercase_first(self, text: str) -> str:
        if not text:
            return text
        return text[0].lower() + text[1:]

    def _metric_label(self, metric_name: str) -> str:
        labels = {
            "capital_ratio": "the synthetic capital-to-liability proxy ratio",
            "reserve_movement": "reserve movement",
            "csm_movement": "CSM movement",
            "claims": "claims",
            "premium": "premium income",
            "reserves": "reserves",
            "capital_proxy": "capital",
        }
        return labels.get(metric_name, metric_name)

    def _format_numeric_value(self, value: object) -> str:
        if isinstance(value, (int, float)):
            return f"{float(value):,.2f}"
        return str(value)

    def _format_relative_change(
        self,
        change_pct: object,
        *,
        baseline_value: object,
        scenario_value: object,
    ) -> str:
        try:
            change_ratio = float(change_pct)
            baseline = float(baseline_value)
            scenario = float(scenario_value)
        except (TypeError, ValueError):
            return "A stable percentage comparison is not available from the saved scenario artifacts."

        if abs(baseline) <= 1e-9:
            return "A stable percentage comparison is not available because the baseline value is near zero."

        percent_change = change_ratio * 100.0
        sign_flip = baseline * scenario < 0
        if abs(change_ratio) >= 2.0 or sign_flip:
            return (
                f"This is equivalent to {percent_change:,.2f}% versus baseline, "
                "which is unusually large because the stressed outcome moves well beyond the baseline level."
            )
        return f"This is equivalent to {percent_change:,.2f}% versus baseline."


@dataclass(slots=True)
class OpenAILLMClient:
    """Optional OpenAI-backed client."""

    model: str = "gpt-4o-mini"

    def generate(self, prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=self.model,
            input=prompt,
        )
        return response.output_text


@dataclass(slots=True)
class GeminiLLMClient:
    """Optional Gemini-backed client via the Gemini REST API."""

    model: str = "gemini-2.5-flash"
    api_base: str = "https://generativelanguage.googleapis.com/v1beta"

    def generate(self, prompt: str) -> str:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")

        endpoint = f"{self.api_base}/models/{self.model}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt,
                        }
                    ]
                }
            ]
        }
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"Gemini API request failed: {message}") from exc

        body = json.loads(raw)
        candidates = body.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [part.get("text", "") for part in parts if part.get("text")]
        return "\n".join(text_parts).strip()


@dataclass(slots=True)
class LocalLLMClient:
    """Wrapper around a local callable generator."""

    generator: Callable[[str], str]

    def generate(self, prompt: str) -> str:
        return self.generator(prompt)


def get_default_llm_client() -> LLMClient:
    """Return the default LLM client for local/offline operation."""

    provider = os.getenv("AIR_LLM_PROVIDER", "mock").lower()
    if provider == "openai":
        return OpenAILLMClient(model=os.getenv("AIR_OPENAI_MODEL", "gpt-4o-mini"))
    if provider == "gemini":
        return GeminiLLMClient(model=os.getenv("AIR_GEMINI_MODEL", "gemini-2.5-flash"))
    return MockLLMClient()
