The Society of Actuaries Research Institute does not warrant these tools as fit for use in any respect, and no warranty whatsoever should be assumed or implied by any individual of this product or its fitness for any particular purpose. Actuaries, insurers, regulators and other parties use this tool at their own risk. The Institute disclaims all responsibility for any party’s use or misuse of its tools and for any work product generated through use or misuse of the tools.



# AI Insurance Reporting Demo

`ai-insurance-reporting` is a modular Python package that implements the synthetic insurance management reporting case study used in this project. It is designed as an illustrative and governed reporting environment rather than as a production actuarial platform. The emphasis is on traceable workflow stages, explainable outputs, reproducible artifacts, and interfaces that support review of generated results.

An introductory video is made available to assist program setup and overview: https://youtu.be/JEerFxuWm18

## Purpose

The project is intended to demonstrate how AI-supported methods can be embedded within a management reporting workflow built on synthetic life insurance data. The implemented workflow includes:

* synthetic data generation for policies, claims, assets, balances, and reporting metrics
* ETL and validation
* forecasting across core reporting targets
* explainability outputs
* movement analysis
* narrative reporting and report assembly
* visualization
* scenario comparison
* retrieval, RAG, and tool-augmented question answering
* LLM evaluation and analyst review
* CLI and Streamlit interfaces

The package should therefore be read as an executable case study. It is not intended to reproduce full valuation logic or regulatory capital production.

## Use of AI

OpenAI GPT-5.2 through GPT-5.4 were used to support the development of the LLM interface for the OpenAI and Google Gemini APIs, as well as the creation of the `pyproject.toml` configuration file summarizing the required Python libraries and dependencies.

## How The System Is Organized

The program is organized as a staged workflow in which each step writes structured outputs that can be used by later stages, the dashboard, and the chatbot.

### Package layout

```text
ai\_insurance\_reporting/
  chatbot/         # Retrieval, RAG, tools, planner, and agent layer
  config/          # Config loading and runtime settings
  data/            # Synthetic data generation, ETL, validation
  explainability/  # SHAP, LIME, PDP and ICE reporting
  interface/       # CLI and Streamlit app
  models/          # Forecasting pipeline
  narrative/       # Narrative generation and quality checks
  orchestration/   # Workflow and scenario orchestration
  reporting/       # Insights, anomalies, movement, LLM evaluation, review queue
  utils/           # Logging and artifact path helpers
configs/
artifacts/
```

### If you want to inspect a specific capability, start here

|Capability|Recommended starting files|
|-|-|
|End-to-end workflow|`ai\_insurance\_reporting/orchestration/workflow.py`, `ai\_insurance\_reporting/orchestration/agents.py`|
|Synthetic data and ETL|`ai\_insurance\_reporting/data/synthetic.py`, `ai\_insurance\_reporting/data/etl.py`|
|Validation and anomaly logic|`ai\_insurance\_reporting/data/validation.py`, `ai\_insurance\_reporting/reporting/anomaly\_investigation.py`|
|Forecasting|`ai\_insurance\_reporting/models/forecasting.py`|
|Explainability|`ai\_insurance\_reporting/explainability/`, `ai\_insurance\_reporting/interface/streamlit\_app.py`|
|Narrative generation|`ai\_insurance\_reporting/narrative/generator.py`, `ai\_insurance\_reporting/narrative/quality\_check.py`, `ai\_insurance\_reporting/reporting/full\_report.py`|
|Movement analysis|`ai\_insurance\_reporting/reporting/movement.py`|
|Chatbot and tools|`ai\_insurance\_reporting/chatbot/agent.py`, `ai\_insurance\_reporting/chatbot/planner.py`, `ai\_insurance\_reporting/chatbot/tools.py`, `ai\_insurance\_reporting/chatbot/indexing.py`|
|Dashboard and CLI|`ai\_insurance\_reporting/interface/streamlit\_app.py`, `ai\_insurance\_reporting/interface/cli.py`|

## End-To-End Workflow

The baseline workflow is:

1. Generate synthetic raw datasets.
2. Build a cleaned quarterly reporting dataset.
3. Run data validation and anomaly checks.
4. Build the forecasting training frame.
5. Train forecasting models, backtest, and write forecast outputs.
6. Generate explainability artifacts.
7. Detect notable projected movements.
8. Build anomaly investigation summaries.
9. Build movement analysis and bridge summaries.
10. Create management commentary with traceability.
11. Run narrative quality checks and assemble a full report package.
12. Save reporting figures and metadata.
13. Build the chatbot index.
14. Run LLM evaluation benchmarks.
15. Build the analyst review queue.

### Implemented workflow agents

|Stage|Agent|Main purpose|Primary outputs|
|-|-|-|-|
|1|`IngestionAgent`|Generate or load source datasets|raw data files|
|2|`ValidationAgent`|Perform validation and anomaly logging|`quarterly\_validation\_summary`, `anomaly\_table`|
|3|`FeatureEngineeringAgent`|Build the training frame|`forecast\_training\_frame`|
|4|`ForecastingAgent`|Train, compare, and apply forecast models|`model\_evaluation`, `forecast\_output\_table`, `backtest\_predictions`|
|5|`ExplainabilityAgent`|Produce interpretation outputs|SHAP, LIME, PDP and ICE artifacts|
|6|`InsightDetectionAgent`|Identify notable projected movements|`insight\_summary`|
|7|`AnomalyInvestigationAgent`|Add first-pass anomaly explanations|`anomaly\_investigation`|
|8|`MovementAnalysisAgent`|Build opening-to-closing bridges|`movement\_analysis`, `movement\_bridge\_summary`|
|9|`NarrativeAgent`|Generate deterministic commentary|`narrative\_statements`, management report|
|10|`NarrativeQualityAgent`|Check commentary consistency|`narrative\_quality\_check`|
|11|`VisualizationAgent`|Persist charts and figure metadata|figure files, `figure\_metadata`|
|12|`FullReportAgent`|Assemble a fuller report package|`management\_report\_full`, report sections, manifest|
|13|`ChatbotIndexingAgent`|Refresh the retrieval corpus|chatbot index artifacts|
|14|`LLMEvaluationAgent`|Score benchmark chatbot answers|`llm\_eval\_results`, `llm\_eval\_summary`|
|15|`AnalystReviewQueueAgent`|Consolidate reviewable items|`analyst\_review\_queue`, review summary|

Scenario questions can also trigger isolated reruns under `artifacts/scenarios/`, allowing baseline versus stressed comparisons without overwriting the main outputs.

## Main Artifacts Written By The Workflow

The term `artifact` refers to a saved workflow output that can be used by later stages, the dashboard, or the chatbot. In the current implementation, all tabular workflow artifacts are persisted as CSV files. Text-based outputs remain Markdown, JSON, or JSONL as appropriate, and figures remain PNG files.

### Data and validation artifacts

|Artifact group|Representative outputs|Purpose|
|-|-|-|
|Raw data|`data/raw/policy\_data.csv` and related files|synthetic source inputs|
|Cleaned data|`data/processed/curated\_reporting\_dataset.csv`|common downstream reporting dataset|
|Validation|`validation\_flags.csv`, `quarterly\_validation\_summary.csv`, `anomaly\_table.csv`|data quality and exception monitoring|
|Anomaly review|`reports/reporting/anomaly\_investigation.csv`|first-pass investigation support|

### Forecasting and explainability artifacts

|Artifact group|Representative outputs|Purpose|
|-|-|-|
|Forecasting|`model\_evaluation.csv`, `backtest\_predictions.csv`, `forecast\_output\_table.csv`|model comparison and future values|
|Insights|`reports/reporting/insight\_summary.csv`|material projected movement detection|
|Explainability|SHAP, LIME, PDP and ICE outputs|model interpretation and review|
|Movement|`movement\_analysis.csv`, `movement\_bridge\_summary.csv`, `movement\_llm\_summary.md`|opening-to-closing movement decomposition|

### Reporting, scenario, and governance artifacts

|Artifact group|Representative outputs|Purpose|
|-|-|-|
|Narrative reporting|`narrative\_statements.csv`, markdown management reports|structured commentary|
|Full reports|`management\_report\_sections.csv`, full Markdown and HTML reports|assembled report package|
|Scenario reporting|`scenario\_impact\_summary.csv`, `scenario\_top\_impacts.csv`, `scenario\_narrative\_summary.json`|stressed comparison outputs|
|Figures|reporting PNGs and `figure\_metadata.csv`|visual layer for the dashboard and report|
|Chatbot|chatbot index, vector store, chat logs, demo query artifacts|retrieval and conversational access|
|Evaluation and review|`llm\_eval\_results.csv`, `llm\_feedback\_summary.csv`, `analyst\_review\_queue.csv`|governance, benchmarking, and review|



## Installation

### Environment requirements

The package metadata in `pyproject.toml` declares `Python >=3.13`. The project tooling is aligned to Python `3.13`, and Python `3.14` is also a reasonable target. For reproducibility, it is best to use a dedicated virtual environment rather than a shared global Python installation.

### Recommended virtual environment setup

Create a virtual environment from the project root:

```powershell
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\\.venv\\Scripts\\Activate.ps1
```

If you already have a project `.venv`, you can reuse it instead of creating a new one.

### Install the package

The recommended installation path is editable mode:

```powershell
.\\.venv\\Scripts\\python.exe -m pip install -e .
```

This makes the console scripts such as `case-study` available in the virtual environment and is the best option when you want to inspect or modify the code.

A simpler fallback is to install from `requirements.txt`:

```powershell
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
```

This installs the dependencies, but editable installation is still preferred if you want the package entry points and the most convenient local-development setup.

### Verify the environment

A quick check is to run:

```powershell
.\\.venv\\Scripts\\python.exe -m ai\_insurance\_reporting.interface.cli --help
```

If the editable installation succeeded and the virtual environment is active, the console script should also work:

```powershell
case-study --help
```

## How To Run The Program

### Baseline full run

Generate the baseline workflow outputs:

```powershell
.\\.venv\\Scripts\\python.exe -m ai\_insurance\_reporting.interface.cli run
```

If the console script is installed, the equivalent is:

```powershell
case-study run
```

If `case-study` is not available in your shell, use the `python -m ...` form above. The console script works only after the package has been installed into the environment.

### Launch the dashboard

```powershell
.\\.venv\\Scripts\\python.exe -m streamlit run ai\_insurance\_reporting\\interface\\streamlit\_app.py
```

### Common targeted reruns

Forecast only:

```powershell
case-study forecast
```

Explainability only:

```powershell
case-study explain
```

Visualization only:

```powershell
case-study visualize
```

Chatbot index rebuild:

```powershell
case-study chatbot-index
```

LLM evaluation only:

```powershell
case-study evaluate-llm
```

If `case-study` is not on your `PATH`, use:

```powershell
.\\.venv\\Scripts\\python.exe -m ai\_insurance\_reporting.interface.cli <command>
```

### Scenario and conversational reruns

The chatbot can trigger controlled workflow execution rather than only answering questions from existing artifacts.

Examples:

```powershell
case-study chatbot-ask "Rerun the full workflow" --show-tools
case-study chatbot-ask "Run validation and forecasting again" --show-tools
case-study chatbot-ask "Refresh the narrative and charts" --show-tools
case-study chatbot-ask "Rebuild the chatbot index" --show-tools
```

Scenario questions can also be expressed conversationally, for example:

```powershell
case-study chatbot-ask "What if claims increase 20% and asset returns fall 5%?" --show-tools
```

Workflow execution remains constrained to approved workflow stages. The assistant does not execute arbitrary shell commands.

## Interface And Usage Pathways

### Command-line usage

The CLI is the best path for reproducible execution, targeted reruns, and scripted workflows.

### Streamlit dashboard usage

The Streamlit dashboard is the best path for exploring outputs interactively. Pages currently include:

* data overview
* validation results
* forecast results
* explainability
* narrative reporting
* visualizations
* scenario comparison
* chatbot Q\&A
* LLM evaluation
* analyst review

### Chatbot usage and limits

The chatbot supports:

* retrieval-grounded answers
* tool-augmented analytical answers
* controlled scenario reruns
* controlled full and partial workflow reruns
* selectable LLM backends in the UI

The chatbot is designed not to answer without evidence from retrieved artifacts or structured tool outputs.

## Configuration And Override Controls

Default settings live in `configs/default.yaml`. Runtime behaviour is driven by the orchestrator, CLI commands, and interface actions rather than by generic feature flags, so the configuration focuses on paths, tolerances, reporting controls, and related parameters.

Supported environment overrides include:

* `AIR\_ENV`
* `AIR\_LOG\_LEVEL`
* `AIR\_ARTIFACTS\_DIR`
* `AIR\_DATA\_INPUT\_DIR`
* `AIR\_DATA\_PROCESSED\_DIR`
* `AIR\_INTERFACE\_DEFAULT`

AIR stands for AI insurance reporting

### CLI overrides for conversational reruns

`case-study chatbot-ask` supports structured overrides.

Synthetic assumption overrides:

* `--premium-multiplier`
* `--claims-multiplier`
* `--reserve-multiplier`
* `--csm-multiplier`
* `--asset-return-shift`
* `--capital-multiplier`

Validation overrides:

* `--reserve-tolerance`
* `--csm-tolerance`
* `--capital-tolerance`

Forecasting overrides:

* `--forecast-selection-metric`
* `--forecast-error-tolerance-pct`
* `--forecast-gb-max-depth`
* `--forecast-gb-n-estimators`
* `--forecast-gb-learning-rate`
* `--forecast-horizon-quarters`

Replacement raw input paths:

* `--policy-data-path`
* `--claims-data-path`
* `--asset-data-path`
* `--financial-balances-path`
* `--reporting-metrics-path`



## Forecasting, Validation, And Chatbot Notes

### Forecasting notes

Forecasting uses three model families:

* `baseline\_actuarial`
* `time\_series`
* `gradient\_boosting`

Targets include:

* `claims`
* `premium`
* `reserve\_movement`
* `csm\_movement`
* `capital\_ratio`

The current evaluation framework uses rolling backtests rather than a single one-quarter holdout. Because the dataset is synthetic and relatively short, some targets may still show weak or unstable `R2` values even when absolute error metrics are acceptable.

### Validation notes

The synthetic data generator includes a small number of deliberate data-quality issues so that the validation workflow produces visible examples. These include missing required values, negative values, reserve reconciliation mismatches, CSM reconciliation mismatches, and capital consistency mismatches.

### Chatbot notes

The chatbot is built in layers:

1. document indexing
2. lexical and semantic retrieval
3. optional structured tools
4. answer generation

`capital\_ratio` is currently a synthetic capital-to-liability proxy ratio, not a regulatory solvency coverage ratio.

## Governance And Evaluation Controls

Governance controls currently include:

* source citations
* tool trace capture
* chat history logging
* deterministic baseline reporting
* optional LLM-assisted draft separation
* fallback responses when evidence is insufficient
* benchmark scoring for chatbot answers
* analyst review queue and persistent reviewer status updates

The workflow now includes a deterministic LLM evaluation layer that benchmarks grounded chatbot answers against the curated demo query set and expected evidence mappings. Outputs include `llm\_eval\_results`, `llm\_feedback\_summary`, and `llm\_eval\_summary.json`. Reviewer feedback entered in Streamlit is logged separately and summarized for governance review.

### LLM providers

The chatbot supports three LLM modes:

* `mock`
* `openai`
* `gemini`

If no provider is configured, the project uses the deterministic local mock client.

The mock client exists so the chatbot and report-generation paths can run in a fully offline and reproducible way. It is useful for demonstration, testing, and governed benchmarking because it does not require credentials, network access, or a separate model server. It also keeps answer structure stable enough for regression tests.

The main limitation is that the mock client is not a real generative model. It does not reason or rewrite in the same way an external LLM would. Instead, it synthesizes answers from retrieved artifacts and structured tool outputs using deterministic formatting rules. That makes it suitable for controlled demos, but less natural for open-ended explanation, summarization, or nuanced narrative wording.

Some of the main mock-LLM logic is worth making explicit:

* it first checks whether structured tool outputs are available and, if so, prefers those over free-text retrieval
* for forecast questions, it can summarize the selected target, the best model, the saved forecast rows, and insight-detection candidates
* for validation questions, it can summarize the latest validation pass-rate view and highlight the most material saved exception
* for movement questions, it can summarize bridge outputs using opening value, closing value, net movement, main upward or downward contributors, and offsetting steps
* for explainability questions, it can summarize the leading saved SHAP drivers for the selected target
* if structured tool outputs are not available, it falls back to retrieved document snippets and joins them with simple rule-based connector phrases

This means the mock client is best understood as a deterministic answer synthesizer over workflow artifacts rather than as a general-purpose language model. Its outputs are usually traceable and stable, but they may sound repetitive, may not adapt gracefully to unusual phrasing, and may be less effective when a question requires broader inference across several artifacts.

#### Mock

No credentials required.

```powershell
$env:AIR\_LLM\_PROVIDER = "mock"
```

#### Local LLM

If you want a real local model instead of the mock client, the codebase already includes a `LocalLLMClient` abstraction in `ai\_insurance\_reporting/chatbot/llm\_client.py`. This is intended as the hook point for connecting a local runtime such as Ollama, LM Studio, or another callable wrapper around a local model.

This is not exposed as a turnkey environment-variable option in the current implementation. To use it, you would typically:

1. implement a small local generator function that accepts a prompt string and returns a response string
2. wrap that function with `LocalLLMClient`
3. inject that client into the chatbot path, for example where the assistant or RAG pipeline is constructed

In other words, local-model integration is supported as an extension point, but it currently requires a small amount of code wiring rather than only configuration.

#### OpenAI

Required environment variables:

* `AIR\_LLM\_PROVIDER=openai`
* `OPENAI\_API\_KEY=<your key>`

Optional:

* `AIR\_OPENAI\_MODEL=gpt-4o-mini`

This is the easiest path if you want stronger answer quality without maintaining a local model runtime.

#### Gemini

Required environment variables:

* `AIR\_LLM\_PROVIDER=gemini`
* `GEMINI\_API\_KEY=<your key>`

Also supported:

* `GOOGLE\_API\_KEY=<your key>`

Optional:

* `AIR\_GEMINI\_MODEL=gemini-2.5-flash`

This is an alternative hosted-provider path if you prefer Gemini rather than OpenAI.

The Streamlit chatbot page also allows provider selection and optional API key entry for the current local session.

## Practical extension points

The current codebase is designed so that readers can extend it in bounded ways. Typical extension points include:

* adding new synthetic drivers in `data/synthetic.py`
* extending validation rules in `data/validation.py`
* adding forecast features or targets in `models/forecasting.py`
* extending explainability outputs
* adding new chatbot tools or routing rules
* adding new scenario definitions and reporting summaries
* adjusting evaluation thresholds and review prioritisation

## Troubleshooting Notes

* If `case-study ...` does not work but `.\\.venv\\Scripts\\python.exe -m ai\_insurance\_reporting.interface.cli ...` does, the console script is not installed or not on your `PATH`.
* If the dashboard appears stale after a rerun, rerun the workflow and then refresh artifacts in Streamlit.
* If a page still shows old outputs, check whether the relevant stage has been rerun and whether older artifacts are still being displayed.
* External LLM-backed report drafts require the appropriate provider credentials; otherwise the workflow falls back to the deterministic baseline path.

