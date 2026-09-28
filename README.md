# XAU/USD and Oil & Gas Multi-Agent Forecasting System

A production-grade, modular multi-agent pipeline that forecasts gold (XAU/USD)
price trends using the Groq API. Specialist agents analyze technicals,
macro fundamentals, and news sentiment in parallel, a Risk Manager agent
stress-tests their conclusions, a Lead Strategist agent runs a debate cycle to
surface conflicts, and a final synthesis produces a directional forecast with
a confidence score, price target range, and risk warnings — all backed by a
full audit log of every agent's reasoning.

> ⚠️ **Not financial advice.** This is a research/educational tool. Gold price
> forecasting is inherently uncertain; do not use this system's output as the
> sole basis for real trading or investment decisions.

## Architecture

```
config.py                  # API keys, model names, tunables (env-driven)
schemas/models.py          # Pydantic schemas for all data + agent outputs + shared state
tools/
  market_data.py           # yfinance: gold, DXY, US10Y, crude oil
  technical_indicators.py  # RSI, MACD, SMA/EMA, Bollinger Bands, support/resistance
  news_sentiment.py        # Tavily (optional) or DuckDuckGo news search
agents/
  prompts.py                # System prompts per agent role
  base_agent.py              # Groq wrapper: structured output, error classification, token throttle, retries
  specialists.py             # Data_Ingestion, Technical, Macro, Sentiment, Risk agents
  strategist.py               # Lead_Strategist: debate turn + final synthesis
orchestration/
  workflow.py                # Async orchestration graph wiring everything together
storage/
  database.py                 # SQLite persistence for full state + audit log
app.py                        # Streamlit dashboard
```

### Pipeline flow

1. **Data collection** (parallel): gold OHLCV history, DXY / US10Y / crude oil,
   and gold-relevant news headlines are fetched concurrently.
2. **Data_Ingestion_Agent** normalizes the raw payload into a clean brief.
3. **Technical_Analyst_Agent**, **Macro_Fundamentals_Agent**, and
   **Sentiment_News_Agent** run **concurrently** (`asyncio.gather`), each
   returning a structured, schema-validated opinion with an explicit
   step-by-step `reasoning_chain`.
4. **Risk_Manager_Agent** reviews all three conclusions looking for bull/bear
   traps, crowded positioning, and conflicting signals.
5. **Lead_Strategist_Agent** runs a **debate turn**, explicitly surfacing
   conflicts between agents (e.g. "Technical says bullish breakout, but Macro
   flags a hawkish Fed").
6. **Lead_Strategist_Agent** runs **synthesis**, resolving the debate into a
   final direction, confidence score, price target range, timeframe, risk
   warnings, and an executive summary.
7. Every step is persisted to SQLite (`audit_log.sqlite3` by default) as it
   happens, so the dashboard can show live progress and past runs can be
   re-opened later.

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
```

Edit `.env` and set:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a key at https://console.groq.com/keys. `TAVILY_API_KEY` is optional
— if you don't set it, the system automatically falls back to free
DuckDuckGo news search.

### 3. Run the dashboard

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`), then click
**"Run new analysis"** in the sidebar. A full run typically takes 30–90
seconds depending on API latency and how much price history needs fetching.

### 4. (Optional) Run the pipeline headlessly

```bash
python -c "
from orchestration.workflow import run_pipeline_sync
state = run_pipeline_sync()
print(state.synthesis_report.executive_summary)
"
```

## Configuration reference

All settings in `config.py` can be overridden via environment variables (see
`.env.example`). Defaults:

| Purpose | Variable | Default |
|---|---|---|
| Specialist agents | `GOLD_AGENT_FAST_MODEL` | `openai/gpt-oss-20b` |
| Debate + synthesis | `GOLD_AGENT_PRO_MODEL` | `openai/gpt-oss-120b` |
| Reasoning effort (gpt-oss) | `GOLD_AGENT_REASONING_EFFORT` | `low` |
| Requests/min per model | `GOLD_AGENT_MAX_RPM` | `30` |
| Tokens/min per model | `GOLD_AGENT_MAX_TPM` | `8000` |

**Structured output.** Models whose id starts with `openai/gpt-oss` or
`qwen/qwen3.8` are called with `response_format=json_schema` (best-effort mode,
`strict: false`, because our schemas have optional fields). Any other model
automatically falls back to JSON-object mode with the schema embedded in the
prompt; every response is validated with Pydantic and retried on mismatch.

## Resilience notes

- **Token-aware throttle, per model.** Groq's free tier for gpt-oss allows
  30 requests/min but only **8,000 tokens/min per model**, so the token limit
  is what actually binds. Every call first waits for a free request *and* token
  slot in a rolling 60 s window (using real `usage.total_tokens` from previous
  calls), instead of firing and recovering from a `429`. Because limits are per
  model, the strategist (`gpt-oss-120b`) has its own budget separate from the
  analysts (`gpt-oss-20b`).
- **Compact prompts.** Candles, headlines and inter-agent hand-offs are sent in
  a trimmed form to stay inside the token budget.
- **Retry policy.** Transient errors (`429` per-minute limits, `5xx`,
  timeouts, off-schema JSON) are retried; for a `429` the wait honors Groq's
  own `retry-after` / "try again in 8.5s".
- **Fail-fast cases** (no pointless retrying): daily quota reached
  (`RPD`/`TPD`, waits of many minutes), request larger than the per-minute
  token budget (`413`), invalid API key, unknown/decommissioned model.
- If an agent still fails, it returns a clearly-labeled fallback output
  (confidence `0.0`, direction `Sideways`) instead of crashing the run.
- Data-fetching failures are recorded in `raw_data.data_warnings` and shown in
  the dashboard rather than halting the pipeline.
- `Data_Ingestion_Agent` defaults to free local formatting (no LLM call); set
  `GOLD_AGENT_LLM_INGESTION=true` to use the LLM.

Expect a full run on the free tier to take roughly 1-3 minutes, because the
throttle deliberately waits for token budget. On the Developer plan, raise
`GOLD_AGENT_MAX_RPM` / `GOLD_AGENT_MAX_TPM` for much faster runs.

## Troubleshooting

**`Groq quota exhausted ... DAILY limit`** — free tier gpt-oss allows
1,000 requests/day and 200,000 tokens/day per model. Wait for the reset, point
`GOLD_AGENT_FAST_MODEL` / `GOLD_AGENT_PRO_MODEL` at a model with quota left, or
upgrade the plan.

**`Request too large ... tokens per minute`** — the single request exceeds the
model's per-minute token budget. Lower `GOLD_AGENT_MAX_TOKENS` or upgrade the plan.

**`model ... does not exist / decommissioned`** — Groq retires models over
time. Pick a current one from https://console.groq.com/docs/models and set the
two model variables in `.env`.

**`Groq rejected the API key`** — check `GROQ_API_KEY` in `.env` (the old
`GEMINI_API_KEY` is no longer used).

## Extending the system

- **Swap orchestration engines:** the workflow in `orchestration/workflow.py`
  is deliberately plain `asyncio` so it's easy to port into LangGraph or
  CrewAI if you want their built-in graph visualization / checkpointing —
  each function already maps cleanly to a graph node.
- **Add a new specialist agent:** define its Pydantic output schema in
  `schemas/models.py`, its prompt in `agents/prompts.py`, an implementation
  in `agents/specialists.py`, and wire it into `orchestration/workflow.py`'s
  parallel `asyncio.gather` step and into the Risk Manager / debate inputs.
- **Change the gold ticker:** `DATA_CONFIG.gold_ticker` defaults to `GC=F`
  (COMEX futures); `XAUUSD=X` is also available on Yahoo Finance if you
  prefer spot pricing (data quality/availability varies by data provider).
