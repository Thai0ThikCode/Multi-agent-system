"""
config.py
Central configuration for the Gold Price Multi-Agent System (Groq backend).
All values can be overridden via environment variables (see .env.example).
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ModelConfig:
    # Fast model for the specialist agents (Technical / Macro / Sentiment / Risk).
    # openai/gpt-oss-20b supports Groq Structured Outputs (json_schema).
    fast_model: str = os.getenv("GOLD_AGENT_FAST_MODEL", "openai/gpt-oss-20b")
    # Stronger model for the debate + final synthesis. Groq applies rate limits PER MODEL,
    # so using a different model here also gives the strategist its own token budget.
    pro_model: str = os.getenv("GOLD_AGENT_PRO_MODEL", "openai/gpt-oss-120b")
    temperature_analysis: float = float(os.getenv("GOLD_AGENT_TEMP_ANALYSIS", "0.3"))
    temperature_debate: float = float(os.getenv("GOLD_AGENT_TEMP_DEBATE", "0.5"))
    # Upper bound for generated tokens per call (reasoning tokens of gpt-oss count too).
    max_output_tokens: int = int(os.getenv("GOLD_AGENT_MAX_TOKENS", "3000"))
    # Rough expected output size, only used by the client-side token throttle.
    expected_output_tokens: int = int(os.getenv("GOLD_AGENT_EXPECTED_OUTPUT_TOKENS", "1500"))
    # gpt-oss reasoning effort: "low" | "medium" | "high" ("" = don't send). Low saves tokens.
    reasoning_effort: str = os.getenv("GOLD_AGENT_REASONING_EFFORT", "low")
    # Model-id prefixes that support response_format=json_schema on Groq. Any other model
    # automatically falls back to JSON-object mode with the schema embedded in the prompt.
    json_schema_model_prefixes: tuple = tuple(
        p.strip()
        for p in os.getenv("GOLD_AGENT_JSON_SCHEMA_MODELS", "openai/gpt-oss,qwen/qwen3.8").split(",")
        if p.strip()
    )


@dataclass(frozen=True)
class DataConfig:
    gold_ticker: str = "GC=F"          # COMEX Gold futures (proxy for XAU/USD); "XAUUSD=X" also usable
    dxy_ticker: str = "DX-Y.NYB"       # US Dollar Index
    yield_ticker: str = "^TNX"         # US 10-Year Treasury Yield
    oil_ticker: str = "CL=F"           # WTI Crude Oil
    lookback_days: int = 180
    news_max_results: int = 12
    news_query_terms: tuple = (
        "gold price forecast",
        "federal reserve interest rate decision",
        "central bank gold reserves purchase",
        "geopolitical risk safe haven gold",
    )


@dataclass(frozen=True)
class AppConfig:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")  # optional, falls back to DuckDuckGo
    db_path: str = os.getenv("GOLD_AGENT_DB_PATH", "audit_log.sqlite3")
    request_timeout_s: int = int(os.getenv("GOLD_AGENT_TIMEOUT", "60"))
    max_retries: int = int(os.getenv("GOLD_AGENT_MAX_RETRIES", "5"))
    # Client-side throttle, tracked PER MODEL (Groq limits are per model per organization).
    # Groq free tier (gpt-oss): 30 requests/min and only 8,000 tokens/min -> the token limit
    # is the one that actually binds. Raise both on the Developer plan; 0 disables a limit.
    max_requests_per_minute: int = int(os.getenv("GOLD_AGENT_MAX_RPM", "30"))
    max_tokens_per_minute: int = int(os.getenv("GOLD_AGENT_MAX_TPM", "8000"))
    # If Groq says "try again in <longer than this>" (e.g. daily quota hit), fail fast
    # instead of blocking the dashboard for minutes.
    max_retry_wait_s: float = float(os.getenv("GOLD_AGENT_MAX_RETRY_WAIT", "90"))
    # Data_Ingestion_Agent uses free local formatting by default; set true to use the LLM.
    enable_llm_ingestion: bool = _env_bool("GOLD_AGENT_LLM_INGESTION", False)


MODEL_CONFIG = ModelConfig()
DATA_CONFIG = DataConfig()
APP_CONFIG = AppConfig()


def validate_config() -> list:
    """Returns a list of human-readable configuration problems (empty = OK)."""
    problems = []
    if not APP_CONFIG.groq_api_key:
        problems.append(
            "GROQ_API_KEY is not set. Create a .env file (see .env.example) "
            "or export it in your shell before running the app. "
            "Get a key at https://console.groq.com/keys"
        )
    return problems
