"""
agents/base_agent.py
Single place that talks to the Groq API (OpenAI-compatible chat completions):
 - structured output via response_format=json_schema (strict:false, best-effort) on models
   that support it, otherwise JSON-object mode with the schema embedded in the prompt
 - Pydantic validation of every structured response
 - Groq-aware error classification (429 "try again in 8.5s", 413, daily quotas, bad key...)
 - per-model client-side throttle on requests/min AND tokens/min (free tier = 8k TPM)
 - tenacity retries that honor Groq's own retry-after hint
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Type, TypeVar

from groq import Groq
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from config import APP_CONFIG, MODEL_CONFIG

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        if not APP_CONFIG.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not configured. Set it in your environment or .env file.")
        # max_retries=0: retries are owned by tenacity + the rate limiter below.
        _client = Groq(api_key=APP_CONFIG.groq_api_key, timeout=APP_CONFIG.request_timeout_s, max_retries=0)
    return _client


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class RetryableLLMError(Exception):
    """Transient error worth retrying. `retry_after` (seconds), when known, is the wait
    suggested by Groq itself (retry-after header / 'try again in 8.5s')."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class QuotaExhaustedError(Exception):
    """Non-retryable quota/size problem (daily limit hit, request larger than the per-minute
    token budget, ...). Retrying within this run cannot help, so tenacity does not retry it."""


_JSON_OBJECT_ONLY_MODELS: set[str] = set()   # models where json_schema turned out unsupported

_DURATION_SEGMENT = re.compile(r"(?:try again|retry) in ([0-9hms.]+)", re.IGNORECASE)
_DURATION_TOKEN = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def _parse_retry_after(raw_msg: str) -> float | None:
    """Parses Groq hints like 'Please try again in 8.5s', '1m30.5s', '14m2.5s', '320ms'."""
    match = _DURATION_SEGMENT.search(raw_msg)
    if not match:
        return None
    tokens = _DURATION_TOKEN.findall(match.group(1))
    if not tokens:
        return None
    return sum(float(value) * _UNIT_SECONDS[unit] for value, unit in tokens)


def _header_retry_after(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("retry-after")
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _classify_error(exc: Exception, model: str = "") -> Exception:
    if isinstance(exc, (RetryableLLMError, QuotaExhaustedError)):
        return exc
    raw = str(exc)
    low = raw.lower()
    status = getattr(exc, "status_code", None)

    if status in (401, 403) or "invalid api key" in low or "invalid_api_key" in low:
        return RuntimeError(
            "Groq rejected the API key (401/403). Check GROQ_API_KEY in your .env "
            f"(create one at https://console.groq.com/keys). Original error: {raw}"
        )

    if status == 413 or "request too large" in low or "reduce your message size" in low:
        return QuotaExhaustedError(
            "The request is larger than this model's per-minute token budget on your Groq plan "
            "(free tier gpt-oss: 8,000 tokens/min). Reduce GOLD_AGENT_MAX_TOKENS / news items, "
            "or upgrade the Groq plan. Original error: " + raw
        )

    if status == 429 or "rate_limit_exceeded" in low or "rate limit reached" in low:
        retry_after = _header_retry_after(exc)
        if retry_after is None:
            retry_after = _parse_retry_after(raw)
        daily = "(tpd)" in low or "(rpd)" in low or "per day" in low
        if (retry_after is not None and retry_after > APP_CONFIG.max_retry_wait_s) or (daily and retry_after is None):
            return QuotaExhaustedError(
                f"Groq quota exhausted for model '{model}' (would need to wait "
                f"{'~%d s' % retry_after if retry_after else 'a long time'}; usually the DAILY limit: "
                "free tier gpt-oss = 1,000 requests/day and 200,000 tokens/day). Wait for the reset, "
                "switch GOLD_AGENT_FAST_MODEL/PRO_MODEL to a model with quota left, or upgrade the plan. "
                f"Original error: {raw}"
            )
        return RetryableLLMError(raw, retry_after=retry_after)

    if status == 404 or "model_not_found" in low or "does not exist" in low or "decommissioned" in low:
        return RuntimeError(
            f"Groq model '{model}' is not available (removed/renamed/decommissioned or no access). "
            "Pick a current model from https://console.groq.com/docs/models and set "
            f"GOLD_AGENT_FAST_MODEL / GOLD_AGENT_PRO_MODEL. Original error: {raw}"
        )

    if status == 400:
        if "does not match the expected schema" in low or "json_validate_failed" in low or "failed to generate json" in low:
            return RetryableLLMError(raw, retry_after=0.5)   # model produced off-schema JSON; try again
        if "json_schema" in low or "response_format" in low:
            _JSON_OBJECT_ONLY_MODELS.add(model)              # degrade to JSON-object mode and retry
            logger.warning("Model %s rejected json_schema; falling back to JSON-object mode.", model)
            return RetryableLLMError(raw, retry_after=0.0)
        return exc

    transient_markers = ("timeout", "timed out", "connection", "temporarily", "unavailable", "overloaded")
    if (isinstance(status, int) and status >= 500) or any(m in low for m in transient_markers):
        return RetryableLLMError(raw, retry_after=_header_retry_after(exc))
    return exc


# --------------------------------------------------------------------------
# Client-side throttle (per model: requests/min + tokens/min)
# --------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    return int(len(text) / 3.2) + 1   # deliberately conservative


class _RateLimiter:
    """Thread-safe sliding-window limiter. Groq limits are per model, so each model gets its
    own request/token window. Blocks the calling thread until a slot is free instead of
    firing the request and recovering from a 429 afterwards."""

    def __init__(self, max_rpm: int, max_tpm: int, window_seconds: float = 60.0):
        self._max_rpm = max_rpm
        self._max_tpm = max_tpm
        self._window = window_seconds
        self._events: dict[str, list[list]] = {}   # model -> [[timestamp, tokens], ...]
        self._lock = threading.Lock()

    def _fits(self, events: list[list], est_tokens: int) -> bool:
        rpm_ok = self._max_rpm <= 0 or len(events) < self._max_rpm
        # an empty window always admits the call, otherwise a request bigger than the whole
        # TPM budget would wait forever (Groq itself will then answer 413, handled above)
        tpm_ok = self._max_tpm <= 0 or not events or sum(e[1] for e in events) + est_tokens <= self._max_tpm
        return rpm_ok and tpm_ok

    def acquire(self, model: str, est_tokens: int) -> list:
        while True:
            with self._lock:
                now = time.monotonic()
                events = [e for e in self._events.get(model, []) if now - e[0] < self._window]
                self._events[model] = events
                if self._fits(events, est_tokens):
                    ticket = [now, est_tokens]
                    events.append(ticket)
                    return ticket
                sleep_for = 0.05
                for i, e in enumerate(events):
                    if self._fits(events[i + 1:], est_tokens):
                        sleep_for = max(e[0] + self._window - now, 0.0) + 0.05
                        break
            logger.info("Rate limit guard (%s): waiting %.1fs for token/request budget.", model, sleep_for)
            time.sleep(max(sleep_for, 0.05))


_rate_limiter = _RateLimiter(APP_CONFIG.max_requests_per_minute, APP_CONFIG.max_tokens_per_minute)


def _wait_for_llm(retry_state):
    """Honor Groq's own retry hint when we have one; otherwise exponential backoff."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is not None:
        return min(retry_after + 0.5, APP_CONFIG.max_retry_wait_s)
    return wait_exponential(multiplier=1.5, min=2, max=30)(retry_state)


# --------------------------------------------------------------------------
# Request execution
# --------------------------------------------------------------------------

def _clean_schema(node):
    """Strips Pydantic-only noise ('title', 'default') so the schema stays inside the JSON-Schema
    subset Groq accepts; keeps $defs/$ref, enum, anyOf, minimum/maximum, descriptions."""
    if isinstance(node, dict):
        return {k: _clean_schema(v) for k, v in node.items() if k not in ("title", "default")}
    if isinstance(node, list):
        return [_clean_schema(v) for v in node]
    return node


def _supports_json_schema(model: str) -> bool:
    return model not in _JSON_OBJECT_ONLY_MODELS and model.startswith(MODEL_CONFIG.json_schema_model_prefixes)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def digest_output(output: BaseModel, max_steps: int = 5, max_chars: int = 240) -> str:
    """Compact JSON of an agent's output for feeding other agents: no indentation and a trimmed
    reasoning chain. Keeps prompts small, which matters with Groq's 8k tokens/min free tier."""
    data = output.model_dump(mode="json")
    data["reasoning_chain"] = [str(s)[:max_chars] for s in data.get("reasoning_chain", [])[:max_steps]]
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


@retry(
    stop=stop_after_attempt(APP_CONFIG.max_retries),
    wait=_wait_for_llm,
    retry=retry_if_exception_type(RetryableLLMError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _execute(model: str, system_instruction: str, user_content: str, schema: Type[T] | None, temperature: float):
    client = get_client()
    system = system_instruction
    extra: dict = {}

    if schema is not None:
        json_schema = _clean_schema(schema.model_json_schema())
        if _supports_json_schema(model):
            extra["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "strict": False, "schema": json_schema},
            }
        else:
            extra["response_format"] = {"type": "json_object"}
            system += (
                "\n\nRespond ONLY with a single JSON object that conforms to this JSON Schema:\n"
                + json.dumps(json_schema, separators=(",", ":"))
            )

    if MODEL_CONFIG.reasoning_effort and "gpt-oss" in model:
        extra["reasoning_effort"] = MODEL_CONFIG.reasoning_effort

    est_tokens = _estimate_tokens(system + user_content) + MODEL_CONFIG.expected_output_tokens
    ticket = _rate_limiter.acquire(model, est_tokens)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_content}],
            temperature=temperature,
            max_completion_tokens=MODEL_CONFIG.max_output_tokens,
            **extra,
        )
    except Exception as e:
        raise _classify_error(e, model) from e

    usage = getattr(response, "usage", None)
    total = getattr(usage, "total_tokens", None) if usage else None
    if total:
        ticket[1] = total   # replace the estimate with the real usage for accurate throttling

    text = ((response.choices[0].message.content if response.choices else "") or "").strip()
    if not text:
        raise RetryableLLMError("Empty response from Groq (reasoning may have used the whole token budget).")
    if schema is None:
        return text

    try:
        return schema.model_validate(json.loads(_FENCE.sub("", text)))
    except ValueError as e:   # JSONDecodeError and pydantic.ValidationError both subclass ValueError
        raise RetryableLLMError(f"Failed to parse/validate structured output: {e}", retry_after=0.5) from e


def run_structured_agent(
    *,
    model: str,
    system_instruction: str,
    user_content: str,
    schema: Type[T],
    temperature: float = MODEL_CONFIG.temperature_analysis,
) -> T:
    """Public entry point used by every agent. Synchronous — callers running inside asyncio
    should wrap with `asyncio.to_thread`."""
    return _execute(model, system_instruction, user_content, schema, temperature)


def run_text_agent(*, model: str, system_instruction: str, user_content: str, temperature: float = 0.1) -> str:
    """Plain-text completion (used by the optional LLM ingestion step)."""
    return _execute(model, system_instruction, user_content, None, temperature)


def fallback_instance(schema: Type[T], **overrides) -> T:
    """Builds a minimal, clearly-labeled fallback instance of `schema` for use when the LLM call
    fails after all retries — keeps the pipeline resilient instead of crashing the whole run."""
    defaults: dict = {}
    for name, field in schema.model_fields.items():
        if name in overrides:
            continue
        # Only truly required fields (no default, no default_factory) need a synthesized value.
        if not field.is_required():
            continue
        annotation = field.annotation
        origin = getattr(annotation, "__origin__", None)
        if origin in (list, set):
            defaults[name] = []
        elif annotation in (float, int):
            defaults[name] = 0.0 if annotation is float else 0
        elif annotation is str:
            defaults[name] = "UNAVAILABLE: agent call failed after retries."
        else:
            defaults[name] = None
    defaults.update(overrides)
    return schema.model_validate(defaults)
