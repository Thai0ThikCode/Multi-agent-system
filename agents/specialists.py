"""
agents/specialists.py
Domain specialist agents: Data_Ingestion, Technical_Analyst, Macro_Fundamentals,
Sentiment_News, Risk_Manager.

Each function is synchronous (the Groq call is blocking); the orchestration layer runs the
three domain analysts concurrently via `asyncio.to_thread` + `asyncio.gather`.
Payloads are deliberately compact (short candle lines, trimmed headlines) because Groq's
free tier only allows ~8,000 tokens/minute per model.
"""
from __future__ import annotations

import logging

from config import MODEL_CONFIG, APP_CONFIG
from schemas import (
    RawData,
    Direction,
    TechnicalAnalysisOutput,
    MacroFundamentalsOutput,
    SentimentNewsOutput,
    OilGasAnalysisOutput,
    RiskManagerOutput,
)
from agents.base_agent import run_structured_agent, run_text_agent, fallback_instance, digest_output
from agents.prompts import (
    DATA_INGESTION_PROMPT,
    TECHNICAL_ANALYST_PROMPT,
    MACRO_FUNDAMENTALS_PROMPT,
    SENTIMENT_NEWS_PROMPT,
    OIL_GAS_ANALYST_PROMPT,
    RISK_MANAGER_PROMPT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- compact payload helpers

def _compact_candles(raw_data: RawData, n: int = 15) -> str:
    rows = [
        f"{p.date} O{p.open:.2f} H{p.high:.2f} L{p.low:.2f} C{p.close:.2f}"
        for p in raw_data.gold_price_history[-n:]
    ]
    return "\n".join(rows) if rows else "UNAVAILABLE"


def _compact_news(raw_data: RawData, limit: int = 8, snippet_chars: int = 160) -> str:
    lines = []
    for item in raw_data.news_items[:limit]:
        snippet = (item.snippet or "").replace("\n", " ")[:snippet_chars]
        lines.append(f"- [{item.source}] {item.title} :: {snippet}")
    return "\n".join(lines) if lines else "UNAVAILABLE"


def _snapshot_json(model_obj) -> str:
    return model_obj.model_dump_json(exclude_none=True) if model_obj else "{}"


# ---------------------------------------------------------------- ingestion

def _format_ingestion_brief_locally(raw_data: RawData) -> str:
    """Normalized brief without any LLM call — plain templating over the already-validated
    RawData. Default path: free, instant, and enough for what is a data-shaping step."""
    tech = raw_data.technical_snapshot
    macro = raw_data.macro_snapshot
    lines: list[str] = ["=== Normalized Data Brief (locally formatted) ==="]

    lines.append(f"Price history points: {len(raw_data.gold_price_history)}")
    if tech:
        lines.append(
            f"Technical: last_close={tech.last_close}, RSI14={tech.rsi_14}, "
            f"MACD={tech.macd}/{tech.macd_signal}, SMA20/50/200={tech.sma_20}/{tech.sma_50}/{tech.sma_200}, "
            f"trend={tech.trend_20_50}, support={tech.support_levels}, resistance={tech.resistance_levels}"
        )
    else:
        lines.append("Technical: UNAVAILABLE")

    if macro:
        lines.append(
            f"Macro: DXY={macro.dxy_last} ({macro.dxy_change_pct_5d}% 5d), "
            f"US10Y={macro.us10y_yield_last} (chg {macro.us10y_change_5d} 5d), "
            f"Crude={macro.crude_oil_last} ({macro.crude_oil_change_pct_5d}% 5d)"
        )
        for note in macro.notes:
            lines.append(f"  macro note: {note}")
    else:
        lines.append("Macro: UNAVAILABLE")

    # Global Energy Commodities
    if raw_data.oil_price_history:
        lines.append(f"WTI Crude (CL=F): latest_close=${raw_data.oil_price_history[-1].close:.2f} ({len(raw_data.oil_price_history)} candles)")
    if raw_data.gas_price_history:
        lines.append(f"Natural Gas (NG=F): latest_close=${raw_data.gas_price_history[-1].close:.2f} ({len(raw_data.gas_price_history)} candles)")

    # Domestic Vietnamese Retail Fuel (PVOIL)
    if raw_data.domestic_fuel_snapshot and raw_data.domestic_fuel_snapshot.items:
        snap = raw_data.domestic_fuel_snapshot
        lines.append(f"Domestic Fuel (PVOIL - Effective {snap.effective_date}):")
        for it in snap.items[:5]:
            lines.append(f"  - {it.product_name}: Vùng 1={it.zone_1_price:,.0f} {it.unit}, Vùng 2={it.zone_2_price:,.0f} {it.unit}")

    lines.append(f"News headlines ({len(raw_data.news_items)}):")
    for item in raw_data.news_items[:10]:
        lines.append(f"  - [{item.source}] {item.title}")

    if raw_data.data_warnings:
        lines.append("Data warnings:")
        for w in raw_data.data_warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)


def run_data_ingestion(raw_data: RawData) -> str:
    """Returns a normalized text brief consumed as plain context by the specialists. Defaults to
    free local formatting; set GOLD_AGENT_LLM_INGESTION=true for an LLM-written brief."""
    if not APP_CONFIG.enable_llm_ingestion:
        return _format_ingestion_brief_locally(raw_data)
    try:
        return run_text_agent(
            model=MODEL_CONFIG.fast_model,
            system_instruction=DATA_INGESTION_PROMPT,
            user_content=f"Raw data payload:\n{_format_ingestion_brief_locally(raw_data)}",
            temperature=0.1,
        )
    except Exception as e:
        logger.warning("LLM ingestion failed, falling back to local formatting: %s", e)
        return _format_ingestion_brief_locally(raw_data)


# ---------------------------------------------------------------- specialists

def run_technical_analyst(ingestion_brief: str, raw_data: RawData) -> TechnicalAnalysisOutput:
    user_content = (
        f"Normalized data brief:\n{ingestion_brief}\n\n"
        f"Technical snapshot (JSON): {_snapshot_json(raw_data.technical_snapshot)}\n\n"
        f"Recent daily candles (last 15):\n{_compact_candles(raw_data)}"
    )
    try:
        return run_structured_agent(
            model=MODEL_CONFIG.fast_model,
            system_instruction=TECHNICAL_ANALYST_PROMPT,
            user_content=user_content,
            schema=TechnicalAnalysisOutput,
            temperature=MODEL_CONFIG.temperature_analysis,
        )
    except Exception as e:
        logger.error("Technical_Analyst_Agent failed after retries: %s", e)
        return fallback_instance(
            TechnicalAnalysisOutput,
            reasoning_chain=[f"Agent call failed: {e}"],
            direction=Direction.SIDEWAYS,
            confidence=0.0,
        )


def run_macro_fundamentals(ingestion_brief: str, raw_data: RawData) -> MacroFundamentalsOutput:
    user_content = (
        f"Normalized data brief:\n{ingestion_brief}\n\n"
        f"Macro snapshot (JSON): {_snapshot_json(raw_data.macro_snapshot)}\n\n"
        f"Macro-relevant news headlines:\n{_compact_news(raw_data)}"
    )
    try:
        return run_structured_agent(
            model=MODEL_CONFIG.fast_model,
            system_instruction=MACRO_FUNDAMENTALS_PROMPT,
            user_content=user_content,
            schema=MacroFundamentalsOutput,
            temperature=MODEL_CONFIG.temperature_analysis,
        )
    except Exception as e:
        logger.error("Macro_Fundamentals_Agent failed after retries: %s", e)
        return fallback_instance(
            MacroFundamentalsOutput,
            reasoning_chain=[f"Agent call failed: {e}"],
            direction=Direction.SIDEWAYS,
            confidence=0.0,
        )


def run_sentiment_news(ingestion_brief: str, raw_data: RawData) -> SentimentNewsOutput:
    user_content = (
        f"Normalized data brief:\n{ingestion_brief}\n\n"
        f"News headlines/snippets:\n{_compact_news(raw_data, limit=10, snippet_chars=200)}"
    )
    try:
        return run_structured_agent(
            model=MODEL_CONFIG.fast_model,
            system_instruction=SENTIMENT_NEWS_PROMPT,
            user_content=user_content,
            schema=SentimentNewsOutput,
            temperature=MODEL_CONFIG.temperature_analysis,
        )
    except Exception as e:
        logger.error("Sentiment_News_Agent failed after retries: %s", e)
        return fallback_instance(
            SentimentNewsOutput,
            reasoning_chain=[f"Agent call failed: {e}"],
            direction=Direction.SIDEWAYS,
            confidence=0.0,
            sentiment_score=0.0,
        )


def run_oil_gas_analyst(ingestion_brief: str, raw_data: RawData) -> OilGasAnalysisOutput:
    oil_summary = "UNAVAILABLE"
    if raw_data.oil_price_history:
        last = raw_data.oil_price_history[-1]
        recent_15 = raw_data.oil_price_history[-15:]
        min_p = min(p.low for p in recent_15)
        max_p = max(p.high for p in recent_15)
        oil_summary = f"WTI Crude (CL=F) latest_close: ${last.close:.2f} | 15-day range: [${min_p:.2f} - ${max_p:.2f}]"

    gas_summary = "UNAVAILABLE"
    if raw_data.gas_price_history:
        last_g = raw_data.gas_price_history[-1]
        gas_summary = f"Natural Gas (NG=F) latest_close: ${last_g.close:.2f}"

    fuel_summary = "UNAVAILABLE"
    if raw_data.domestic_fuel_snapshot and raw_data.domestic_fuel_snapshot.items:
        snap = raw_data.domestic_fuel_snapshot
        items_str = ", ".join(f"{it.product_name}: {it.zone_1_price:,.0f} VNĐ" for it in snap.items[:4])
        fuel_summary = f"PVOIL Domestic Fuel (Effective {snap.effective_date}): {items_str}"

    user_content = (
        f"Normalized data brief:\n{ingestion_brief}\n\n"
        f"Crude Oil Market:\n{oil_summary}\n\n"
        f"Natural Gas Market:\n{gas_summary}\n\n"
        f"Vietnam Domestic Fuel (PVOIL):\n{fuel_summary}\n\n"
        f"Macro snapshot:\n{_snapshot_json(raw_data.macro_snapshot)}"
    )
    try:
        return run_structured_agent(
            model=MODEL_CONFIG.fast_model,
            system_instruction=OIL_GAS_ANALYST_PROMPT,
            user_content=user_content,
            schema=OilGasAnalysisOutput,
            temperature=MODEL_CONFIG.temperature_analysis,
        )
    except Exception as e:
        logger.error("Oil_Gas_Analyst_Agent failed after retries: %s", e)
        return fallback_instance(
            OilGasAnalysisOutput,
            reasoning_chain=[f"Agent call failed: {e}"],
            direction=Direction.SIDEWAYS,
            confidence=0.0,
            crude_trend="Data unavailable or call failed",
            natural_gas_trend="Data unavailable or call failed",
            domestic_fuel_assessment="PVOIL analysis unavailable",
            macro_inflation_impact="Energy impact assessment unavailable",
            executive_summary="Energy market analysis is unavailable because the model call failed; domestic fuel and inflation implications remain uncertain.",
        )


def run_risk_manager(
    technical: TechnicalAnalysisOutput,
    macro: MacroFundamentalsOutput,
    sentiment: SentimentNewsOutput,
    oil_gas: Optional[OilGasAnalysisOutput] = None,
) -> RiskManagerOutput:
    parts = [
        f"Technical_Analyst_Agent conclusion: {digest_output(technical)}",
        f"Macro_Fundamentals_Agent conclusion: {digest_output(macro)}",
        f"Sentiment_News_Agent conclusion: {digest_output(sentiment)}",
    ]
    if oil_gas is not None:
        parts.append(f"Oil_Gas_Analyst_Agent conclusion: {digest_output(oil_gas)}")

    user_content = "\n\n".join(parts)
    try:
        return run_structured_agent(
            model=MODEL_CONFIG.fast_model,
            system_instruction=RISK_MANAGER_PROMPT,
            user_content=user_content,
            schema=RiskManagerOutput,
            temperature=MODEL_CONFIG.temperature_analysis,
        )
    except Exception as e:
        logger.error("Risk_Manager_Agent failed after retries: %s", e)
        return fallback_instance(
            RiskManagerOutput,
            reasoning_chain=[f"Agent call failed: {e}"],
            direction=Direction.SIDEWAYS,
            confidence=0.0,
            bull_trap_risk=0.5,
            bear_trap_risk=0.5,
        )
