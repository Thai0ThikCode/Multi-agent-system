"""
orchestration/workflow.py
Custom async orchestration graph (no LangGraph/CrewAI dependency required)
for the gold prediction multi-agent pipeline:

    fetch data (parallel: price, macro, news)
        -> Data_Ingestion_Agent
        -> [Technical | Macro | Sentiment] agents run concurrently
        -> Risk_Manager_Agent (reviews the three above)
        -> Lead_Strategist_Agent: debate turn
        -> Lead_Strategist_Agent: synthesis
        -> persist full AgentState + audit log to SQLite

Every step appends an AgentLogEntry to the shared AgentState and immediately
persists it, so a UI can poll the database mid-run for live progress.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Callable, Optional

from config import DATA_CONFIG
from schemas import (
    AgentState,
    AgentLogEntry,
    RawData,
)
from tools import market_data, technical_indicators, news_sentiment
from agents import specialists, strategist
from storage import database

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[str, AgentState], None]]


def _notify(callback: ProgressCallback, stage: str, state: AgentState) -> None:
    if callback:
        try:
            callback(stage, state)
        except Exception:
            logger.exception("Progress callback raised — ignoring so the run continues.")


async def _fetch_raw_data() -> RawData:
    """Runs all data collection concurrently in worker threads (the underlying
    libraries — yfinance, ddgs — are synchronous/blocking)."""
    history_task = asyncio.to_thread(market_data.fetch_gold_history, DATA_CONFIG.lookback_days)
    macro_task = asyncio.to_thread(market_data.fetch_macro_snapshot)
    news_task = asyncio.to_thread(news_sentiment.fetch_news)

    history, macro_snapshot, (news_items, news_warnings) = await asyncio.gather(
        history_task, macro_task, news_task
    )

    warnings = list(news_warnings)
    if not history:
        warnings.append("Gold price history could not be retrieved — technical analysis will be degraded.")

    technical_snapshot = technical_indicators.compute_technical_snapshot(history)
    if technical_snapshot is None:
        warnings.append("Technical indicators could not be computed from available history.")

    return RawData(
        gold_price_history=history,
        technical_snapshot=technical_snapshot,
        macro_snapshot=macro_snapshot,
        news_items=news_items,
        data_warnings=warnings,
    )


async def run_pipeline(
    db_path: Optional[str] = None,
    on_progress: ProgressCallback = None,
) -> AgentState:
    """Runs the full multi-agent pipeline end-to-end and returns the final,
    fully-populated AgentState (also persisted to SQLite as it progresses)."""
    run_id = str(uuid.uuid4())
    state = AgentState(run_id=run_id)
    database.init_db(db_path)
    database.upsert_run_state(state, db_path)

    # ---- Stage 1: data collection -----------------------------------------
    _notify(on_progress, "fetching_data", state)
    try:
        state.raw_data = await _fetch_raw_data()
    except Exception as e:
        logger.exception("Fatal error during data fetching.")
        state.errors.append(f"Data fetching failed: {e}")
        state.raw_data = RawData(data_warnings=[f"Fatal fetch error: {e}"])
    _log(state, "Data_Fetcher", "ingestion", [f"Collected raw data with {len(state.raw_data.data_warnings)} warning(s)."], db_path)
    database.upsert_run_state(state, db_path)
    _notify(on_progress, "data_fetched", state)

    # ---- Stage 2: ingestion normalization ----------------------------------
    ingestion_brief = await asyncio.to_thread(specialists.run_data_ingestion, state.raw_data)
    _log(state, "Data_Ingestion_Agent", "ingestion", [ingestion_brief[:1000]], db_path)
    database.upsert_run_state(state, db_path)
    _notify(on_progress, "ingested", state)

    # ---- Stage 3: parallel specialist analysis -----------------------------
    _notify(on_progress, "analyzing", state)
    technical_task = asyncio.to_thread(specialists.run_technical_analyst, ingestion_brief, state.raw_data)
    macro_task = asyncio.to_thread(specialists.run_macro_fundamentals, ingestion_brief, state.raw_data)
    sentiment_task = asyncio.to_thread(specialists.run_sentiment_news, ingestion_brief, state.raw_data)

    state.technical_output, state.macro_output, state.sentiment_output = await asyncio.gather(
        technical_task, macro_task, sentiment_task
    )

    _log(state, "Technical_Analyst_Agent", "analysis", state.technical_output.reasoning_chain,
         db_path, final_output=state.technical_output.model_dump())
    _log(state, "Macro_Fundamentals_Agent", "analysis", state.macro_output.reasoning_chain,
         db_path, final_output=state.macro_output.model_dump())
    _log(state, "Sentiment_News_Agent", "analysis", state.sentiment_output.reasoning_chain,
         db_path, final_output=state.sentiment_output.model_dump())
    database.upsert_run_state(state, db_path)
    _notify(on_progress, "analyzed", state)

    # ---- Stage 4: risk review ----------------------------------------------
    state.risk_output = await asyncio.to_thread(
        specialists.run_risk_manager, state.technical_output, state.macro_output, state.sentiment_output
    )
    _log(state, "Risk_Manager_Agent", "analysis", state.risk_output.reasoning_chain,
         db_path, final_output=state.risk_output.model_dump())
    database.upsert_run_state(state, db_path)
    _notify(on_progress, "risk_reviewed", state)

    # ---- Stage 5: debate ----------------------------------------------------
    _notify(on_progress, "debating", state)
    state.debate_transcript = await asyncio.to_thread(
        strategist.run_debate_turn,
        state.technical_output, state.macro_output, state.sentiment_output, state.risk_output,
    )
    for rebuttal in state.debate_transcript:
        _log(
            state, "Lead_Strategist_Agent", "debate",
            [f"{rebuttal.from_agent} -> {rebuttal.to_agent}: {rebuttal.challenge}"],
            db_path,
            debate_response=rebuttal.rebuttal_or_concession,
        )
    database.upsert_run_state(state, db_path)
    _notify(on_progress, "debated", state)

    # ---- Stage 6: synthesis --------------------------------------------------
    _notify(on_progress, "synthesizing", state)
    state.synthesis_report = await asyncio.to_thread(
        strategist.run_synthesis,
        state.technical_output, state.macro_output, state.sentiment_output,
        state.risk_output, state.debate_transcript,
    )
    _log(state, "Lead_Strategist_Agent", "synthesis", state.synthesis_report.reasoning_chain,
         db_path, final_output=state.synthesis_report.model_dump())
    database.upsert_run_state(state, db_path)
    _notify(on_progress, "complete", state)

    return state


def _log(
    state: AgentState,
    agent_name: str,
    step: str,
    inner_thought: list[str],
    db_path: Optional[str],
    final_output: Optional[dict] = None,
    debate_response: Optional[str] = None,
) -> None:
    entry = AgentLogEntry(
        agent_name=agent_name,
        step=step,
        inner_thought=inner_thought,
        final_agent_output=final_output,
        debate_response=debate_response,
    )
    state.log(entry)
    try:
        database.append_agent_log(state.run_id, entry, db_path)
    except Exception:
        logger.exception("Failed to persist agent log entry (continuing).")


def run_pipeline_sync(db_path: Optional[str] = None, on_progress: ProgressCallback = None) -> AgentState:
    """Convenience sync wrapper for callers (e.g. Streamlit) that are not
    themselves running inside an event loop."""
    return asyncio.run(run_pipeline(db_path=db_path, on_progress=on_progress))
