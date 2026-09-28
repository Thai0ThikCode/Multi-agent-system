"""
schemas/models.py
Pydantic v2 schemas shared across the whole system:
 - raw data payloads
 - structured LLM outputs (one schema per agent)
 - the shared AgentState that flows through the orchestration graph
 - audit log entry shape
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Raw data payloads (produced by tools/, consumed by Data_Ingestion_Agent)
# --------------------------------------------------------------------------

class PricePoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class TechnicalSnapshot(BaseModel):
    last_close: float
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_20: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_mid: Optional[float] = None
    trend_20_50: Optional[str] = None   # "golden_cross" | "death_cross" | "neutral"
    support_levels: list[float] = Field(default_factory=list)
    resistance_levels: list[float] = Field(default_factory=list)


class MacroSnapshot(BaseModel):
    dxy_last: Optional[float] = None
    dxy_change_pct_5d: Optional[float] = None
    us10y_yield_last: Optional[float] = None
    us10y_change_5d: Optional[float] = None
    crude_oil_last: Optional[float] = None
    crude_oil_change_pct_5d: Optional[float] = None
    notes: list[str] = Field(default_factory=list)


class NewsItem(BaseModel):
    title: str
    source: str = ""
    url: str = ""
    published: str = ""
    snippet: str = ""


class RawData(BaseModel):
    fetched_at: str = Field(default_factory=utc_now_iso)
    gold_price_history: list[PricePoint] = Field(default_factory=list)
    technical_snapshot: Optional[TechnicalSnapshot] = None
    macro_snapshot: Optional[MacroSnapshot] = None
    news_items: list[NewsItem] = Field(default_factory=list)
    data_warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Structured LLM outputs — one schema per agent, all share a reasoning chain
# --------------------------------------------------------------------------

class Direction(str, Enum):
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    SIDEWAYS = "Sideways"


class AgentOpinionBase(BaseModel):
    """Common fields every specialist agent must return."""
    reasoning_chain: list[str] = Field(
        description="Explicit, ordered step-by-step reasoning that led to the conclusion. "
                    "Each item is one discrete reasoning step."
    )
    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the direction, 0-1.")
    key_points: list[str] = Field(default_factory=list, description="Bullet summary of the case made.")


class TechnicalAnalysisOutput(AgentOpinionBase):
    key_support: list[float] = Field(default_factory=list)
    key_resistance: list[float] = Field(default_factory=list)
    momentum_assessment: str = ""


class MacroFundamentalsOutput(AgentOpinionBase):
    usd_impact: str = Field(description="How USD strength/weakness is affecting gold right now.")
    rate_environment_impact: str = ""
    geopolitical_catalysts: list[str] = Field(default_factory=list)


class SentimentNewsOutput(AgentOpinionBase):
    sentiment_score: float = Field(ge=-1.0, le=1.0, description="-1 fully bearish, +1 fully bullish.")
    influential_headlines: list[str] = Field(default_factory=list)


class RiskFlag(BaseModel):
    description: str
    severity: str = Field(description="'low' | 'medium' | 'high'")


class RiskManagerOutput(AgentOpinionBase):
    """The Risk Manager is a skeptic — its 'direction' expresses which side it thinks
    is most likely to be a trap, not necessarily its own market call."""
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    bull_trap_risk: float = Field(ge=0.0, le=1.0)
    bear_trap_risk: float = Field(ge=0.0, le=1.0)
    invalidation_conditions: list[str] = Field(default_factory=list)


class DebateRebuttal(BaseModel):
    from_agent: str
    to_agent: str
    challenge: str
    rebuttal_or_concession: str = ""


class DebateRebuttalList(BaseModel):
    """Wrapper so the debate output has an object root (required by Groq json_schema mode)."""
    rebuttals: list[DebateRebuttal] = Field(
        description="The 2-4 most decision-relevant conflicts between the specialist agents."
    )


class SynthesisReport(BaseModel):
    reasoning_chain: list[str]
    debate_summary: list[DebateRebuttal] = Field(default_factory=list)
    final_direction: Direction
    confidence_score: float = Field(ge=0.0, le=1.0)
    price_target_low: Optional[float] = None
    price_target_high: Optional[float] = None
    timeframe: str = Field(description="e.g. '1-2 weeks', '1 month'")
    risk_warnings: list[str] = Field(default_factory=list)
    dissenting_views: list[str] = Field(default_factory=list)
    executive_summary: str


# --------------------------------------------------------------------------
# Audit log + shared orchestration state
# --------------------------------------------------------------------------

class AgentLogEntry(BaseModel):
    agent_name: str
    step: str  # "ingestion" | "analysis" | "debate" | "synthesis"
    timestamp: str = Field(default_factory=utc_now_iso)
    inner_thought: list[str] = Field(default_factory=list)
    debate_response: Optional[str] = None
    final_agent_output: Optional[dict] = None


class AgentState(BaseModel):
    """The single object threaded through the whole workflow. Every node reads
    from and appends to this state; it is persisted in full for auditing."""
    run_id: str
    started_at: str = Field(default_factory=utc_now_iso)
    raw_data: Optional[RawData] = None
    technical_output: Optional[TechnicalAnalysisOutput] = None
    macro_output: Optional[MacroFundamentalsOutput] = None
    sentiment_output: Optional[SentimentNewsOutput] = None
    risk_output: Optional[RiskManagerOutput] = None
    debate_transcript: list[DebateRebuttal] = Field(default_factory=list)
    synthesis_report: Optional[SynthesisReport] = None
    agent_logs: list[AgentLogEntry] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def log(self, entry: AgentLogEntry) -> None:
        self.agent_logs.append(entry)
