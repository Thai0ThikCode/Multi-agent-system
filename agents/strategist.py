"""
agents/strategist.py
Lead_Strategist_Agent (Executive Orchestrator):
 1. Debate phase — surfaces conflicts between the four specialist opinions.
 2. Synthesis phase — resolves conflicts and produces the final SynthesisReport.

Both phases use the heavier `pro` model (default openai/gpt-oss-120b on Groq).
"""
from __future__ import annotations

import json
import logging

from config import MODEL_CONFIG
from schemas import (
    Direction,
    TechnicalAnalysisOutput,
    MacroFundamentalsOutput,
    SentimentNewsOutput,
    RiskManagerOutput,
    DebateRebuttal,
    DebateRebuttalList,
    SynthesisReport,
)
from agents.base_agent import run_structured_agent, fallback_instance, digest_output
from agents.prompts import DEBATE_TURN_PROMPT, SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)


def _specialists_block(
    technical: TechnicalAnalysisOutput,
    macro: MacroFundamentalsOutput,
    sentiment: SentimentNewsOutput,
    risk: RiskManagerOutput,
) -> str:
    # Compact digests (trimmed reasoning, no indentation) keep the prompt well inside the
    # per-minute token budget of Groq's free tier.
    return (
        f"Technical_Analyst_Agent: {digest_output(technical)}\n\n"
        f"Macro_Fundamentals_Agent: {digest_output(macro)}\n\n"
        f"Sentiment_News_Agent: {digest_output(sentiment)}\n\n"
        f"Risk_Manager_Agent: {digest_output(risk)}"
    )


def run_debate_turn(
    technical: TechnicalAnalysisOutput,
    macro: MacroFundamentalsOutput,
    sentiment: SentimentNewsOutput,
    risk: RiskManagerOutput,
) -> list[DebateRebuttal]:
    user_content = _specialists_block(technical, macro, sentiment, risk)
    try:
        result = run_structured_agent(
            model=MODEL_CONFIG.pro_model,
            system_instruction=DEBATE_TURN_PROMPT,
            user_content=user_content,
            schema=DebateRebuttalList,
            temperature=MODEL_CONFIG.temperature_debate,
        )
        return result.rebuttals
    except Exception as e:
        logger.error("Debate turn failed after retries: %s", e)
        return [
            DebateRebuttal(
                from_agent="Lead_Strategist_Agent",
                to_agent="System",
                challenge="Debate phase failed to produce output.",
                rebuttal_or_concession=f"Error: {e}",
            )
        ]


def run_synthesis(
    technical: TechnicalAnalysisOutput,
    macro: MacroFundamentalsOutput,
    sentiment: SentimentNewsOutput,
    risk: RiskManagerOutput,
    debate: list[DebateRebuttal],
) -> SynthesisReport:
    debate_json = json.dumps([d.model_dump() for d in debate], separators=(",", ":"), ensure_ascii=False)
    user_content = f"{_specialists_block(technical, macro, sentiment, risk)}\n\nDebate transcript: {debate_json}"
    try:
        return run_structured_agent(
            model=MODEL_CONFIG.pro_model,
            system_instruction=SYNTHESIS_PROMPT,
            user_content=user_content,
            schema=SynthesisReport,
            temperature=MODEL_CONFIG.temperature_debate,
        )
    except Exception as e:
        logger.error("Synthesis failed after retries: %s", e)
        return fallback_instance(
            SynthesisReport,
            reasoning_chain=[f"Synthesis agent call failed: {e}"],
            final_direction=Direction.SIDEWAYS,
            confidence_score=0.0,
            timeframe="unknown",
            executive_summary=(
                "The synthesis step could not complete because the underlying model call failed "
                "after multiple retries. Please check your Groq API key, quota, and network connection, "
                "then re-run the analysis."
            ),
        )
