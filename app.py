"""
app.py
Streamlit dashboard for the Gold (XAU/USD) Multi-Agent Prediction System.

Run with:  streamlit run app.py

The dashboard:
 - triggers a full multi-agent run (data fetch -> analysis -> debate -> synthesis)
 - shows a live-updating progress trace as each agent finishes
 - renders the executive summary, price chart with indicator overlays,
   per-agent reasoning chains, the debate log, and risk warnings
 - lets the user browse past runs persisted in SQLite
"""
from __future__ import annotations

import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import APP_CONFIG, validate_config
from orchestration.workflow import run_pipeline_sync
from schemas import AgentState
from storage import database
from tools.technical_indicators import compute_indicator_frame

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="XAU/USD Multi-Agent Forecaster", page_icon="🪙", layout="wide")

STAGE_LABELS = {
    "fetching_data": "Fetching market data, macro indicators, and news...",
    "data_fetched": "Market data collected.",
    "ingested": "Data Ingestion Agent normalized the raw payload.",
    "analyzing": "Technical, Macro, and Sentiment agents analyzing in parallel...",
    "analyzed": "Specialist analysis complete.",
    "risk_reviewed": "Risk Manager reviewed the specialists' conclusions.",
    "debating": "Lead Strategist is running the multi-agent debate...",
    "debated": "Debate complete — conflicts surfaced.",
    "synthesizing": "Lead Strategist synthesizing the final forecast...",
    "complete": "Run complete.",
}

DIRECTION_COLOR = {"Bullish": "#16a34a", "Bearish": "#dc2626", "Sideways": "#6b7280"}


def render_price_chart(state: AgentState) -> None:
    if not state.raw_data or not state.raw_data.gold_price_history:
        st.info("No price history available to chart.")
        return
    df = compute_indicator_frame(state.raw_data.gold_price_history)
    if df.empty:
        st.info("No price history available to chart.")
        return

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.2, 0.25], vertical_spacing=0.03,
        subplot_titles=("Gold Price & Moving Averages", "RSI (14)", "MACD"),
    )
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="XAU/USD"
    ), row=1, col=1)
    for col, name, color in [("sma_20", "SMA 20", "#2563eb"), ("sma_50", "SMA 50", "#f59e0b"),
                              ("bb_upper", "BB Upper", "rgba(107,114,128,0.5)"), ("bb_lower", "BB Lower", "rgba(107,114,128,0.5)")]:
        if col in df:
            fig.add_trace(go.Scatter(x=df["date"], y=df[col], name=name, line=dict(color=color, width=1)), row=1, col=1)

    if "rsi_14" in df:
        fig.add_trace(go.Scatter(x=df["date"], y=df["rsi_14"], name="RSI 14", line=dict(color="#7c3aed")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    if "macd" in df:
        fig.add_trace(go.Scatter(x=df["date"], y=df["macd"], name="MACD", line=dict(color="#0891b2")), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["macd_signal"], name="Signal", line=dict(color="#ea580c")), row=3, col=1)
        fig.add_trace(go.Bar(x=df["date"], y=df["macd_hist"], name="Hist", marker_color="#94a3b8"), row=3, col=1)

    fig.update_layout(height=750, xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, width="stretch")


def render_executive_summary(state: AgentState) -> None:
    report = state.synthesis_report
    if not report:
        st.warning("No synthesis report available for this run.")
        return

    c1, c2, c3, c4 = st.columns(4)
    color = DIRECTION_COLOR.get(report.final_direction.value, "#6b7280")
    c1.markdown(
        f"<h2 style='color:{color};margin-bottom:0'>{report.final_direction.value}</h2>"
        f"<span style='color:#6b7280'>Predicted direction</span>",
        unsafe_allow_html=True,
    )
    c2.metric("Confidence", f"{report.confidence_score * 100:.0f}%")
    if report.price_target_low is not None and report.price_target_high is not None:
        c3.metric("Target Range", f"${report.price_target_low:,.2f} – ${report.price_target_high:,.2f}")
    else:
        c3.metric("Target Range", "N/A")
    c4.metric("Timeframe", report.timeframe or "N/A")

    st.markdown(f"**Executive Summary:** {report.executive_summary}")

    if report.risk_warnings:
        with st.container(border=True):
            st.markdown("**⚠️ Risk Warnings**")
            for w in report.risk_warnings:
                st.markdown(f"- {w}")

    if report.dissenting_views:
        with st.expander("Dissenting views"):
            for d in report.dissenting_views:
                st.markdown(f"- {d}")


def render_agent_reasoning(state: AgentState) -> None:
    specialists = [
        ("📈 Technical Analyst", state.technical_output),
        ("💵 Macro Fundamentals", state.macro_output),
        ("📰 Sentiment & News", state.sentiment_output),
        ("🛡️ Risk Manager", state.risk_output),
    ]
    for label, output in specialists:
        if output is None:
            continue
        direction = getattr(output, "direction", None)
        badge = f":{'green' if direction and direction.value == 'Bullish' else 'red' if direction and direction.value == 'Bearish' else 'gray'}[{direction.value if direction else 'N/A'}]"
        with st.expander(f"{label} — {badge} (confidence {getattr(output, 'confidence', 0) * 100:.0f}%)"):
            st.markdown("**Reasoning chain:**")
            for i, step in enumerate(output.reasoning_chain, 1):
                st.markdown(f"{i}. {step}")
            if output.key_points:
                st.markdown("**Key points:**")
                for kp in output.key_points:
                    st.markdown(f"- {kp}")
            extra_fields = output.model_dump(exclude={"reasoning_chain", "direction", "confidence", "key_points"})
            if extra_fields:
                st.markdown("**Details:**")
                st.json(extra_fields)


def render_debate_log(state: AgentState) -> None:
    if not state.debate_transcript:
        st.info("No debate turns recorded for this run.")
        return
    for turn in state.debate_transcript:
        with st.chat_message("assistant"):
            st.markdown(f"**{turn.from_agent} → {turn.to_agent}**")
            st.markdown(f"*Challenge:* {turn.challenge}")
            if turn.rebuttal_or_concession:
                st.markdown(f"*Response:* {turn.rebuttal_or_concession}")

    if state.synthesis_report and state.synthesis_report.reasoning_chain:
        st.markdown("---")
        st.markdown("**Lead Strategist's final synthesis reasoning:**")
        for i, step in enumerate(state.synthesis_report.reasoning_chain, 1):
            st.markdown(f"{i}. {step}")


def render_data_warnings(state: AgentState) -> None:
    warnings = (state.raw_data.data_warnings if state.raw_data else []) + state.errors
    if warnings:
        with st.expander(f"⚠️ Data / system warnings ({len(warnings)})"):
            for w in warnings:
                st.markdown(f"- {w}")


def render_state(state: AgentState) -> None:
    st.subheader("Executive Summary")
    render_executive_summary(state)
    st.subheader("Market Data")
    render_price_chart(state)
    st.subheader("Agent Reasoning")
    render_agent_reasoning(state)
    st.subheader("Live Debate Log")
    render_debate_log(state)
    render_data_warnings(state)


def main() -> None:
    st.title("🪙 XAU/USD Multi-Agent Forecasting System")
    st.caption(
        "Multi-agent Groq pipeline: Technical, Macro, Sentiment, and Risk specialists debate "
        "and synthesize a gold price forecast. **Not financial advice** — for research/education only."
    )

    problems = validate_config()
    if problems:
        for p in problems:
            st.error(p)
        st.stop()

    database.init_db()

    with st.sidebar:
        st.header("Controls")
        run_clicked = st.button("▶️ Run new analysis", type="primary", width="stretch")
        st.markdown("---")
        st.subheader("Past runs")
        past_runs = database.list_runs(limit=20)
        run_options = {
            f"{r['started_at']} — {r['final_direction'] or 'incomplete'} ({(r['confidence_score'] or 0) * 100:.0f}%)": r["run_id"]
            for r in past_runs
        }
        selected_label = st.selectbox("View a past run", ["(current)"] + list(run_options.keys()))

    if run_clicked:
        progress_placeholder = st.empty()
        stage_log: list[str] = []

        def on_progress(stage: str, _state: AgentState) -> None:
            stage_log.append(STAGE_LABELS.get(stage, stage))
            with progress_placeholder.container():
                st.info("\n\n".join(f"✅ {s}" for s in stage_log[:-1]) or "")
                st.write(f"⏳ {stage_log[-1]}")

        with st.spinner("Running multi-agent analysis — this can take 30-90 seconds..."):
            try:
                final_state = run_pipeline_sync(on_progress=on_progress)
                st.session_state["last_state"] = final_state
                progress_placeholder.empty()
                st.success("Analysis complete.")
            except Exception as e:
                progress_placeholder.empty()
                st.error(f"Pipeline failed: {e}")
                st.stop()

    state_to_render: AgentState | None = None
    if selected_label != "(current)":
        run_id = run_options[selected_label]
        state_to_render = database.load_run_state(run_id)
    elif "last_state" in st.session_state:
        state_to_render = st.session_state["last_state"]

    if state_to_render:
        render_state(state_to_render)
    else:
        st.info("Click **Run new analysis** in the sidebar to generate a forecast.")


if __name__ == "__main__":
    main()
