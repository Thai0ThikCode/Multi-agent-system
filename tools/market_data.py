"""
tools/market_data.py
Pulls historical price data for gold and macro proxies via yfinance.
All functions are synchronous (yfinance is not natively async) — callers should
wrap them with `asyncio.to_thread` to keep the orchestration loop non-blocking.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import DATA_CONFIG
from schemas import PricePoint, MacroSnapshot

logger = logging.getLogger(__name__)

_RETRYABLE = (ConnectionError, TimeoutError, ValueError)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
)
def _download(ticker: str, period_days: int) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=f"{period_days}d",
        interval="1d",
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker {ticker}")
    # yfinance may return MultiIndex columns for a single ticker in newer versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _history_from_dataframe(df: pd.DataFrame) -> list[PricePoint]:
    points: list[PricePoint] = []
    for idx, row in df.iterrows():
        try:
            points.append(
                PricePoint(
                    date=str(idx.date()) if hasattr(idx, "date") else str(idx),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row.get("Volume", 0.0) or 0.0),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return points


def fetch_gold_history(period_days: Optional[int] = None) -> list[PricePoint]:
    """Fetch OHLCV history for the configured gold ticker."""
    period_days = period_days or DATA_CONFIG.lookback_days
    try:
        df = _download(DATA_CONFIG.gold_ticker, period_days)
        return _history_from_dataframe(df)
    except Exception as e:
        logger.warning("Gold history fetch failed for %s: %s", DATA_CONFIG.gold_ticker, e)
        return []


def fetch_oil_history(period_days: Optional[int] = None) -> list[PricePoint]:
    """Fetch OHLCV history for WTI Crude Oil."""
    period_days = period_days or DATA_CONFIG.lookback_days
    try:
        df = _download(DATA_CONFIG.oil_ticker, period_days)
        return _history_from_dataframe(df)
    except Exception as e:
        logger.warning("Crude oil history fetch failed for %s: %s", DATA_CONFIG.oil_ticker, e)
        return []


def fetch_gas_history(period_days: Optional[int] = None) -> list[PricePoint]:
    """Fetch OHLCV history for Natural Gas futures."""
    period_days = period_days or DATA_CONFIG.lookback_days
    try:
        df = _download(DATA_CONFIG.gas_ticker, period_days)
        return _history_from_dataframe(df)
    except Exception as e:
        logger.warning("Natural gas history fetch failed for %s: %s", DATA_CONFIG.gas_ticker, e)
        return []


def _pct_change(series: pd.Series, periods: int) -> Optional[float]:
    if len(series) <= periods:
        return None
    try:
        old = float(series.iloc[-periods - 1])
        new = float(series.iloc[-1])
        if old == 0:
            return None
        return round((new - old) / abs(old) * 100.0, 3)
    except (IndexError, TypeError, ValueError):
        return None


def _diff_change(series: pd.Series, periods: int) -> Optional[float]:
    """Absolute difference (useful for bond yields)."""
    if len(series) <= periods:
        return None
    try:
        old = float(series.iloc[-periods - 1])
        new = float(series.iloc[-1])
        return round(new - old, 3)
    except (IndexError, TypeError, ValueError):
        return None


def fetch_macro_snapshot(period_days: Optional[int] = None) -> MacroSnapshot:
    """Fetch DXY, 10Y yield, and crude oil, and derive short-term change metrics."""
    period_days = period_days or max(DATA_CONFIG.lookback_days, 30)
    snapshot = MacroSnapshot()
    notes: list[str] = []

    def _safe_last_and_change(ticker: str, is_diff: bool = False):
        try:
            df = _download(ticker, period_days)
            close = df["Close"].dropna()
            last = float(close.iloc[-1])
            chg = _diff_change(close, 5) if is_diff else _pct_change(close, 5)
            return last, chg
        except Exception as e:
            notes.append(f"Failed to fetch {ticker}: {e}")
            return None, None

    snapshot.dxy_last, snapshot.dxy_change_pct_5d = _safe_last_and_change(DATA_CONFIG.dxy_ticker, is_diff=False)
    snapshot.us10y_yield_last, snapshot.us10y_change_5d = _safe_last_and_change(DATA_CONFIG.yield_ticker, is_diff=True)
    snapshot.crude_oil_last, snapshot.crude_oil_change_pct_5d = _safe_last_and_change(DATA_CONFIG.oil_ticker, is_diff=False)
    snapshot.notes = notes
    return snapshot
