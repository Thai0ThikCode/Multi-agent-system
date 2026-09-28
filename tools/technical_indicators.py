"""
tools/technical_indicators.py
Computes RSI, MACD, SMA/EMA, Bollinger Bands, and naive support/resistance
levels from a list of PricePoint objects.

Indicators are computed with plain pandas so the module has no hard
dependency on any particular TA library version (avoids pandas-ta's
occasional breakage on newer pandas/numpy). If `ta` is installed it is
used for cross-validation-free convenience, but the fallback below is
self-contained and always available.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from schemas import PricePoint, TechnicalSnapshot

logger = logging.getLogger(__name__)


def _to_dataframe(history: list[PricePoint]) -> pd.DataFrame:
    if not history:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame([p.model_dump() for p in history])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = close.rolling(window=period, min_periods=max(2, period // 2)).mean()
    std = close.rolling(window=period, min_periods=max(2, period // 2)).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def _naive_support_resistance(df: pd.DataFrame, window: int = 5, top_n: int = 3) -> tuple[list[float], list[float]]:
    """Finds local swing lows/highs using a simple rolling-window pivot method."""
    if len(df) < window * 2 + 1:
        return [], []
    lows, highs = [], []
    for i in range(window, len(df) - window):
        low_slice = df["low"].iloc[i - window : i + window + 1]
        high_slice = df["high"].iloc[i - window : i + window + 1]
        if df["low"].iloc[i] == low_slice.min():
            lows.append(float(df["low"].iloc[i]))
        if df["high"].iloc[i] == high_slice.max():
            highs.append(float(df["high"].iloc[i]))
    supports = sorted(set(round(x, 2) for x in lows))[:top_n]
    resistances = sorted(set(round(x, 2) for x in highs), reverse=True)[:top_n]
    return supports, resistances


def compute_technical_snapshot(history: list[PricePoint]) -> Optional[TechnicalSnapshot]:
    df = _to_dataframe(history)
    if df.empty or len(df) < 5:
        logger.warning("Insufficient price history to compute technical indicators.")
        return None

    close = df["close"]
    last_close = float(close.iloc[-1])

    rsi_series = _rsi(close, 14)
    macd_line, signal_line, hist = _macd(close)
    bb_upper, bb_mid, bb_lower = _bollinger(close)

    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    ema_20 = close.ewm(span=20, adjust=False).mean()

    def _last_or_none(series: pd.Series) -> Optional[float]:
        val = series.iloc[-1] if len(series) else None
        return None if val is None or pd.isna(val) else round(float(val), 4)

    trend = "neutral"
    s20, s50 = _last_or_none(sma_20), _last_or_none(sma_50)
    if s20 is not None and s50 is not None:
        trend = "golden_cross" if s20 > s50 else "death_cross" if s20 < s50 else "neutral"

    supports, resistances = _naive_support_resistance(df)

    return TechnicalSnapshot(
        last_close=round(last_close, 4),
        rsi_14=_last_or_none(rsi_series),
        macd=_last_or_none(macd_line),
        macd_signal=_last_or_none(signal_line),
        macd_hist=_last_or_none(hist),
        sma_20=s20,
        sma_50=s50,
        sma_200=_last_or_none(sma_200),
        ema_20=_last_or_none(ema_20),
        bb_upper=_last_or_none(bb_upper),
        bb_lower=_last_or_none(bb_lower),
        bb_mid=_last_or_none(bb_mid),
        trend_20_50=trend,
        support_levels=supports,
        resistance_levels=resistances,
    )


def compute_indicator_frame(history: list[PricePoint]) -> pd.DataFrame:
    """Returns a full DataFrame with indicator columns attached, for charting."""
    df = _to_dataframe(history)
    if df.empty:
        return df
    df["rsi_14"] = _rsi(df["close"], 14)
    macd_line, signal_line, hist = _macd(df["close"])
    df["macd"], df["macd_signal"], df["macd_hist"] = macd_line, signal_line, hist
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()
    bb_upper, bb_mid, bb_lower = _bollinger(df["close"])
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = bb_upper, bb_mid, bb_lower
    return df
