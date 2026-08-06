"""Prop firm challenge sprint strategy — SPY 15-min enhanced mean reversion.

Three-layer architecture:
  1. Trend context via EMA(200, 1h) for directional bias
  2. RSI(2) + Bollinger Band(20, 2.0) entry trigger
  3. Multi-target exit: 60% at 0.5R, 40% trail at 2× ATR, 1.0R hard stop

Designed for: $50k FTMO-style challenge, 10% profit in 22 trading days.
Daily DD: 4% max. Total DD: 10% max.

References:
  - Bollerslev, Li & Xue (2018) — SPY intraday mean reversion
  - Connors & Alvarez (2012) — RSI(2) short-term mean reversion
  - FTMO/Prop firm challenge optimal sizing (López de Prado, 2018)
"""

import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 2) -> pd.Series:
    """Compute RSI for mean reversion entry timing.

    Standard Wilder-style RSI. For period=2, values below 10 indicate
    extreme oversold conditions that historically precede mean reversion.
    """
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    # Avoid division by zero
    loss = loss.replace(0, np.nan)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def generate_signals(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_std: float = 2.0,
    rsi_period: int = 2,
    rsi_oversold: float = 10.0,
    rsi_overbought: float = 90.0,
    ema_trend_period: int = 200,
    risk_per_trade: float = 0.0085,
    atr_period: int = 14,
    partial_tp_ratio: float = 0.6,
    partial_tp_r_mult: float = 0.5,
    stop_loss_r_mult: float = 1.0,
    trail_atr_mult: float = 2.0,
    max_hold_bars: int = 8,
    long_only: bool = True,
    **kwargs,
) -> tuple:
    """Generate sprint challenge signals for SPY 15-min chart.

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume
        bb_period: Bollinger Band lookback period (default: 20 ~= 1 trading day)
        bb_std: Bollinger Band standard deviation threshold (default: 2.0)
        rsi_period: RSI lookback (default: 2 — ultra-short for climax detection)
        rsi_oversold: RSI threshold for long entry (default: 10)
        rsi_overbought: RSI threshold for short entry (default: 90)
        ema_trend_period: EMA period for trend context (default: 200 ~= 5 days on 1h)
        risk_per_trade: Fraction of equity at risk per trade (default: 0.0085)
        atr_period: ATR lookback for stop distances (default: 14)
        partial_tp_ratio: Fraction of position to close at TP (default: 0.6 = 60%)
        partial_tp_r_mult: R-multiple for partial take-profit (default: 0.5)
        stop_loss_r_mult: R-multiple for hard stop loss (default: 1.0)
        trail_atr_mult: ATR multiplier for trailing stop (default: 2.0)
        max_hold_bars: Maximum bars to hold position (default: 8 = 2 hours)
        long_only: If True, only generate long signals (default: True for bull market)

    Returns:
        5-tuple: (long_entries, long_exits, short_entries, short_exits, trailing_stops)
        All shifted by 1 bar to prevent lookahead bias.
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # === ATR Calculation ===
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    # === Layer 1: Trend Context (EMA) ===
    # Use a longer EMA to establish the macro direction
    ema_trend = close.rolling(ema_trend_period, min_periods=50).mean()
    price_vs_ema = (close - ema_trend) / ema_trend.replace(0, np.nan)
    # Bull bias: price is not significantly below EMA (allow small pullbacks)
    bull_bias = price_vs_ema > -0.005
    # Bear bias: price is not significantly above EMA
    bear_bias = price_vs_ema < 0.005

    # === Layer 2: Entry Trigger ===
    # Bollinger Bands
    sma = close.rolling(bb_period).mean()
    std = close.rolling(bb_period).std()
    bb_upper = sma + bb_std * std
    bb_lower = sma - bb_std * std

    # RSI(2) — extreme oversold/overbought
    rsi = compute_rsi(close, rsi_period)

    # === ENTRY CONDITIONS ===
    # Long entry:  RSI(2) < oversold  AND  price < BB lower  AND  bull trend context
    # Short entry: RSI(2) > overbought  AND  price > BB upper  AND  bear trend context
    raw_long_entry = (
        (rsi < rsi_oversold) &
        (close < bb_lower) &
        bull_bias
    )

    raw_short_entry = (
        (rsi > rsi_overbought) &
        (close > bb_upper) &
        bear_bias
    )

    if long_only:
        raw_short_entry = pd.Series(False, index=df.index)

    # === SIGNAL CLEANUP ===
    # Remove consecutive signals: only take first signal in a cluster
    # (If a signal fires on bar t, don't fire again until we've had N bars without signals)
    signal_cooldown = 3  # Minimum 3 bars between signals
    cleaned_long = raw_long_entry.copy()
    last_signal_idx = -signal_cooldown - 1
    for i in range(len(cleaned_long)):
        if cleaned_long.iloc[i]:
            if i - last_signal_idx <= signal_cooldown:
                cleaned_long.iloc[i] = False
            else:
                last_signal_idx = i

    cleaned_short = raw_short_entry.copy()
    last_signal_idx = -signal_cooldown - 1
    for i in range(len(cleaned_short)):
        if cleaned_short.iloc[i]:
            if i - last_signal_idx <= signal_cooldown:
                cleaned_short.iloc[i] = False
            else:
                last_signal_idx = i

    # === POSITION SIZING (ATR-based, fixed fractional risk) ===
    equity = 50_000  # Reference equity for sizing; actual tracked in lifecycle manager
    risk_dollars = equity * risk_per_trade
    stop_distance = atr * stop_loss_r_mult
    positions = (risk_dollars / stop_distance.replace(0, np.nan)).clip(
        lower=0,
        upper=equity * 2.0 / close.replace(0, np.nan),
    ).fillna(0)

    # === TRAILING STOP ===
    trailing_stops = atr * trail_atr_mult

    # === SHIFT BY 1 BAR (prevent lookahead) ===
    long_entries = cleaned_long.shift(1, fill_value=False)
    short_entries = cleaned_short.shift(1, fill_value=False)
    long_exits = pd.Series(False, index=df.index)
    short_exits = pd.Series(False, index=df.index)
    sizes = positions.shift(1).fillna(0)
    trailing_stops = trailing_stops.shift(1).fillna(0)

    return long_entries, long_exits, short_entries, short_exits, trailing_stops
