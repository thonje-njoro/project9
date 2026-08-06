"""Time-Series Momentum (TSMOM) — Moskowitz, Ooi & Pedersen (2012).

OPERATES ON DAILY DATA. Do NOT use on intraday bars — the signal-to-noise
ratio is too low and you get hundreds of false signals.

Reference:
  Moskowitz, Ooi, Pedersen (2012), Journal of Financial Economics
  Hurst, Ooi, Pedersen (2017), "A Century of Evidence on Trend-Following"

Standard approach:
  - Daily close prices
  - 12-month lookback (252 trading days)
  - Volatility normalization to 15-20% target annualized vol
  - Entry/exit: signal crosses zero OR exceeds threshold
"""

import numpy as np
import pandas as pd


def compute_vol_target_factor(
    close: pd.Series,
    target_vol: float = 0.18,
    vol_span: int = 60,
    min_factor: float = 0.1,
    max_factor: float = 3.0,
) -> pd.Series:
    """Volatility scaling factor per the AQR TSMOM framework.

    Scale position size so each instrument contributes equal risk.
    """
    daily_ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
    ann_vol = daily_ret.ewm(span=vol_span, min_periods=vol_span).std() * np.sqrt(252)
    factor = target_vol / ann_vol.replace(0, np.nan)
    return factor.clip(min_factor, max_factor).bfill().fillna(1.0)


def generate_signals(
    df: pd.DataFrame,
    lookbacks: list = None,
    target_vol: float = 0.18,
    signal_threshold: float = 0.0,
    vol_span: int = 60,
    use_trailing_stop: bool = True,
    trail_atr_mult: float = 3.0,
    long_only: bool = True,
) -> tuple:
    """TSMOM trend-following on daily data.

    Lookbacks are in TRADING DAYS (not bars). Standard values:
      21 = 1 month, 63 = 3 months, 126 = 6 months, 252 = 12 months

    The signal is the average of sign(return) across all lookbacks,
    weighted by 1/sqrt(lookback) for responsiveness.

    Entry: blended signal goes from ≤ threshold to > threshold.
    Exit: blended signal goes from ≥ -threshold to < -threshold OR
          crosses zero after an extended positive run.
    """
    if lookbacks is None:
        lookbacks = [21, 63, 126, 252]  # standard AQR monthly lookbacks

    close = df["close"]
    idx = close.index
    n = len(close)

    min_bars_needed = max(lookbacks) + 5
    if n < min_bars_needed:
        empty = pd.Series(False, index=idx)
        zero = pd.Series(0.0, index=idx)
        return (empty, empty, empty, empty, zero)

    # --- 1. Multi-lookback blended signal (weighted by 1/sqrt(L)) ---
    signal_sum = pd.Series(0.0, index=idx)
    total_weight = 0.0

    for lb in lookbacks:
        ret = df["close"].pct_change(lb)
        # Directional signal: +1 bullish, -1 bearish, 0 neutral
        s = pd.Series(0.0, index=idx)
        s[ret > 0] = 1.0
        s[ret < 0] = -1.0

        # 1/sqrt(L) weighting: shorter lookbacks contribute more per-bar signal
        w = 1.0 / np.sqrt(lb)
        signal_sum += s.fillna(0) * w
        total_weight += w

    # Normalize to [-1, 1] range
    blended = signal_sum / total_weight if total_weight > 0 else signal_sum

    # --- 2. Smooth signal with 3-bar MA to reduce flip noise ---
    blended = blended.rolling(3, min_periods=1, center=False).mean()

    # --- 3. Generate entry/exit signals ---
    # Entry: signal crosses above entry_threshold
    entry_th = signal_threshold
    # Exit: signal crosses below -entry_th (or zero crossing on downswing)
    exit_th = -entry_th

    long_entries_raw = (blended > entry_th) & (blended.shift(1) <= entry_th)
    long_exits_raw = (blended < exit_th) & (blended.shift(1) >= exit_th)
    # Also exit if signal stayed positive but has been declining and crosses zero
    long_exits_raw = long_exits_raw | ((blended <= 0) & (blended.shift(1) > 0))

    if long_only:
        short_entries_raw = pd.Series(False, index=idx)
        short_exits_raw = pd.Series(False, index=idx)
    else:
        short_entries_raw = (blended < -entry_th) & (blended.shift(1) >= -entry_th)
        short_exits_raw = (blended > -exit_th) & (blended.shift(1) <= -exit_th)
        short_exits_raw = short_exits_raw | ((blended >= 0) & (blended.shift(1) < 0))

    # --- 4. Trailing stop based on ATR (wider for longer lookbacks) ---
    trailing_stops = pd.Series(0.0, index=idx)
    if use_trailing_stop:
        from risk.position_sizer import compute_atr
        atr = compute_atr(df, 14)
        # Scale trail by current vol level
        vf = compute_vol_target_factor(close, target_vol=target_vol, vol_span=vol_span)
        effective_mult = trail_atr_mult * vf.clip(0.5, 2.0)
        trailing_stops = atr * effective_mult

    return (
        long_entries_raw.shift(1).fillna(False).astype(bool),
        long_exits_raw.shift(1).fillna(False).astype(bool),
        short_entries_raw.shift(1).fillna(False).astype(bool),
        short_exits_raw.shift(1).fillna(False).astype(bool),
        trailing_stops.shift(1).fillna(0),
    )
