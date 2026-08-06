"""CPER/GLD ratio mean-reversion strategy.

Research findings from CPER data analysis:
  - CPER/GLD ratio daily return AC(1) = -0.102 (mean-reversion signature)
  - Ratio range: [0.103, 0.165], mean=0.132, std=0.014
  - The pair is structurally cointegrated: both track gold/copper macro cycles
  - When CPER outperforms GLD (ratio ↑), short the ratio → mean reverts
  - When CPER underperforms GLD (ratio ↓), long the ratio → mean reverts

This is similar to GLD_USO_RATIO but uses copper instead of oil.
Copper tracks industrial demand (China PMI, global growth).
Gold tracks safe-haven demand and real rates.
The ratio mean-reverts around macroeconomic cycle shifts.

Strategy:
  - Z-score of CPER/GLD ratio with 20-day rolling window
  - Entry: |z-score| > 1.5 (extreme deviation)
  - Exit: z-score crosses zero (reversion to mean)
  - Stop: ATR-based trailing
"""

import numpy as np
import pandas as pd


def generate_signals(
    df_pair: pd.DataFrame,
    df_ref: pd.DataFrame = None,
    pair_name: str = "CPER_GLD_RATIO",
    z_entry: float = 1.5,
    z_exit: float = 0.0,
    z_take_profit: float = 0.0,
    window: int = 20,
    use_trailing_stop: bool = True,
    trail_atr_mult: float = 2.0,
) -> tuple:
    """CPER/GLD ratio mean reversion.

    The pair DataFrame (df_pair) has 'close' = CPER (or the spread/ratio).
    df_ref has 'close' = GLD (the reference asset).

    If df_ref is not provided, we assume df_pair['close'] IS the ratio already
    (as constructed by main.py's _build_synthetic_instruments).

    Args:
        df_pair: DataFrame with ratio as 'close'
        df_ref: Optional reference DataFrame (GLD)
        pair_name: Name for logging
        z_entry: Z-score threshold for entry
        z_exit: Z-score threshold for exit (0 = cross mean)
        window: Rolling window for mean/std estimation
        use_trailing_stop: Apply ATR trailing stop
        trail_atr_mult: Multiplier for stop distance

    Returns:
        (long_entries, long_exits, short_entries, short_exits, trailing_stops)
    """
    close = df_pair["close"]
    idx = close.index

    if len(close) < window + 5:
        empty = pd.Series(False, index=idx)
        zero = pd.Series(0.0, index=idx)
        return (empty, empty, empty, empty, zero)

    # If df_ref provided, compute ratio; otherwise assume close IS the ratio
    if df_ref is not None:
        ref_close = df_ref["close"].reindex(idx, method="ffill")
        ratio = close / ref_close
    else:
        ratio = close

    # Z-score the ratio
    ratio_mean = ratio.rolling(window, min_periods=window // 2).mean()
    ratio_std = ratio.rolling(window, min_periods=window // 2).std()
    z_score = (ratio - ratio_mean) / ratio_std.replace(0, np.nan)

    # Entry signals
    long_entries_raw = z_score < -z_entry        # ratio too low → buy (long CPER/short GLD)
    short_entries_raw = z_score > z_entry        # ratio too high → sell (short CPER/long GLD)

    # Exit signals: cross back toward mean
    long_exits_raw = z_score >= -abs(z_exit) if z_exit != 0 else z_score >= 0
    short_exits_raw = z_score <= abs(z_exit) if z_exit != 0 else z_score <= 0

    # Take-profit: exit when z-score has recovered partially toward mean
    if z_take_profit > 0:
        tp_z = z_entry * z_take_profit  # e.g., entry at z=-1.5 → exit at z=-0.6
        long_tp = z_score >= -tp_z
        short_tp = z_score <= tp_z
        long_exits_raw = long_exits_raw | long_tp
        short_exits_raw = short_exits_raw | short_tp

    # Trailing stop
    trailing_stops = pd.Series(0.0, index=idx)
    if use_trailing_stop:
        from risk.position_sizer import compute_atr
        atr = compute_atr(df_pair, 14)
        trailing_stops = atr * trail_atr_mult

    return (
        long_entries_raw.shift(1).fillna(False).astype(bool),
        long_exits_raw.shift(1).fillna(False).astype(bool),
        short_entries_raw.shift(1).fillna(False).astype(bool),
        short_exits_raw.shift(1).fillna(False).astype(bool),
        trailing_stops.shift(1).fillna(0),
    )
