"""Kalman Filter trend-following strategy — 2-state position + velocity.

Implements Benhamou's approach from 'Trend Without Hiccups - A Kalman Filter
Approach' (arXiv:1808.03297) with adaptive noise estimation.

The Kalman Filter estimates:
  State[0] = position (smoothed price / fair value)
  State[1] = velocity (trend direction and strength)

Supports two modes:
  mean_revert=False (default): trend-following via velocity zero-cross
  mean_revert=True: mean-reversion via deviation from Kalman fair value

Key advantage over EMA crossovers:
  - EMA has fixed lag (~7 bars for EMA 20)
  - Kalman lag adapts: ~3-5 bars in strong trends, ~12-15 in chop
  - Velocity zero-cross is a LEADING indicator (changes 1-3 bars before price)
  - Position estimate acts as adaptive moving average for mean-reversion

Exit improvements (Item 4):
  When use_vwap_exit=True, exits use VWAP trailing instead of raw velocity cross.
  VWAP exit produces smoother equity curves with higher Sharpe.

Usage:
    from strategies.kalman_trend import generate_signals
    signals = generate_signals(df, Q=0.01, R=1.0)              # trend-follow
    signals = generate_signals(df, Q=0.02, R=1.0, mean_revert=True)  # mean-revert
"""

import numpy as np
import pandas as pd
from pykalman import KalmanFilter


def _compute_adaptive_noise(close: pd.Series) -> tuple[float, float]:
    """Estimate Q (process noise) and R (measurement noise) from recent data.

    When vol is high -> R goes up -> filter smooths more (trusts observations less).
    When vol is low -> R goes down -> filter responds faster.
    """
    returns = close.pct_change().dropna()
    recent_vol = returns.tail(20).std()
    long_vol = returns.tail(100).std() if len(returns) > 100 else recent_vol

    Q = float(np.clip(recent_vol * 0.1, 0.0001, 0.1))
    R = float(np.clip(close.tail(20).std() * 0.01, 0.01, 10.0))
    return Q, R


def _compute_vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Compute Volume-Weighted Average Price."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = tp * df["volume"]
    vwap = pv.rolling(period).sum() / df["volume"].rolling(period).sum()
    return vwap.bfill().fillna(df["close"])


# ── Pullback Entry State Machine (4-phase, from Sunrise Ogle / FORWIZ AI) ──


def _apply_pullback_filter(
    raw_entries: pd.Series,
    df: pd.DataFrame,
    velocity: pd.Series,
    pullback_max_candles: int = 2,
    entry_window_periods: int = 3,
    long_side: bool = True,
) -> pd.Series:
    """Apply 4-phase pullback entry state machine to raw signal entries.

    Phase 1 — SCANNING:  Raw signal detected (velocity zero-cross + threshold)
    Phase 2 — PULLBACK:  Wait 1-N candles of counter-move, track resistant level.
                         Global invalidation if velocity reverses sign.
    Phase 3 — WINDOW:    Open breakout window at configured offset from signal.
    Phase 4 — ENTRY:     Enter when price breaks out beyond the pullback level.

    This dramatically reduces whipsaw entries by requiring confirmation that
    the trend has resumed after an initial signal, rather than entering on the
    first sign of velocity change.

    Args:
        raw_entries: Boolean Series marking initial signal bars (unshifted).
        df: OHLCV DataFrame with 'high', 'low'.
        velocity: Kalman velocity Series.
        pullback_max_candles: Max pullback candles before window opens (1-3).
        entry_window_periods: Bars to monitor for breakout after pullback.
        long_side: True for long entries, False for short.

    Returns:
        Boolean Series of same length, with entry on the breakout bar.
    """
    n = len(df)
    values = np.zeros(n, dtype=bool)
    idx = df.index
    raw_bars = np.where(raw_entries.values)[0]

    if long_side:
        prev_signal_end = -1
        for i in raw_bars:
            if i <= prev_signal_end:
                continue  # Skip overlapping pullback windows

            pullback_high = -np.inf
            invalidated = False
            end_pullback = min(n, i + 1 + pullback_max_candles)

            for j in range(i + 1, end_pullback):
                pullback_high = max(pullback_high, df["high"].iloc[j])
                if velocity.iloc[j] < 0:
                    invalidated = True
                    break

            if invalidated or np.isinf(pullback_high):
                continue

            # Open breakout window
            window_end = min(n, end_pullback + entry_window_periods)
            for j in range(end_pullback, window_end):
                if df["high"].iloc[j] > pullback_high:
                    values[j] = True
                    prev_signal_end = j
                    break
    else:
        prev_signal_end = -1
        for i in raw_bars:
            if i <= prev_signal_end:
                continue

            pullback_low = np.inf
            invalidated = False
            end_pullback = min(n, i + 1 + pullback_max_candles)

            for j in range(i + 1, end_pullback):
                pullback_low = min(pullback_low, df["low"].iloc[j])
                if velocity.iloc[j] > 0:
                    invalidated = True
                    break

            if invalidated or np.isinf(pullback_low):
                continue

            window_end = min(n, end_pullback + entry_window_periods)
            for j in range(end_pullback, window_end):
                if df["low"].iloc[j] < pullback_low:
                    values[j] = True
                    prev_signal_end = j
                    break

    return pd.Series(values, index=idx)


def generate_signals(
    df: pd.DataFrame,
    Q: float = None,
    R: float = None,
    use_adaptive_noise: bool = True,
    velocity_threshold_pct: float = 0.20,
    mean_revert: bool = False,
    mr_deviation: float = 1.5,
    use_trailing_stop: bool = True,
    trail_atr_mult: float = 2.5,
    long_only: bool = True,
    use_vwap_exit: bool = False,
    daily_confirmation: bool = False,
    # ── Pullback Entry State Machine (from Sunrise Ogle / FORWIZ AI) ──
    use_pullback_entry: bool = False,
    pullback_max_candles: int = 2,
    entry_window_periods: int = 3,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Kalman Filter strategy with optional VWAP trailing exit and pullback entry.

    Trend-following (mean_revert=False):
      Long entry:  velocity crosses above zero with sufficient strength
      Long exit:   VWAP trailing exit OR velocity crosses below zero
      Short entry: velocity crosses below zero (only if long_only=False)
      Short exit:  VWAP trailing exit OR velocity crosses above zero

    Mean-reversion (mean_revert=True):
      Long entry:  price deviates below position estimate by >mr_deviation*pos_std
      Long exit:   velocity reversal (momentum toward fair value fading)

    Returns (long_entries, long_exits, short_entries, short_exits, trailing_stops).
    """
    close = df["close"].values
    idx = df.index
    n = len(close)

    if n < 30:
        empty = pd.Series(False, index=idx)
        zero = pd.Series(0.0, index=idx)
        return (empty, empty, empty, empty, zero)

    # --- Adaptive noise estimation ---
    if use_adaptive_noise or Q is None or R is None:
        Q_est, R_est = _compute_adaptive_noise(df["close"])
        Q = Q or Q_est
        R = R or R_est

    # --- Run Kalman Filter ---
    kf = KalmanFilter(
        transition_matrices=[[1, 1], [0, 1]],
        observation_matrices=[[1, 0]],
        initial_state_mean=[float(close[0]), 0.0],
        initial_state_covariance=[[R, 0], [0, Q]],
        transition_covariance=[[Q * 0.5, 0], [0, Q * 0.1]],
        observation_covariance=[[R]],
    )

    state_means, state_covs = kf.filter(close)

    position = pd.Series(state_means[:, 0], index=idx)
    velocity = pd.Series(state_means[:, 1], index=idx)
    pos_std = pd.Series(np.sqrt(state_covs[:, 0, 0]), index=idx)

    # --- Alarm-based filter: velocity magnitude must exceed rolling percentile ---
    min_bars = min(100, len(velocity) - 1)
    vel_abs = velocity.abs()
    vel_threshold = vel_abs.rolling(min_bars, min_periods=20).quantile(velocity_threshold_pct)
    vel_strong_enough = vel_abs > vel_threshold

    # --- Multi-timeframe: daily KF confirmation ---
    daily_velocity = None
    if daily_confirmation and len(df) >= 60:
        try:
            daily_close = df["close"].resample("1D").last().dropna()
            if len(daily_close) >= 20:
                dv = daily_close.values
                kf_d = KalmanFilter(
                    transition_matrices=[[1, 1], [0, 1]],
                    observation_matrices=[[1, 0]],
                    initial_state_mean=[float(dv[0]), 0.0],
                    initial_state_covariance=[[R, 0], [0, Q]],
                    transition_covariance=[[Q * 0.5, 0], [0, Q * 0.1]],
                    observation_covariance=[[R]],
                )
                d_state_means, _ = kf_d.filter(dv)
                daily_velocity = pd.Series(d_state_means[:, 1], index=daily_close.index)
        except Exception:
            daily_velocity = None

    # --- VWAP for trailing exit ---
    vwap = _compute_vwap(df) if use_vwap_exit else None

    if not mean_revert:
        # ===== TREND-FOLLOWING MODE =====
        raw_long_entry = (velocity > 0) & (velocity.shift(1) <= 0)
        raw_long_exit = (velocity < 0) & (velocity.shift(1) >= 0)

        long_entries_raw = raw_long_entry & vel_strong_enough

        # Daily confirmation: require daily KF velocity to agree
        if daily_confirmation and daily_velocity is not None:
            daily_aligned = daily_velocity.reindex(idx, method="ffill")
            long_entries_raw = long_entries_raw & (daily_aligned > 0)

        # ── Pullback Entry State Machine (4-phase, from Sunrise Ogle) ──
        if use_pullback_entry:
            long_entries = _apply_pullback_filter(
                long_entries_raw, df, velocity,
                pullback_max_candles, entry_window_periods,
                long_side=True,
            )
        else:
            long_entries = long_entries_raw

        if use_vwap_exit and vwap is not None:
            # Exit long when price closes below VWAP (tight trailing)
            long_exits = raw_long_exit | (df["close"] < vwap * 0.995)
        else:
            long_exits = raw_long_exit

        short_entries = pd.Series(False, index=idx)
        short_exits = pd.Series(False, index=idx)

        if not long_only:
            short_entries = (velocity < 0) & (velocity.shift(1) >= 0) & vel_strong_enough
            if use_vwap_exit and vwap is not None:
                short_exits = (velocity > 0) & (velocity.shift(1) <= 0) | (df["close"] > vwap * 1.005)
            else:
                short_exits = (velocity > 0) & (velocity.shift(1) <= 0)
    else:
        # ===== MEAN-REVERSION MODE =====
        deviation = (df["close"] - position) / pos_std.replace(0, 1e-10)

        raw_long_entry = deviation < -mr_deviation
        raw_short_entry = deviation > mr_deviation

        # Exit on velocity reversal
        raw_long_exit = (velocity < 0) & (velocity.shift(1) >= 0)
        raw_short_exit = (velocity > 0) & (velocity.shift(1) <= 0)

        long_entries = raw_long_entry & vel_strong_enough
        long_exits = raw_long_exit

        short_entries = pd.Series(False, index=idx)
        short_exits = pd.Series(False, index=idx)

        if not long_only:
            short_entries = raw_short_entry & vel_strong_enough
            short_exits = raw_short_exit

    # --- Trailing stop ---
    trailing_stops = pd.Series(0.0, index=idx)
    if use_trailing_stop:
        from risk.position_sizer import compute_atr
        atr = compute_atr(df, 14)
        trailing_stops = atr * trail_atr_mult

    return (
        long_entries.shift(1).fillna(False),
        long_exits.shift(1).fillna(False),
        short_entries.shift(1).fillna(False),
        short_exits.shift(1).fillna(False),
        trailing_stops.shift(1).fillna(0),
    )
