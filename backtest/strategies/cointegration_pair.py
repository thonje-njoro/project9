"""Cointegration-based pairs trading with Kalman filter hedge ratio (Item 7).

Extends the macro pair approach (GLD_USO_RATIO) to additional pairs
formed from existing instruments: CPER/Gold, CPER/XAU, SLV/XAU, etc.

Uses a Kalman filter to track the time-varying hedge ratio — superior
to static OLS because the ratio drifts over months.

Key research:
- Kalman-filtered cointegration pairs outperform static-hedge-ratio pairs
- Cross-asset pairs (equity vs commodity, commodity vs currency) show
  more persistent cointegration than equity-equity pairs (TildAlice 2024)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def _compute_vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Compute Volume-Weighted Average Price."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = tp * df["volume"]
    vwap = pv.rolling(period).sum() / df["volume"].rolling(period).sum()
    return vwap.fillna(method="bfill").fillna(df["close"])


def kalman_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    delta: float = 0.0001,
) -> Tuple[pd.Series, pd.Series, float]:
    """Estimate time-varying hedge ratio between two assets via Kalman filter.

    State-space model:
      State: hedge_ratio_t = hedge_ratio_{t-1} + noise
      Observation: y_t = hedge_ratio_t * x_t + noise

    Args:
        y: Dependent asset price (e.g. CPER)
        x: Independent asset price (e.g. XAU/USD as GC=F)
        delta: Transition covariance (higher = ratio changes faster)

    Returns:
        (spread, hedge_ratios, half_life)
        spread = y - hedge_ratio * x (mean-reverting if cointegrated)
        hedge_ratios = time-varying hedge ratio estimates
        half_life = mean reversion half-life in bars (diagnostic)
    """
    try:
        from pykalman import KalmanFilter
    except ImportError:
        raise ImportError("pip install pykalman")

    # Align both series
    common_idx = y.dropna().index.intersection(x.dropna().index)
    y_aligned = y.loc[common_idx].values
    x_aligned = x.loc[common_idx].values

    if len(y_aligned) < 30:
        empty = pd.Series(0.0, index=common_idx)
        return empty, empty, 0.0

    # Observation matrix: we observe y_t, and x_t is the input
    obs_mat = x_aligned.reshape(-1, 1)

    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=np.ones((1, 1)),  # will be updated per-step
        initial_state_mean=1.0,
        initial_state_covariance=1.0,
        transition_covariance=delta,
        observation_covariance=1.0,
    )

    # Run Kalman filter with time-varying observation matrix
    n = len(y_aligned)
    filtered_states = np.zeros(n)
    filtered_covs = np.zeros(n)

    # Manual Kalman filter iteration (pykalman annoyingly doesn't support
    # time-varying observation matrices easily, so we DIY)
    state_mean = 1.0
    state_cov = 1.0

    for t in range(n):
        # Predict
        state_mean = state_mean  # transition matrix = 1
        state_cov = state_cov + delta

        # Update with current observation
        H = x_aligned[t]  # observation matrix = x_t
        innovation = y_aligned[t] - H * state_mean
        innovation_cov = H * state_cov * H + 1.0  # R = 1.0
        kalman_gain = state_cov * H / innovation_cov

        state_mean = state_mean + kalman_gain * innovation
        state_cov = (1 - kalman_gain * H) * state_cov

        filtered_states[t] = state_mean
        filtered_covs[t] = state_cov

    hedge_ratios = pd.Series(filtered_states, index=common_idx)
    spread = pd.Series(y_aligned - filtered_states * x_aligned, index=common_idx)

    # Compute half-life of mean reversion for the spread
    half_life = _compute_half_life(spread)

    return spread, hedge_ratios, half_life


def _compute_half_life(spread: pd.Series) -> float:
    """Compute mean reversion half-life from spread auto-regression.

    Regress spread_{t} - spread_{t-1} on spread_{t-1}.
    Half-life = -ln(2) / ln(1 + coefficient)
    """
    spread = spread.dropna()
    if len(spread) < 30:
        return 0.0

    y = spread.diff().iloc[1:]
    x = spread.shift(1).iloc[1:]

    # OLS: y = alpha + beta * x
    x_sm = np.column_stack([np.ones(len(x)), x.values])
    try:
        beta = np.linalg.lstsq(x_sm, y.values, rcond=None)[0]
        theta = beta[1]  # coefficient on lagged spread
        if theta >= 0:
            return 999.0  # not mean-reverting
        hl = -np.log(2) / np.log(1 + theta)
        return float(min(max(hl, 1), 500))
    except Exception:
        return 0.0


def generate_signals(
    df_pair: pd.DataFrame,
    df_hedge: pd.DataFrame,
    pair_name: str = "pair",
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    use_adaptive_hedge: bool = True,
    hedge_delta: float = 0.0001,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Generate mean-reversion signals for a cointegrated pair.

    This function is called from main.py with pre-constructed spread data.
    The 'df_pair' DataFrame already contains the spread as its 'close' column.

    Args:
        df_pair: DataFrame with 'close' = spread of the pair
        df_hedge: DataFrame with 'close' = hedge ratio (for reference)
        pair_name: Name for logging
        z_entry: Z-score threshold for entry (default 2.0)
        z_exit: Z-score threshold for exit (default 0.5)
        use_adaptive_hedge: Whether to use adaptive Kalman hedge ratio
        hedge_delta: Kalman transition covariance

    Returns:
        (long_entries, long_exits, short_entries, short_exits, trailing_stops)
    """
    close = df_pair["close"]
    idx = close.index

    # Z-score the spread
    spread_mean = close.rolling(20).mean()
    spread_std = close.rolling(20).std()
    z_score = (close - spread_mean) / spread_std.replace(0, np.nan)

    # Entry signals: spread is > z_entry std from mean
    long_entries_raw = z_score < -z_entry  # spread too low = long the pair
    short_entries_raw = z_score > z_entry  # spread too high = short the pair

    # Exit signals: spread reverts to within z_exit std of mean
    long_exits_raw = z_score > -z_exit
    short_exits_raw = z_score < z_exit

    # Trailing stop from ATR of the spread (computed by engine via trail_atr_mult)
    trailing_stops = pd.Series(0.0, index=idx)

    return (
        long_entries_raw.shift(1).fillna(False),
        long_exits_raw.shift(1).fillna(False),
        short_entries_raw.shift(1).fillna(False),
        short_exits_raw.shift(1).fillna(False),
        trailing_stops.shift(1).fillna(0),
    )
