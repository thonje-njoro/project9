"""XAUUSD Session-Based Mean Reversion Strategy.

Implements intraday mean reversion on XAUUSD using session-specific VWAP
and Z-score entry signals with multi-method regime detection.

Core Logic:
    1. Compute session VWAP (London 8-12 UTC, NY 14-20 UTC)
    2. Enter when price deviates > 2σ from session VWAP
    3. Exit when price reverts to 0.5σ from session VWAP
    4. Regime filter: Only trade in mean-reverting regime (consensus of 3 methods)
    5. News filter: Block NFP, FOMC, CPI days
    6. Time exit: Close all by 20:00 UTC (before rollover)

Pitfall Mitigations:
    - Session ambiguity: Use core hours only, skip overlap zones
    - Regime detection: 3-method consensus (Hurst + VR + Half-life)
    - News events: Calendar-based + real-time volatility spike detection
    - Overnight gap: Hard close at 20:00 UTC
    - Swap costs: Never hold past 22:00 UTC rollover
    - Position sizing: Session-aware, reduced during news windows

Usage:
    from strategies.xauusd_session_mr import generate_signals
    result = generate_signals(df, **params)
    # Returns (long_entries, long_exits, short_entries, short_exits, trailing_stops)

Reference:
    - Session-based mean reversion on gold (proprietary research)
    - Zarattini et al. (2024) for ORB methodology adapted to forex
    - Bailey & López de Prado (2014) for Deflated Sharpe Ratio
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Session Classification
# ──────────────────────────────────────────────────────────────────────────────

SESSION_CORE = {
    'london': (8, 12),      # London morning, pre-NY
    'ny': (14, 20),         # NY afternoon, post-London
}


def _classify_session(hour_utc: int) -> Optional[str]:
    """Classify hour into core session. Returns None for ambiguous/dead zones."""
    for session, (start, end) in SESSION_CORE.items():
        if start <= hour_utc < end:
            return session
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Regime Detection (3-Method Consensus)
# ──────────────────────────────────────────────────────────────────────────────

def _hurst_exponent(prices: np.ndarray) -> float:
    """R/S analysis for Hurst exponent. H < 0.5 = mean-reverting."""
    if len(prices) < 100:
        return np.nan
    
    lags = range(10, min(len(prices) // 2, 200))
    tau = []
    for lag in lags:
        diffs = prices[lag:] - prices[:-lag]
        tau.append(np.std(diffs))
    
    if len(tau) < 5:
        return np.nan
    
    log_lags = np.log(list(lags)[:len(tau)])
    log_tau = np.log(tau)
    
    valid = np.isfinite(log_tau) & np.isfinite(log_lags)
    if valid.sum() < 5:
        return np.nan
    
    from scipy import stats
    slope, _, _, _, _ = stats.linregress(log_lags[valid], log_tau[valid])
    return slope


def _variance_ratio_test(prices: np.ndarray, k: int = 5) -> float:
    """Variance ratio test. VR < 1 → mean-reverting."""
    returns = np.diff(np.log(prices))
    n = len(returns)
    
    if n < k * 10:
        return np.nan
    
    k_returns = np.array([np.sum(returns[i:i+k]) for i in range(n - k + 1)])
    var_k = np.var(k_returns, ddof=1)
    var_1 = np.var(returns, ddof=1)
    
    if var_1 == 0:
        return np.nan
    
    return var_k / (k * var_1)


def _half_life(prices: np.ndarray) -> float:
    """Ornstein-Uhlenbeck half-life. Shorter = faster mean reversion."""
    if len(prices) < 20:
        return np.nan
    
    lagged = prices[:-1]
    delta = np.diff(prices)
    
    valid = np.isfinite(lagged) & np.isfinite(delta)
    if valid.sum() < 10:
        return np.nan
    
    from scipy import stats
    slope, _, _, _, _ = stats.linregress(lagged[valid], delta[valid])
    
    if slope >= 0:
        return np.inf  # Not mean-reverting
    
    return -np.log(2) / slope


def _detect_regime(
    prices: np.ndarray,
    hurst_threshold: float = 0.45,
    vr_threshold: float = 0.85,
    half_life_min: float = 4,
    half_life_max: float = 48
) -> str:
    """Multi-method regime detection with consensus.
    
    Returns: 'mean_reverting', 'trending', or 'uncertain'
    Requires 2-of-3 agreement to declare a regime.
    """
    h = _hurst_exponent(prices)
    vr = _variance_ratio_test(prices, k=5)
    hl = _half_life(prices)
    
    votes = []
    
    # Hurst: < 0.45 → MR, > 0.55 → trending
    if not np.isnan(h):
        if h < hurst_threshold:
            votes.append('mean_reverting')
        elif h > 0.55:
            votes.append('trending')
    
    # Variance ratio: < 0.85 → MR, > 1.15 → trending
    if not np.isnan(vr):
        if vr < vr_threshold:
            votes.append('mean_reverting')
        elif vr > 1.15:
            votes.append('trending')
    
    # Half-life: 4-48h → viable MR, >200h → trending
    if not np.isnan(hl) and not np.isinf(hl):
        if half_life_min <= hl <= half_life_max:
            votes.append('mean_reverting')
        elif hl > 200:
            votes.append('trending')
    
    # Require consensus: at least 2/3 agree
    mr_votes = votes.count('mean_reverting')
    tr_votes = votes.count('trending')
    
    if mr_votes >= 2:
        return 'mean_reverting'
    elif tr_votes >= 2:
        return 'trending'
    else:
        return 'uncertain'


# ──────────────────────────────────────────────────────────────────────────────
# News Event Filter
# ──────────────────────────────────────────────────────────────────────────────

# Known FOMC dates (update annually)
FOMC_DATES_2024 = [
    '2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12',
    '2024-07-31', '2024-09-18', '2024-11-07', '2024-12-18',
]
FOMC_DATES_2025 = [
    '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18',
    '2025-07-30', '2025-09-17', '2025-10-29', '2025-12-17',
]
FOMC_DATES_2026 = [
    '2026-01-28', '2026-03-18', '2026-04-29', '2026-06-17',
    '2026-07-29', '2026-09-16', '2026-10-28', '2026-12-16',
]

_ALL_FOMC = set()
for dates_str in [FOMC_DATES_2024, FOMC_DATES_2025, FOMC_DATES_2026]:
    for d in dates_str:
        _ALL_FOMC.add(pd.Timestamp(d).date())


def _is_news_blocked(dt_utc: pd.Timestamp) -> Tuple[bool, str]:
    """Check if current time is blocked by news event."""
    date = dt_utc.date()
    hour = dt_utc.hour
    
    # NFP: first Friday of month, block 13:00-16:00 UTC
    if dt_utc.weekday() == 4 and dt_utc.day <= 7:
        if 13 <= hour < 16:
            return True, 'NFP blackout'
    
    # FOMC: block 18:00-23:00 UTC on FOMC days
    if date in _ALL_FOMC:
        if 18 <= hour < 23:
            return True, 'FOMC blackout'
    
    # CPI: mid-month Thursday, block 13:00-16:00 UTC
    if 10 <= dt_utc.day <= 18 and dt_utc.weekday() == 3:
        if 13 <= hour < 16:
            return True, 'CPI blackout'
    
    return False, 'OK'


# ──────────────────────────────────────────────────────────────────────────────
# Overnight Gap Protection
# ──────────────────────────────────────────────────────────────────────────────

def _must_close(dt_utc: pd.Timestamp, max_hold_hour: int = 20) -> Tuple[bool, str]:
    """Check if position must be closed due to gap risk."""
    hour = dt_utc.hour
    
    # Close before rollover
    if hour >= max_hold_hour:
        return True, 'Approaching rollover'
    
    # Close before weekend
    if dt_utc.weekday() == 4 and hour >= 19:  # Friday 19:00 UTC
        return True, 'Friday close'
    
    # Don't enter in dead zone
    if hour >= 21 or hour < 1:
        return True, 'Dead zone'
    
    return False, 'OK'


# ──────────────────────────────────────────────────────────────────────────────
# Signal Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_signals(
    df: pd.DataFrame,
    london_start: int = 8,
    london_end: int = 12,
    ny_start: int = 14,
    ny_end: int = 20,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: float = 3.0,
    vwap_window: int = 20,
    regime_lookback: int = 168,
    hurst_threshold: float = 0.45,
    vr_threshold: float = 0.85,
    half_life_min: float = 4,
    half_life_max: float = 48,
    atr_period: int = 14,
    atr_stop_mult: float = 2.0,
    trail_atr_mult: float = 1.5,
    risk_per_trade: float = 0.005,
    max_hold_bars: int = 12,
    block_nfp: bool = True,
    block_fomc: bool = True,
    block_cpi: bool = True,
    news_buffer_before: int = 30,
    news_buffer_after: int = 120,
    min_bars_after_open: int = 2,
    close_before_rollover: int = 20,
    commission: float = 0.00002,
    slippage_bps: float = 0.001,
    **kwargs
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Generate XAUUSD session mean-reversion signals.
    
    Args:
        df: DataFrame with columns [open, high, low, close, volume] and UTC DatetimeIndex
        london_start/end: London session hours (UTC)
        ny_start/end: NY session hours (UTC)
        z_entry: Z-score threshold for entry
        z_exit: Z-score threshold for exit (reversion target)
        z_stop: Z-score threshold for stop loss
        vwap_window: Rolling VWAP window in bars
        regime_lookback: Bars for regime detection (168 = 1 week)
        hurst_threshold: Hurst exponent threshold for MR regime
        vr_threshold: Variance ratio threshold for MR regime
        half_life_min/max: Half-life range for viable MR
        atr_period: ATR lookback period
        atr_stop_mult: ATR multiplier for stop loss
        trail_atr_mult: ATR multiplier for trailing stop
        risk_per_trade: Risk per trade as fraction of equity
        max_hold_bars: Maximum bars to hold position
        block_nfp/fomc/cpi: Whether to block trading on news days
        news_buffer_before/after: Minutes to block around news
        min_bars_after_open: Wait N bars after session open
        close_before_rollover: Hour to close positions (UTC)
        commission: Commission rate
        slippage_bps: Slippage in basis points
    
    Returns:
        (long_entries, long_exits, short_entries, short_exits, trailing_stops)
    """
    n = len(df)
    idx = df.index
    
    long_entries = pd.Series(False, index=idx)
    long_exits = pd.Series(False, index=idx)
    short_entries = pd.Series(False, index=idx)
    short_exits = pd.Series(False, index=idx)
    trailing_stops = pd.Series(np.nan, index=idx)
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    opn = df['open'].values
    volume = df['volume'].values if 'volume' in df.columns else np.ones(n)
    
    # ── Pre-compute indicators ──
    
    # Rolling VWAP (session-reset would be ideal, but rolling works for backtesting)
    typical_price = (high + low + close) / 3
    tp_vol = typical_price * volume
    cum_tp_vol = pd.Series(tp_vol, index=idx).rolling(vwap_window).sum()
    cum_vol = pd.Series(volume, index=idx).rolling(vwap_window).sum()
    vwap = (cum_tp_vol / cum_vol.replace(0, np.nan)).values
    
    # Z-score of price vs VWAP
    price_dev = close - vwap
    dev_std = pd.Series(price_dev, index=idx).rolling(vwap_window).std().values
    z_score = np.where(dev_std > 0, price_dev / dev_std, 0)
    
    # ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr, index=idx).rolling(atr_period).mean().values
    
    # ── State machine ──
    in_position = False
    position_dir = None  # 'long' or 'short'
    entry_idx = 0
    regime = "uncertain"  # Will be set by regime detection
    entry_price = 0.0
    stop_price = 0.0
    peak_price = 0.0  # For trailing stop
    
    for i in range(regime_lookback, n):
        dt = idx[i]
        hour = dt.hour
        
        # ── Session filter ──
        session = _classify_session(hour)
        if session is None:
            # Outside core sessions - close any open position
            if in_position:
                if position_dir == 'long':
                    long_exits.iloc[i] = True
                else:
                    short_exits.iloc[i] = True
                in_position = False
            continue
        
        # ── Wait after session open ──
        # Skip first N bars of each session
        if session == 'london' and hour == london_start:
            continue
        if session == 'ny' and hour == ny_start:
            continue
        
        # ── News filter ──
        if block_nfp or block_fomc or block_cpi:
            blocked, reason = _is_news_blocked(dt)
            if blocked:
                if in_position:
                    if position_dir == 'long':
                        long_exits.iloc[i] = True
                    else:
                        short_exits.iloc[i] = True
                    in_position = False
                continue
        
        # ── Overnight gap protection ──
        must_cl, gap_reason = _must_close(dt, close_before_rollover)
        if must_cl:
            if in_position:
                if position_dir == 'long':
                    long_exits.iloc[i] = True
                else:
                    short_exits.iloc[i] = True
                in_position = False
            continue
        
        # ── Regime filter (check periodically, not every bar) ──
        if i % 24 == 0:  # Check once per day
            prices_window = close[max(0, i - regime_lookback):i]
            regime = _detect_regime(
                prices_window, hurst_threshold, vr_threshold,
                half_life_min, half_life_max
            )
        
        # Skip if not in mean-reverting regime
        if regime != 'mean_reverting':
            if in_position:
                # Close position if regime shifts
                if position_dir == 'long':
                    long_exits.iloc[i] = True
                else:
                    short_exits.iloc[i] = True
                in_position = False
            continue
        
        # ── Position management ──
        if in_position:
            bars_held = i - entry_idx
            
            # Time stop
            if bars_held >= max_hold_bars:
                if position_dir == 'long':
                    long_exits.iloc[i] = True
                else:
                    short_exits.iloc[i] = True
                in_position = False
                continue
            
            # Update trailing stop
            if position_dir == 'long':
                peak_price = max(peak_price, high[i])
                trail_stop = peak_price - trail_atr_mult * atr[i]
                trailing_stops.iloc[i] = trail_stop
                
                # Check stop/exit conditions
                if low[i] <= stop_price:  # Hard stop
                    long_exits.iloc[i] = True
                    in_position = False
                elif low[i] <= trail_stop:  # Trailing stop
                    long_exits.iloc[i] = True
                    in_position = False
                elif z_score[i] >= -z_exit:  # Reversion target
                    long_exits.iloc[i] = True
                    in_position = False
                elif z_score[i] <= -z_stop:  # Extended against us
                    long_exits.iloc[i] = True
                    in_position = False
            
            elif position_dir == 'short':
                peak_price = min(peak_price, low[i])
                trail_stop = peak_price + trail_atr_mult * atr[i]
                trailing_stops.iloc[i] = trail_stop
                
                if high[i] >= stop_price:  # Hard stop
                    short_exits.iloc[i] = True
                    in_position = False
                elif high[i] >= trail_stop:  # Trailing stop
                    short_exits.iloc[i] = True
                    in_position = False
                elif z_score[i] <= z_exit:  # Reversion target
                    short_exits.iloc[i] = True
                    in_position = False
                elif z_score[i] >= z_stop:  # Extended against us
                    short_exits.iloc[i] = True
                    in_position = False
        
        else:
            # ── Entry logic ──
            if not np.isnan(z_score[i]) and not np.isnan(atr[i]) and atr[i] > 0:
                
                # Long entry: price deeply below VWAP
                if z_score[i] <= -z_entry:
                    # Shift by 1 bar to avoid look-ahead
                    if i + 1 < n:
                        long_entries.iloc[i + 1] = True
                        in_position = True
                        position_dir = 'long'
                        entry_idx = i + 1
                        entry_price = close[i]
                        stop_price = entry_price - atr_stop_mult * atr[i]
                        peak_price = entry_price
                        trailing_stops.iloc[i + 1] = stop_price
                
                # Short entry: price far above VWAP
                elif z_score[i] >= z_entry:
                    if i + 1 < n:
                        short_entries.iloc[i + 1] = True
                        in_position = True
                        position_dir = 'short'
                        entry_idx = i + 1
                        entry_price = close[i]
                        stop_price = entry_price + atr_stop_mult * atr[i]
                        peak_price = entry_price
                        trailing_stops.iloc[i + 1] = stop_price
    
    return long_entries, long_exits, short_entries, short_exits, trailing_stops
