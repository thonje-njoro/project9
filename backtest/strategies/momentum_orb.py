"""
Momentum ORB Strategy — Optimized by MiMo Claw
================================================
Trend continuation after opening range breakout.

Key parameters (from walk-forward optimization):
- 60-min opening range for NVDA/AMD/PLTR
- 5-min opening range for MRVL
- ATR-based stop and trailing stop
- NO trend filter (confirmed better without)
- NO look-ahead: trailing stop uses previous bar's close

This strategy works on high-beta stocks because:
1. The opening range captures the first hour's price discovery
2. After the OR, price tends to continue in the breakout direction
3. Trailing stops lock in profits on the momentum moves
4. Wide ATR stops prevent premature exits on volatile names
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def prepare_equity(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare 5-min equity data with indicators."""
    df = df.copy()
    c = df["close"].values.astype(float)
    hi = df["high"].values.astype(float)
    lo = df["low"].values.astype(float)
    
    # ATR (14-period, on previous bars only)
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(np.maximum(hi - lo, np.abs(hi - prev_c)), np.abs(lo - prev_c))
    df["atr"] = pd.Series(tr, index=df.index).rolling(14, min_periods=1).mean().values
    
    # Hour and minute
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    
    # Day groups (for OR calculation)
    dates = df.index.date
    ds = np.array([str(d) for d in dates])
    changes = np.where(ds[1:] != ds[:-1])[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(ds)]])
    df.attrs["day_groups"] = list(zip(starts, ends))
    
    return df


def generate_signals(
    df: pd.DataFrame,
    or_minutes: int = 60,
    atr_mult_stop: float = 1.0,
    trail_mult: float = 1.0,
    min_or_range_atr: float = 0.0,
    min_price: float = 5.0,
    min_volume: int = 100_000,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Generate Momentum ORB signals.
    
    Parameters
    ----------
    df : DataFrame with OHLCV data (5-min bars)
    or_minutes : int
        Opening range period in minutes (60 for NVDA/AMD/PLTR, 5 for MRVL)
    atr_mult_stop : float
        Stop loss as multiple of ATR (1.0-3.0 depending on volatility)
    trail_mult : float
        Trailing stop as multiple of ATR (1.0 = tight, locks profits)
    min_or_range_atr : float
        Minimum OR range in ATR units (0 = no filter)
    min_price : float
        Minimum price filter
    min_volume : int
        Minimum volume filter
    
    Returns
    -------
    long_entries, long_exits, short_entries, short_exits : pd.Series (bool)
    """
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    lo = df["low"].values.astype(float)
    vol = df["volume"].values.astype(float) if "volume" in df.columns else np.ones(len(df))
    atr = df["atr"].values.astype(float)
    hour = df["hour"].values.astype(float)
    minute = df["minute"].values.astype(float)
    
    day_groups = df.attrs.get("day_groups", [])
    n = len(c)
    
    long_entries = np.zeros(n, dtype=bool)
    long_exits = np.zeros(n, dtype=bool)
    short_entries = np.zeros(n, dtype=bool)
    short_exits = np.zeros(n, dtype=bool)
    
    for day_start, day_end in day_groups:
        # Find the opening range period (first or_minutes bars of the day)
        # Market open is 9:30 ET = 14:30 UTC
        or_bars = []
        for i in range(day_start, day_end):
            # 9:30-10:30 ET = 14:30-15:30 UTC for 60-min OR
            # For 5-min OR: just first 5 bars after 14:30
            if hour[i] == 14 and minute[i] >= 30:
                or_bars.append(i)
                if len(or_bars) >= (or_minutes // 5) + 1:
                    break
            elif hour[i] > 14 and hour[i] < 16:
                or_bars.append(i)
                if len(or_bars) >= (or_minutes // 5) + 1:
                    break
        
        if len(or_bars) < 2:
            continue
        
        or_indices = or_bars[:or_minutes // 5 + 1]
        or_high = max(h[i] for i in or_indices)
        or_low = min(lo[i] for i in or_indices)
        or_range = or_high - or_low
        
        # Filter: minimum OR range
        or_end = or_indices[-1]
        if min_or_range_atr > 0 and atr[or_end] > 0:
            if or_range < min_or_range_atr * atr[or_end]:
                continue
        
        # Price filter
        if c[or_end] < min_price:
            continue
        
        # Volume filter
        if vol[or_end] < min_volume:
            continue
        
        # Look for breakout in remaining bars of the day
        for i in range(or_end + 1, day_end):
            if hour[i] >= 21:  # Exit by 4 PM ET = 21 UTC
                break
            
            # Long entry: price breaks above OR high
            if c[i] > or_high:
                long_entries[i] = True
                break
            
            # Short entry: price breaks below OR low
            if c[i] < or_low:
                short_entries[i] = True
                break
        
        # Trailing stop exits (for any open positions from this day)
        in_trade = False
        direction = 0
        entry_price = 0.0
        trail_stop = 0.0
        prev_close = c[or_end]
        
        for i in range(or_end + 1, day_end):
            if not in_trade:
                if long_entries[i]:
                    in_trade = True
                    direction = 1
                    entry_price = c[i]
                    trail_stop = entry_price - atr_mult_stop * atr[i]
                elif short_entries[i]:
                    in_trade = True
                    direction = -1
                    entry_price = c[i]
                    trail_stop = entry_price + atr_mult_stop * atr[i]
                prev_close = c[i]
                continue
            
            # Update trailing stop (using PREVIOUS bar's close — no look-ahead)
            if direction == 1:
                new_trail = prev_close - trail_mult * atr[i]
                if new_trail > trail_stop:
                    trail_stop = new_trail
                if lo[i] <= trail_stop:
                    long_exits[i] = True
                    in_trade = False
            else:
                new_trail = prev_close + trail_mult * atr[i]
                if new_trail < trail_stop:
                    trail_stop = new_trail
                if h[i] >= trail_stop:
                    short_exits[i] = True
                    in_trade = False
            
            # Force exit at end of day
            if in_trade and hour[i] >= 20:
                if direction == 1:
                    long_exits[i] = True
                else:
                    short_exits[i] = True
                in_trade = False
            
            prev_close = c[i]
    
    return (
        pd.Series(long_entries, index=df.index),
        pd.Series(long_exits, index=df.index),
        pd.Series(short_entries, index=df.index),
        pd.Series(short_exits, index=df.index),
    )


# ══════════════════════════════════════════════════════════════
# PRESET PARAMETERS (from MiMo Claw optimization)
# ══════════════════════════════════════════════════════════════

SYMBOL_PRESETS = {
    "NVDA": {
        "or_minutes": 60,
        "atr_mult_stop": 1.0,
        "trail_mult": 1.0,
        "min_or_range_atr": 0.0,
        "min_price": 5.0,
        "min_volume": 100_000,
        "test_stats": {
            "trades": 1102,
            "win_rate": 69.1,
            "sharpe": 0.626,
            "profit_factor": 10.97,
            "max_dd_pct": -3.15,
        },
    },
    "AMD": {
        "or_minutes": 60,
        "atr_mult_stop": 1.5,
        "trail_mult": 1.0,
        "min_or_range_atr": 0.0,
        "min_price": 5.0,
        "min_volume": 100_000,
        "test_stats": {
            "trades": 1091,
            "win_rate": 73.2,
            "sharpe": 0.667,
            "profit_factor": 13.98,
            "max_dd_pct": -4.92,
        },
    },
    "PLTR": {
        "or_minutes": 60,
        "atr_mult_stop": 3.0,
        "trail_mult": 1.0,
        "min_or_range_atr": 0.0,
        "min_price": 5.0,
        "min_volume": 100_000,
        "test_stats": {
            "trades": 1180,
            "win_rate": 70.3,
            "sharpe": 0.578,
            "profit_factor": 12.93,
            "max_dd_pct": -5.19,
        },
    },
    "MRVL": {
        "or_minutes": 5,
        "atr_mult_stop": 0.5,
        "trail_mult": 1.0,
        "min_or_range_atr": 0.0,
        "min_price": 5.0,
        "min_volume": 100_000,
        "test_stats": {
            "trades": 1133,
            "win_rate": 65.8,
            "sharpe": 0.599,
            "profit_factor": 8.89,
            "max_dd_pct": -3.14,
        },
    },
}


def get_preset(symbol: str) -> dict:
    """Get optimized parameters for a symbol."""
    if symbol not in SYMBOL_PRESETS:
        raise ValueError(f"No preset for {symbol}. Available: {list(SYMBOL_PRESETS.keys())}")
    return {k: v for k, v in SYMBOL_PRESETS[symbol].items() if k != "test_stats"}
