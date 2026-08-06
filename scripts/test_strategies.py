"""Quick validation backtest for ORB and XAUUSD strategies."""
import sys, os
sys.path.insert(0, "/home/admin1/project9/backtest")

import pandas as pd
import numpy as np

# ── Test ORB Strategy on SPY ──
print("=" * 60)
print("ORB STRATEGY TEST — SPY 15-min (2024)")
print("=" * 60)

try:
    from strategies.orb_strategy import generate_signals
    
    spy = pd.read_parquet("/home/admin1/project9/backtest/data/cache/SPY_2024-01-01_2024-12-31.parquet")
    spy.index = pd.to_datetime(spy.index)
    
    params = {
        "orb_period": 1,
        "session_open_hour": 14,
        "session_open_minute": 30,
        "session_close_hour": 21,
        "rel_vol_lookback": 14,
        "min_rel_volume": 1.0,
        "atr_period": 14,
        "atr_stop_pct": 0.10,
        "risk_per_trade": 0.01,
        "min_price": 5.0,
        "min_avg_volume": 1_000_000,
    }
    
    long_ent, long_ext, short_ent, short_ext = generate_signals(spy, **params)
    
    print(f"Long entries: {long_ent.sum()}")
    print(f"Long exits: {long_ext.sum()}")
    print(f"Short entries: {short_ent.sum()}")
    print(f"Short exits: {short_ext.sum()}")
    print(f"Total trades: {long_ent.sum() + short_ent.sum()}")
    
    # Check no look-ahead (first bar should never have signal)
    first_bar_signal = long_ent.iloc[0] or short_ent.iloc[0]
    print(f"First bar signal (should be False): {first_bar_signal}")
    
    # Verify signals are boolean
    assert long_ent.dtype == bool, f"long_ent dtype: {long_ent.dtype}"
    assert long_ext.dtype == bool, f"long_ext dtype: {long_ext.dtype}"
    assert short_ent.dtype == bool, f"short_ent dtype: {short_ent.dtype}"
    assert short_ext.dtype == bool, f"short_ext dtype: {short_ext.dtype}"
    print("Signal types: OK (all boolean)")
    
    # Balance check
    if long_ent.sum() > 0:
        balance = long_ent.cumsum() - long_ext.cumsum()
        print(f"Long position balance range: {balance.min()} to {balance.max()}")
    
    print("\nORB Strategy: PASS")
    
except Exception as e:
    print(f"\nORB Strategy: FAIL — {e}")
    import traceback
    traceback.print_exc()

# ── Test XAUUSD Strategy ──
print("\n" + "=" * 60)
print("XAUUSD SESSION MR TEST — 1H (2019-2025)")
print("=" * 60)

try:
    from strategies.xauusd_session_mr import generate_signals
    
    xau = pd.read_parquet("/home/admin1/project9/trading-system/data/XAUUSD_1h_raw.parquet")
    xau["timestamp"] = pd.to_datetime(xau["timestamp"])
    xau = xau.set_index("timestamp")
    
    # Use last 2 years for quick test
    xau_test = xau[xau.index >= "2024-01-01"].copy()
    
    params = {
        "london_start": 8, "london_end": 12,
        "ny_start": 14, "ny_end": 20,
        "z_entry": 2.0, "z_exit": 0.5, "z_stop": 3.0,
        "vwap_window": 20,
        "regime_lookback": 168,
        "atr_period": 14,
        "atr_stop_mult": 2.0,
        "trail_atr_mult": 1.5,
        "max_hold_bars": 12,
        "block_nfp": True, "block_fomc": True, "block_cpi": True,
    }
    
    result = generate_signals(xau_test, **params)
    
    if len(result) == 5:
        long_ent, long_ext, short_ent, short_ext, trailing = result
        print(f"Trailing stops provided: Yes")
    else:
        long_ent, long_ext, short_ent, short_ext = result
        print(f"Trailing stops provided: No")
    
    print(f"Long entries: {long_ent.sum()}")
    print(f"Long exits: {long_ext.sum()}")
    print(f"Short entries: {short_ent.sum()}")
    print(f"Short exits: {short_ext.sum()}")
    print(f"Total trades: {long_ent.sum() + short_ent.sum()}")
    
    # Check no look-ahead
    first_bar_signal = long_ent.iloc[0] or short_ent.iloc[0]
    print(f"First bar signal (should be False): {first_bar_signal}")
    
    # Verify signal types
    assert long_ent.dtype == bool
    assert long_ext.dtype == bool
    assert short_ent.dtype == bool
    assert short_ext.dtype == bool
    print("Signal types: OK (all boolean)")
    
    print("\nXAUUSD Session MR: PASS")
    
except Exception as e:
    print(f"\nXAUUSD Session MR: FAIL — {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)
