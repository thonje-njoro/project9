#!/usr/bin/env python3
"""Quick validation of Momentum ORB strategy on WSL."""
import sys
sys.path.insert(0, "/home/admin1/project9/backtest")
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from strategies.momentum_orb import generate_signals, prepare_equity, SYMBOL_PRESETS

print("=" * 70)
print("MOMENTUM ORB VALIDATION — All 4 Symbols")
print("=" * 70)

results = {}

for symbol in ["NVDA", "AMD", "PLTR", "MRVL"]:
    print(f"\n--- {symbol} ---")
    
    # Try to load data
    data_file = f"/home/admin1/project9/backtest/data/cache/{symbol}_5min_2022_2024.parquet"
    try:
        df = pd.read_parquet(data_file)
    except FileNotFoundError:
        # Try alternative paths
        alt_paths = [
            f"/home/admin1/project9/trading-system/data/{symbol}_5min.parquet",
            f"/home/admin1/backtest/{symbol}_5min.parquet",
        ]
        df = None
        for path in alt_paths:
            try:
                df = pd.read_parquet(path)
                print(f"  Found data at: {path}")
                break
            except:
                continue
        
        if df is None:
            print(f"  No data found for {symbol}, skipping")
            continue
    
    print(f"  Data: {len(df)} bars, {df.index[0]} to {df.index[-1]}")
    
    # Prepare
    df = prepare_equity(df)
    
    # Get preset params
    params = {k: v for k, v in SYMBOL_PRESETS[symbol].items() if k != "test_stats"}
    
    # Generate signals
    le, lx, se, sx = generate_signals(df, **params)
    
    # Count trades
    long_entries = le.sum()
    short_entries = se.sum()
    total_trades = long_entries + short_entries
    
    print(f"  Trades: {total_trades} (long={long_entries}, short={short_entries})")
    
    if total_trades > 0:
        # Walk-forward split
        train_mask = df.index < "2023-07-01"
        test_mask = df.index >= "2023-07-01"
        
        # Train stats
        le_tr, lx_tr, se_tr, sx_tr = generate_signals(df[train_mask], **params)
        le_te, lx_te, se_te, sx_te = generate_signals(df[test_mask], **params)
        
        train_trades = le_tr.sum() + se_tr.sum()
        test_trades = le_te.sum() + se_te.sum()
        
        print(f"  Train: {train_trades} trades")
        print(f"  Test: {test_trades} trades")
        
        # Check against expected
        expected = SYMBOL_PRESETS[symbol]["test_stats"]
        print(f"  Expected test trades: {expected['trades']}")
        print(f"  Expected test WR: {expected['win_rate']}%")
        
        results[symbol] = {
            "total_trades": total_trades,
            "train_trades": train_trades,
            "test_trades": test_trades,
            "expected_test_trades": expected["trades"],
        }

print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)
for sym, res in results.items():
    match = "✓" if abs(res["test_trades"] - res["expected_test_trades"]) < res["expected_test_trades"] * 0.3 else "?"
    print(f"{sym}: {res['total_trades']} total trades ({match})")

print("\nValidation complete.")
