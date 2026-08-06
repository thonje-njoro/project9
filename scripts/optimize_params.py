"""Debug QQQ and optimize parameters."""
import sys
sys.path.insert(0, "/home/admin1/project9/backtest")
import pandas as pd
import numpy as np
from strategies.orb_strategy import generate_signals, _compute_relative_volume, _compute_daily_atr

# Check QQQ data
qqq = pd.read_parquet("/home/admin1/project9/backtest/data/cache/QQQ_2024-01-01_2024-12-31.parquet")
qqq.index = pd.to_datetime(qqq.index)

print("=== QQQ Data Debug ===")
print(f"Shape: {qqq.shape}")
print(f"Date range: {qqq.index[0]} to {qqq.index[-1]}")
print(f"Unique dates: {len(set(qqq.index.date))}")
print(f"Hours: {sorted(qqq.index.hour.unique())}")
print(f"Minutes: {sorted(qqq.index.minute.unique())}")

# Check first bar of each day
first_bars = qqq.groupby(qqq.index.date).first()
print(f"\nFirst bar hour/minute distribution:")
first_times = qqq.groupby(qqq.index.date).apply(lambda x: f"{x.index[0].hour}:{x.index[0].minute:02d}")
print(first_times.value_counts().head(10))

# Check volume
print(f"\nVolume stats:")
print(f"  Mean: {qqq['volume'].mean():.0f}")
print(f"  Median: {qqq['volume'].median():.0f}")
print(f"  Min: {qqq['volume'].min():.0f}")
print(f"  Zero volume bars: {(qqq['volume'] == 0).sum()}")

# Check relative volume
rv = _compute_relative_volume(qqq, 14, 30, 14)
print(f"\nRelative volume stats:")
print(f"  Count: {rv.count()}")
print(f"  Mean: {rv.mean():.2f}")
print(f"  Median: {rv.median():.2f}")
print(f"  > 1.0: {(rv > 1.0).sum()}")
print(f"  > 0.5: {(rv > 0.5).sum()}")

# Check ATR
atr = _compute_daily_atr(qqq, 14)
print(f"\nATR stats:")
print(f"  Count: {atr.count()}")
print(f"  Mean: {atr.mean():.2f}")

# Try ORB with relaxed params
print("\n=== ORB with Relaxed Params ===")
params = {
    "orb_period": 1, "session_open_hour": 14, "session_open_minute": 30,
    "session_close_hour": 21, "rel_vol_lookback": 14, "min_rel_volume": 0.5,  # Relaxed
    "atr_period": 14, "atr_stop_pct": 0.10, "risk_per_trade": 0.01,
    "min_price": 5.0, "min_avg_volume": 500_000,  # Relaxed
}

le, lx, se, sx = generate_signals(qqq, **params)
print(f"Long entries: {le.sum()}")
print(f"Short entries: {se.sum()}")
print(f"Total: {le.sum() + se.sum()}")

# Also check SPY with same relaxed params
spy = pd.read_parquet("/home/admin1/project9/backtest/data/cache/SPY_2024-01-01_2024-12-31.parquet")
spy.index = pd.to_datetime(spy.index)

le2, lx2, se2, sx2 = generate_signals(spy, **params)
print(f"\nSPY relaxed: Long={le2.sum()}, Short={se2.sum()}, Total={le2.sum()+se2.sum()}")

# Parameter sweep for XAUUSD
print("\n=== XAUUSD Parameter Sweep ===")
xau = pd.read_parquet("/home/admin1/project9/trading-system/data/XAUUSD_1h_raw.parquet")
xau["timestamp"] = pd.to_datetime(xau["timestamp"])
xau = xau.set_index("timestamp")

from strategies.xauusd_session_mr import generate_signals as xau_signals

best_sharpe = -999
best_params = {}
results = []

for z_entry in [1.5, 2.0, 2.5, 3.0]:
    for z_exit in [0.0, 0.3, 0.5, 0.8]:
        for atr_mult in [1.5, 2.0, 2.5, 3.0]:
            for max_hold in [6, 12, 18, 24]:
                try:
                    params = {
                        "london_start": 8, "london_end": 12, "ny_start": 14, "ny_end": 20,
                        "z_entry": z_entry, "z_exit": z_exit, "z_stop": 3.0, "vwap_window": 20,
                        "regime_lookback": 168, "atr_period": 14, "atr_stop_mult": atr_mult,
                        "trail_atr_mult": 1.5, "max_hold_bars": max_hold,
                        "block_nfp": True, "block_fomc": True, "block_cpi": True,
                    }
                    
                    result = xau_signals(xau, **params)
                    le, lx, se, sx, ts = result
                    
                    trades = []
                    in_trade = False
                    for i in range(len(xau)):
                        if le.iloc[i] and not in_trade:
                            in_trade = True
                            entry_price = xau['close'].iloc[i]
                            direction = 1
                        elif se.iloc[i] and not in_trade:
                            in_trade = True
                            entry_price = xau['close'].iloc[i]
                            direction = -1
                        elif (lx.iloc[i] or sx.iloc[i]) and in_trade:
                            exit_price = xau['close'].iloc[i]
                            pnl = (exit_price - entry_price) / entry_price * 100 * direction
                            trades.append(pnl)
                            in_trade = False
                    
                    if trades and len(trades) > 20:
                        t = np.array(trades)
                        sharpe = t.mean() / t.std() if t.std() > 0 else 0
                        
                        results.append({
                            "z_entry": z_entry, "z_exit": z_exit,
                            "atr_mult": atr_mult, "max_hold": max_hold,
                            "trades": len(t), "win_rate": len(t[t>0])/len(t)*100,
                            "total_return": t.sum(), "sharpe": sharpe,
                            "profit_factor": abs(t[t>0].sum()/t[t<=0].sum()) if t[t<=0].sum() != 0 else 999,
                        })
                        
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params = params.copy()
                            best_params["_trades"] = len(t)
                            best_params["_return"] = t.sum()
                            best_params["_sharpe"] = sharpe
                            best_params["_wr"] = len(t[t>0])/len(t)*100
                            best_params["_pf"] = abs(t[t>0].sum()/t[t<=0].sum()) if t[t<=0].sum() != 0 else 999
                except Exception as e:
                    pass

print(f"\nBest XAUUSD params (by Sharpe):")
print(f"  z_entry={best_params.get('z_entry')}, z_exit={best_params.get('z_exit')}")
print(f"  atr_mult={best_params.get('atr_stop_mult')}, max_hold={best_params.get('max_hold_bars')}")
print(f"  Trades={best_params.get('_trades')}, WR={best_params.get('_wr', 0):.1f}%")
print(f"  Total return={best_params.get('_return', 0):.2f}%")
print(f"  Sharpe={best_params.get('_sharpe', 0):.3f}")
print(f"  Profit factor={best_params.get('_pf', 0):.2f}")

# Top 5 results
if results:
    results_df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    print("\nTop 5 by Sharpe:")
    print(results_df.head(5).to_string(index=False))
