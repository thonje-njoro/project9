"""Quick XAUUSD parameter optimization (reduced search space)."""
import sys
sys.path.insert(0, "/home/admin1/project9/backtest")
import pandas as pd
import numpy as np

xau = pd.read_parquet("/home/admin1/project9/trading-system/data/XAUUSD_1h_raw.parquet")
xau["timestamp"] = pd.to_datetime(xau["timestamp"])
xau = xau.set_index("timestamp")

from strategies.xauusd_session_mr import generate_signals as xau_signals

best_sharpe = -999
best_params = {}
results = []

# Reduced search: 3*3*3*3 = 81 combinations
for z_entry in [1.5, 2.0, 2.5]:
    for z_exit in [0.0, 0.3, 0.5]:
        for atr_mult in [1.5, 2.0, 2.5]:
            for max_hold in [8, 12, 18]:
                try:
                    params = {
                        "london_start": 8, "london_end": 12,
                        "ny_start": 14, "ny_end": 20,
                        "z_entry": z_entry, "z_exit": z_exit, "z_stop": 3.0,
                        "vwap_window": 20, "regime_lookback": 168,
                        "atr_period": 14, "atr_stop_mult": atr_mult,
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
                        wr = len(t[t>0])/len(t)*100
                        pf = abs(t[t>0].sum()/t[t<=0].sum()) if t[t<=0].sum() != 0 else 999
                        
                        results.append({
                            "z_entry": z_entry, "z_exit": z_exit,
                            "atr_mult": atr_mult, "max_hold": max_hold,
                            "trades": len(t), "wr": round(wr, 1),
                            "return": round(t.sum(), 2),
                            "sharpe": round(sharpe, 3),
                            "pf": round(pf, 2),
                        })
                        
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params = {
                                "z_entry": z_entry, "z_exit": z_exit,
                                "atr_mult": atr_mult, "max_hold": max_hold,
                                "trades": len(t), "wr": wr,
                                "return": t.sum(), "sharpe": sharpe, "pf": pf,
                            }
                except Exception as e:
                    pass

print("=" * 70)
print("XAUUSD PARAMETER OPTIMIZATION RESULTS")
print("=" * 70)

if best_params:
    print(f"\nBest params (by Sharpe):")
    print(f"  z_entry={best_params['z_entry']}, z_exit={best_params['z_exit']}")
    print(f"  atr_mult={best_params['atr_mult']}, max_hold={best_params['max_hold']}")
    print(f"  Trades={best_params['trades']}, WR={best_params['wr']:.1f}%")
    print(f"  Total return={best_params['return']:.2f}%")
    print(f"  Sharpe={best_params['sharpe']:.3f}")
    print(f"  Profit factor={best_params['pf']:.2f}")

if results:
    df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    print("\nTop 10 by Sharpe:")
    print(df.head(10).to_string(index=False))
    
    # Also show best by return
    print("\nTop 5 by Total Return:")
    print(df.sort_values("return", ascending=False).head(5).to_string(index=False))
    
    # Show best by profit factor
    print("\nTop 5 by Profit Factor:")
    print(df.sort_values("pf", ascending=False).head(5).to_string(index=False))

# Now test best params on walk-forward splits
print("\n" + "=" * 70)
print("WALK-FORWARD VALIDATION (Best Params)")
print("=" * 70)

if best_params:
    # Split: 2019-2022 train, 2023-2025 test
    train = xau[xau.index < "2023-01-01"]
    test = xau[xau.index >= "2023-01-01"]
    
    for label, data in [("Train (2019-2022)", train), ("Test (2023-2025)", test)]:
        params = {
            "london_start": 8, "london_end": 12,
            "ny_start": 14, "ny_end": 20,
            "z_entry": best_params["z_entry"], "z_exit": best_params["z_exit"],
            "z_stop": 3.0, "vwap_window": 20, "regime_lookback": 168,
            "atr_period": 14, "atr_stop_mult": best_params["atr_mult"],
            "trail_atr_mult": 1.5, "max_hold_bars": best_params["max_hold"],
            "block_nfp": True, "block_fomc": True, "block_cpi": True,
        }
        
        result = xau_signals(data, **params)
        le, lx, se, sx, ts = result
        
        trades = []
        in_trade = False
        for i in range(len(data)):
            if le.iloc[i] and not in_trade:
                in_trade = True
                entry_price = data['close'].iloc[i]
                direction = 1
            elif se.iloc[i] and not in_trade:
                in_trade = True
                entry_price = data['close'].iloc[i]
                direction = -1
            elif (lx.iloc[i] or sx.iloc[i]) and in_trade:
                exit_price = data['close'].iloc[i]
                pnl = (exit_price - entry_price) / entry_price * 100 * direction
                trades.append(pnl)
                in_trade = False
        
        if trades:
            t = np.array(trades)
            sharpe = t.mean() / t.std() if t.std() > 0 else 0
            wr = len(t[t>0])/len(t)*100
            pf = abs(t[t>0].sum()/t[t<=0].sum()) if t[t<=0].sum() != 0 else 0
            print(f"\n{label}:")
            print(f"  Trades={len(t)}, WR={wr:.1f}%, Return={t.sum():.2f}%")
            print(f"  Sharpe={sharpe:.3f}, PF={pf:.2f}")
