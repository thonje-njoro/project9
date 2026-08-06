"""Full backtest for ORB + XAUUSD strategies with performance metrics."""
import sys
sys.path.insert(0, "/home/admin1/project9/backtest")

import pandas as pd
import numpy as np

from strategies.orb_strategy import generate_signals as orb_signals
from strategies.xauusd_session_mr import generate_signals as xau_signals

def compute_metrics(entries, exits, prices, label=""):
    """Compute basic performance metrics from signal arrays."""
    trades = []
    in_trade = False
    entry_price = 0
    direction = 0
    
    for i in range(len(entries)):
        if entries.iloc[i] and not in_trade:
            in_trade = True
            entry_price = prices.iloc[i]
            direction = 1 if entries.name.startswith("long") else -1
        elif exits.iloc[i] and in_trade:
            exit_price = prices.iloc[i]
            pnl = (exit_price - entry_price) / entry_price * 100 * direction
            trades.append(pnl)
            in_trade = False
    
    if not trades:
        return {"trades": 0, "label": label}
    
    trades = np.array(trades)
    wins = trades[trades > 0]
    losses = trades[trades <= 0]
    
    return {
        "label": label,
        "trades": len(trades),
        "win_rate": f"{len(wins)/len(trades)*100:.1f}%",
        "avg_return": f"{trades.mean():.3f}%",
        "total_return": f"{trades.sum():.2f}%",
        "avg_win": f"{wins.mean():.3f}%" if len(wins) > 0 else "N/A",
        "avg_loss": f"{losses.mean():.3f}%" if len(losses) > 0 else "N/A",
        "profit_factor": f"{abs(wins.sum()/losses.sum()):.2f}" if len(losses) > 0 and losses.sum() != 0 else "N/A",
        "max_loss": f"{trades.min():.3f}%",
        "max_win": f"{trades.max():.3f}%",
        "sharpe_approx": f"{trades.mean()/trades.std():.2f}" if trades.std() > 0 else "N/A",
    }

# ══════════════════════════════════════════════════════════════
# TEST 1: ORB on SPY 15-min (2024)
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("ORB STRATEGY — SPY 15-min (2024)")
print("=" * 70)

spy = pd.read_parquet("/home/admin1/project9/backtest/data/cache/SPY_2024-01-01_2024-12-31.parquet")
spy.index = pd.to_datetime(spy.index)

orb_params = {
    "orb_period": 1, "session_open_hour": 14, "session_open_minute": 30,
    "session_close_hour": 21, "rel_vol_lookback": 14, "min_rel_volume": 1.0,
    "atr_period": 14, "atr_stop_pct": 0.10, "risk_per_trade": 0.01,
    "min_price": 5.0, "min_avg_volume": 1_000_000,
}

le, lx, se, sx = orb_signals(spy, **orb_params)

# Compute combined metrics
all_entries = le | se
all_exits = lx | sx

# Direction-aware P&L
trades = []
in_trade = False
entry_price = 0
direction = 0

for i in range(len(spy)):
    if le.iloc[i] and not in_trade:
        in_trade = True
        entry_price = spy['close'].iloc[i]
        direction = 1
    elif se.iloc[i] and not in_trade:
        in_trade = True
        entry_price = spy['close'].iloc[i]
        direction = -1
    elif (lx.iloc[i] or sx.iloc[i]) and in_trade:
        exit_price = spy['close'].iloc[i]
        pnl = (exit_price - entry_price) / entry_price * 100 * direction
        trades.append(pnl)
        in_trade = False

if trades:
    trades_arr = np.array(trades)
    wins = trades_arr[trades_arr > 0]
    losses = trades_arr[trades_arr <= 0]
    
    print(f"Total trades: {len(trades)}")
    print(f"Long trades: {le.sum()}, Short trades: {se.sum()}")
    print(f"Win rate: {len(wins)/len(trades)*100:.1f}%")
    print(f"Avg return/trade: {trades_arr.mean():.3f}%")
    print(f"Total return: {trades_arr.sum():.2f}%")
    print(f"Avg win: {wins.mean():.3f}%" if len(wins) > 0 else "No wins")
    print(f"Avg loss: {losses.mean():.3f}%" if len(losses) > 0 else "No losses")
    if len(losses) > 0 and losses.sum() != 0:
        print(f"Profit factor: {abs(wins.sum()/losses.sum()):.2f}")
    print(f"Max drawdown (single trade): {trades_arr.min():.3f}%")
    print(f"Best trade: {trades_arr.max():.3f}%")
    if trades_arr.std() > 0:
        print(f"Sharpe (per-trade): {trades_arr.mean()/trades_arr.std():.2f}")
    
    # Monthly breakdown
    print("\n--- Monthly Breakdown ---")
    trade_dates = spy.index[le | se][:len(trades)]
    for i, (t, p) in enumerate(zip(trades, trade_dates)):
        month = p.strftime("%Y-%m")
    
    # Count by month
    monthly = {}
    for t_val, p in zip(trades, spy.index[le | se][:len(trades)]):
        m = p.strftime("%Y-%m")
        if m not in monthly:
            monthly[m] = []
        monthly[m].append(t_val)
    
    for m in sorted(monthly.keys()):
        mt = np.array(monthly[m])
        print(f"  {m}: {len(mt)} trades, {mt.sum():.2f}% return, WR={len(mt[mt>0])/len(mt)*100:.0f}%")
else:
    print("No trades!")

# ══════════════════════════════════════════════════════════════
# TEST 2: ORB on QQQ 15-min (2024)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ORB STRATEGY — QQQ 15-min (2024)")
print("=" * 70)

qqq = pd.read_parquet("/home/admin1/project9/backtest/data/cache/QQQ_2024-01-01_2024-12-31.parquet")
qqq.index = pd.to_datetime(qqq.index)

le_q, lx_q, se_q, sx_q = orb_signals(qqq, **orb_params)

trades_q = []
in_trade = False
for i in range(len(qqq)):
    if le_q.iloc[i] and not in_trade:
        in_trade = True
        entry_price = qqq['close'].iloc[i]
        direction = 1
    elif se_q.iloc[i] and not in_trade:
        in_trade = True
        entry_price = qqq['close'].iloc[i]
        direction = -1
    elif (lx_q.iloc[i] or sx_q.iloc[i]) and in_trade:
        exit_price = qqq['close'].iloc[i]
        pnl = (exit_price - entry_price) / entry_price * 100 * direction
        trades_q.append(pnl)
        in_trade = False

if trades_q:
    t = np.array(trades_q)
    w = t[t > 0]
    l = t[t <= 0]
    print(f"Trades: {len(t)}, Win rate: {len(w)/len(t)*100:.1f}%, Total: {t.sum():.2f}%")
    print(f"Sharpe: {t.mean()/t.std():.2f}" if t.std() > 0 else "Sharpe: N/A")

# ══════════════════════════════════════════════════════════════
# TEST 3: XAUUSD Session MR (2024)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("XAUUSD SESSION MR — 1H (2024-2025)")
print("=" * 70)

xau = pd.read_parquet("/home/admin1/project9/trading-system/data/XAUUSD_1h_raw.parquet")
xau["timestamp"] = pd.to_datetime(xau["timestamp"])
xau = xau.set_index("timestamp")
xau_2024 = xau[xau.index >= "2024-01-01"].copy()

xau_params = {
    "london_start": 8, "london_end": 12, "ny_start": 14, "ny_end": 20,
    "z_entry": 2.0, "z_exit": 0.5, "z_stop": 3.0, "vwap_window": 20,
    "regime_lookback": 168, "atr_period": 14, "atr_stop_mult": 2.0,
    "trail_atr_mult": 1.5, "max_hold_bars": 12,
    "block_nfp": True, "block_fomc": True, "block_cpi": True,
}

result = xau_signals(xau_2024, **xau_params)
le_x, lx_x, se_x, sx_x, ts_x = result

trades_x = []
in_trade = False
for i in range(len(xau_2024)):
    if le_x.iloc[i] and not in_trade:
        in_trade = True
        entry_price = xau_2024['close'].iloc[i]
        direction = 1
    elif se_x.iloc[i] and not in_trade:
        in_trade = True
        entry_price = xau_2024['close'].iloc[i]
        direction = -1
    elif (lx_x.iloc[i] or sx_x.iloc[i]) and in_trade:
        exit_price = xau_2024['close'].iloc[i]
        pnl = (exit_price - entry_price) / entry_price * 100 * direction
        trades_x.append(pnl)
        in_trade = False

if trades_x:
    t = np.array(trades_x)
    w = t[t > 0]
    l = t[t <= 0]
    print(f"Trades: {len(t)}, Win rate: {len(w)/len(t)*100:.1f}%")
    print(f"Total return: {t.sum():.2f}%")
    print(f"Avg return/trade: {t.mean():.3f}%")
    if len(w) > 0: print(f"Avg win: {w.mean():.3f}%")
    if len(l) > 0: print(f"Avg loss: {l.mean():.3f}%")
    if len(l) > 0 and l.sum() != 0: print(f"Profit factor: {abs(w.sum()/l.sum()):.2f}")
    print(f"Max loss: {t.min():.3f}%")
    print(f"Best trade: {t.max():.3f}%")
    if t.std() > 0: print(f"Sharpe (per-trade): {t.mean()/t.std():.2f}")
    
    # Monthly
    print("\n--- Monthly Breakdown ---")
    monthly = {}
    for t_val, p in zip(trades_x, xau_2024.index[le_x | se_x][:len(trades_x)]):
        m = p.strftime("%Y-%m")
        if m not in monthly: monthly[m] = []
        monthly[m].append(t_val)
    for m in sorted(monthly.keys()):
        mt = np.array(monthly[m])
        print(f"  {m}: {len(mt)} trades, {mt.sum():.2f}% return, WR={len(mt[mt>0])/len(mt)*100:.0f}%")
else:
    print("No trades!")

# ══════════════════════════════════════════════════════════════
# TEST 4: XAUUSD on longer history (2019-2025)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("XAUUSD SESSION MR — 1H FULL (2019-2025)")
print("=" * 70)

result_full = xau_signals(xau, **xau_params)
le_f, lx_f, se_f, sx_f, ts_f = result_full

trades_f = []
in_trade = False
for i in range(len(xau)):
    if le_f.iloc[i] and not in_trade:
        in_trade = True
        entry_price = xau['close'].iloc[i]
        direction = 1
    elif se_f.iloc[i] and not in_trade:
        in_trade = True
        entry_price = xau['close'].iloc[i]
        direction = -1
    elif (lx_f.iloc[i] or sx_f.iloc[i]) and in_trade:
        exit_price = xau['close'].iloc[i]
        pnl = (exit_price - entry_price) / entry_price * 100 * direction
        trades_f.append(pnl)
        in_trade = False

if trades_f:
    t = np.array(trades_f)
    w = t[t > 0]
    l = t[t <= 0]
    print(f"Trades: {len(t)}, Win rate: {len(w)/len(t)*100:.1f}%")
    print(f"Total return: {t.sum():.2f}%")
    print(f"Avg return/trade: {t.mean():.3f}%")
    if len(w) > 0: print(f"Avg win: {w.mean():.3f}%")
    if len(l) > 0: print(f"Avg loss: {l.mean():.3f}%")
    if len(l) > 0 and l.sum() != 0: print(f"Profit factor: {abs(w.sum()/l.sum()):.2f}")
    if t.std() > 0: print(f"Sharpe (per-trade): {t.mean()/t.std():.2f}")
    
    # Yearly breakdown
    print("\n--- Yearly Breakdown ---")
    yearly = {}
    for t_val, p in zip(trades_f, xau.index[le_f | se_f][:len(trades_f)]):
        y = p.strftime("%Y")
        if y not in yearly: yearly[y] = []
        yearly[y].append(t_val)
    for y in sorted(yearly.keys()):
        yt = np.array(yearly[y])
        print(f"  {y}: {len(yt)} trades, {yt.sum():.2f}% return, WR={len(yt[yt>0])/len(yt)*100:.0f}%")
else:
    print("No trades!")

print("\n" + "=" * 70)
print("BACKTEST COMPLETE")
print("=" * 70)
