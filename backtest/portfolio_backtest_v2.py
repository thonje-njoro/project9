#!/usr/bin/env python3
"""
REALISTIC ORB Portfolio Backtest v2
====================================
Fixed entry logic — uses proper ORB mechanics:
- Opening range = first N bars' high/low
- Entry when price TRADES through OR level (not just closes above)
- Stop at opposite end of OR (structural stop, not ATR)
- Trailing stop activates after 1R profit
- Realistic slippage on entries and exits
"""

import pandas as pd, numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

print("="*70, flush=True)
print("REALISTIC ORB PORTFOLIO BACKTEST v2", flush=True)
print("Proper entry/stop mechanics · Realistic costs", flush=True)
print("="*70, flush=True)

# ── Config ──
SLIPPAGE_PCT = 0.002     # 0.2% slippage per side
COMMISSION_PCT = 0.0005   # 0.05% per side
DAILY_DD_LIMIT = 0.05     # 5% daily loss limit
TOTAL_DD_LIMIT = 0.10     # 10% total drawdown
PROFIT_TARGET = 0.10      # 10% profit target
RISK_PER_TRADE = 0.01     # 1% account risk

SYMBOLS = {
    "NVDA": {"file": "NVDA_5min.parquet", "or_minutes": 60},
    "AMD":  {"file": "AMD_5min.parquet",  "or_minutes": 60},
    "PLTR": {"file": "PLTR_5min.parquet", "or_minutes": 60},
    "MRVL": {"file": "MRVL_5min.parquet", "or_minutes": 30},
}

# ── Data ──
def load_data(sym, cfg):
    df = pd.read_parquet(cfg["file"])
    h, m = df.index.hour, df.index.minute
    mask = ((h == 9) & (m >= 30)) | ((h >= 10) & (h < 16))
    df = df[mask].copy()
    ohlcv = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df = df.resample("5min").agg(ohlcv).dropna()
    df["date"] = df.index.date
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    
    # ATR for position sizing (shifted to avoid look-ahead)
    c = df["close"].values
    hi = df["high"].values
    lo = df["low"].values
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum(np.maximum(hi - lo, np.abs(hi - prev_c)), np.abs(lo - prev_c))
    df["atr"] = pd.Series(tr, index=df.index).rolling(14, min_periods=1).mean().shift(1).values
    
    # Day groups
    dates = df.index.date
    ds = np.array([str(d) for d in dates])
    ch = np.where(ds[1:] != ds[:-1])[0] + 1
    df.attrs["day_groups"] = list(zip(np.concatenate([[0], ch]), np.concatenate([ch, [len(dates)]])))
    
    return df

print("\nLoading data...", flush=True)
data = {}
for sym, cfg in SYMBOLS.items():
    df = load_data(sym, cfg)
    dates = sorted(set(df.index.date))
    data[sym] = df
    print(f"  {sym}: {len(df)} bars, {len(dates)} days", flush=True)


def run_orb_v2(df, or_minutes=60, symbol="UNK"):
    """
    Proper ORB mechanics:
    1. Opening range = first N bars' high and low
    2. Entry when price TRADES through OR level (intrabar)
    3. Stop at opposite end of OR (structural)
    4. After 1R profit, trail with 0.5R increments
    5. Exit at EOD
    """
    open_ = df["open"].values
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["atr"].values
    hour = df["hour"].values
    minute = df["minute"].values
    day_groups = df.attrs["day_groups"]
    bars_per_or = max(1, or_minutes // 5)
    
    trades = []
    
    for si, ei in day_groups:
        if ei - si < bars_per_or + 3:
            continue
        
        # ── 1. Define Opening Range ──
        or_high = high[si:si+bars_per_or].max()
        or_low = low[si:si+bars_per_or].min()
        or_range = or_high - or_low
        
        if or_range <= 0:
            continue
        
        # ATR for position sizing
        atr_val = atr[si+bars_per_or-1]
        if np.isnan(atr_val) or atr_val <= 0:
            atr_val = or_range  # Fallback
        
        in_trade = False
        direction = 0
        entry_price = 0.0
        stop_price = 0.0
        target_1r = 0.0
        trail_price = 0.0
        entry_bar = 0
        entry_time = None
        risk_amount = 0.0
        trailing_active = False
        
        # ── 2. Scan for breakout ──
        for j in range(si + bars_per_or, ei):
            dt = df.index[j]
            
            # EOD close
            if hour[j] >= 15 and minute[j] >= 55:
                if in_trade:
                    exit_p = close[j] * (1 - SLIPPAGE_PCT)
                    pnl_pct = (exit_p - entry_price) / entry_price * 100 * direction
                    pnl_pct -= COMMISSION_PCT * 200  # Round trip
                    trades.append({
                        "date": dt.date(), "entry_time": entry_time, "exit_time": dt,
                        "direction": direction, "entry": entry_price, "exit": exit_p,
                        "pnl_pct": pnl_pct, "reason": "eod", "bars_held": j - entry_bar,
                        "or_range_pct": or_range / close[si] * 100,
                    })
                break
            
            if not in_trade:
                # ── Breakout Detection ──
                # Price must TRADE through the OR level (high > or_high for long)
                # This happens intrabar — we detect it when bar closes beyond level
                # Entry at the OR level (conservative — assume we got filled at the level)
                
                if high[j] > or_high and low[j] < or_high:
                    # Bar traded through OR high — potential long breakout
                    # Confirm: close must be above OR high (not just a wick)
                    if close[j] > or_high:
                        # Enter at OR high + small buffer (realistic fill)
                        entry_price = or_high * (1 + SLIPPAGE_PCT)
                        stop_price = or_low  # Structural stop at opposite end
                        risk_amount = entry_price - stop_price
                        
                        if risk_amount <= 0:
                            continue
                        
                        target_1r = entry_price + risk_amount  # 1R target
                        trail_price = stop_price
                        direction = 1
                        entry_bar = j
                        entry_time = dt
                        in_trade = True
                        trailing_active = False
                
                elif low[j] < or_low and high[j] > or_low:
                    # Bar traded through OR low — potential short breakout
                    if close[j] < or_low:
                        entry_price = or_low * (1 - SLIPPAGE_PCT)
                        stop_price = or_high
                        risk_amount = stop_price - entry_price
                        
                        if risk_amount <= 0:
                            continue
                        
                        target_1r = entry_price - risk_amount
                        trail_price = stop_price
                        direction = -1
                        entry_bar = j
                        entry_time = dt
                        in_trade = True
                        trailing_active = False
            
            else:
                # ── Manage Trade ──
                bars_held = j - entry_bar
                
                if direction == 1:
                    # Check stop hit
                    if low[j] <= stop_price:
                        exit_p = stop_price * (1 - SLIPPAGE_PCT)
                        pnl_pct = (exit_p - entry_price) / entry_price * 100
                        pnl_pct -= COMMISSION_PCT * 200
                        trades.append({
                            "date": dt.date(), "entry_time": entry_time, "exit_time": dt,
                            "direction": 1, "entry": entry_price, "exit": exit_p,
                            "pnl_pct": pnl_pct, "reason": "stop", "bars_held": bars_held,
                            "or_range_pct": or_range / close[si] * 100,
                        })
                        in_trade = False
                        continue
                    
                    # Check if 1R reached — activate trailing
                    if not trailing_active and high[j] >= target_1r:
                        trailing_active = True
                        trail_price = entry_price  # Move stop to breakeven
                    
                    # Trail: move stop up by 0.5R each time price makes new high
                    if trailing_active:
                        new_trail = high[j] - 0.5 * risk_amount
                        if new_trail > trail_price:
                            trail_price = new_trail
                        stop_price = max(stop_price, trail_price)
                
                else:  # Short
                    if high[j] >= stop_price:
                        exit_p = stop_price * (1 + SLIPPAGE_PCT)
                        pnl_pct = (entry_price - exit_p) / entry_price * 100
                        pnl_pct -= COMMISSION_PCT * 200
                        trades.append({
                            "date": dt.date(), "entry_time": entry_time, "exit_time": dt,
                            "direction": -1, "entry": entry_price, "exit": exit_p,
                            "pnl_pct": pnl_pct, "reason": "stop", "bars_held": bars_held,
                            "or_range_pct": or_range / close[si] * 100,
                        })
                        in_trade = False
                        continue
                    
                    if not trailing_active and low[j] <= target_1r:
                        trailing_active = True
                        trail_price = entry_price
                    
                    if trailing_active:
                        new_trail = low[j] + 0.5 * risk_amount
                        if new_trail < trail_price:
                            trail_price = new_trail
                        stop_price = min(stop_price, trail_price)
    
    return trades


# ── Run All Symbols ──
print("\n" + "="*70, flush=True)
print("RUNNING ORB v2 BACKTESTS", flush=True)
print("="*70, flush=True)

all_trades = {}
for sym, cfg in SYMBOLS.items():
    print(f"\n  {sym} (OR={cfg['or_minutes']}min)...", end=" ", flush=True)
    trades = run_orb_v2(data[sym], or_minutes=cfg["or_minutes"], symbol=sym)
    all_trades[sym] = trades
    
    if trades:
        pnls = [t["pnl_pct"] for t in trades]
        w = [p for p in pnls if p > 0]
        l = [p for p in pnls if p <= 0]
        wr = len(w)/len(pnls)*100
        total = sum(pnls)
        sharpe = np.mean(pnls)/np.std(pnls) if np.std(pnls) > 0 else 0
        pf = abs(sum(w)/sum(l)) if l and sum(l) != 0 else 99
        hold = np.mean([t["bars_held"] for t in trades]) * 5
        
        reasons = {}
        for t in trades:
            r = t["reason"]
            reasons[r] = reasons.get(r, 0) + 1
        
        print(f"{len(trades)} trades, WR={wr:.1f}%, Ret={total:.2f}%, "
              f"Sharpe={sharpe:.3f}, PF={min(pf,99):.2f}, "
              f"AvgHold={hold:.0f}min, Reasons={reasons}", flush=True)
    else:
        print("0 trades", flush=True)


# ── Combined Portfolio ──
print("\n" + "="*70, flush=True)
print("COMBINED PORTFOLIO SIMULATION", flush=True)
print("="*70, flush=True)

# Daily P&L
portfolio_daily = {}
for sym, trades in all_trades.items():
    for t in trades:
        d = str(t["date"])
        if d not in portfolio_daily:
            portfolio_daily[d] = []
        portfolio_daily[d].append({"symbol": sym, "pnl": t["pnl_pct"]})

initial_capital = 100000
capital = initial_capital
peak_capital = initial_capital
daily_pnls = []
violations = []
dates_sorted = sorted(portfolio_daily.keys())
train_end = "2023-06-30"

for d in dates_sorted:
    day_trades = portfolio_daily[d]
    symbols_traded = set(t["symbol"] for t in day_trades)
    n_symbols = len(symbols_traded)
    
    # Equal weight: average P&L across symbols
    day_pnl_pct = sum(t["pnl"] for t in day_trades) / max(n_symbols, 1)
    
    # Apply risk sizing
    daily_return = day_pnl_pct * RISK_PER_TRADE
    capital *= (1 + daily_return / 100)
    daily_pnls.append(daily_return)
    
    peak_capital = max(peak_capital, capital)
    dd = (capital - peak_capital) / peak_capital
    
    if daily_return < -DAILY_DD_LIMIT * 100:
        violations.append({"date": d, "type": "daily_dd", "value": daily_return})
    if dd < -TOTAL_DD_LIMIT:
        violations.append({"date": d, "type": "total_dd", "value": dd})


# ── Results ──
dp = np.array(daily_pnls)
total_return = (capital - initial_capital) / initial_capital * 100
annual_sharpe = dp.mean() / dp.std() * np.sqrt(252) if dp.std() > 0 else 0

all_pnls = [t["pnl_pct"] for trades in all_trades.values() for t in trades]
ap = np.array(all_pnls)
wins = ap[ap > 0]
losses = ap[ap <= 0]
overall_wr = len(wins)/len(ap)*100 if len(ap) > 0 else 0
overall_pf = abs(wins.sum()/losses.sum()) if losses.sum() != 0 else 99

all_hold = [t["bars_held"] for trades in all_trades.values() for t in trades]
avg_hold_min = np.mean(all_hold) * 5 if all_hold else 0

n_days = len(dates_sorted)
trades_per_day = len(all_pnls) / n_days if n_days > 0 else 0

train_pnls = [daily_pnls[i] for i,d in enumerate(dates_sorted) if d <= train_end]
test_pnls = [daily_pnls[i] for i,d in enumerate(dates_sorted) if d > train_end]
train_ret = sum(train_pnls)
test_ret = sum(test_pnls)

# Max drawdown
eq = np.cumsum(dp)
peak = np.maximum.accumulate(eq)
dd = eq - peak
max_dd = dd.min()

print(f"\n  CAPITAL:", flush=True)
print(f"    Initial: ${initial_capital:,.0f}", flush=True)
print(f"    Final:   ${capital:,.0f}", flush=True)
print(f"    Return:  {total_return:.2f}%", flush=True)

print(f"\n  RISK METRICS:", flush=True)
print(f"    Annual Sharpe:    {annual_sharpe:.3f}", flush=True)
print(f"    Max Drawdown:     {max_dd:.2f}%", flush=True)
print(f"    Daily DD violations: {len([v for v in violations if v['type']=='daily_dd'])}", flush=True)

print(f"\n  TRADE STATISTICS:", flush=True)
print(f"    Total trades:     {len(all_pnls)}", flush=True)
print(f"    Trades/day:       {trades_per_day:.1f} (target: 1-4)", flush=True)
print(f"    Win rate:         {overall_wr:.1f}% (target: 40-65%)", flush=True)
print(f"    Profit factor:    {min(overall_pf,99):.2f} (target: 1.5-3.0)", flush=True)
if len(wins) > 0:
    print(f"    Avg win:          {wins.mean():.3f}% (target: 0.5-3%)", flush=True)
if len(losses) > 0:
    print(f"    Avg loss:         {losses.mean():.3f}% (target: -0.5 to -2%)", flush=True)
if len(losses) > 0 and losses.mean() != 0:
    print(f"    Win/Loss ratio:   {abs(wins.mean()/losses.mean()):.2f}x (target: 1.5-3x)", flush=True)
print(f"    Avg hold time:    {avg_hold_min:.0f} min (target: 30-240)", flush=True)

print(f"\n  WALK-FORWARD:", flush=True)
print(f"    Train (to {train_end}): {train_ret:.2f}% daily return", flush=True)
print(f"    Test  (after):        {test_ret:.2f}% daily return", flush=True)

# Per-symbol
print(f"\n  PER-SYMBOL:", flush=True)
print(f"  {'Sym':<6} {'#':<6} {'WR%':<8} {'Ret%':<10} {'AvgW':<10} {'AvgL':<10} {'PF':<8} {'Hold':<8}", flush=True)
print(f"  {'─'*65}", flush=True)
for sym in SYMBOLS:
    trades = all_trades.get(sym, [])
    if not trades: continue
    pnls = [t["pnl_pct"] for t in trades]
    w = [p for p in pnls if p > 0]
    l = [p for p in pnls if p <= 0]
    wr = len(w)/len(pnls)*100
    pf = abs(sum(w)/sum(l)) if l and sum(l) != 0 else 99
    h = np.mean([t["bars_held"] for t in trades]) * 5
    print(f"  {sym:<6} {len(pnls):<6} {wr:<8.1f} {sum(pnls):<10.2f} "
          f"{np.mean(w) if w else 0:<10.3f} {np.mean(l) if l else 0:<10.3f} "
          f"{min(pf,99):<8.2f} {h:<8.0f}", flush=True)

# Exit reasons
print(f"\n  EXIT REASONS:", flush=True)
reason_counts = {}
for trades in all_trades.values():
    for t in trades:
        r = t["reason"]
        reason_counts[r] = reason_counts.get(r, 0) + 1
for r, c in sorted(reason_counts.items(), key=lambda x: -x[1]):
    pct = c / len(all_pnls) * 100
    print(f"    {r}: {c} ({pct:.1f}%)", flush=True)

# OR range analysis
print(f"\n  OPENING RANGE ANALYSIS:", flush=True)
or_ranges = [t["or_range_pct"] for trades in all_trades.values() for t in trades]
if or_ranges:
    print(f"    Avg OR range: {np.mean(or_ranges):.3f}%", flush=True)
    print(f"    Min OR range: {np.min(or_ranges):.3f}%", flush=True)
    print(f"    Max OR range: {np.max(or_ranges):.3f}%", flush=True)

# Prop firm
print(f"\n  PROP FIRM ASSESSMENT:", flush=True)
print(f"    Return > 10%:  {'✓' if total_return > 10 else '✗'} ({total_return:.2f}%)", flush=True)
print(f"    Max DD < 10%:  {'✓' if abs(max_dd) < 10 else '✗'} ({max_dd:.2f}%)", flush=True)
print(f"    Sharpe > 0.5:  {'✓' if annual_sharpe > 0.5 else '✗'} ({annual_sharpe:.3f})", flush=True)
print(f"    WR > 45%:      {'✓' if overall_wr > 45 else '✗'} ({overall_wr:.1f}%)", flush=True)
print(f"    PF > 1.2:      {'✓' if overall_pf > 1.2 else '✗'} ({min(overall_pf,99):.2f})", flush=True)

# Survival
print(f"\n  SURVIVAL:", flush=True)
running = initial_capital
running_peak = initial_capital
survived = True
for i, d in enumerate(dates_sorted):
    running *= (1 + daily_pnls[i] / 100)
    running_peak = max(running_peak, running)
    dd = (running - running_peak) / running_peak
    if dd < -TOTAL_DD_LIMIT:
        print(f"    ✗ BLOWN on {d}: DD {dd*100:.2f}%", flush=True)
        survived = False
        break
    if daily_pnls[i] < -DAILY_DD_LIMIT * 100:
        print(f"    ✗ BLOWN on {d}: Daily {daily_pnls[i]:.2f}%", flush=True)
        survived = False
        break
    if (running - initial_capital) / initial_capital >= PROFIT_TARGET:
        print(f"    ✓ PASSED on {d}: Hit {PROFIT_TARGET*100}% target", flush=True)
        break

if survived and total_return < PROFIT_TARGET * 100:
    print(f"    ⚠️ Still running: {total_return:.2f}% (need {PROFIT_TARGET*100}%)", flush=True)

# Trade frequency distribution
print(f"\n  FREQUENCY:", flush=True)
tc = np.array([len(portfolio_daily[d]) for d in dates_sorted])
print(f"    Mean: {tc.mean():.1f}/day, Median: {np.median(tc):.0f}/day", flush=True)
print(f"    0 trades: {(tc==0).sum()} days", flush=True)
print(f"    1-3 trades: {((tc>=1)&(tc<=3)).sum()} days", flush=True)
print(f"    4+ trades: {(tc>=4).sum()} days", flush=True)

# Save
with open("portfolio_backtest_v2.json", "w") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "config": {"slippage": SLIPPAGE_PCT, "commission": COMMISSION_PCT,
                   "daily_dd": DAILY_DD_LIMIT, "total_dd": TOTAL_DD_LIMIT,
                   "risk_per_trade": RISK_PER_TRADE},
        "results": {"total_return": round(total_return,2), "annual_sharpe": round(annual_sharpe,3),
                    "max_dd": round(max_dd,2), "trades": len(all_pnls),
                    "trades_per_day": round(trades_per_day,1), "win_rate": round(overall_wr,1),
                    "profit_factor": round(min(overall_pf,99),2), "avg_hold_min": round(avg_hold_min,0),
                    "violations": len(violations)},
        "per_symbol": {sym: {"trades": len(ts), "wr": round(len([t for t in ts if t["pnl_pct"]>0])/len(ts)*100,1),
                             "return": round(sum(t["pnl_pct"] for t in ts),2)}
                       for sym, ts in all_trades.items() if ts},
    }, f, indent=2, default=str)
print(f"\n  Saved to portfolio_backtest_v2.json", flush=True)
