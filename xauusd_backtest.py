#!/usr/bin/env python3
"""XAUUSD Donchian Breakout Backtest — direct LSE API call, no framework dependency"""

import os
import json
import sys
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path

# Load API key from .env
def load_env():
    env_path = Path(__file__).parent / "backtest" / ".env"
    if not env_path.exists():
        print(f"ERROR: {env_path} not found")
        sys.exit(1)
    env = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env

ENV = load_env()
LSE_API_KEY = ENV.get("LSE_API_KEY", "")
LSE_BASE_URL = "https://api.londonstrategicedge.com/vault/candles"

# Strategy parameters
LOOKBACK = 50
RR_TARGET = 3.0
STOP_ATR_MULT = 2.0
ATR_PERIOD = 14
COST_RT = 0.002  # 0.1% commission + 0.1% slippage round-trip

def fetch_lse(symbol, tf, start, end):
    """Fetch candles from LSE API. Returns list of dicts."""
    headers = {"x-api-key": LSE_API_KEY}
    all_candles = []
    
    cur = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    
    while cur < end_dt:
        nxt = min(cur + timedelta(days=365), end_dt)
        params = {
            "symbol": symbol,
            "from": cur.strftime("%Y-%m-%d"),
            "to": nxt.strftime("%Y-%m-%d"),
            "timeframe": tf
        }
        try:
            r = requests.get(LSE_BASE_URL, headers=headers, params=params, timeout=60)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and len(data) > 0:
                    all_candles.extend(data)
                    print(f"  Chunk {cur.strftime('%Y-%m-%d')} to {nxt.strftime('%Y-%m-%d')}: {len(data)} bars")
                else:
                    print(f"  Chunk {cur.strftime('%Y-%m-%d')}: empty response")
            else:
                print(f"  Chunk {cur.strftime('%Y-%m-%d')}: HTTP {r.status_code}")
        except Exception as e:
            print(f"  Chunk failed: {e}")
        cur = nxt
    
    return all_candles

def parse_candles(candles):
    """Parse LSE JSON array into parallel arrays."""
    # Sort by timestamp
    candles.sort(key=lambda c: c["ts"])
    
    times = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    for c in candles:
        times.append(c["ts"])
        opens.append(float(c["open"]))
        highs.append(float(c["high"]))
        lows.append(float(c["low"]))
        closes.append(float(c["close"]))
        volumes.append(float(c.get("volume", 0)))
    
    return {
        "times": times,
        "opens": np.array(opens),
        "highs": np.array(highs),
        "lows": np.array(lows),
        "closes": np.array(closes),
        "volumes": np.array(volumes),
        "n": len(times)
    }

def rolling_max(arr, window):
    """Rolling max over array."""
    result = np.full_like(arr, np.nan)
    for i in range(window - 1, len(arr)):
        result[i] = np.max(arr[i - window + 1:i + 1])
    return result

def rolling_min(arr, window):
    """Rolling min over array."""
    result = np.full_like(arr, np.nan)
    for i in range(window - 1, len(arr)):
        result[i] = np.min(arr[i - window + 1:i + 1])
    return result

def rolling_mean(arr, window):
    """Rolling mean over array."""
    result = np.full_like(arr, np.nan)
    for i in range(window - 1, len(arr)):
        result[i] = np.mean(arr[i - window + 1:i + 1])
    return result

def run_backtest(data):
    """Run Donchian breakout strategy."""
    n = data["n"]
    closes = data["closes"]
    highs = data["highs"]
    lows = data["lows"]
    
    high_n = rolling_max(highs, LOOKBACK)
    low_n = rolling_min(lows, LOOKBACK)
    atr = rolling_mean(highs - lows, ATR_PERIOD)
    
    trades = []
    pos = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    for i in range(LOOKBACK, n):
        price = closes[i]
        h = high_n[i]
        l = low_n[i]
        a = atr[i]
        
        if np.isnan(h) or np.isnan(a) or a == 0:
            continue
        
        if pos == 0:
            # Entry logic
            if price >= h:
                pos = 1
                entry_price = price
                stop_price = price - STOP_ATR_MULT * a
                target_price = price + RR_TARGET * STOP_ATR_MULT * a
            elif price <= l:
                pos = -1
                entry_price = price
                stop_price = price + STOP_ATR_MULT * a
                target_price = price - RR_TARGET * STOP_ATR_MULT * a
        elif pos == 1:
            # Long trade management
            if price <= stop_price:
                pnl = (stop_price - entry_price) / entry_price - COST_RT
                trades.append(pnl)
                pos = 0
            elif price >= target_price:
                pnl = (target_price - entry_price) / entry_price - COST_RT
                trades.append(pnl)
                pos = 0
            else:
                # Trailing stop
                new_stop = price - STOP_ATR_MULT * a
                if new_stop > stop_price:
                    stop_price = new_stop
        elif pos == -1:
            # Short trade management
            if price >= stop_price:
                pnl = (entry_price - stop_price) / entry_price - COST_RT
                trades.append(pnl)
                pos = 0
            elif price <= target_price:
                pnl = (entry_price - target_price) / entry_price - COST_RT
                trades.append(pnl)
                pos = 0
            else:
                # Trailing stop
                new_stop = price + STOP_ATR_MULT * a
                if new_stop < stop_price:
                    stop_price = new_stop
    
    return trades

def walk_forward(data, n_windows=3):
    """Walk-forward validation."""
    n = data["n"]
    step = n // n_windows
    results = []
    
    for i in range(n_windows):
        start_idx = i * step
        end_idx = min((i + 1) * step + LOOKBACK, n)
        
        window_data = {
            "opens": data["opens"][start_idx:end_idx],
            "highs": data["highs"][start_idx:end_idx],
            "lows": data["lows"][start_idx:end_idx],
            "closes": data["closes"][start_idx:end_idx],
            "volumes": data["volumes"][start_idx:end_idx],
            "times": data["times"][start_idx:end_idx],
            "n": end_idx - start_idx
        }
        
        if window_data["n"] < LOOKBACK + 10:
            results.append({"window": i+1, "trades": 0, "pass": False})
            continue
        
        trades = run_backtest(window_data)
        
        if len(trades) < 3:
            results.append({"window": i+1, "trades": len(trades), "pass": False})
            continue
        
        wins = sum(1 for t in trades if t > 0)
        wr = wins / len(trades)
        avg = np.mean(trades)
        std = np.std(trades)
        sharpe = (avg / std) * np.sqrt(252) if std > 0 else 0
        gross_profit = sum(t for t in trades if t > 0)
        gross_loss = abs(sum(t for t in trades if t < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        passed = wr >= 0.45 and sharpe >= 0.5 and pf >= 1.5
        
        results.append({
            "window": i+1,
            "trades": len(trades),
            "wr": round(wr, 3),
            "sharpe": round(sharpe, 3),
            "pf": round(pf, 3),
            "pass": passed
        })
    
    return results

def monte_carlo(trades, n_sims=2000):
    """Monte Carlo simulation."""
    sharpes = []
    max_dds = []
    
    for _ in range(n_sims):
        sampled = np.random.choice(trades, size=len(trades), replace=True)
        cum = np.cumsum(sampled)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        max_dd = dd.max() if len(dd) > 0 else 0
        avg = sampled.mean()
        std = sampled.std()
        s = (avg / std) * np.sqrt(252) if std > 0 else 0
        sharpes.append(s)
        max_dds.append(max_dd)
    
    return {
        "sharpe_median": round(float(np.median(sharpes)), 3),
        "sharpe_5pct": round(float(np.percentile(sharpes, 5)), 3),
        "sharpe_95pct": round(float(np.percentile(sharpes, 95)), 3),
        "dd_median": round(float(np.median(max_dds)), 3),
        "dd_95pct": round(float(np.percentile(max_dds, 95)), 3),
        "dd_99pct": round(float(np.percentile(max_dds, 99)), 3),
        "survival_rate": round(float(np.mean([1 if s > 0 else 0 for s in sharpes])), 3)
    }

def main():
    print("=== XAUUSD Donchian(50, R:R=3.0) Backtest ===")
    print(f"Parameters: lookback={LOOKBACK}, RR={RR_TARGET}, stop_atr_mult={STOP_ATR_MULT}")
    print(f"Costs: {COST_RT*100}% round-trip")
    print(f"API: {LSE_BASE_URL}")
    print(f"API Key: {LSE_API_KEY[:8]}...{LSE_API_KEY[-4:]}")
    print()
    
    # Fetch data
    print("Fetching XAUUSD data from LSE API...")
    candles = fetch_lse("XAU/USD", "1h", "2020-01-01", "2025-06-16")
    
    if len(candles) == 0:
        print("ERROR: No data returned from API")
        sys.exit(1)
    
    data = parse_candles(candles)
    print(f"Total: {data['n']} bars from {data['times'][0]} to {data['times'][-1]}")
    print()
    
    # Run backtest
    print("Running backtest...")
    trades = run_backtest(data)
    
    if len(trades) < 5:
        print(f"ERROR: Only {len(trades)} trades — need more data")
        sys.exit(1)
    
    # Performance metrics
    wins = sum(1 for t in trades if t > 0)
    wr = wins / len(trades)
    avg_ret = np.mean(trades)
    std_ret = np.std(trades)
    sharpe = (avg_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
    
    gross_profit = sum(t for t in trades if t > 0)
    gross_loss = abs(sum(t for t in trades if t < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    
    cum = np.cumsum(trades)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = dd.max() if len(dd) > 0 else 0
    
    total_ret = sum(trades)
    
    print(f"=== RESULTS ===")
    print(f"Trades: {len(trades)}")
    print(f"Win Rate: {wr*100:.1f}%")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"Max Drawdown: {max_dd*100:.1f}%")
    print(f"Total Return: {total_ret*100:.1f}%")
    print(f"Avg Trade: {avg_ret*100:.3f}%")
    print()
    
    # Walk-forward
    print(f"=== WALK-FORWARD VALIDATION ===")
    wf = walk_forward(data)
    for w in wf:
        status = "✅" if w.get("pass", False) else "❌"
        print(f"  Window {w['window']}: {w['trades']} trades, WR={w.get('wr',0)*100:.1f}%, Sharpe={w.get('sharpe',0):.3f}, PF={w.get('pf',0):.2f} {status}")
    wf_passes = sum(1 for w in wf if w.get("pass", False))
    print(f"  Walk-forward: {wf_passes}/{len(wf)} windows pass")
    print()
    
    # Monte Carlo
    print(f"=== MONTE CARLO (2000 sims) ===")
    mc = monte_carlo(trades)
    print(f"  Sharpe median: {mc['sharpe_median']}, 5th-95th: [{mc['sharpe_5pct']}, {mc['sharpe_95pct']}]")
    print(f"  Max DD median: {mc['dd_median']*100:.1f}%, 99th: {mc['dd_99pct']*100:.1f}%")
    print(f"  Survival rate: {mc['survival_rate']*100:.1f}%")
    print()
    
    # Acceptance criteria
    accept = wr >= 0.45 and sharpe >= 0.5 and pf >= 1.5 and max_dd <= 0.20
    print(f"{'✅ ACCEPTED' if accept else '❌ REJECTED'}")
    print(f"  WR={wr*100:.1f}% {'✅' if wr >= 0.45 else '❌ (need 45%)'}")
    print(f"  Sharpe={sharpe:.3f} {'✅' if sharpe >= 0.5 else '❌ (need 0.5)'}")
    print(f"  PF={pf:.2f} {'✅' if pf >= 1.5 else '❌ (need 1.5)'}")
    print(f"  DD={max_dd*100:.1f}% {'✅' if max_dd <= 0.20 else '❌ (need <20%)'}")

if __name__ == "__main__":
    main()
