#!/usr/bin/env python3
"""XAUUSD Donchian Breakout Backtest — LSE API with proper start-based pagination"""

import os
import json
import sys
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path

# Load API key
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

LOOKBACK = 50
RR_TARGET = 3.0
STOP_ATR_MULT = 2.0
ATR_PERIOD = 14
COST_RT = 0.002  # 0.1% + 0.1%

def fetch_lse(symbol, tf, start, end):
    """Paginated fetch using 'start' parameter. API ignores from/to, returns 5K bar pages."""
    headers = {"x-api-key": LSE_API_KEY}
    all_candles = []
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    cur_start = start
    max_pages = 20
    seen_ts = set()
    
    for page in range(max_pages):
        params = {"symbol": symbol, "timeframe": tf, "start": cur_start}
        try:
            r = requests.get(LSE_BASE_URL, headers=headers, params=params, timeout=60)
            if r.status_code != 200:
                print(f"  Page {page}: HTTP {r.status_code}")
                break
            data = r.json()
            if not isinstance(data, list) or len(data) == 0:
                print(f"  Page {page}: empty")
                break
        except Exception as e:
            print(f"  Page {page} failed: {e}")
            break
        
        # Add new candles (dedup + filter to end date)
        new_count = 0
        last_ts = None
        for c in data:
            ts = c["ts"]
            if ts in seen_ts:
                continue
            seen_ts.add(ts)
            dt = datetime.strptime(ts[:10], "%Y-%m-%d")
            if dt > end_dt:
                continue
            all_candles.append(c)
            new_count += 1
            last_ts = ts
        
        first_ts = data[0]["ts"][:10]
        last_page_ts = data[-1]["ts"][:10]
        print(f"  Page {page}: {len(data)} bars ({first_ts} → {last_page_ts}), {new_count} new, total={len(all_candles)}")
        
        if new_count == 0:
            break
        
        # Next page starts after last timestamp
        if last_ts:
            cur_start = (datetime.strptime(last_ts[:10], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            if datetime.strptime(last_ts[:10], "%Y-%m-%d") >= end_dt:
                break
        
        if len(data) < 4900:
            break
    
    return all_candles

def parse_candles(candles):
    candles.sort(key=lambda c: c["ts"])
    n = len(candles)
    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    closes = np.zeros(n)
    times = []
    for i, c in enumerate(candles):
        times.append(c["ts"])
        opens[i] = float(c["open"])
        highs[i] = float(c["high"])
        lows[i] = float(c["low"])
        closes[i] = float(c["close"])
    return {"times": times, "opens": opens, "highs": highs, "lows": lows, "closes": closes, "n": n}

def rolling_max(arr, w):
    out = np.full(len(arr), np.nan)
    for i in range(w-1, len(arr)):
        out[i] = np.max(arr[i-w+1:i+1])
    return out

def rolling_min(arr, w):
    out = np.full(len(arr), np.nan)
    for i in range(w-1, len(arr)):
        out[i] = np.min(arr[i-w+1:i+1])
    return out

def rolling_mean(arr, w):
    out = np.full(len(arr), np.nan)
    for i in range(w-1, len(arr)):
        out[i] = np.mean(arr[i-w+1:i+1])
    return out

def run_backtest(d):
    n = d["n"]
    c, h, l = d["closes"], d["highs"], d["lows"]
    high_n = rolling_max(h, LOOKBACK)
    low_n = rolling_min(l, LOOKBACK)
    atr = rolling_mean(h - l, ATR_PERIOD)
    
    trades = []
    pos = 0
    entry = stop = target = 0.0
    
    for i in range(LOOKBACK, n):
        price = c[i]
        hn, ln, a = high_n[i], low_n[i], atr[i]
        if np.isnan(hn) or np.isnan(a) or a == 0:
            continue
        if pos == 0:
            if price >= hn:
                pos = 1; entry = price; stop = price - STOP_ATR_MULT*a; target = price + RR_TARGET*STOP_ATR_MULT*a
            elif price <= ln:
                pos = -1; entry = price; stop = price + STOP_ATR_MULT*a; target = price - RR_TARGET*STOP_ATR_MULT*a
        elif pos == 1:
            if price <= stop:
                trades.append((stop-entry)/entry - COST_RT); pos = 0
            elif price >= target:
                trades.append((target-entry)/entry - COST_RT); pos = 0
            else:
                ns = price - STOP_ATR_MULT*a
                if ns > stop: stop = ns
        elif pos == -1:
            if price >= stop:
                trades.append((entry-stop)/entry - COST_RT); pos = 0
            elif price <= target:
                trades.append((entry-target)/entry - COST_RT); pos = 0
            else:
                ns = price + STOP_ATR_MULT*a
                if ns < stop: stop = ns
    return trades

def walk_forward(d, n_windows=3):
    n = d["n"]
    step = n // n_windows
    results = []
    for i in range(n_windows):
        s = i * step
        e = min((i+1)*step + LOOKBACK, n)
        wd = {k: d[k][s:e] if isinstance(d[k], np.ndarray) else d[k][s:e] for k in ["opens","highs","lows","closes"]}
        wd["n"] = e - s
        if wd["n"] < LOOKBACK + 10:
            results.append({"w": i+1, "trades": 0, "pass": False}); continue
        trades = run_backtest(wd)
        if len(trades) < 3:
            results.append({"w": i+1, "trades": len(trades), "pass": False}); continue
        wr = sum(1 for t in trades if t > 0) / len(trades)
        avg, std = np.mean(trades), np.std(trades)
        sharpe = (avg/std)*np.sqrt(252) if std > 0 else 0
        gp = sum(t for t in trades if t > 0)
        gl = abs(sum(t for t in trades if t < 0))
        pf = gp/gl if gl > 0 else float("inf")
        results.append({"w": i+1, "trades": len(trades), "wr": round(wr,3), "sharpe": round(sharpe,3),
                        "pf": round(pf,3), "pass": wr>=0.45 and sharpe>=0.5 and pf>=1.5})
    return results

def monte_carlo(trades, n_sims=2000):
    sharpes, dds = [], []
    for _ in range(n_sims):
        s = np.random.choice(trades, size=len(trades), replace=True)
        cum = np.cumsum(s); pk = np.maximum.accumulate(cum)
        dd = (pk - cum).max()
        avg, std = s.mean(), s.std()
        sharpes.append((avg/std)*np.sqrt(252) if std > 0 else 0)
        dds.append(dd)
    return {"sharpe_med": round(float(np.median(sharpes)),3),
            "sharpe_5": round(float(np.percentile(sharpes,5)),3),
            "sharpe_95": round(float(np.percentile(sharpes,95)),3),
            "dd_med": round(float(np.median(dds)),3),
            "dd_99": round(float(np.percentile(dds,99)),3),
            "survival": round(float(np.mean([1 if s>0 else 0 for s in sharpes])),3)}

def main():
    print("=== XAUUSD Donchian(50, R:R=3.0) Backtest ===")
    print(f"Params: lookback={LOOKBACK}, RR={RR_TARGET}, stop_atr={STOP_ATR_MULT}, cost={COST_RT*100}%")
    print(f"API: {LSE_BASE_URL}\n")
    
    print("Fetching XAUUSD 1h data (paginated with start=)...")
    candles = fetch_lse("XAU/USD", "1h", "2020-01-01", "2025-06-16")
    
    if len(candles) < 100:
        print(f"ERROR: Only {len(candles)} candles"); sys.exit(1)
    
    d = parse_candles(candles)
    print(f"\nTotal: {d['n']} bars, {d['times'][0][:10]} to {d['times'][-1][:10]}\n")
    
    trades = run_backtest(d)
    if len(trades) < 5:
        print(f"ERROR: {len(trades)} trades"); sys.exit(1)
    
    wins = sum(1 for t in trades if t > 0)
    wr = wins/len(trades)
    avg, std = np.mean(trades), np.std(trades)
    sharpe = (avg/std)*np.sqrt(252) if std > 0 else 0
    gp = sum(t for t in trades if t > 0)
    gl = abs(sum(t for t in trades if t < 0))
    pf = gp/gl if gl > 0 else float("inf")
    cum = np.cumsum(trades); pk = np.maximum.accumulate(cum)
    max_dd = (pk - cum).max()
    
    print(f"=== RESULTS ({d['times'][0][:10]} to {d['times'][-1][:10]}) ===")
    print(f"Trades: {len(trades)}  |  WR: {wr*100:.1f}%  |  PF: {pf:.2f}  |  Sharpe: {sharpe:.3f}")
    print(f"Max DD: {max_dd*100:.1f}%  |  Total Return: {sum(trades)*100:.1f}%  |  Avg: {avg*100:.3f}%\n")
    
    print("=== WALK-FORWARD ===")
    wf = walk_forward(d)
    for w in wf:
        s = "✅" if w.get("pass") else "❌"
        print(f"  W{w['w']}: {w['trades']}T, WR={w.get('wr',0)*100:.1f}%, Sh={w.get('sharpe',0):.3f}, PF={w.get('pf',0):.2f} {s}")
    wfp = sum(1 for w in wf if w.get("pass"))
    print(f"  {wfp}/{len(wf)} pass\n")
    
    print("=== MONTE CARLO ===")
    mc = monte_carlo(trades)
    print(f"  Sharpe: med={mc['sharpe_med']}, [{mc['sharpe_5']}, {mc['sharpe_95']}]")
    print(f"  DD: med={mc['dd_med']*100:.1f}%, 99th={mc['dd_99']*100:.1f}%")
    print(f"  Survival: {mc['survival']*100:.1f}%\n")
    
    ok = wr>=0.45 and sharpe>=0.5 and pf>=1.5 and max_dd<=0.20
    print(f"{'✅ ACCEPTED' if ok else '❌ REJECTED'}")
    print(f"  WR {wr*100:.1f}% {'✅' if wr>=0.45 else '❌'}  Sharpe {sharpe:.3f} {'✅' if sharpe>=0.5 else '❌'}  PF {pf:.2f} {'✅' if pf>=1.5 else '❌'}  DD {max_dd*100:.1f}% {'✅' if max_dd<=0.20 else '❌'}")

if __name__ == "__main__":
    main()
