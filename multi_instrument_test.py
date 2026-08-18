#!/usr/bin/env python3
"""
Multi-Instrument Strategy Test on Real LSE Data
Option A: Equities (SPY, QQQ, NVDA, AMD, AAPL, MSFT) — daily timeframe
Option D: Forex (EUR/USD, GBP/USD, USD/JPY, AUD/USD, NZD/USD) — daily timeframe
All with Donchian, EMA, Momentum, Bollinger, RSI strategies.
"""

import os
import sys
import json
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

# ─── Config ──────────────────────────────────────────────────────────────────

def load_env():
    for p in [Path("backtest/.env"), Path(".env")]:
        if p.exists():
            env = {}
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
            if "LSE_API_KEY" in env:
                return env
    sys.exit(1)

ENV = load_env()
API_KEY = ENV["LSE_API_KEY"]
BASE_URL = "https://api.londonstrategicedge.com/vault/candles"
COST_RT = 0.002  # 0.1% round-trip

# ─── Data ────────────────────────────────────────────────────────────────────

def fetch_lse(symbol, tf="1d", start="2015-01-01", end="2025-06-16"):
    """Paginated fetch."""
    headers = {"x-api-key": API_KEY}
    all_candles = []
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    cur_start = start
    seen_ts = set()

    for page in range(20):
        params = {"symbol": symbol, "timeframe": tf, "start": cur_start}
        try:
            r = requests.get(BASE_URL, headers=headers, params=params, timeout=60)
            if r.status_code != 200:
                break
            data = r.json()
            if not isinstance(data, list) or len(data) == 0:
                break
        except:
            break

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

        if new_count == 0:
            break
        if last_ts:
            cur_start = (datetime.strptime(last_ts[:10], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            if datetime.strptime(last_ts[:10], "%Y-%m-%d") >= end_dt:
                break
        if len(data) < 4900:
            break

    if not all_candles:
        return None

    all_candles.sort(key=lambda c: c["ts"])
    n = len(all_candles)
    o = np.zeros(n); h = np.zeros(n); l = np.zeros(n); c = np.zeros(n)
    times = []
    for i, candle in enumerate(all_candles):
        times.append(candle["ts"])
        o[i] = float(candle["open"])
        h[i] = float(candle["high"])
        l[i] = float(candle["low"])
        c[i] = float(candle["close"])
    return {"o": o, "h": h, "l": l, "c": c, "n": n, "times": times}

# ─── Indicators ──────────────────────────────────────────────────────────────

def rolling_max(arr, w):
    out = np.full(len(arr), np.nan)
    for i in range(w - 1, len(arr)):
        out[i] = np.max(arr[i - w + 1:i + 1])
    return out

def rolling_min(arr, w):
    out = np.full(len(arr), np.nan)
    for i in range(w - 1, len(arr)):
        out[i] = np.min(arr[i - w + 1:i + 1])
    return out

def rolling_mean(arr, w):
    out = np.full(len(arr), np.nan)
    for i in range(w - 1, len(arr)):
        out[i] = np.mean(arr[i - w + 1:i + 1])
    return out

def rolling_std(arr, w):
    out = np.full(len(arr), np.nan)
    for i in range(w - 1, len(arr)):
        out[i] = np.std(arr[i - w + 1:i + 1], ddof=0)
    return out

def ema(arr, span):
    out = np.full(len(arr), np.nan)
    k = 2.0 / (span + 1)
    out[span - 1] = np.mean(arr[:span])
    for i in range(span, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out

# ─── Strategies ──────────────────────────────────────────────────────────────

def strategy_donchian(data, lookback=20, rr=2.0, stop_mult=1.5):
    """Donchian breakout with trailing stop."""
    h, l, c = data["h"], data["l"], data["c"]
    n = data["n"]
    high_n = rolling_max(h, lookback)
    low_n = rolling_min(l, lookback)
    atr = rolling_mean(h - l, 14)

    trades = []
    pos = 0; entry = stop = target = 0.0

    for i in range(max(lookback, 14), n):
        price = c[i]
        hn, ln, a = high_n[i], low_n[i], atr[i]
        if np.isnan(hn) or np.isnan(a) or a == 0:
            continue

        if pos == 0:
            if price >= hn:
                pos = 1; entry = price; stop = price - stop_mult * a; target = price + rr * stop_mult * a
            elif price <= ln:
                pos = -1; entry = price; stop = price + stop_mult * a; target = price - rr * stop_mult * a
        elif pos == 1:
            new_stop = price - stop_mult * a
            if new_stop > stop: stop = new_stop
            if price <= stop:
                trades.append((stop - entry) / entry - COST_RT); pos = 0
            elif price >= target:
                trades.append((target - entry) / entry - COST_RT); pos = 0
        elif pos == -1:
            new_stop = price + stop_mult * a
            if new_stop < stop: stop = new_stop
            if price >= stop:
                trades.append((entry - stop) / entry - COST_RT); pos = 0
            elif price <= target:
                trades.append((entry - target) / entry - COST_RT); pos = 0

    return trades

def strategy_ema_crossover(data, fast=10, slow=30):
    """EMA crossover with trailing stop."""
    c = data["c"]
    h = data["h"]
    l = data["l"]
    n = data["n"]
    ema_fast = ema(c, fast)
    ema_slow = ema(c, slow)
    atr = rolling_mean(h - l, 14)

    trades = []
    pos = 0; entry = stop = 0.0

    for i in range(max(slow, 14), n):
        price = c[i]
        a = atr[i]
        if np.isnan(ema_fast[i]) or np.isnan(a) or a == 0:
            continue

        if pos == 0:
            if ema_fast[i] > ema_slow[i] and ema_fast[i - 1] <= ema_slow[i - 1]:
                pos = 1; entry = price; stop = price - 2.0 * a
            elif ema_fast[i] < ema_slow[i] and ema_fast[i - 1] >= ema_slow[i - 1]:
                pos = -1; entry = price; stop = price + 2.0 * a
        elif pos == 1:
            new_stop = price - 2.0 * a
            if new_stop > stop: stop = new_stop
            if price <= stop:
                trades.append((stop - entry) / entry - COST_RT); pos = 0
        elif pos == -1:
            new_stop = price + 2.0 * a
            if new_stop < stop: stop = new_stop
            if price >= stop:
                trades.append((entry - stop) / entry - COST_RT); pos = 0

    return trades

def strategy_momentum(data, lookback=20):
    """Momentum: long if N-bar return > 0, short if < 0, with trailing stop."""
    c = data["c"]
    h = data["h"]
    l = data["l"]
    n = data["n"]
    atr = rolling_mean(h - l, 14)

    trades = []
    pos = 0; entry = stop = 0.0

    for i in range(max(lookback, 14), n):
        price = c[i]
        a = atr[i]
        if np.isnan(a) or a == 0:
            continue
        ret = (c[i] - c[i - lookback]) / c[i - lookback]

        if pos == 0:
            if ret > 0:
                pos = 1; entry = price; stop = price - 2.0 * a
            elif ret < 0:
                pos = -1; entry = price; stop = price + 2.0 * a
        elif pos == 1:
            new_stop = price - 2.0 * a
            if new_stop > stop: stop = new_stop
            if price <= stop:
                trades.append((stop - entry) / entry - COST_RT); pos = 0
            elif ret < 0:
                trades.append((price - entry) / entry - COST_RT); pos = 0
        elif pos == -1:
            new_stop = price + 2.0 * a
            if new_stop < stop: stop = new_stop
            if price >= stop:
                trades.append((entry - stop) / entry - COST_RT); pos = 0
            elif ret > 0:
                trades.append((entry - price) / entry - COST_RT); pos = 0

    return trades

def strategy_bollinger(data, lookback=20, bb_mult=2.0):
    """Bollinger band mean reversion."""
    c = data["c"]
    ma = rolling_mean(c, lookback)
    sd = rolling_std(c, lookback)
    upper = ma + bb_mult * sd
    lower = ma - bb_mult * sd
    atr = rolling_mean(data["h"] - data["l"], 14)

    trades = []
    pos = 0; entry = stop = 0.0

    for i in range(max(lookback, 14), len(c)):
        price = c[i]
        if np.isnan(upper[i]) or np.isnan(atr[i]) or atr[i] == 0:
            continue

        if pos == 0:
            if price <= lower[i]:
                pos = 1; entry = price; stop = price - 2.0 * atr[i]
            elif price >= upper[i]:
                pos = -1; entry = price; stop = price + 2.0 * atr[i]
        elif pos == 1:
            if price >= ma[i]:
                trades.append((price - entry) / entry - COST_RT); pos = 0
            elif price <= stop:
                trades.append((stop - entry) / entry - COST_RT); pos = 0
        elif pos == -1:
            if price <= ma[i]:
                trades.append((entry - price) / entry - COST_RT); pos = 0
            elif price >= stop:
                trades.append((entry - stop) / entry - COST_RT); pos = 0

    return trades

def strategy_rsi(data, period=14, oversold=30, overbought=70):
    """RSI mean reversion with trailing stop."""
    c = data["c"]
    n = len(c)
    h = data["h"]
    l = data["l"]
    atr = rolling_mean(h - l, 14)

    deltas = np.diff(c)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    rsi_arr = np.full(n, np.nan)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi_arr[i + 1] = 100
        else:
            rs = avg_gain / avg_loss
            rsi_arr[i + 1] = 100 - 100 / (1 + rs)

    trades = []
    pos = 0; entry = stop = 0.0

    for i in range(period + 14, n):
        price = c[i]
        a = atr[i]
        if np.isnan(rsi_arr[i]) or np.isnan(a) or a == 0:
            continue

        if pos == 0:
            if rsi_arr[i] < oversold:
                pos = 1; entry = price; stop = price - 2.0 * a
            elif rsi_arr[i] > overbought:
                pos = -1; entry = price; stop = price + 2.0 * a
        elif pos == 1:
            if rsi_arr[i] > 50:
                trades.append((price - entry) / entry - COST_RT); pos = 0
            elif price <= stop:
                trades.append((stop - entry) / entry - COST_RT); pos = 0
        elif pos == -1:
            if rsi_arr[i] < 50:
                trades.append((entry - price) / entry - COST_RT); pos = 0
            elif price >= stop:
                trades.append((entry - stop) / entry - COST_RT); pos = 0

    return trades

# ─── Analysis ────────────────────────────────────────────────────────────────

def analyze(trades):
    if len(trades) < 5:
        return None
    wins = sum(1 for t in trades if t > 0)
    wr = wins / len(trades)
    avg, std = np.mean(trades), np.std(trades)
    sharpe = (avg / std) * np.sqrt(252) if std > 0 else 0
    gp = sum(t for t in trades if t > 0)
    gl = abs(sum(t for t in trades if t < 0))
    pf = gp / gl if gl > 0 else float("inf")
    cum = np.cumsum(trades)
    pk = np.maximum.accumulate(cum)
    max_dd = (pk - cum).max() if len(cum) > 0 else 0
    return {
        "trades": len(trades), "wr": round(wr * 100, 1),
        "pf": round(pf, 2), "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd * 100, 1), "ret": round(sum(trades) * 100, 1)
    }

def walk_forward(data, strategy_fn, n_windows=3, **kwargs):
    """Walk-forward validation."""
    n = data["n"]
    step = n // n_windows
    results = []

    for i in range(n_windows):
        s = i * step
        e = min((i + 1) * step + 50, n)
        wd = {k: data[k][s:e] if isinstance(data[k], np.ndarray) else data[k][s:e]
              for k in ["o", "h", "l", "c"]}
        wd["n"] = e - s
        wd["times"] = data["times"][s:e]

        if wd["n"] < 50:
            results.append({"w": i + 1, "pass": False})
            continue

        trades = strategy_fn(wd, **kwargs)
        a = analyze(trades)
        if a is None:
            results.append({"w": i + 1, "trades": len(trades), "pass": False})
            continue

        passed = a["sharpe"] > 0 and a["pf"] >= 1.2
        results.append({"w": i + 1, **a, "pass": passed})

    passes = sum(1 for r in results if r.get("pass"))
    return results, passes

# ─── Main ────────────────────────────────────────────────────────────────────

def test_instrument(symbol, tf, start, end, label):
    """Test all strategies on one instrument."""
    print(f"\n{'='*70}")
    print(f"  {label} ({symbol}, {tf})")
    print(f"{'='*70}")

    data = fetch_lse(symbol, tf, start, end)
    if data is None:
        print(f"  ❌ No data available")
        return []

    print(f"  {data['n']} bars ({data['times'][0][:10]} → {data['times'][-1][:10]})")

    strategies = [
        ("Donchian(10,rr=2)", strategy_donchian, {"lookback": 10, "rr": 2.0, "stop_mult": 1.5}),
        ("Donchian(20,rr=2)", strategy_donchian, {"lookback": 20, "rr": 2.0, "stop_mult": 1.5}),
        ("Donchian(20,rr=3)", strategy_donchian, {"lookback": 20, "rr": 3.0, "stop_mult": 2.0}),
        ("Donchian(50,rr=3)", strategy_donchian, {"lookback": 50, "rr": 3.0, "stop_mult": 2.0}),
        ("EMA(10/30)", strategy_ema_crossover, {"fast": 10, "slow": 30}),
        ("EMA(20/50)", strategy_ema_crossover, {"fast": 20, "slow": 50}),
        ("Momentum(20)", strategy_momentum, {"lookback": 20}),
        ("Momentum(50)", strategy_momentum, {"lookback": 50}),
        ("Bollinger(20,2.0)", strategy_bollinger, {"lookback": 20, "bb_mult": 2.0}),
        ("RSI(14,30/70)", strategy_rsi, {"period": 14, "oversold": 30, "overbought": 70}),
    ]

    results = []
    for name, fn, kwargs in strategies:
        trades = fn(data, **kwargs)
        a = analyze(trades)
        if a is None:
            continue

        wf_results, wf_passes = walk_forward(data, fn, **kwargs)

        entry = {"instrument": label, "symbol": symbol, "strategy": name, **a,
                 "wf_passes": wf_passes, "wf_total": len(wf_results)}
        results.append(entry)

        s = "✅" if a["sharpe"] > 0.5 and a["pf"] >= 1.3 and wf_passes >= 2 else "  "
        print(f"  {s} {name:<22} T={a['trades']:>3}  WR={a['wr']:>5}%  PF={a['pf']:>5}  Sh={a['sharpe']:>7}  DD={a['max_dd']:>5}%  Ret={a['ret']:>6}%  WF={wf_passes}/{len(wf_results)}")

    return results

def main():
    print("=" * 70)
    print("  MULTI-INSTRUMENT STRATEGY TEST — REAL LSE DATA")
    print("  Testing equities (daily) + forex pairs (daily)")
    print("=" * 70)

    all_results = []

    # ─── Option A: Equities ───────────────────────────────────────────────────
    equities = [
        ("SPY", "SPY"),
        ("QQQ", "QQQ"),
        ("NVDA", "NVDA"),
        ("AMD", "AMD"),
        ("AAPL", "AAPL"),
        ("MSFT", "MSFT"),
    ]

    print("\n" + "=" * 70)
    print("  OPTION A: EQUITIES (Daily, 2015-2025)")
    print("=" * 70)

    for symbol, label in equities:
        results = test_instrument(symbol, "1d", "2015-01-01", "2025-06-16", label)
        all_results.extend(results)

    # ─── Option D: Forex ──────────────────────────────────────────────────────
    forex = [
        ("EUR/USD", "EUR/USD"),
        ("GBP/USD", "GBP/USD"),
        ("USD/JPY", "USD/JPY"),
        ("AUD/USD", "AUD/USD"),
        ("NZD/USD", "NZD/USD"),
    ]

    print("\n" + "=" * 70)
    print("  OPTION D: FOREX PAIRS (Daily, 2015-2025)")
    print("=" * 70)

    for symbol, label in forex:
        results = test_instrument(symbol, "1d", "2015-01-01", "2025-06-16", label)
        all_results.extend(results)

    # ─── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  GLOBAL SUMMARY — ALL INSTRUMENTS × ALL STRATEGIES")
    print("=" * 70)

    # Sort by Sharpe
    all_results.sort(key=lambda x: x["sharpe"], reverse=True)

    print(f"\n{'Instrument':<12} {'Strategy':<22} {'T':>4} {'WR%':>5} {'PF':>5} {'Sharpe':>7} {'DD%':>5} {'Ret%':>6} {'WF':>4}")
    print("-" * 75)

    for r in all_results[:30]:
        wf_s = f"{r['wf_passes']}/{r['wf_total']}"
        star = ""
        if r["sharpe"] > 1.0 and r["pf"] >= 1.3 and r["wf_passes"] >= 2:
            star = " ⭐"
        elif r["sharpe"] > 0.5 and r["pf"] >= 1.2:
            star = " ✅"
        print(f"{r['instrument']:<12} {r['strategy']:<22} {r['trades']:>4} {r['wr']:>5} {r['pf']:>5} {r['sharpe']:>7} {r['max_dd']:>5} {r['ret']:>6} {wf_s:>4}{star}")

    # Winners
    winners = [r for r in all_results if r["sharpe"] > 0.5 and r["pf"] >= 1.2 and r["wf_passes"] >= 2]
    strong = [r for r in all_results if r["sharpe"] > 1.0 and r["pf"] >= 1.3 and r["wf_passes"] >= 2]

    print()
    if strong:
        print(f"⭐ STRONG SIGNALS ({len(strong)}):")
        for r in strong:
            print(f"  {r['instrument']} / {r['strategy']}: Sharpe={r['sharpe']}, PF={r['pf']}, WR={r['wr']}%, Ret={r['ret']}%, WF={r['wf_passes']}/{r['wf_total']}")
    elif winners:
        print(f"✅ CANDIDATES ({len(winners)}):")
        for r in winners:
            print(f"  {r['instrument']} / {r['strategy']}: Sharpe={r['sharpe']}, PF={r['pf']}, WR={r['wr']}%, Ret={r['ret']}%, WF={r['wf_passes']}/{r['wf_total']}")
    else:
        print("❌ NO STRONG EDGES FOUND across any instrument/strategy combination")
        print("\n  Best candidates (by Sharpe):")
        for r in all_results[:5]:
            print(f"  {r['instrument']} / {r['strategy']}: Sharpe={r['sharpe']}, PF={r['pf']}, WF={r['wf_passes']}/{r['wf_total']}")

    # Save results
    out = [r for r in all_results if r["sharpe"] > -999]
    print(f"\nTotal combinations tested: {len(all_results)}")

if __name__ == "__main__":
    main()
