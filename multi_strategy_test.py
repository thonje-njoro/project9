#!/usr/bin/env python3
"""
Multi-Strategy Backtest on Real LSE Data
Tests: Mean Reversion, Momentum, EMA Trend, Donchian variants, Bollinger, RSI
All on XAU/USD hourly data from LSE API.
"""

import os
import sys
import json
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path

# Load API key
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
    print("ERROR: No .env with LSE_API_KEY found")
    sys.exit(1)

ENV = load_env()
API_KEY = ENV["LSE_API_KEY"]
BASE_URL = "https://api.londonstrategicedge.com/vault/candles"
COST_RT = 0.002  # 0.1% commission + 0.1% slippage

# ─── Data Fetching ───────────────────────────────────────────────────────────

def fetch_lse(symbol="XAU/USD", tf="1h", start="2020-01-01", end="2025-06-16"):
    """Paginated fetch using 'start' parameter."""
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
        except Exception as e:
            print(f"  Fetch page {page} failed: {e}")
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

        print(f"  Page {page}: {len(data)} bars ({data[0]['ts'][:10]}→{data[-1]['ts'][:10]}), total={len(all_candles)}")
        if new_count == 0:
            break
        if last_ts:
            cur_start = (datetime.strptime(last_ts[:10], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            if datetime.strptime(last_ts[:10], "%Y-%m-%d") >= end_dt:
                break
        if len(data) < 4900:
            break

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

def rolling(arr, w, fn):
    out = np.full(len(arr), np.nan)
    for i in range(w - 1, len(arr)):
        out[i] = fn(arr[i - w + 1:i + 1])
    return out

def ema(arr, span):
    out = np.full(len(arr), np.nan)
    k = 2.0 / (span + 1)
    out[span - 1] = np.mean(arr[:span])
    for i in range(span, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out

def std_dev(arr, w):
    out = np.full(len(arr), np.nan)
    for i in range(w - 1, len(arr)):
        out[i] = np.std(arr[i - w + 1:i + 1], ddof=0)
    return out

# ─── Strategy: helpers ───────────────────────────────────────────────────────

def execute_trades(signals, closes, cost=COST_RT):
    """Given entry/exit signals, compute trade PnLs.
    signals: array of +1 (long entry), -1 (short entry), 0 (flat/no signal)
    Uses next-bar execution (shift by 1).
    """
    trades = []
    pos = 0
    entry_price = 0.0

    for i in range(1, len(closes)):
        sig = signals[i - 1]  # signal from previous bar, execute on this bar
        price = closes[i]

        if pos == 0 and sig == 1:
            pos = 1
            entry_price = price
        elif pos == 0 and sig == -1:
            pos = -1
            entry_price = price
        elif pos == 1 and sig == -1:
            pnl = (price - entry_price) / entry_price - cost
            trades.append(pnl)
            pos = -1
            entry_price = price
        elif pos == -1 and sig == 1:
            pnl = (entry_price - price) / entry_price - cost
            trades.append(pnl)
            pos = 1
            entry_price = price
        elif pos == 1 and sig == 0:
            pnl = (price - entry_price) / entry_price - cost
            trades.append(pnl)
            pos = 0
        elif pos == -1 and sig == 0:
            pnl = (entry_price - price) / entry_price - cost
            trades.append(pnl)
            pos = 0

    return trades

def trailing_stop_trades(data, entry_fn, stop_fn, exit_fn, cost=COST_RT):
    """Generic trailing stop backtest. Functions take index and return price or None."""
    c = data["c"]
    n = data["n"]
    trades = []
    pos = 0
    entry_price = 0.0
    stop_price = 0.0

    for i in range(1, n):
        price = c[i]

        if pos == 0:
            entry_signal = entry_fn(i)
            if entry_signal == 1:
                pos = 1
                entry_price = price
                stop_price = stop_fn(i, price, "long")
            elif entry_signal == -1:
                pos = -1
                entry_price = price
                stop_price = stop_fn(i, price, "short")
        elif pos == 1:
            new_stop = stop_fn(i, price, "long")
            if new_stop is not None and new_stop > stop_price:
                stop_price = new_stop
            if price <= stop_price:
                trades.append((stop_price - entry_price) / entry_price - cost)
                pos = 0
            elif exit_fn(i, price, "long"):
                trades.append((price - entry_price) / entry_price - cost)
                pos = 0
        elif pos == -1:
            new_stop = stop_fn(i, price, "short")
            if new_stop is not None and new_stop < stop_price:
                stop_price = new_stop
            if price >= stop_price:
                trades.append((entry_price - stop_price) / entry_price - cost)
                pos = 0
            elif exit_fn(i, price, "short"):
                trades.append((entry_price - price) / entry_price - cost)
                pos = 0

    return trades

# ─── Strategies ──────────────────────────────────────────────────────────────

def strategy_zscore_mr(data, z_entry=2.0, z_exit=0.5, lookback=20):
    """Z-Score Mean Reversion: enter when z-score exceeds threshold, exit when reverts."""
    c = data["c"]
    ma = ema(c, lookback)
    sd = std_dev(c, lookback)
    z = (c - ma) / sd
    signals = np.zeros(len(c))
    for i in range(lookback, len(c)):
        if np.isnan(z[i]):
            continue
        if z[i] < -z_entry:
            signals[i] = 1   # oversold → long
        elif z[i] > z_entry:
            signals[i] = -1  # overbought → short
        elif abs(z[i]) < z_exit:
            signals[i] = 0   # exit
    return execute_trades(signals, c)

def strategy_ema_crossover(data, fast=12, slow=26):
    """EMA Crossover: long when fast > slow, short when fast < slow."""
    c = data["c"]
    ema_fast = ema(c, fast)
    ema_slow = ema(c, slow)
    signals = np.zeros(len(c))
    for i in range(slow, len(c)):
        if ema_fast[i] > ema_slow[i]:
            signals[i] = 1
        elif ema_fast[i] < ema_slow[i]:
            signals[i] = -1
    return execute_trades(signals, c)

def strategy_donchian(data, lookback=20, rr=2.0, stop_mult=1.5):
    """Donchian Channel Breakout with trailing stop."""
    h, l, c_arr = data["h"], data["l"], data["c"]
    high_n = rolling(h, lookback, np.max)
    low_n = rolling(l, lookback, np.min)
    atr = rolling(h - l, 14, np.mean)

    def entry_fn(i):
        if np.isnan(high_n[i]) or np.isnan(atr[i]):
            return 0
        if c_arr[i] >= high_n[i]:
            return 1
        elif c_arr[i] <= low_n[i]:
            return -1
        return 0

    def stop_fn(i, price, side):
        if np.isnan(atr[i]):
            return None
        a = atr[i]
        if side == "long":
            return price - stop_mult * a
        else:
            return price + stop_mult * a

    def exit_fn(i, price, side):
        return False  # only exit on stop

    return trailing_stop_trades(data, entry_fn, stop_fn, exit_fn)

def strategy_bollinger(data, lookback=20, bb_mult=2.0):
    """Bollinger Band Mean Reversion: buy at lower band, sell at upper band."""
    c = data["c"]
    ma = rolling(c, lookback, np.mean)
    sd = std_dev(c, lookback)
    upper = ma + bb_mult * sd
    lower = ma - bb_mult * sd
    signals = np.zeros(len(c))
    pos = 0
    for i in range(lookback, len(c)):
        if np.isnan(upper[i]):
            continue
        if c[i] <= lower[i] and pos <= 0:
            signals[i] = 1
            pos = 1
        elif c[i] >= upper[i] and pos >= 0:
            signals[i] = -1
            pos = -1
        elif pos == 1 and c[i] >= ma[i]:
            signals[i] = 0
            pos = 0
        elif pos == -1 and c[i] <= ma[i]:
            signals[i] = 0
            pos = 0
    return execute_trades(signals, c)

def strategy_rsi(data, period=14, oversold=30, overbought=70):
    """RSI Mean Reversion."""
    c = data["c"]
    n = len(c)
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

    signals = np.zeros(n)
    for i in range(period + 1, n):
        if np.isnan(rsi_arr[i]):
            continue
        if rsi_arr[i] < oversold:
            signals[i] = 1
        elif rsi_arr[i] > overbought:
            signals[i] = -1
    return execute_trades(signals, c)

def strategy_momentum(data, lookback=20, threshold=0.02):
    """Simple momentum: long if return over lookback > threshold, short if < -threshold."""
    c = data["c"]
    ret = np.full(len(c), np.nan)
    for i in range(lookback, len(c)):
        ret[i] = (c[i] - c[i - lookback]) / c[i - lookback]
    signals = np.zeros(len(c))
    for i in range(lookback, len(c)):
        if np.isnan(ret[i]):
            continue
        if ret[i] > threshold:
            signals[i] = 1
        elif ret[i] < -threshold:
            signals[i] = -1
    return execute_trades(signals, c)

def strategy_session_mr(data, session_start=7, session_end=20, z_entry=2.0, z_exit=0.5, max_hold=12, atr_trail=2.0):
    """Session Mean Reversion (from xauusd_session_mr): z-score within trading session."""
    c = data["c"]
    h = data["h"]
    l = data["l"]
    times = data["times"]
    n = len(c)

    # Compute hourly returns and session stats
    returns = np.diff(c) / c[:-1]
    returns = np.insert(returns, 0, 0)

    # Session z-score using rolling window
    lookback = 50
    ma = ema(c, lookback)
    sd = std_dev(c, lookback)
    z = (c - ma) / sd
    atr = rolling(h - l, 14, np.mean)

    def entry_fn(i):
        if np.isnan(z[i]) or np.isnan(atr[i]):
            return 0
        # Check if within session hours (approximate from timestamp)
        try:
            hour = int(times[i][11:13])
        except:
            return 0
        if hour < session_start or hour >= session_end:
            return 0
        if z[i] < -z_entry:
            return 1
        elif z[i] > z_entry:
            return -1
        return 0

    def stop_fn(i, price, side):
        if np.isnan(atr[i]):
            return None
        if side == "long":
            return price - atr_trail * atr[i]
        else:
            return price + atr_trail * atr[i]

    def exit_fn(i, price, side):
        if np.isnan(z[i]):
            return False
        return abs(z[i]) < z_exit

    return trailing_stop_trades(data, entry_fn, stop_fn, exit_fn)

def strategy_nr7_breakout(data):
    """NR7 Breakout: enter on range contraction then directional break."""
    h, l, c_arr = data["h"], data["l"], data["c"]
    ranges = h - l
    atr = rolling(ranges, 14, np.mean)

    def entry_fn(i):
        if i < 7 or np.isnan(atr[i]):
            return 0
        # NR7: today's range is smallest of last 7 days
        recent_ranges = ranges[i - 6:i + 1]
        if ranges[i] != np.min(recent_ranges):
            return 0
        # Breakout direction
        if c_arr[i] > h[i - 1]:
            return 1
        elif c_arr[i] < l[i - 1]:
            return -1
        return 0

    def stop_fn(i, price, side):
        if np.isnan(atr[i]):
            return None
        if side == "long":
            return price - 1.5 * atr[i]
        else:
            return price + 1.5 * atr[i]

    def exit_fn(i, price, side):
        return False

    return trailing_stop_trades(data, entry_fn, stop_fn, exit_fn)

# ─── Analysis ────────────────────────────────────────────────────────────────

def analyze_trades(trades, name):
    if len(trades) < 3:
        return {"name": name, "trades": len(trades), "pass": False, "reason": "too few trades"}

    wins = sum(1 for t in trades if t > 0)
    wr = wins / len(trades)
    avg = np.mean(trades)
    std = np.std(trades)
    sharpe = (avg / std) * np.sqrt(252) if std > 0 else 0
    gp = sum(t for t in trades if t > 0)
    gl = abs(sum(t for t in trades if t < 0))
    pf = gp / gl if gl > 0 else float("inf")

    cum = np.cumsum(trades)
    pk = np.maximum.accumulate(cum)
    max_dd = (pk - cum).max()

    passed = wr >= 0.45 and sharpe >= 0.5 and pf >= 1.5 and max_dd <= 0.20

    return {
        "name": name,
        "trades": len(trades),
        "wr": round(wr * 100, 1),
        "pf": round(pf, 2),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd * 100, 1),
        "total_ret": round(sum(trades) * 100, 1),
        "avg_trade": round(avg * 100, 3),
        "pass": passed
    }

def walk_forward(data, strategy_fn, n_windows=3, **kwargs):
    """Walk-forward validation."""
    n = data["n"]
    step = n // n_windows
    results = []

    for i in range(n_windows):
        s = i * step
        e = min((i + 1) * step + 100, n)
        wd = {k: data[k][s:e] if isinstance(data[k], np.ndarray) else data[k][s:e]
              for k in ["o", "h", "l", "c"]}
        wd["n"] = e - s
        wd["times"] = data["times"][s:e]

        if wd["n"] < 100:
            results.append({"w": i + 1, "trades": 0, "pass": False})
            continue

        trades = strategy_fn(wd, **kwargs)
        if len(trades) < 3:
            results.append({"w": i + 1, "trades": len(trades), "pass": False})
            continue

        wins = sum(1 for t in trades if t > 0)
        wr = wins / len(trades)
        avg, std = np.mean(trades), np.std(trades)
        sharpe = (avg / std) * np.sqrt(252) if std > 0 else 0
        gp = sum(t for t in trades if t > 0)
        gl = abs(sum(t for t in trades if t < 0))
        pf = gp / gl if gl > 0 else float("inf")

        results.append({
            "w": i + 1, "trades": len(trades), "wr": round(wr, 3),
            "sharpe": round(sharpe, 3), "pf": round(pf, 3),
            "pass": wr >= 0.45 and sharpe >= 0.5 and pf >= 1.5
        })

    passes = sum(1 for r in results if r.get("pass"))
    return results, passes

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  MULTI-STRATEGY BACKTEST — REAL LSE DATA (XAU/USD 1h)")
    print("=" * 70)
    print()

    # Fetch data
    print("Fetching XAU/USD 1h data from LSE API...")
    data = fetch_lse("XAU/USD", "1h", "2020-01-01", "2025-06-16")
    print(f"Total: {data['n']} bars ({data['times'][0][:10]} → {data['times'][-1][:10]})")
    print()

    # Define strategies to test
    strategies = [
        ("ZScore MR (z=2.0, lb=20)", strategy_zscore_mr, {"z_entry": 2.0, "z_exit": 0.5, "lookback": 20}),
        ("ZScore MR (z=1.5, lb=30)", strategy_zscore_mr, {"z_entry": 1.5, "z_exit": 0.3, "lookback": 30}),
        ("ZScore MR (z=2.5, lb=50)", strategy_zscore_mr, {"z_entry": 2.5, "z_exit": 0.5, "lookback": 50}),
        ("EMA Cross (12/26)", strategy_ema_crossover, {"fast": 12, "slow": 26}),
        ("EMA Cross (8/21)", strategy_ema_crossover, {"fast": 8, "slow": 21}),
        ("EMA Cross (5/13)", strategy_ema_crossover, {"fast": 5, "slow": 13}),
        ("Donchian (20, R:R=2)", strategy_donchian, {"lookback": 20, "rr": 2.0, "stop_mult": 1.5}),
        ("Donchian (50, R:R=3)", strategy_donchian, {"lookback": 50, "rr": 3.0, "stop_mult": 2.0}),
        ("Donchian (10, R:R=2)", strategy_donchian, {"lookback": 10, "rr": 2.0, "stop_mult": 1.5}),
        ("Bollinger (20, 2.0)", strategy_bollinger, {"lookback": 20, "bb_mult": 2.0}),
        ("Bollinger (20, 2.5)", strategy_bollinger, {"lookback": 20, "bb_mult": 2.5}),
        ("RSI (14, 30/70)", strategy_rsi, {"period": 14, "oversold": 30, "overbought": 70}),
        ("RSI (14, 25/75)", strategy_rsi, {"period": 14, "oversold": 25, "overbought": 75}),
        ("Momentum (20, 2%)", strategy_momentum, {"lookback": 20, "threshold": 0.02}),
        ("Momentum (50, 3%)", strategy_momentum, {"lookback": 50, "threshold": 0.03}),
        ("Session MR (z=2.0)", strategy_session_mr, {"session_start": 7, "session_end": 20, "z_entry": 2.0, "z_exit": 0.5}),
        ("NR7 Breakout", strategy_nr7_breakout, {}),
    ]

    results = []
    for name, fn, kwargs in strategies:
        print(f"--- {name} ---")
        trades = fn(data, **kwargs)
        r = analyze_trades(trades, name)
        results.append(r)

        # Walk-forward
        wf_results, wf_passes = walk_forward(data, fn, **kwargs)

        status = "✅" if r["pass"] else "❌"
        print(f"  Trades={r['trades']}  WR={r['wr']}%  PF={r['pf']}  Sharpe={r['sharpe']}  DD={r['max_dd']}%  Ret={r['total_ret']}%  WF={wf_passes}/3  {status}")

        for wf in wf_results:
            ws = "✅" if wf.get("pass") else "❌"
            print(f"    W{wf['w']}: {wf['trades']}T, WR={wf.get('wr', 0) * 100:.1f}%, Sh={wf.get('sharpe', 0):.2f}, PF={wf.get('pf', 0):.2f} {ws}")
        print()

    # Summary table
    print("=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Strategy':<30} {'T':>4} {'WR%':>5} {'PF':>5} {'Sh':>7} {'DD%':>5} {'Ret%':>6} {'WF':>3} {'':>2}")
    print("-" * 70)
    for r in results:
        s = "✅" if r["pass"] else "❌"
        print(f"{r['name']:<30} {r['trades']:>4} {r['wr']:>5} {r['pf']:>5} {r['sharpe']:>7} {r['max_dd']:>5} {r['total_ret']:>6} {'':>3} {s}")

    # Winners
    winners = [r for r in results if r["pass"]]
    print()
    if winners:
        print(f"✅ PASSED: {len(winners)} strategies")
        for r in winners:
            print(f"  - {r['name']}: WR={r['wr']}%, PF={r['pf']}, Sharpe={r['sharpe']}")
    else:
        print("❌ NO STRATEGY PASSED acceptance criteria")
        print("  Best candidates (by Sharpe):")
        sorted_r = sorted(results, key=lambda x: x.get("sharpe", -999), reverse=True)
        for r in sorted_r[:3]:
            print(f"  - {r['name']}: WR={r['wr']}%, PF={r['pf']}, Sharpe={r['sharpe']}")

if __name__ == "__main__":
    main()
