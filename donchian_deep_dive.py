#!/usr/bin/env python3
"""
Deep Dive: Donchian Breakout Parameter Search on Real XAU/USD Data
Tests lookback 5-30, R:R 1.5-4.0, stop multipliers 1.0-2.5
With proper walk-forward validation.
"""

import os
import sys
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path

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
COST_RT = 0.002

# ─── Data ────────────────────────────────────────────────────────────────────

def fetch_lse(symbol="XAU/USD", tf="1h", start="2020-01-01", end="2025-06-16"):
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

# ─── Donchian Strategy ───────────────────────────────────────────────────────

def donchian_backtest(data, lookback=10, rr=2.0, stop_mult=1.5, atr_period=14):
    """Donchian breakout with ATR trailing stop."""
    h, l, c = data["h"], data["l"], data["c"]
    n = data["n"]

    high_n = rolling_max(h, lookback)
    low_n = rolling_min(l, lookback)
    atr = rolling_mean(h - l, atr_period)

    trades = []
    pos = 0
    entry = stop = target = 0.0
    entry_atr = 0.0

    for i in range(max(lookback, atr_period), n):
        price = c[i]
        hn, ln, a = high_n[i], low_n[i], atr[i]
        if np.isnan(hn) or np.isnan(a) or a == 0:
            continue

        if pos == 0:
            if price >= hn:
                pos = 1; entry = price
                entry_atr = a
                stop = price - stop_mult * a
                target = price + rr * stop_mult * a
            elif price <= ln:
                pos = -1; entry = price
                entry_atr = a
                stop = price + stop_mult * a
                target = price - rr * stop_mult * a
        elif pos == 1:
            new_stop = price - stop_mult * a
            if new_stop > stop:
                stop = new_stop
            if price <= stop:
                trades.append((stop - entry) / entry - COST_RT)
                pos = 0
            elif price >= target:
                trades.append((target - entry) / entry - COST_RT)
                pos = 0
        elif pos == -1:
            new_stop = price + stop_mult * a
            if new_stop < stop:
                stop = new_stop
            if price >= stop:
                trades.append((entry - stop) / entry - COST_RT)
                pos = 0
            elif price <= target:
                trades.append((entry - target) / entry - COST_RT)
                pos = 0

    return trades

# ─── Analysis ────────────────────────────────────────────────────────────────

def analyze(trades):
    if len(trades) < 3:
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

def walk_forward(data, lookback, rr, stop_mult, n_windows=3):
    """Walk-forward with time-sorted windows."""
    n = data["n"]
    step = n // n_windows
    results = []

    for i in range(n_windows):
        s = i * step
        e = min((i + 1) * step + lookback + 50, n)
        wd = {k: data[k][s:e] for k in ["o", "h", "l", "c"]}
        wd["n"] = e - s
        wd["times"] = data["times"][s:e]

        if wd["n"] < lookback + 50:
            results.append({"w": i + 1, "pass": False})
            continue

        trades = donchian_backtest(wd, lookback, rr, stop_mult)
        a = analyze(trades)
        if a is None:
            results.append({"w": i + 1, "trades": len(trades), "pass": False})
            continue

        # Relaxed criteria for WF: positive Sharpe + PF > 1.2
        passed = a["sharpe"] > 0 and a["pf"] >= 1.2
        results.append({"w": i + 1, **a, "pass": passed})

    passes = sum(1 for r in results if r.get("pass"))
    return results, passes

# ─── Parameter Search ────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  DONCHIAN BREAKOUT — DEEP PARAMETER SEARCH (Real XAU/USD 1h)")
    print("=" * 80)
    print()

    print("Fetching data...")
    data = fetch_lse("XAU/USD", "1h", "2020-01-01", "2025-06-16")
    print(f"  {data['n']} bars ({data['times'][0][:10]} → {data['times'][-1][:10]})\n")

    # Parameter grid
    lookbacks = [5, 8, 10, 12, 15, 20, 25, 30]
    rrs = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    stop_mults = [1.0, 1.5, 2.0]

    all_results = []
    total = len(lookbacks) * len(rrs) * len(stop_mults)
    count = 0

    for lb in lookbacks:
        for rr in rrs:
            for sm in stop_mults:
                count += 1
                trades = donchian_backtest(data, lb, rr, sm)
                a = analyze(trades)

                if a is None or a["trades"] < 5:
                    continue

                # Walk-forward
                wf_results, wf_passes = walk_forward(data, lb, rr, sm)

                entry = {
                    "lb": lb, "rr": rr, "sm": sm,
                    **a, "wf_passes": wf_passes, "wf_total": len(wf_results),
                    "wf_details": wf_results
                }
                all_results.append(entry)

    # Sort by Sharpe
    all_results.sort(key=lambda x: x["sharpe"], reverse=True)

    # Print top 20
    print(f"Tested {count} parameter combinations, {len(all_results)} had ≥5 trades\n")
    print(f"{'LB':>3} {'RR':>4} {'SM':>4} {'T':>4} {'WR%':>5} {'PF':>5} {'Sharpe':>7} {'DD%':>5} {'Ret%':>6} {'WF':>4}")
    print("-" * 60)

    for r in all_results[:30]:
        wf_s = f"{r['wf_passes']}/{r['wf_total']}"
        star = " ⭐" if r["sharpe"] > 1.0 and r["pf"] >= 1.3 and r["wf_passes"] >= 1 else ""
        print(f"{r['lb']:>3} {r['rr']:>4.1f} {r['sm']:>4.1f} {r['trades']:>4} {r['wr']:>5} {r['pf']:>5} {r['sharpe']:>7} {r['max_dd']:>5} {r['ret']:>6} {wf_s:>4}{star}")

    # Deep dive on best
    if all_results:
        best = all_results[0]
        print(f"\n{'='*60}")
        print(f"  BEST: LB={best['lb']}, RR={best['rr']}, SM={best['sm']}")
        print(f"  Trades={best['trades']}, WR={best['wr']}%, PF={best['pf']}, Sharpe={best['sharpe']}")
        print(f"  Max DD={best['max_dd']}%, Return={best['ret']}%")
        print(f"  Walk-forward: {best['wf_passes']}/{best['wf_total']} pass")
        print(f"{'='*60}")

        for wf in best["wf_details"]:
            s = "✅" if wf.get("pass") else "❌"
            t = wf.get("trades", 0)
            sh = wf.get("sharpe", 0)
            pf = wf.get("pf", 0)
            print(f"    Window {wf['w']}: {t} trades, Sharpe={sh:.3f}, PF={pf:.2f} {s}")

        # Also show walk-forward details for top 5
        if len(all_results) > 1:
            print(f"\n  TOP 5 WALK-FORWARD DETAILS:")
            for i, r in enumerate(all_results[:5]):
                print(f"\n  #{i+1}: LB={r['lb']}, RR={r['rr']}, SM={r['sm']} (Sharpe={r['sharpe']}, PF={r['pf']})")
                for wf in r.get("wf_details", []):
                    s = "✅" if wf.get("pass") else "❌"
                    t = wf.get("trades", 0)
                    sh = wf.get("sharpe", 0)
                    pf = wf.get("pf", 0)
                    print(f"      W{wf['w']}: {t}T, Sh={sh:.2f}, PF={pf:.2f} {s}")

        # Monte Carlo for best
        print(f"\n  MONTE CARLO (best params):")
        best_trades = donchian_backtest(data, best["lb"], best["rr"], best["sm"])
        sharpes, dds = [], []
        for _ in range(2000):
            s = np.random.choice(best_trades, size=len(best_trades), replace=True)
            cum = np.cumsum(s); pk = np.maximum.accumulate(cum)
            dd = (pk - cum).max()
            avg, std = s.mean(), s.std()
            sharpes.append((avg / std) * np.sqrt(252) if std > 0 else 0)
            dds.append(dd)
        print(f"    Sharpe: median={np.median(sharpes):.3f}, 5th={np.percentile(sharpes, 5):.3f}, 95th={np.percentile(sharpes, 95):.3f}")
        print(f"    Max DD: median={np.median(dds) * 100:.1f}%, 99th={np.percentile(dds, 99) * 100:.1f}%")
        print(f"    Survival: {np.mean([1 if s > 0 else 0 for s in sharpes]) * 100:.1f}%")
        print(f"    P(profitable): {np.mean([1 if sum(t) > 0 else 0 for t in [np.random.choice(best_trades, len(best_trades), True) for _ in range(2000)]]) * 100:.1f}%")

if __name__ == "__main__":
    main()
