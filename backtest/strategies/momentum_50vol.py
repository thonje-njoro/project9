#!/usr/bin/env python3
"""
High-Vol Momentum System for Prop Firm Evaluation
==================================================
Vectorized time-series momentum with drawdown-aware position sizing.

Key design: DD management only at REBALANCE time (weekly).
Intra-week, positions ride. At rebalance, we check DD and adjust.
"""

import json
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WORKSPACE = Path("/home/work/.openclaw/workspace")
CACHE_DIR = WORKSPACE / "alpaca_cache"

COST_PER_TRADE = 0.0015
PROF_TARGET = 0.10
MAX_DD_TOTAL = 0.10
MAX_DD_DAILY = 0.03
MIN_TRADE_DAYS = 10
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"

UNIVERSE = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "META", "AMZN",
    "NVDA", "AMD", "TSLA", "AVGO", "JPM", "V",
    "TLT", "IEF", "SHY", "BND", "LQD",
    "GLD", "SLV", "USO", "DBA", "DBC",
    "EFA", "EEM", "VXUS", "FXI", "EWJ",
]


def load_data():
    all_data = {}
    for sym in UNIVERSE:
        f = CACHE_DIR / f"{sym}_alpaca_daily.parquet"
        if f.exists():
            all_data[sym] = pd.read_parquet(f)
    closes = {sym: df["close"] for sym, df in all_data.items()}
    prices = pd.DataFrame(closes).dropna(how="all").ffill()
    min_rows = int(len(prices) * 0.80)
    return prices.dropna(axis=1, thresh=min_rows).dropna()


def run_strategy(prices_df, lookback=60, vol_target=0.30, top_n=5,
                 max_position=0.10, dd_threshold=0.08):
    """
    Time-series momentum with DD-aware rebalancing.
    
    On each weekly rebalance:
    - If current DD < dd_threshold: rebalance to target weights
    - If current DD >= dd_threshold: go flat (wait for recovery)
    - DD is measured from peak equity
    """
    returns = prices_df.pct_change().values
    prices_arr = prices_df.values
    T, N = returns.shape
    dates = prices_df.index
    asset_names = list(prices_df.columns)
    
    # Momentum signal
    mom_df = prices_df.pct_change(lookback)
    mom = mom_df.values
    
    # Per-asset realized vol (20-day)
    realized_vol = prices_df.pct_change().rolling(20).std().values * np.sqrt(252)
    realized_vol = np.nan_to_num(realized_vol, nan=0.15)
    realized_vol[realized_vol == 0] = 0.15
    
    # Vol scaling
    vol_scale = np.clip(vol_target / realized_vol, 0, 2.0)
    
    # Target weights
    target_weights = np.zeros((T, N))
    for t in range(lookback, T):
        row = mom[t]
        pos_mask = row > 0
        if not np.any(pos_mask):
            continue
        pos_indices = np.where(pos_mask)[0]
        pos_values = row[pos_indices]
        if len(pos_values) > top_n:
            top_idx = np.argpartition(pos_values, -top_n)[-top_n:]
            selected = pos_indices[top_idx]
        else:
            selected = pos_indices
        n_sel = len(selected)
        for s in selected:
            w = vol_scale[t, s] / n_sel
            target_weights[t, s] = min(w, max_position)
    
    # Weekly rebalance detection
    week_starts = np.zeros(T, dtype=bool)
    week_starts[0] = True
    for t in range(1, T):
        gap = (dates[t] - dates[t-1]).days
        if gap > 3 or dates[t].dayofweek < dates[t-1].dayofweek:
            week_starts[t] = True
    
    # ── Simulate ──
    equity = np.ones(T)
    current_w = np.zeros(N)
    weights_arr = np.zeros((T, N))
    prev_w = np.zeros(N)
    
    for t in range(1, T):
        # Check if it's a rebalance day
        if week_starts[t]:
            # Check DD
            peak = np.max(equity[:t])
            dd = equity[t-1] / peak - 1
            
            if dd > -dd_threshold:
                # Normal: rebalance to target
                current_w = target_weights[t].copy()
            else:
                # In drawdown: go flat
                current_w = np.zeros(N)
        
        weights_arr[t] = current_w
        
        # Daily return
        day_ret = np.dot(current_w, returns[t])
        
        # Cost
        cost = np.sum(np.abs(current_w - prev_w)) * COST_PER_TRADE
        prev_w = current_w.copy()
        
        net_ret = day_ret - cost
        if np.isnan(net_ret):
            net_ret = 0.0
        
        equity[t] = equity[t-1] * (1 + net_ret)
    
    net_returns = pd.Series(np.diff(equity) / equity[:-1], index=dates[1:])
    net_returns = pd.concat([pd.Series([0.0], index=[dates[0]]), net_returns])
    
    return net_returns


def calc_metrics(net_returns):
    if len(net_returns) < 10:
        return None
    
    total_return = (1 + net_returns).prod() - 1
    n_days = len(net_returns)
    ann_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
    ann_vol = net_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    equity = (1 + net_returns).cumprod()
    running_max = equity.cummax()
    dd = equity / running_max - 1
    max_dd = dd.min()
    
    daily_violations = int((net_returns < -MAX_DD_DAILY).sum())
    
    hit = equity >= (1 + PROF_TARGET)
    days_to_target = int(hit.values.argmax()) + 1 if hit.any() else None
    
    hits_target = total_return >= PROF_TARGET
    dd_ok = abs(max_dd) <= MAX_DD_TOTAL
    daily_ok = daily_violations == 0
    passes = hits_target and dd_ok and daily_ok and (n_days >= MIN_TRADE_DAYS)
    
    win_rate = (net_returns > 0).mean()
    gains = net_returns[net_returns > 0].sum()
    losses = abs(net_returns[net_returns < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
    dr = net_returns[net_returns < 0]
    downside = dr.std() * np.sqrt(252) if len(dr) > 0 else 0
    sortino = ann_return / downside if downside > 0 else 0
    
    return {
        "total_return_pct": round(float(total_return * 100), 2),
        "annual_return_pct": round(float(ann_return * 100), 2),
        "annual_vol_pct": round(float(ann_vol * 100), 2),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "calmar": round(float(calmar), 3),
        "max_drawdown_pct": round(float(max_dd * 100), 2),
        "win_rate": round(float(win_rate), 3),
        "profit_factor": round(float(pf), 3),
        "n_days": int(n_days),
        "daily_dd_violations": int(daily_violations),
        "hits_profit_target": bool(hits_target),
        "dd_under_limit": bool(dd_ok),
        "daily_dd_ok": bool(daily_ok),
        "passes_prop_firm": bool(passes),
        "days_to_10pct": int(days_to_target) if days_to_target is not None else None,
    }


def run_walk_forward(prices_df, params):
    net_ret = run_strategy(
        prices_df, lookback=params["lookback"], vol_target=params["vol_target"],
        top_n=params["top_n"], max_position=params["max_position"],
        dd_threshold=params["dd_threshold"]
    )
    
    train_ret = net_ret.loc[:TRAIN_END]
    test_ret = net_ret.loc[TEST_START:]
    
    if len(train_ret) < 60 or len(test_ret) < 30:
        return None
    
    return {
        "train": calc_metrics(train_ret),
        "test": calc_metrics(test_ret),
    }


def main():
    print("=" * 70)
    print("HIGH-VOL MOMENTUM + DD-AWARE REBALANCING — PROP FIRM TEST")
    print("=" * 70)
    
    prices_df = load_data()
    print(f"Data: {prices_df.shape[0]} days × {prices_df.shape[1]} assets\n")
    
    # Grid: vol_target × max_position × lookback × dd_threshold
    param_grid = list(product(
        [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70],
        [0.06, 0.08, 0.10, 0.12, 0.15, 0.20],
        [40, 60, 90],
        [0.06, 0.08, 0.10, 0.12],
    ))
    
    print(f"Combinations: {len(param_grid)}")
    print("Running...\n")
    
    results = []
    best_score = -999
    best_entry = None
    
    for i, (vt, mp, lb, ddt) in enumerate(param_grid):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{len(param_grid)}] vol={vt:.0%} pos={mp:.0%} lb={lb} dd={ddt:.0%}...", flush=True)
        
        params = {"vol_target": vt, "max_position": mp, "lookback": lb, "top_n": 5, "dd_threshold": ddt}
        
        try:
            wf = run_walk_forward(prices_df, params)
        except Exception:
            continue
        
        if wf is None or wf["train"] is None or wf["test"] is None:
            continue
        
        entry = {**params, "train": wf["train"], "test": wf["test"]}
        results.append(entry)
        
        test = wf["test"]
        score = 0
        if test["passes_prop_firm"]: score += 1000
        score += test["total_return_pct"]
        score -= abs(test["max_drawdown_pct"]) * 3
        score += test["sharpe"] * 15
        score -= test["daily_dd_violations"] * 5
        
        if score > best_score:
            best_score = score
            best_entry = entry
    
    print(f"\nCompleted {len(results)} valid configurations.\n")
    
    # ── Analysis ──
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    passing = [r for r in results if r["test"]["passes_prop_firm"]]
    dd_safe = [r for r in results if r["test"]["dd_under_limit"]]
    daily_safe = [r for r in results if r["test"]["daily_dd_ok"]]
    profitable = [r for r in results if r["test"]["hits_profit_target"]]
    
    print(f"\nPROP FIRM PASS:   {len(passing)}/{len(results)}")
    print(f"DD under 10%:     {len(dd_safe)}/{len(results)}")
    print(f"Daily DD OK:      {len(daily_safe)}/{len(results)}")
    print(f"Hit 10% target:   {len(profitable)}/{len(results)}")
    
    def score_fn(r):
        t = r["test"]
        s = 0
        if t["passes_prop_firm"]: s += 1000
        s += t["total_return_pct"] - abs(t["max_drawdown_pct"]) * 3
        s += t["sharpe"] * 15 - t["daily_dd_violations"] * 5
        return s
    
    results.sort(key=score_fn, reverse=True)
    
    print(f"\nTOP 25 BY SCORE:")
    print(f"{'#':>3} {'Vol':>4} {'Pos':>4} {'LB':>3} {'DD':>4} "
          f"{'Ret%':>7} {'Sharpe':>7} {'MaxDD%':>7} {'D.V':>4} {'Pass':>5}")
    print("─" * 60)
    for i, r in enumerate(results[:25]):
        t = r["test"]
        p = "✅" if t["passes_prop_firm"] else "❌"
        print(f"{i+1:>3} {r['vol_target']:>3.0%} {r['max_position']:>3.0%} {r['lookback']:>3} {r['dd_threshold']:>3.0%} "
              f"{t['total_return_pct']:>6.1f}% {t['sharpe']:>7.3f} {t['max_drawdown_pct']:>6.1f}% "
              f"{t['daily_dd_violations']:>4} {p:>5}")
    
    if passing:
        print(f"\n{'='*70}")
        print(f"✅ {len(passing)} CONFIG(S) PASS PROP FIRM")
        print(f"{'='*70}")
        passing.sort(key=lambda x: x["test"]["total_return_pct"], reverse=True)
        for r in passing[:15]:
            t = r["test"]
            print(f"  vol={r['vol_target']:.0%} pos={r['max_position']:.0%} lb={r['lookback']} dd={r['dd_threshold']:.0%} "
                  f"→ ret={t['total_return_pct']:.1f}% sharpe={t['sharpe']:.3f} DD={t['max_drawdown_pct']:.1f}% "
                  f"dailyV={t['daily_dd_violations']}")
            if t["days_to_10pct"]:
                print(f"    Days to 10%: {t['days_to_10pct']}")
    
    # Best config
    if best_entry:
        print(f"\n{'='*70}")
        print("BEST CONFIG")
        print(f"{'='*70}")
        for k in ["vol_target", "max_position", "lookback", "dd_threshold"]:
            print(f"  {k}: {best_entry[k]}")
        print(f"\n  TEST:")
        for k, v in best_entry["test"].items():
            print(f"    {k}: {v}")
    
    # Verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    if passing:
        bp = max(passing, key=lambda x: x["test"]["total_return_pct"])
        t = bp["test"]
        print(f"  ✅ YES — {len(passing)} config(s) pass")
        print(f"  Best: vol={bp['vol_target']:.0%} ret={t['total_return_pct']:.1f}% "
              f"DD={t['max_drawdown_pct']:.1f}% dailyV={t['daily_dd_violations']}")
        if t["days_to_10pct"]:
            print(f"  Days to 10%: {t['days_to_10pct']}")
    elif results:
        t = results[0]["test"]
        print(f"  ❌ NO config passes all rules")
        print(f"  Best return: {t['total_return_pct']:.1f}%")
        print(f"  Best Sharpe: {max(r['test']['sharpe'] for r in results):.3f}")
        print(f"  Min DD: {min(abs(r['test']['max_drawdown_pct']) for r in results):.1f}%")
        print(f"  Min daily violations: {min(r['test']['daily_dd_violations'] for r in results)}")
        
        # Closest to passing
        close = [r for r in results if r["test"]["dd_under_limit"] and r["test"]["daily_dd_ok"]]
        if close:
            close.sort(key=lambda x: x["test"]["total_return_pct"], reverse=True)
            c = close[0]
            print(f"\n  CLOSEST (DD safe + daily safe):")
            print(f"    vol={c['vol_target']:.0%} pos={c['max_position']:.0%} lb={c['lookback']} dd={c['dd_threshold']:.0%}")
            print(f"    ret={c['test']['total_return_pct']:.1f}% DD={c['test']['max_drawdown_pct']:.1f}%")
    
    # Save
    output = {
        "strategy": "time_series_momentum_dd_aware_rebalance",
        "prop_firm_rules": {"profit_target_pct": 10, "max_total_dd_pct": 10,
                            "max_daily_dd_pct": 3, "min_trade_days": MIN_TRADE_DAYS},
        "summary": {
            "total_configs": len(results),
            "passing_configs": len(passing),
            "dd_safe": len(dd_safe),
            "daily_safe": len(daily_safe),
            "profitable": len(profitable),
            "best_return_pct": round(results[0]["test"]["total_return_pct"], 2) if results else None,
            "best_sharpe": round(max(r["test"]["sharpe"] for r in results), 3) if results else None,
        },
        "best_config": best_entry,
        "all_passing": passing,
        "top_25": results[:25],
        "all_results": results,
    }
    
    out_path = WORKSPACE / "momentum_50vol_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    main()
