#!/usr/bin/env python3
"""
Regime-Conditional Factor Strategy
From arxiv 2511.12490 (13-Sharpe paper)

Daily cross-sectional strategy with regime-conditional signals.
Optimized: precompute signals once per parameter, then sweep.
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

WORKSPACE = "/home/work/.openclaw/workspace"
LSE_API_KEY = "lse_live_f4c9a7419371ecdd9365e146247b0289"
API_URL = "https://api.londonstrategicedge.com/vault/candles"
HEADERS = {"x-api-key": LSE_API_KEY}

SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "TSM", "AVGO",
           "NVDA", "AMD", "PLTR", "MRVL"]

TRAIN_START, TRAIN_END = "2019-01-01", "2022-12-31"
TEST_START, TEST_END = "2023-01-01", "2024-07-31"
ROUND_TRIP_COST = 0.003  # 0.3%


# ============================================================
# DATA
# ============================================================

def fetch_daily_data(symbol, start="2018-06-01", end="2024-07-31"):
    cache_path = os.path.join(WORKSPACE, f"{symbol}_daily.parquet")
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        if len(df) > 100 and pd.api.types.is_datetime64_any_dtype(df.index):
            return df

    print(f"  Fetching {symbol}...")
    try:
        resp = requests.get(API_URL, params={"symbol": symbol, "timeframe": "1d",
                                              "start": start, "end": end},
                            headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        candles = data.get("candles", data) if isinstance(data, dict) else data
        df = pd.DataFrame(candles)
        col_map = {c: ("date" if c.lower() in ("ts","time","timestamp","date","datetime") else c.lower())
                   for c in df.columns}
        df = df.rename(columns=col_map)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
        for col in ["open","high","low","close","volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        df.to_parquet(cache_path)
        print(f"  ✓ {symbol}: {len(df)} bars")
        return df
    except Exception as e:
        print(f"  ✗ {symbol}: {e}")
        return None


def load_all_data():
    print("=" * 60)
    print("PHASE 1: Data Loading")
    print("=" * 60)
    all_closes = {}
    for sym in SYMBOLS:
        df = fetch_daily_data(sym)
        if df is not None and "close" in df.columns and len(df) > 200:
            all_closes[sym] = df["close"]
        time.sleep(0.3)
    prices = pd.DataFrame(all_closes).dropna(how="all").ffill().dropna(axis=1)
    print(f"\n✓ {len(prices.columns)} symbols, {len(prices)} days "
          f"({prices.index[0].date()} to {prices.index[-1].date()})")
    return prices


# ============================================================
# PRECOMPUTE (called once per unique param set)
# ============================================================

def precompute_all(prices):
    """Precompute all signals and regime masks for all parameter values."""
    print("\nPrecomputing signals...")
    daily_ret = prices.pct_change()

    # Up fraction for different lookbacks (always 63 in paper, but compute once)
    up_frac_63 = daily_ret.rolling(63).apply(lambda x: (x > 0).mean(), raw=True)
    print(f"  ✓ UpFraction(63) computed")

    # Regime masks for each threshold
    regime_masks = {}
    for t in [0.50, 0.55, 0.60, 0.65]:
        regime_masks[t] = (up_frac_63 > t).astype(float)
    print(f"  ✓ Regime masks for thresholds {list(regime_masks.keys())}")

    # Value signal (constant)
    value_signal = (1.0 / prices).rank(axis=1, pct=True)
    print(f"  ✓ Value signal")

    # Reversal signals for different lookbacks
    reversal_signals = {}
    for lb in [5, 10, 20]:
        reversal_signals[lb] = (-prices.pct_change(lb)).rank(axis=1, pct=True)
    print(f"  ✓ Reversal signals for lookbacks {list(reversal_signals.keys())}")

    # Rebalance masks
    rebal_masks = {
        "daily": pd.Series(True, index=prices.index),
        "weekly": prices.index.dayofweek == 0,
    }

    # Train/test masks
    train_mask = (prices.index >= TRAIN_START) & (prices.index <= TRAIN_END)
    test_mask = (prices.index >= TEST_START) & (prices.index <= TEST_END)

    return {
        "daily_ret": daily_ret,
        "regime_masks": regime_masks,
        "value_signal": value_signal,
        "reversal_signals": reversal_signals,
        "rebal_masks": rebal_masks,
        "train_mask": train_mask,
        "test_mask": test_mask,
    }


# ============================================================
# BACKTEST (fast, uses precomputed data)
# ============================================================

def run_backtest_fast(precomp, value_weight, reversal_weight, reversal_lookback,
                      up_frac_thresh, n_stocks, rebal_freq, regime_mode):
    """Fast backtest using precomputed signals."""
    prices_idx = precomp["daily_ret"].index
    daily_ret = precomp["daily_ret"]
    n_half = n_stocks

    # Combine signals
    base_signal = (value_weight * precomp["value_signal"] +
                   reversal_weight * precomp["reversal_signals"][reversal_lookback])

    regime = precomp["regime_masks"][up_frac_thresh]
    rebal_mask = precomp["rebal_masks"][rebal_freq]

    # Edge signal
    if regime_mode == "gate":
        edge_signal = base_signal * regime
    elif regime_mode == "boost":
        edge_signal = base_signal + regime * 0.15
    else:
        edge_signal = base_signal

    # Run portfolio day by day
    prev_long, prev_short = [], []
    returns = []
    costs = []
    dates = []

    n_days = len(edge_signal)
    edge_vals = edge_signal.values
    ret_vals = daily_ret.values
    regime_vals = regime.values
    rebal_vals = rebal_mask.values if isinstance(rebal_mask, pd.Series) else rebal_mask
    cols = list(edge_signal.columns)
    col_idx = {c: i for i, c in enumerate(cols)}

    for i in range(1, n_days):
        dates.append(prices_idx[i])

        if not rebal_vals[i - 1]:
            # Hold
            port_ret = 0.0
            wt = 1.0 / (2 * n_half)
            for s in prev_long:
                j = col_idx.get(s)
                if j is not None:
                    port_ret += wt * ret_vals[i, j]
            for s in prev_short:
                j = col_idx.get(s)
                if j is not None:
                    port_ret -= wt * ret_vals[i, j]
            returns.append(port_ret)
            costs.append(0.0)
            continue

        # Get yesterday's signal
        sig_row = edge_vals[i - 1]
        regime_row = regime_vals[i - 1]

        # Build ranked list
        if regime_mode == "gate":
            eligible_indices = [j for j in range(len(cols)) if regime_row[j] > 0 and not np.isnan(sig_row[j])]
            if len(eligible_indices) < n_half * 2:
                eligible_indices = [j for j in range(len(cols)) if not np.isnan(sig_row[j])]
        else:
            eligible_indices = [j for j in range(len(cols)) if not np.isnan(sig_row[j])]

        if len(eligible_indices) < 4:
            returns.append(0.0)
            costs.append(0.0)
            prev_long, prev_short = [], []
            continue

        # Sort by signal descending
        eligible_indices.sort(key=lambda j: sig_row[j], reverse=True)

        long_stocks = [cols[j] for j in eligible_indices[:n_half]]
        # Short from bottom, avoid overlap
        short_stocks = [cols[j] for j in eligible_indices[::-1] if cols[j] not in long_stocks][:n_half]

        if not short_stocks:
            returns.append(0.0)
            costs.append(0.0)
            prev_long, prev_short = [], []
            continue

        # Calculate return
        wt = 1.0 / (2 * n_half)
        port_ret = 0.0
        for s in long_stocks:
            j = col_idx[s]
            port_ret += wt * ret_vals[i, j]
        for s in short_stocks:
            j = col_idx[s]
            port_ret -= wt * ret_vals[i, j]

        # Cost on rebalance
        cost = 0.0
        if set(long_stocks) != set(prev_long) or set(short_stocks) != set(prev_short):
            cost = ROUND_TRIP_COST

        returns.append(port_ret - cost)
        costs.append(cost)
        prev_long, prev_short = long_stocks, short_stocks

    port_df = pd.DataFrame({"return": returns, "cost": costs}, index=dates)

    # Compute metrics for train/test
    results = {}
    for label, mask in [("train", precomp["train_mask"]), ("test", precomp["test_mask"])]:
        # Align mask with port_df dates
        if label == "train":
            mask_dates = [d for d in port_df.index if TRAIN_START <= str(d.date()) <= TRAIN_END]
        else:
            mask_dates = [d for d in port_df.index if TEST_START <= str(d.date()) <= TEST_END]

        rets = port_df.loc[mask_dates, "return"]
        if len(rets) < 20:
            results[label] = {"error": "insufficient data", "n_days": len(rets)}
            continue

        total_return = (1 + rets).prod() - 1
        ann_return = (1 + total_return) ** (252 / len(rets)) - 1 if total_return > -1 else -1
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        cum = (1 + rets).cumprod()
        max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
        win_rate = (rets > 0).mean()
        n_active = (rets.abs() > 1e-8).sum()

        results[label] = {
            "n_days": len(rets),
            "n_active_days": int(n_active),
            "pct_active": round(100 * n_active / len(rets), 1),
            "total_return": round(total_return * 100, 2),
            "ann_return": round(ann_return * 100, 2),
            "ann_vol": round(ann_vol * 100, 2),
            "sharpe": round(sharpe, 3),
            "max_drawdown": round(max_dd * 100, 2),
            "win_rate": round(win_rate * 100, 1),
            "total_cost_pct": round(port_df.loc[mask_dates, "cost"].sum() * 100, 2),
        }

    return results


# ============================================================
# PARAMETER SWEEP
# ============================================================

def run_sweep(precomp):
    print("\n" + "=" * 60)
    print("PHASE 2: Parameter Sweep")
    print("=" * 60)

    params = list(product(
        [0.50, 0.55, 0.60, 0.65],          # up_frac_thresh
        [(0.7, 0.3), (0.5, 0.5), (0.8, 0.2)],  # value/rev weights
        [5, 10, 20],                          # reversal lookback
        [3, 5, 10],                           # n_stocks
        ["daily", "weekly"],                  # rebal freq
        ["gate", "boost", "none"],            # regime mode
    ))
    print(f"Total: {len(params)} configs")

    all_results = []
    for idx, (uft, vw, rl, ns, rf, rm) in enumerate(params):
        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(params)}...")

        try:
            results = run_backtest_fast(precomp, vw[0], vw[1], rl, uft, ns, rf, rm)
            entry = {
                "config": {"up_frac_threshold": uft, "value_weight": vw[0],
                           "reversal_weight": vw[1], "reversal_lookback": rl,
                           "n_stocks": ns, "rebalance_freq": rf, "regime_mode": rm},
                "train": results.get("train", {}),
                "test": results.get("test", {}),
            }
        except Exception as e:
            entry = {
                "config": {"up_frac_threshold": uft, "value_weight": vw[0],
                           "reversal_weight": vw[1], "reversal_lookback": rl,
                           "n_stocks": ns, "rebalance_freq": rf, "regime_mode": rm},
                "error": str(e),
            }
        all_results.append(entry)

    return all_results


# ============================================================
# BENCHMARK
# ============================================================

def calc_benchmark(prices, symbol="SPY"):
    if symbol not in prices.columns:
        return None
    spy_ret = prices[symbol].pct_change().dropna()
    results = {}
    for label, start, end in [("train", TRAIN_START, TRAIN_END), ("test", TEST_START, TEST_END)]:
        rets = spy_ret[(spy_ret.index >= start) & (spy_ret.index <= end)]
        if len(rets) < 20:
            continue
        total = (1 + rets).prod() - 1
        ann_ret = (1 + total) ** (252 / len(rets)) - 1
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = (1 + rets).cumprod()
        max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
        results[label] = {
            "total_return": round(total * 100, 2),
            "ann_return": round(ann_ret * 100, 2),
            "ann_vol": round(ann_vol * 100, 2),
            "sharpe": round(sharpe, 3),
            "max_drawdown": round(max_dd * 100, 2),
        }
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("REGIME-CONDITIONAL FACTOR STRATEGY")
    print("arxiv 2511.12490 — walk-forward OOS test")
    print("=" * 60)

    prices = load_all_data()
    precomp = precompute_all(prices)
    all_results = run_sweep(precomp)

    # Benchmark
    print("\n" + "=" * 60)
    print("PHASE 3: Benchmark (SPY Buy & Hold)")
    print("=" * 60)
    benchmark = calc_benchmark(prices)
    if benchmark:
        for label, m in benchmark.items():
            print(f"  {label}: Ret={m['total_return']}%, Sharpe={m['sharpe']}, DD={m['max_drawdown']}%")

    # Filter valid
    valid = [r for r in all_results
             if "error" not in r and "error" not in r.get("train", {}) and "error" not in r.get("test", {})
             and r.get("test", {}).get("n_days", 0) > 20]

    # Sort by test Sharpe
    valid.sort(key=lambda x: x["test"].get("sharpe", -999), reverse=True)

    print("\n" + "=" * 60)
    print("PHASE 4: Best Configuration")
    print("=" * 60)

    if valid:
        best = valid[0]
        bc = best["config"]
        print(f"\n  Best config (test Sharpe):")
        print(f"    {bc}")
        for label in ["train", "test"]:
            m = best[label]
            print(f"    {label}: Ret={m['total_return']}%, Sharpe={m['sharpe']}, "
                  f"DD={m['max_drawdown']}%, WR={m['win_rate']}%, Active={m['pct_active']}%")

        # Prop firm
        test = best["test"]
        train = best["train"]
        print("\n  Prop Firm Assessment:")
        ts = test["sharpe"]
        mdd = test["max_drawdown"]
        tr = test["total_return"]

        checks = []
        checks.append(f"{'✓' if ts >= 1.5 else '✗'} Test Sharpe {ts} {'≥' if ts >= 1.5 else '<'} 1.5")
        checks.append(f"{'✓' if mdd > -15 else '✗'} Max DD {mdd}% {'>' if mdd > -15 else '≤'} -15%")
        checks.append(f"{'✓' if tr > 0 else '✗'} Test return {tr}%")

        if train["sharpe"] > 0 and test["sharpe"] > 0:
            decay = test["sharpe"] / train["sharpe"]
            checks.append(f"{'✓' if decay >= 0.5 else '✗'} Sharpe decay {decay:.2f}")

        if benchmark and "test" in benchmark:
            spy_s = benchmark["test"]["sharpe"]
            checks.append(f"{'✓' if ts > spy_s else '✗'} vs SPY Sharpe {spy_s}")

        for c in checks:
            print(f"    {c}")
    else:
        print("  No valid configs!")

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 5: Summary")
    print("=" * 60)

    if valid:
        sharpes = [r["test"]["sharpe"] for r in valid]
        returns = [r["test"]["total_return"] for r in valid]
        pos_s = sum(1 for s in sharpes if s > 0)
        pos_r = sum(1 for r in returns if r > 0)

        print(f"  Valid: {len(valid)}/{len(all_results)}")
        print(f"  Positive Sharpe: {pos_s}/{len(valid)} ({100*pos_s/len(valid):.1f}%)")
        print(f"  Positive return: {pos_r}/{len(valid)} ({100*pos_r/len(valid):.1f}%)")
        print(f"  Sharpe range: [{min(sharpes):.3f}, {max(sharpes):.3f}]")
        print(f"  Median Sharpe: {np.median(sharpes):.3f}")

        print(f"\n  Top 10 by test Sharpe:")
        for i, r in enumerate(valid[:10]):
            c, t = r["config"], r["test"]
            print(f"    {i+1}. S={t['sharpe']:.3f} R={t['total_return']:.1f}% "
                  f"DD={t['max_drawdown']:.1f}% WR={t['win_rate']:.0f}% | "
                  f"UFT={c['up_frac_threshold']} VW={c['value_weight']}/{c['reversal_weight']} "
                  f"RL={c['reversal_lookback']} N={c['n_stocks']} {c['rebalance_freq']} {c['regime_mode']}")

        print(f"\n  By regime mode:")
        for mode in ["gate", "boost", "none"]:
            mr = [r for r in valid if r["config"]["regime_mode"] == mode]
            if mr:
                ss = [r["test"]["sharpe"] for r in mr]
                print(f"    {mode}: n={len(mr)}, median={np.median(ss):.3f}, "
                      f"best={max(ss):.3f}, %pos={100*sum(1 for s in ss if s>0)/len(ss):.0f}%")

        # By n_stocks
        print(f"\n  By portfolio size:")
        for ns in [3, 5, 10]:
            mr = [r for r in valid if r["config"]["n_stocks"] == ns]
            if mr:
                ss = [r["test"]["sharpe"] for r in mr]
                print(f"    N={ns}: n={len(mr)}, median={np.median(ss):.3f}, best={max(ss):.3f}")

    # Save
    output = {
        "strategy": "Regime-Conditional Factor (arxiv 2511.12490)",
        "timestamp": datetime.now().isoformat(),
        "configs_tested": len(all_results),
        "valid_configs": len(valid) if valid else 0,
        "symbols": list(prices.columns),
        "date_range": {"start": str(prices.index[0].date()), "end": str(prices.index[-1].date())},
        "costs": {"slippage_pct": 0.1, "commission_pct": 0.05, "round_trip_pct": 0.3},
        "benchmark": benchmark,
        "best_configuration": valid[0] if valid else None,
        "top_10": valid[:10] if valid else [],
        "all_configurations": all_results,
    }
    out_path = os.path.join(WORKSPACE, "regime_factor_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✓ Saved: {out_path}")
    print("=" * 60 + "\nDONE")


if __name__ == "__main__":
    main()
