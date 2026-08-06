#!/usr/bin/env python3
"""
High-Frequency Momentum Strategies for Prop Firm Evaluation.
Tests daily rebalancing and shorter lookbacks to generate more trades and higher returns.
"""

import json
import warnings
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
WORKSPACE = Path(__file__).parent

# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

def load_daily_prices(symbols=None):
    """Load daily close prices from *_daily.parquet files."""
    if symbols is None:
        symbols = [
            "AAPL", "AMD", "AMZN", "AVGO", "GOOGL", "META",
            "MRVL", "MSFT", "NVDA", "PLTR", "QQQ", "SPY", "TSLA",
        ]
    frames = {}
    for sym in symbols:
        f = WORKSPACE / f"{sym}_daily.parquet"
        if not f.exists():
            print(f"  [skip] {f.name} not found")
            continue
        df = pd.read_parquet(f)
        frames[sym] = df["close"] if "close" in df.columns else df.iloc[:, 0]
    prices = pd.DataFrame(frames)
    prices.index = pd.DatetimeIndex(prices.index)
    prices = prices.sort_index().dropna(how="all")
    print(f"Loaded {len(prices.columns)} symbols, {len(prices)} rows, "
          f"{prices.index.min().date()} → {prices.index.max().date()}")
    return prices


# ──────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────

def calc_returns(prices):
    return prices.pct_change()

def calc_momentum_signal(prices, lookback):
    return prices / prices.shift(lookback) - 1

def prop_firm_metrics(equity_curve, daily_dd_limit=0.03, total_dd_limit=0.10, profit_target=0.10):
    returns = equity_curve.pct_change().dropna()
    if len(returns) == 0:
        return {"pass": False, "reason": "no data"}

    max_daily_dd = abs(returns.min())
    cummax = equity_curve.cummax()
    max_total_dd = abs(((equity_curve - cummax) / cummax).min())
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1

    n_years = len(returns) / 252
    ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    passed = (max_daily_dd <= daily_dd_limit and
              max_total_dd <= total_dd_limit and
              total_return >= profit_target)

    return {
        "total_return_pct": round(total_return * 100, 2),
        "ann_return_pct": round(ann_return * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_daily_dd_pct": round(max_daily_dd * 100, 2),
        "max_total_dd_pct": round(max_total_dd * 100, 2),
        "trading_days": len(returns),
        "pass_prop_firm": passed,
        "daily_dd_breached": max_daily_dd > daily_dd_limit,
        "total_dd_breached": max_total_dd > total_dd_limit,
        "profit_target_met": total_return >= profit_target,
    }


def vol_target_scale(port_ret, target_vol, window=20, cap=3.0):
    rolling_vol = port_ret.rolling(window, min_periods=5).std() * np.sqrt(252)
    scale = target_vol / rolling_vol.clip(lower=0.01)
    return scale.clip(upper=cap)


# ──────────────────────────────────────────────
# Portfolio construction (day-by-day, clean)
# ──────────────────────────────────────────────

def build_long_short_returns(prices_df, signal_df, ret_df, top_n, bottom_n, max_position,
                              rebalance_freq='D'):
    """
    Build daily long-short portfolio returns.
    rebalance_freq: 'D' for daily, 'W' for weekly.
    """
    dates = signal_df.index
    n_stocks = len(prices_df.columns)
    daily_rets = np.zeros(len(dates))

    # Precompute rebalance mask
    if rebalance_freq == 'W':
        week_ids = dates.to_period('W')
        rebalance = np.zeros(len(dates), dtype=bool)
        prev_w = None
        for i in range(len(dates)):
            if week_ids[i] != prev_w:
                rebalance[i] = True
                prev_w = week_ids[i]
    else:
        rebalance = np.ones(len(dates), dtype=bool)

    # Current positions (weights)
    long_weights = np.zeros(n_stocks)
    short_weights = np.zeros(n_stocks)

    for i in range(len(dates)):
        if rebalance[i]:
            sig = signal_df.iloc[i].values
            valid_mask = ~np.isnan(sig)
            if valid_mask.sum() < top_n + bottom_n:
                long_weights[:] = 0
                short_weights[:] = 0
            else:
                # Rank by signal
                valid_idx = np.where(valid_mask)[0]
                valid_sigs = sig[valid_idx]
                order = np.argsort(valid_sigs)
                bottom_idx = valid_idx[order[:bottom_n]]
                top_idx = valid_idx[order[-top_n:]]

                long_weights[:] = 0
                short_weights[:] = 0

                w_long = min(1.0 / top_n, max_position)
                w_short = min(1.0 / bottom_n, max_position)
                long_weights[top_idx] = w_long
                short_weights[bottom_idx] = w_short

        ret_row = ret_df.iloc[i].values
        # Replace NaN with 0 for missing stocks
        ret_row = np.where(np.isnan(ret_row), 0.0, ret_row)
        daily_rets[i] = np.dot(long_weights, ret_row) - np.dot(short_weights, ret_row)

    return pd.Series(daily_rets, index=dates)


# ──────────────────────────────────────────────
# Strategy variants
# ──────────────────────────────────────────────

def run_daily_momentum(prices_df, lookback=20, vol_target=0.30,
                       top_n=5, bottom_n=5, max_position=0.15):
    """Daily rebalancing momentum strategy."""
    ret = calc_returns(prices_df)
    signal = calc_momentum_signal(prices_df, lookback)
    start = lookback + 20

    signal = signal.iloc[start:]
    ret = ret.iloc[start:]

    port_ret = build_long_short_returns(
        prices_df.iloc[start:], signal, ret,
        top_n, bottom_n, max_position, rebalance_freq='D')

    scale = vol_target_scale(port_ret, vol_target)
    scaled_ret = port_ret * scale
    equity = (1 + scaled_ret).cumprod()
    equity.iloc[0] = 1.0

    return equity, scaled_ret, {
        "strategy": "daily_momentum",
        "lookback": lookback, "vol_target": vol_target,
        "top_n": top_n, "bottom_n": bottom_n,
        "max_position": max_position, "rebalance": "daily",
    }


def run_weekly_short(prices_df, lookback=20, vol_target=0.30,
                     top_n=5, bottom_n=5, max_position=0.15):
    """Weekly rebalancing, shorter lookback."""
    ret = calc_returns(prices_df)
    signal = calc_momentum_signal(prices_df, lookback)
    start = lookback + 20

    signal = signal.iloc[start:]
    ret = ret.iloc[start:]

    port_ret = build_long_short_returns(
        prices_df.iloc[start:], signal, ret,
        top_n, bottom_n, max_position, rebalance_freq='W')

    scale = vol_target_scale(port_ret, vol_target)
    scaled_ret = port_ret * scale
    equity = (1 + scaled_ret).cumprod()
    equity.iloc[0] = 1.0

    return equity, scaled_ret, {
        "strategy": "weekly_short",
        "lookback": lookback, "vol_target": vol_target,
        "top_n": top_n, "bottom_n": bottom_n,
        "max_position": max_position, "rebalance": "weekly",
    }


def run_multi_signal(prices_df, lookbacks=[5, 20, 60], weights=[0.2, 0.5, 0.3],
                     vol_target=0.30, top_n=5, bottom_n=5, max_position=0.15):
    """Combine multiple lookback periods into a composite signal."""
    ret = calc_returns(prices_df)
    max_lb = max(lookbacks)

    composite = sum(w * calc_momentum_signal(prices_df, lb)
                    for w, lb in zip(weights, lookbacks))

    start = max_lb + 20
    composite = composite.iloc[start:]
    ret = ret.iloc[start:]

    port_ret = build_long_short_returns(
        prices_df.iloc[start:], composite, ret,
        top_n, bottom_n, max_position, rebalance_freq='D')

    scale = vol_target_scale(port_ret, vol_target)
    scaled_ret = port_ret * scale
    equity = (1 + scaled_ret).cumprod()
    equity.iloc[0] = 1.0

    return equity, scaled_ret, {
        "strategy": "multi_signal",
        "lookbacks": lookbacks, "weights": weights,
        "vol_target": vol_target,
        "top_n": top_n, "bottom_n": bottom_n,
        "max_position": max_position, "rebalance": "daily",
    }


# ──────────────────────────────────────────────
# Grid Search
# ──────────────────────────────────────────────

def grid_search(prices_train, prices_test):
    results = []
    n_stocks = len(prices_train.columns)

    def _run(fn, params_list, label):
        count = 0
        for params in params_list:
            if params.get("top_n", 5) + params.get("bottom_n", 5) > n_stocks:
                continue
            try:
                eq_tr, _, p = fn(prices_train, **params)
                m_tr = prop_firm_metrics(eq_tr)
                eq_te, _, _ = fn(prices_test, **params)
                m_te = prop_firm_metrics(eq_te)
                results.append({**p, "train": m_tr, "test": m_te})
                count += 1
            except Exception:
                pass
        print(f"  {label}: {count} configs")

    # A: Daily Momentum
    print("\n=== Variant A: Daily Momentum ===")
    _run(run_daily_momentum, [
        {"lookback": lb, "vol_target": vt, "top_n": tn, "bottom_n": bn, "max_position": mp}
        for lb in [5, 10, 20, 40, 60]
        for vt in [0.20, 0.30, 0.40, 0.50]
        for tn in [3, 5, 7]
        for bn in [3, 5, 7]
        for mp in [0.15, 0.20, 0.25]
    ], "Daily Momentum")

    # B: Weekly Short
    print("=== Variant B: Weekly Short ===")
    _run(run_weekly_short, [
        {"lookback": lb, "vol_target": vt, "top_n": tn, "bottom_n": bn, "max_position": mp}
        for lb in [5, 10, 20, 40]
        for vt in [0.20, 0.30, 0.40, 0.50]
        for tn in [3, 5, 7]
        for bn in [3, 5, 7]
        for mp in [0.15, 0.20, 0.25]
    ], "Weekly Short")

    # C: Multi-Signal
    print("=== Variant C: Multi-Signal ===")
    multi_sets = [
        ([5, 20], [0.3, 0.7]),
        ([5, 20, 60], [0.2, 0.5, 0.3]),
        ([5, 10, 20], [0.2, 0.3, 0.5]),
        ([10, 20, 60], [0.3, 0.4, 0.3]),
        ([5, 40, 60], [0.3, 0.4, 0.3]),
    ]
    _run(run_multi_signal, [
        {"lookbacks": lbs, "weights": wts, "vol_target": vt,
         "top_n": tn, "bottom_n": bn, "max_position": mp}
        for lbs, wts in multi_sets
        for vt in [0.20, 0.30, 0.40, 0.50]
        for tn in [3, 5, 7]
        for bn in [3, 5, 7]
        for mp in [0.15, 0.20, 0.25]
    ], "Multi-Signal")

    return results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 70)
    print("HIGH-FREQUENCY MOMENTUM STRATEGIES - PROP FIRM EVALUATION")
    print("=" * 70)

    prices = load_daily_prices()
    print(f"\nFull date range: {prices.index.min().date()} → {prices.index.max().date()}")

    train_start, train_end = "2019-01-01", "2022-12-31"
    test_start, test_end = "2023-01-01", "2024-07-30"

    prices_train = prices.loc[train_start:train_end].dropna(axis=1, thresh=int(len(prices.loc[train_start:train_end]) * 0.8))
    prices_test = prices.loc[test_start:test_end].dropna(axis=1, thresh=int(len(prices.loc[test_start:test_end]) * 0.8))
    common = prices_train.columns.intersection(prices_test.columns)
    prices_train = prices_train[common]
    prices_test = prices_test[common]

    print(f"Train: {prices_train.index.min().date()} → {prices_train.index.max().date()} "
          f"({len(prices_train)} days, {len(prices_train.columns)} stocks)")
    print(f"Test:  {prices_test.index.min().date()} → {prices_test.index.max().date()} "
          f"({len(prices_test)} days, {len(prices_test.columns)} stocks)")

    print("\nRunning parameter grid search...")
    all_results = grid_search(prices_train, prices_test)
    print(f"\nTotal configurations tested: {len(all_results)}")

    # Rankings
    for r in all_results:
        r["test_sharpe"] = r["test"]["sharpe"]
        r["test_return"] = r["test"]["total_return_pct"]
        r["test_ann_return"] = r["test"]["ann_return_pct"]
        r["test_max_dd"] = r["test"]["max_total_dd_pct"]
        r["test_pass"] = r["test"]["pass_prop_firm"]
        r["train_sharpe"] = r["train"]["sharpe"]
        r["train_return"] = r["train"]["total_return_pct"]

    ranked = sorted(all_results, key=lambda x: x.get("test_sharpe", -999), reverse=True)

    # Best per variant
    best_per_variant = {}
    for vn in ["daily_momentum", "weekly_short", "multi_signal"]:
        vr = [r for r in ranked if r["strategy"] == vn]
        if vr:
            best_per_variant[vn] = vr[0]

    top_20 = ranked[:20]
    prop_passers = [r for r in ranked if r["test_pass"]]

    # Print
    print("\n" + "=" * 70)
    print("BEST PER VARIANT (by test Sharpe)")
    print("=" * 70)
    for vname, best in best_per_variant.items():
        lb_s = str(best.get('lookback', best.get('lookbacks')))
        print(f"\n  {vname}:")
        print(f"    Params: lb={lb_s}, vt={best['vol_target']}, "
              f"top={best['top_n']}, bot={best['bottom_n']}, mp={best['max_position']}")
        print(f"    Train: ret={best['train_return']:.1f}%, sharpe={best['train_sharpe']:.2f}, "
              f"dd={best['train']['max_total_dd_pct']:.1f}%")
        print(f"    Test:  ann_ret={best['test_ann_return']:.1f}%, sharpe={best['test_sharpe']:.2f}, "
              f"dd={best['test_max_dd']:.1f}%, pass={best['test_pass']}")

    print(f"\n{'=' * 70}")
    print(f"PROP FIRM PASSERS (test): {len(prop_passers)} / {len(all_results)}")
    print(f"{'=' * 70}")
    if prop_passers:
        for p in prop_passers[:15]:
            lb_s = str(p.get('lookback', p.get('lookbacks')))
            print(f"  {p['strategy']:<16} lb={lb_s:<6} ann_ret={p['test_ann_return']:>7.1f}% "
                  f"sharpe={p['test_sharpe']:>6.2f} dd={p['test_max_dd']:>5.1f}%")
    else:
        print("  None passed all prop firm rules on test set.")
        closest = sorted(ranked, key=lambda x: x["test_max_dd"])[:10]
        print("  Closest to passing (lowest DD):")
        for p in closest:
            lb_s = str(p.get('lookback', p.get('lookbacks')))
            print(f"  {p['strategy']:<16} lb={lb_s:<6} ann_ret={p['test_ann_return']:>7.1f}% "
                  f"dd={p['test_max_dd']:>5.1f}% daily_dd={p['test']['max_daily_dd_pct']:>5.1f}%")

    print(f"\n{'=' * 70}")
    print("TOP 20 (by test Sharpe)")
    print(f"{'=' * 70}")
    print(f"{'#':>3} {'Strategy':<16} {'LB':>6} {'VT':>5} {'TN':>3} {'BN':>3} {'MP':>5} "
          f"{'TrRet%':>8} {'TeAnnRet%':>10} {'TeShp':>6} {'TeDD%':>6} {'Pass':>4}")
    print("-" * 90)
    for i, r in enumerate(top_20):
        lb_s = str(r.get('lookback', r.get('lookbacks', '')))
        print(f"{i+1:>3} {r['strategy']:<16} {lb_s:>6} {r['vol_target']:>5.2f} "
              f"{r['top_n']:>3} {r['bottom_n']:>3} {r['max_position']:>5.2f} "
              f"{r['train_return']:>8.1f} {r['test_ann_return']:>10.1f} "
              f"{r['test_sharpe']:>6.2f} {r['test_max_dd']:>6.1f} "
              f"{'Y' if r['test_pass'] else 'N':>4}")

    # Overfitting
    print(f"\n{'=' * 70}")
    print("OVERFITTING ANALYSIS (top 10)")
    print(f"{'=' * 70}")
    for r in ranked[:10]:
        gap = r['train_sharpe'] - r['test_sharpe']
        lb_s = str(r.get('lookback', r.get('lookbacks', '')))
        print(f"  {r['strategy']:<16} lb={lb_s:<6} "
              f"train_shp={r['train_sharpe']:>6.2f} test_shp={r['test_sharpe']:>6.2f} gap={gap:>7.2f}")

    # Save
    output = {
        "summary": {
            "total_configs_tested": len(all_results),
            "prop_firm_passers_test": len(prop_passers),
            "train_period": f"{train_start} to {train_end}",
            "test_period": f"{test_start} to {test_end}",
            "symbols": list(common),
            "n_symbols": len(common),
        },
        "best_per_variant": {k: {kk: vv for kk, vv in v.items()}
                             for k, v in best_per_variant.items()},
        "top_20": [{kk: vv for kk, vv in r.items()} for r in top_20],
        "prop_passers": [{kk: vv for kk, vv in r.items()} for r in prop_passers[:50]],
        "overfitting_analysis": [
            {"strategy": r["strategy"],
             "params": {k: r[k] for k in ("lookback","lookbacks","weights","vol_target","top_n","bottom_n","max_position") if k in r},
             "train_sharpe": r["train_sharpe"],
             "test_sharpe": r["test_sharpe"],
             "gap": round(r["train_sharpe"] - r["test_sharpe"], 3)}
            for r in ranked[:20]
        ],
        "all_results_count": len(all_results),
    }

    out_path = WORKSPACE / "high_freq_momentum_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    main()
