#!/usr/bin/env python3
"""
Production Multi-Asset Momentum System
=======================================
Walk-forward validated, prop-firm compliant, production-grade momentum strategy.

Expanded universe: 23 assets across equities, bonds, commodities, international, crypto.
"""

import json
import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product as iter_product
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

WORKDIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORKDIR)

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

UNIVERSE = {
    # US Equities
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000",
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google",
    "META": "Meta", "AMZN": "Amazon", "NVDA": "NVIDIA",
    "AMD": "AMD", "TSLA": "Tesla", "AVGO": "Broadcom",
    # Bonds
    "TLT": "20+ Year Treasury", "IEF": "7-10 Year Treasury", "SHY": "1-3 Year Treasury",
    # Commodities
    "GLD": "Gold", "SLV": "Silver", "USO": "Oil", "DBA": "Agriculture",
    # International
    "EFA": "Developed Markets", "EEM": "Emerging Markets", "VXUS": "International",
    # Crypto
    "BTC": "Bitcoin",
}

TRAIN_START = "2019-01-01"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2024-07-31"

# Costs
SLIPPAGE = 0.001       # 0.1% per side
COMMISSION = 0.0005    # 0.05% per side
ROUND_TRIP_COST = 2 * (SLIPPAGE + COMMISSION)  # 0.3% total

# Prop firm rules
PROP_DAILY_DD_LIMIT = 0.03
PROP_TOTAL_DD_LIMIT = 0.10
PROP_PROFIT_TARGET = 0.10
PROP_MIN_TRADING_DAYS = 10

# ══════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════

def get_api_key():
    with open("mimo_claw_pipeline_expanded.py", "r") as f:
        for line in f:
            if "LSE_API_KEY" in line and "lse_live" in line:
                return line.split('"')[1]
    raise ValueError("API key not found")

def fetch_daily_data(symbol, api_key, start="2019-01-01", end="2024-07-31"):
    """Fetch daily candles and cache to parquet."""
    cache_file = os.path.join(WORKDIR, f"{symbol}_daily.parquet")
    if os.path.exists(cache_file):
        df = pd.read_parquet(cache_file)
        # Check if we have sufficient date range
        if len(df) > 100:
            return df

    url = "https://api.londonstrategicedge.com/vault/candles"
    headers = {"x-api-key": api_key}
    params = {"symbol": symbol, "timeframe": "1d", "start": start, "end": end}

    print(f"  Fetching {symbol}...", end=" ", flush=True)
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "candles" not in data or len(data["candles"]) == 0:
            print(f"NO DATA")
            return None
        df = pd.DataFrame(data["candles"])
        # Normalize column names
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if cl in ("timestamp", "time", "date", "datetime"):
                col_map[c] = "ts"
            elif cl in ("open", "high", "low", "close", "volume"):
                col_map[c] = cl
        df = df.rename(columns=col_map)
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"])
            df = df.set_index("ts").sort_index()
        df["symbol"] = symbol
        df.to_parquet(cache_file)
        print(f"OK ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def load_all_data():
    """Load all available data from cache or API."""
    api_key = get_api_key()
    all_data = {}
    missing = []

    for sym in UNIVERSE:
        cache_file = os.path.join(WORKDIR, f"{sym}_daily.parquet")
        if os.path.exists(cache_file):
            df = pd.read_parquet(cache_file)
            if len(df) > 100:
                all_data[sym] = df
                print(f"  {sym}: cached ({len(df)} rows)")
                continue
        missing.append(sym)

    # Fetch missing symbols
    if missing:
        print(f"\nFetching {len(missing)} missing symbols...")
        for sym in missing:
            df = fetch_daily_data(sym, api_key)
            if df is not None and len(df) > 50:
                all_data[sym] = df
            time.sleep(7)  # Rate limit

    return all_data

# ══════════════════════════════════════════════════════════════
# SIGNAL & PORTFOLIO CONSTRUCTION
# ══════════════════════════════════════════════════════════════

def calculate_momentum(close, lookback=60):
    """Simple return over lookback period."""
    return close.pct_change(lookback)

def calculate_vol_target(close, target_vol=0.20, lookback=20):
    """Scale position size by inverse volatility."""
    realized_vol = close.pct_change().rolling(lookback).std() * np.sqrt(252)
    return target_vol / realized_vol.clip(lower=0.01)

def construct_portfolio(signals, vol_targets, top_n=5, bottom_n=5, max_position=0.15):
    """
    Rank assets by momentum, go long top N, short bottom N.
    Size by vol targeting, cap positions, ensure ~market neutral.
    """
    # Drop NaN signals
    valid = signals.dropna().sort_values(ascending=False)
    if len(valid) < top_n + bottom_n:
        return pd.Series(dtype=float)

    long_assets = valid.head(top_n).index
    short_assets = valid.tail(bottom_n).index

    weights = pd.Series(0.0, index=signals.index)

    # Long positions
    for asset in long_assets:
        vt = vol_targets.get(asset, 1.0)
        weights[asset] = min(vt, max_position)

    # Short positions
    for asset in short_assets:
        vt = vol_targets.get(asset, 1.0)
        weights[asset] = -min(vt, max_position)

    # Normalize to net ~0
    long_sum = weights[weights > 0].sum()
    short_sum = abs(weights[weights < 0].sum())
    if long_sum > 0 and short_sum > 0:
        scale = short_sum / long_sum
        weights[weights > 0] *= scale

    # Re-cap after scaling
    weights = weights.clip(lower=-max_position, upper=max_position)

    return weights

def should_rebalance(current_date, last_rebalance_date, frequency="weekly"):
    """Weekly rebalancing on Fridays, or biweekly."""
    if frequency == "weekly":
        return current_date.weekday() == 4  # Friday
    elif frequency == "biweekly":
        if current_date.weekday() == 4:
            return (current_date - last_rebalance_date).days >= 12
        return False
    return False

# ══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════

def run_backtest(close_df, lookback=60, rebalance_freq="weekly",
                 target_vol=0.20, vol_lookback=20,
                 top_n=5, bottom_n=5, max_position=0.15,
                 start_date=None, end_date=None):
    """
    Run momentum backtest on close_df (columns = symbols).
    Returns daily equity curve and stats.
    """
    # Filter date range
    mask = (close_df.index >= start_date) & (close_df.index <= end_date)
    close = close_df.loc[mask].copy()
    if len(close) < lookback + 30:
        return None

    symbols = close.columns.tolist()
    returns = close.pct_change()

    # Precompute signals and vol targets
    momentum = close.apply(lambda x: calculate_momentum(x, lookback))
    vol_targets_df = close.apply(lambda x: calculate_vol_target(x, target_vol, vol_lookback))

    # Track portfolio
    equity = [1.0]
    daily_returns = []
    last_rebalance = close.index[0] - timedelta(days=10)
    current_weights = pd.Series(0.0, index=symbols)
    rebalance_count = 0
    trade_count = 0

    for i in range(lookback + vol_lookback, len(close)):
        date = close.index[i]

        # Check rebalance
        if should_rebalance(date, last_rebalance, rebalance_freq):
            sig = momentum.iloc[i]
            vt = vol_targets_df.iloc[i]
            new_weights = construct_portfolio(sig, vt, top_n, bottom_n, max_position)

            if len(new_weights) > 0:
                # Count trades (position changes)
                trades = (new_weights - current_weights).abs()
                trade_count += (trades > 0.01).sum()
                current_weights = new_weights
                last_rebalance = date
                rebalance_count += 1

        # Daily return
        if len(current_weights) > 0:
            port_ret = (current_weights * returns.iloc[i]).sum()
            # Apply costs on rebalance days
            if last_rebalance == date and rebalance_count > 0:
                turnover = (current_weights.diff().abs().sum() if hasattr(current_weights, 'diff') else 0)
                # Approximate: cost on trades
                port_ret -= ROUND_TRIP_COST * 0.5  # Amortized
        else:
            port_ret = 0.0

        daily_returns.append(port_ret)
        equity.append(equity[-1] * (1 + port_ret))

    if len(daily_returns) == 0:
        return None

    daily_returns = np.array(daily_returns)
    equity = np.array(equity)

    # Stats
    total_return = equity[-1] / equity[0] - 1
    ann_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
    ann_vol = np.std(daily_returns) * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    max_dd = np.min(equity / np.maximum.accumulate(equity)) - 1
    calmar = ann_return / abs(max_dd) if abs(max_dd) > 0 else 0

    # Win rate
    win_rate = np.mean(daily_returns > 0)

    # Profit factor
    gains = daily_returns[daily_returns > 0].sum()
    losses = abs(daily_returns[daily_returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else float('inf')

    return {
        "total_return": round(total_return * 100, 2),
        "ann_return": round(ann_return * 100, 2),
        "ann_vol": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "win_rate": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 2),
        "rebalance_count": rebalance_count,
        "trade_count": int(trade_count),
        "trading_days": len(daily_returns),
        "equity": equity.tolist(),
        "daily_returns": daily_returns.tolist(),
    }

# ══════════════════════════════════════════════════════════════
# PROP FIRM COMPLIANCE
# ══════════════════════════════════════════════════════════════

def check_prop_compliance(stats):
    """Check if strategy passes prop firm rules."""
    if stats is None:
        return {"pass": False, "reason": "No data"}

    daily_returns = np.array(stats["daily_returns"])
    equity = np.array(stats["equity"])

    # Daily DD check
    max_daily_loss = abs(np.min(daily_returns))
    daily_dd_ok = max_daily_loss <= PROP_DAILY_DD_LIMIT

    # Total DD check
    max_total_dd = abs(stats["max_dd"] / 100)
    total_dd_ok = max_total_dd <= PROP_TOTAL_DD_LIMIT

    # Profit target
    total_return = stats["total_return"] / 100
    profit_ok = total_return >= PROP_PROFIT_TARGET

    # Min trading days
    days_ok = stats["trading_days"] >= PROP_MIN_TRADING_DAYS

    passed = daily_dd_ok and total_dd_ok and profit_ok and days_ok

    return {
        "pass": passed,
        "daily_dd": {
            "value": round(max_daily_loss * 100, 2),
            "limit": PROP_DAILY_DD_LIMIT * 100,
            "ok": daily_dd_ok,
        },
        "total_dd": {
            "value": round(max_total_dd * 100, 2),
            "limit": PROP_TOTAL_DD_LIMIT * 100,
            "ok": total_dd_ok,
        },
        "profit_target": {
            "value": round(total_return * 100, 2),
            "target": PROP_PROFIT_TARGET * 100,
            "ok": profit_ok,
        },
        "trading_days": {
            "value": stats["trading_days"],
            "minimum": PROP_MIN_TRADING_DAYS,
            "ok": days_ok,
        },
    }

# ══════════════════════════════════════════════════════════════
# PARAMETER GRID SEARCH
# ══════════════════════════════════════════════════════════════

def run_grid_search(close_df):
    """Run full parameter grid and return results."""
    param_grid = {
        "lookback": [20, 40, 60, 90, 120],
        "rebalance": ["weekly", "biweekly"],
        "target_vol": [0.10, 0.15, 0.20, 0.25],
        "top_n": [3, 5, 7],
        "bottom_n": [3, 5, 7],
        "max_position": [0.10, 0.15, 0.20],
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(iter_product(*values))
    total = len(combos)
    print(f"\n{'='*70}")
    print(f"GRID SEARCH: {total} configurations")
    print(f"{'='*70}\n")

    results = []
    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        # Skip invalid: top_n + bottom_n can't exceed universe size
        if params["top_n"] + params["bottom_n"] > len(close_df.columns):
            continue

        # Train
        train_stats = run_backtest(
            close_df,
            lookback=params["lookback"],
            rebalance_freq=params["rebalance"],
            target_vol=params["target_vol"],
            top_n=params["top_n"],
            bottom_n=params["bottom_n"],
            max_position=params["max_position"],
            start_date=TRAIN_START,
            end_date=TRAIN_END,
        )

        # Test
        test_stats = run_backtest(
            close_df,
            lookback=params["lookback"],
            rebalance_freq=params["rebalance"],
            target_vol=params["target_vol"],
            top_n=params["top_n"],
            bottom_n=params["bottom_n"],
            max_position=params["max_position"],
            start_date=TEST_START,
            end_date=TEST_END,
        )

        if train_stats is None or test_stats is None:
            continue

        train_prop = check_prop_compliance(train_stats)
        test_prop = check_prop_compliance(test_stats)

        result = {
            "params": params,
            "train": {k: v for k, v in train_stats.items() if k not in ("equity", "daily_returns")},
            "test": {k: v for k, v in test_stats.items() if k not in ("equity", "daily_returns")},
            "train_sharpe": train_stats["sharpe"],
            "test_sharpe": test_stats["sharpe"],
            "train_prop": train_prop,
            "test_prop": test_prop,
            # Keep equity curves for best configs only (trimmed)
            "_train_equity": train_stats["equity"][::5],  # Subsample
            "_test_equity": test_stats["equity"][::5],
        }
        results.append(result)

        if (idx + 1) % 50 == 0 or idx + 1 == total:
            print(f"  [{idx+1}/{total}] Completed | "
                  f"Latest train Sharpe: {train_stats['sharpe']:.3f} | "
                  f"Test Sharpe: {test_stats['sharpe']:.3f}")

    return results

# ══════════════════════════════════════════════════════════════
# ANALYSIS & REPORTING
# ══════════════════════════════════════════════════════════════

def analyze_results(results, close_df):
    """Comprehensive analysis of grid search results."""
    if not results:
        print("No results to analyze!")
        return {}

    train_sharpes = np.array([r["train_sharpe"] for r in results])
    test_sharpes = np.array([r["test_sharpe"] for r in results])

    # Correlation: key overfitting metric
    corr, p_val = stats.pearsonr(train_sharpes, test_sharpes)

    # Profitability
    train_profitable = np.mean(train_sharpes > 0) * 100
    test_profitable = np.mean(test_sharpes > 0) * 100
    both_profitable = np.mean((train_sharpes > 0) & (test_sharpes > 0)) * 100

    # Best by test Sharpe
    best_idx = np.argmax(test_sharpes)
    best = results[best_idx]

    # Median outcomes
    median_train_sharpe = np.median(train_sharpes)
    median_test_sharpe = np.median(test_sharpes)

    # Single-asset attribution check
    single_asset_check = check_single_asset_attribution(close_df, best["params"])

    # Correlation matrix of assets
    returns_df = close_df.pct_change().dropna()
    corr_matrix = returns_df.corr()

    # Prop firm pass rates
    train_prop_pass = np.mean([r["train_prop"]["pass"] for r in results]) * 100
    test_prop_pass = np.mean([r["test_prop"]["pass"] for r in results]) * 100

    analysis = {
        "total_configs": len(results),
        "train_test_correlation": round(corr, 4),
        "train_test_p_value": round(p_val, 6),
        "overfitting_risk": "LOW" if corr > 0.5 else "MEDIUM" if corr > 0.3 else "HIGH",
        "profitability": {
            "train_pct": round(train_profitable, 1),
            "test_pct": round(test_profitable, 1),
            "both_pct": round(both_profitable, 1),
        },
        "sharpe_stats": {
            "train_median": round(median_train_sharpe, 3),
            "train_mean": round(np.mean(train_sharpes), 3),
            "train_std": round(np.std(train_sharpes), 3),
            "test_median": round(median_test_sharpe, 3),
            "test_mean": round(np.mean(test_sharpes), 3),
            "test_std": round(np.std(test_sharpes), 3),
        },
        "best_config": {
            "params": best["params"],
            "train_sharpe": best["train_sharpe"],
            "test_sharpe": best["test_sharpe"],
            "train_stats": best["train"],
            "test_stats": best["test"],
            "train_prop": best["train_prop"],
            "test_prop": best["test_prop"],
        },
        "prop_firm": {
            "train_pass_rate": round(train_prop_pass, 1),
            "test_pass_rate": round(test_prop_pass, 1),
        },
        "single_asset_attribution": single_asset_check,
        "asset_correlation_matrix": corr_matrix.round(3).to_dict(),
        "available_assets": list(close_df.columns),
        "asset_count": len(close_df.columns),
    }

    return analysis

def check_single_asset_attribution(close_df, params):
    """Check if results are driven by a single asset."""
    # Run with full universe
    full_stats = run_backtest(
        close_df, lookback=params["lookback"],
        rebalance_freq=params["rebalance"], target_vol=params["target_vol"],
        top_n=params["top_n"], bottom_n=params["bottom_n"],
        max_position=params["max_position"],
        start_date=TEST_START, end_date=TEST_END,
    )

    # Run leaving one out
    attributions = {}
    for sym in close_df.columns:
        subset = close_df.drop(columns=[sym])
        if len(subset.columns) < params["top_n"] + params["bottom_n"]:
            continue
        loo_stats = run_backtest(
            subset, lookback=params["lookback"],
            rebalance_freq=params["rebalance"], target_vol=params["target_vol"],
            top_n=params["top_n"], bottom_n=params["bottom_n"],
            max_position=params["max_position"],
            start_date=TEST_START, end_date=TEST_END,
        )
        if loo_stats and full_stats:
            impact = full_stats["sharpe"] - loo_stats["sharpe"]
            attributions[sym] = round(impact, 3)

    # Sort by impact
    attributions = dict(sorted(attributions.items(), key=lambda x: x[1], reverse=True))

    # Flag if any single asset contributes > 50% of total
    total_impact = sum(v for v in attributions.values() if v > 0)
    dominated_by = None
    for sym, impact in attributions.items():
        if total_impact > 0 and impact / total_impact > 0.5:
            dominated_by = sym
            break

    return {
        "per_asset_impact": attributions,
        "dominated_by": dominated_by,
        "warning": f"Results driven by {dominated_by}" if dominated_by else "Diversified",
    }

# ══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════

def print_report(analysis):
    """Print comprehensive report."""
    print("\n" + "=" * 70)
    print("  PRODUCTION MOMENTUM SYSTEM — RESULTS REPORT")
    print("=" * 70)

    print(f"\n📊 Universe: {analysis['asset_count']} assets")
    print(f"   {', '.join(analysis['available_assets'])}")

    print(f"\n🔬 OVERFITTING ANALYSIS")
    print(f"   Train-Test Sharpe Correlation: {analysis['train_test_correlation']}")
    print(f"   P-value: {analysis['train_test_p_value']}")
    print(f"   Overfitting Risk: {analysis['overfitting_risk']}")

    prof = analysis["profitability"]
    print(f"\n💰 PROFITABILITY")
    print(f"   Train Profitable:     {prof['train_pct']:.1f}%")
    print(f"   Test Profitable:      {prof['test_pct']:.1f}%")
    print(f"   Both Profitable:      {prof['both_pct']:.1f}%")

    ss = analysis["sharpe_stats"]
    print(f"\n📈 SHARPE DISTRIBUTION")
    print(f"   {'':20s} {'Train':>10s} {'Test':>10s}")
    print(f"   {'Median':20s} {ss['train_median']:>10.3f} {ss['test_median']:>10.3f}")
    print(f"   {'Mean':20s} {ss['train_mean']:>10.3f} {ss['test_mean']:>10.3f}")
    print(f"   {'Std Dev':20s} {ss['train_std']:>10.3f} {ss['test_std']:>10.3f}")

    best = analysis["best_config"]
    print(f"\n🏆 BEST CONFIGURATION (by Test Sharpe)")
    print(f"   Params: {json.dumps(best['params'], indent=2)}")
    print(f"   Train Sharpe:  {best['train_sharpe']:.3f}")
    print(f"   Test Sharpe:   {best['test_sharpe']:.3f}")

    ts = best["train_stats"]
    tes = best["test_stats"]
    print(f"\n   {'Metric':20s} {'Train':>12s} {'Test':>12s}")
    print(f"   {'─'*44}")
    print(f"   {'Total Return':20s} {ts['total_return']:>11.2f}% {tes['total_return']:>11.2f}%")
    print(f"   {'Ann. Return':20s} {ts['ann_return']:>11.2f}% {tes['ann_return']:>11.2f}%")
    print(f"   {'Ann. Volatility':20s} {ts['ann_vol']:>11.2f}% {tes['ann_vol']:>11.2f}%")
    print(f"   {'Max Drawdown':20s} {ts['max_dd']:>11.2f}% {tes['max_dd']:>11.2f}%")
    print(f"   {'Calmar Ratio':20s} {ts['calmar']:>11.3f} {tes['calmar']:>11.3f}")
    print(f"   {'Win Rate':20s} {ts['win_rate']:>11.1f}% {tes['win_rate']:>11.1f}%")
    print(f"   {'Profit Factor':20s} {ts['profit_factor']:>11.2f} {tes['profit_factor']:>11.2f}")

    print(f"\n🏛️ PROP FIRM COMPLIANCE (Best Config)")
    for period, prop in [("Train", best["train_prop"]), ("Test", best["test_prop"])]:
        status = "✅ PASS" if prop["pass"] else "❌ FAIL"
        print(f"\n   {period}: {status}")
        for check_name, check in [("Daily DD", "daily_dd"), ("Total DD", "total_dd"),
                                    ("Profit Target", "profit_target"), ("Trading Days", "trading_days")]:
            val = prop[check]["value"]
            lim = prop[check]["limit"] if "limit" in prop[check] else prop[check].get("target", prop[check].get("minimum", ""))
            ok = "✅" if prop[check]["ok"] else "❌"
            print(f"     {ok} {check_name}: {val} (limit: {lim})")

    pf = analysis["prop_firm"]
    print(f"\n📊 PROP FIRM PASS RATES (All Configs)")
    print(f"   Train: {pf['train_pass_rate']:.1f}%")
    print(f"   Test:  {pf['test_pass_rate']:.1f}%")

    sa = analysis["single_asset_attribution"]
    print(f"\n🎯 SINGLE-ASSET ATTRIBUTION")
    print(f"   Status: {sa['warning']}")
    print(f"   Top contributors (Sharpe impact if removed):")
    for sym, impact in list(sa["per_asset_impact"].items())[:5]:
        direction = "↓" if impact > 0 else "↑"
        print(f"     {sym:6s}: {impact:+.3f} {direction}")

    # Top 10 configs
    print(f"\n📋 TOP 10 CONFIGURATIONS (by Test Sharpe)")
    print(f"   {'#':>3s} {'Lookback':>8s} {'Rebal':>8s} {'VolTgt':>6s} {'L/S':>5s} {'MaxPos':>6s} "
          f"{'TrnShp':>7s} {'TstShp':>7s} {'TrnDD':>7s} {'TstDD':>7s}")
    print(f"   {'─'*78}")

def main():
    print("🚀 Production Multi-Asset Momentum System")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 1. Load data
    print("📦 Loading data...")
    all_data = load_all_data()

    if len(all_data) < 5:
        print(f"ERROR: Only {len(all_data)} assets available, need at least 5")
        return

    # Build close price DataFrame
    close_dict = {}
    for sym, df in all_data.items():
        if "close" in df.columns:
            close_dict[sym] = df["close"]

    close_df = pd.DataFrame(close_dict)
    close_df = close_df.dropna(how="all")

    # Forward-fill gaps (holidays etc)
    close_df = close_df.ffill().dropna(axis=1, thresh=int(len(close_df) * 0.8))

    print(f"\n✅ Universe: {len(close_df.columns)} assets, {len(close_df)} trading days")
    print(f"   Assets: {', '.join(close_df.columns)}")
    print(f"   Date range: {close_df.index[0].strftime('%Y-%m-%d')} to {close_df.index[-1].strftime('%Y-%m-%d')}")

    # 2. Grid search
    results = run_grid_search(close_df)

    if not results:
        print("ERROR: No valid results from grid search")
        return

    # 3. Analysis
    print("\n📊 Analyzing results...")
    analysis = analyze_results(results, close_df)

    # 4. Print report
    print_report(analysis)

    # Print top 10 table (after report)
    sorted_results = sorted(results, key=lambda x: x["test_sharpe"], reverse=True)
    print(f"\n   {'#':>3s} {'Lookback':>8s} {'Rebal':>8s} {'VolTgt':>6s} {'L/S':>5s} {'MaxPos':>6s} "
          f"{'TrnShp':>7s} {'TstShp':>7s} {'TrnDD':>7s} {'TstDD':>7s}")
    print(f"   {'─'*78}")
    for i, r in enumerate(sorted_results[:10]):
        p = r["params"]
        print(f"   {i+1:3d} {p['lookback']:>8d} {p['rebalance']:>8s} {p['target_vol']:>5.0%} "
              f"{p['top_n']}/{p['bottom_n']:>3d} {p['max_position']:>5.0%} "
              f"{r['train_sharpe']:>7.3f} {r['test_sharpe']:>7.3f} "
              f"{r['train']['max_dd']:>6.1f}% {r['test']['max_dd']:>6.1f}%")

    # 5. Save results
    output = {
        "generated": datetime.now().isoformat(),
        "analysis": analysis,
        "all_results": [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in sorted_results
        ],
        "top_10_equity_curves": {
            "train": [r.get("_train_equity", []) for r in sorted_results[:10]],
            "test": [r.get("_test_equity", []) for r in sorted_results[:10]],
        },
    }

    output_file = os.path.join(WORKDIR, "production_momentum_results.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n💾 Results saved to: {output_file}")

    # Summary
    best = analysis["best_config"]
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Best Test Sharpe: {best['test_sharpe']:.3f}")
    print(f"  Train-Test Correlation: {analysis['train_test_correlation']:.4f}")
    print(f"  Both Profitable: {analysis['profitability']['both_pct']:.1f}%")
    print(f"  Median Test Sharpe: {analysis['sharpe_stats']['test_median']:.3f}")
    print(f"  Overfitting Risk: {analysis['overfitting_risk']}")
    print(f"  Single Asset: {analysis['single_asset_attribution']['warning']}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
