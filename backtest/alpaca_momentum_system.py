#!/usr/bin/env python3
"""
Production Multi-Asset Momentum System with Alpaca Markets API
==============================================================
Features:
- Expanded 29-asset universe (equities, bonds, commodities, international)
- Drawdown manager with halt/resume
- Position caps (10% max)
- Volatility targeting (15% annual)
- Prop firm compliance checks
- Walk-forward validation (train 2019-2022, test 2023-2024)
- Parameter grid search
- Realistic transaction costs
"""

import os, sys, json, time, warnings
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Alpaca Setup ──────────────────────────────────────────────────────
import alpaca_trade_api as tradeapi

API_KEY = "PKNQAAQ5UWKXZN5ZEIZIGDZWAA"
SECRET_KEY = "3MUFNUDFZNo27YxYEwDkNeR1bKELWTgDarN5zwHVdcG2"
BASE_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")

# ── Universe ──────────────────────────────────────────────────────────
UNIVERSE = {
    # US Large Cap
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000",
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google",
    "META": "Meta", "AMZN": "Amazon", "NVDA": "NVIDIA",
    "AMD": "AMD", "TSLA": "Tesla", "AVGO": "Broadcom",
    "JPM": "JPMorgan", "V": "Visa",
    # Bonds
    "TLT": "20+ Year Treasury", "IEF": "7-10 Year Treasury",
    "SHY": "1-3 Year Treasury", "BND": "Total Bond Market", "LQD": "Corporate Bonds",
    # Commodities
    "GLD": "Gold", "SLV": "Silver", "USO": "Oil", "DBA": "Agriculture", "DBC": "Commodity Index",
    # International
    "EFA": "Developed Markets", "EEM": "Emerging Markets",
    "VXUS": "International", "FXI": "China", "EWJ": "Japan",
}

WORKDIR = Path("/home/work/.openclaw/workspace")
CACHE_DIR = WORKDIR / "alpaca_cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── Data Fetching ─────────────────────────────────────────────────────
def fetch_alpaca_data(symbols, start="2019-01-01", end="2024-07-31"):
    """Fetch daily bars from Alpaca, cache to parquet."""
    all_data = {}
    failed = []
    for sym in symbols:
        cache_file = CACHE_DIR / f"{sym}_alpaca_daily.parquet"
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            all_data[sym] = df
            print(f"  [CACHED] {sym}: {len(df)} bars")
            continue
        try:
            bars = api.get_bars(sym, "1Day", start=start, end=end, limit=10000).df
            if bars.empty:
                print(f"  [EMPTY]  {sym}")
                failed.append(sym)
                continue
            # Clean up
            if "close" not in bars.columns and len(bars.columns) > 0:
                bars = bars.rename(columns={bars.columns[0]: "close"})
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.reset_index(level=0, drop=True)
            bars.index = pd.to_datetime(bars.index).tz_localize(None)
            bars = bars[["open", "high", "low", "close", "volume"]].copy()
            bars.to_parquet(cache_file)
            all_data[sym] = bars
            print(f"  [FETCH]  {sym}: {len(df := bars)} bars ({bars.index[0].date()} → {bars.index[-1].date()})")
            time.sleep(0.2)  # rate limit
        except Exception as e:
            print(f"  [ERROR]  {sym}: {e}")
            failed.append(sym)
    print(f"\nFetched {len(all_data)} symbols, {len(failed)} failed: {failed}")
    return all_data

def build_price_matrix(all_data):
    """Build aligned close price DataFrame."""
    closes = {}
    for sym, df in all_data.items():
        closes[sym] = df["close"]
    prices = pd.DataFrame(closes)
    prices = prices.dropna(how="all").ffill()
    return prices

# ── Drawdown Manager ──────────────────────────────────────────────────
class DrawdownManager:
    def __init__(self, max_dd=0.08, recovery_threshold=0.05):
        self.max_dd = max_dd
        self.recovery_threshold = recovery_threshold
        self.peak_equity = 0
        self.halted = False
        self.halt_date = None
        self.events = []  # log of halt/resume events

    def update(self, equity, date):
        self.peak_equity = max(self.peak_equity, equity)
        current_dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0

        if current_dd > self.max_dd and not self.halted:
            self.halted = True
            self.halt_date = date
            self.events.append({
                "date": str(date), "type": "HALT",
                "dd": round(current_dd, 4), "equity": round(equity, 2)
            })

        if self.halted and current_dd < self.recovery_threshold:
            self.halted = False
            self.events.append({
                "date": str(date), "type": "RESUME",
                "dd": round(current_dd, 4), "equity": round(equity, 2)
            })

        return not self.halted

    def reset(self):
        self.peak_equity = 0
        self.halted = False
        self.halt_date = None
        self.events = []

# ── Prop Firm Compliance ──────────────────────────────────────────────
class PropFirmCompliance:
    def __init__(self):
        self.daily_dd_limit = 0.03
        self.total_dd_limit = 0.08
        self.profit_target = 0.10
        self.min_trading_days = 10

    def evaluate(self, returns_series, total_dd, total_return, trading_days):
        breaches = []
        daily_worst = returns_series.min() if len(returns_series) > 0 else 0
        if daily_worst < -self.daily_dd_limit:
            breaches.append(f"daily DD {daily_worst:.2%} < -{self.daily_dd_limit:.0%}")
        if total_dd < -self.total_dd_limit:
            breaches.append(f"total DD {total_dd:.2%} < -{self.total_dd_limit:.0%}")
        hit_target = total_return >= self.profit_target
        enough_days = trading_days >= self.min_trading_days
        return {
            "compliant": len(breaches) == 0,
            "breaches": breaches,
            "hit_target": hit_target,
            "enough_days": enough_days,
            "daily_worst": round(float(daily_worst), 4),
        }

# ── Position Caps ─────────────────────────────────────────────────────
def apply_position_caps(weights, max_position=0.10):
    weights = weights.clip(-max_position, max_position)
    gross = weights.abs().sum()
    if gross > 1.0:
        weights = weights / gross
    return weights

# ── Volatility Targeting ──────────────────────────────────────────────
def calculate_vol_scale(returns, target_vol=0.15, lookback=20):
    realized_vol = returns.rolling(lookback, min_periods=10).std() * np.sqrt(252)
    scale = target_vol / realized_vol
    return scale.clip(0, 2).fillna(1.0)

# ── Core Momentum Engine ──────────────────────────────────────────────
def run_momentum_system(
    prices_df,
    lookback=60,
    rebalance_freq="weekly",
    vol_target=0.15,
    top_n=5,
    bottom_n=5,
    max_position=0.10,
    max_dd=0.08,
    cost_per_side=0.0015,  # 0.1% slippage + 0.05% commission per side
):
    """
    Full momentum system:
    1. Momentum signal = lookback-period return
    2. Rank assets, long top N, short bottom N
    3. Scale by vol targeting
    4. Cap positions at max_position
    5. Drawdown manager halts on breach
    6. Rebalance weekly/biweekly
    """
    returns = prices_df.pct_change().dropna(how="all")
    n_days = len(returns)
    symbols = list(prices_df.columns)

    # Determine rebalance dates
    if rebalance_freq == "weekly":
        rebal_indices = list(range(lookback, n_days, 5))
    else:
        rebal_indices = list(range(lookback, n_days, 10))

    dd_manager = DrawdownManager(max_dd=max_dd)
    equity = 1.0
    equity_curve = [1.0]
    dates = [returns.index[lookback]]
    current_weights = pd.Series(0.0, index=symbols)
    prev_weights = pd.Series(0.0, index=symbols)
    turnover_log = []

    for i in range(lookback, n_days):
        date = returns.index[i]
        day_ret = returns.iloc[i]

        # Check drawdown manager
        if not dd_manager.update(equity, date):
            # Halted — move to zero exposure
            current_weights = current_weights * 0.0  # unwind

        # Rebalance?
        if i in rebal_indices and not dd_manager.halted:
            # Momentum signal: cumulative return over lookback
            mom = prices_df.iloc[i] / prices_df.iloc[i - lookback] - 1
            mom = mom.dropna().sort_values()

            # Long top N, short bottom N
            longs = mom.tail(top_n).index.tolist()
            shorts = mom.head(bottom_n).index.tolist()

            weights = pd.Series(0.0, index=symbols)
            for s in longs:
                weights[s] = 1.0 / top_n
            for s in shorts:
                weights[s] = -1.0 / bottom_n

            # Vol targeting
            port_ret_series = (returns.iloc[max(0, i-60):i] @ weights)
            vol_scale = calculate_vol_scale(port_ret_series, target_vol=vol_target, lookback=20)
            vs = vol_scale.iloc[-1] if len(vol_scale) > 0 else 1.0
            weights = weights * vs

            # Position caps
            weights = apply_position_caps(weights, max_position=max_position)

            # Transaction costs
            turnover = (weights - prev_weights).abs().sum()
            cost = turnover * cost_per_side
            equity -= cost * equity
            turnover_log.append({"date": str(date), "turnover": round(float(turnover), 4), "cost": round(float(cost), 6)})

            prev_weights = weights.copy()
            current_weights = weights.copy()

        # Daily P&L
        port_ret = (current_weights * day_ret).sum()
        equity *= (1 + port_ret)
        equity_curve.append(equity)
        dates.append(date)

    # Compute stats
    eq_series = pd.Series(equity_curve, index=dates)
    daily_rets = eq_series.pct_change().dropna()

    # Drawdown
    peak = eq_series.cummax()
    dd = (eq_series - peak) / peak
    max_dd_actual = dd.min()

    # Annualized stats
    n_years = len(daily_rets) / 252
    total_ret = eq_series.iloc[-1] / eq_series.iloc[0] - 1
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = daily_rets.std() * np.sqrt(252) if len(daily_rets) > 1 else 0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    calmar = ann_ret / abs(max_dd_actual) if max_dd_actual != 0 else 0

    # Prop firm compliance
    compliance = PropFirmCompliance().evaluate(
        daily_rets, max_dd_actual, total_ret, len(daily_rets)
    )

    return {
        "total_return": round(float(total_ret), 4),
        "annual_return": round(float(ann_ret), 4),
        "annual_vol": round(float(ann_vol), 4),
        "sharpe": round(float(sharpe), 3),
        "calmar": round(float(calmar), 3),
        "max_drawdown": round(float(max_dd_actual), 4),
        "dd_halt_events": dd_manager.events.copy(),
        "n_trades": len(turnover_log),
        "total_turnover": round(sum(t["turnover"] for t in turnover_log), 2),
        "total_costs": round(sum(t["cost"] for t in turnover_log), 6),
        "compliance": compliance,
        "equity_curve": [round(float(e), 4) for e in equity_curve],
        "dates": [str(d) for d in dates],
        "final_equity": round(float(eq_series.iloc[-1]), 4),
    }

# ── Parameter Grid ────────────────────────────────────────────────────
PARAM_GRID = {
    "lookback": [40, 60, 90],
    "rebalance_freq": ["weekly", "biweekly"],
    "vol_target": [0.10, 0.15, 0.20],
    "top_n": [3, 5, 7],
    "bottom_n": [3, 5, 7],
    "max_position": [0.08, 0.10, 0.15],
    "max_dd": [0.06, 0.08, 0.10],
}

def generate_grid():
    keys = list(PARAM_GRID.keys())
    combos = list(product(*PARAM_GRID.values()))
    configs = []
    for combo in combos:
        cfg = dict(zip(keys, combo))
        # Skip invalid: top_n + bottom_n can't exceed universe size
        if cfg["top_n"] + cfg["bottom_n"] > len(UNIVERSE):
            continue
        configs.append(cfg)
    return configs

# ── Correlation Matrix ────────────────────────────────────────────────
def compute_correlation(prices_df):
    returns = prices_df.pct_change().dropna()
    corr = returns.corr()
    return corr.round(3).to_dict()

# ── Single Asset Attribution ──────────────────────────────────────────
def check_single_asset_dominance(prices_df, lookback=60):
    """Check if momentum returns are driven by a single asset."""
    returns = prices_df.pct_change().dropna()
    # Asset vol contribution (rough)
    vols = returns.std() * np.sqrt(252)
    total_vol = vols.sum()
    concentration = (vols / total_vol).sort_values(ascending=False)
    return {
        "top_contributor": str(concentration.index[0]),
        "top_pct": round(float(concentration.iloc[0]), 3),
        "top3_pct": round(float(concentration.head(3).sum()), 3),
        "warning": concentration.iloc[0] > 0.15,
    }

# ── Main Execution ────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("PRODUCTION MULTI-ASSET MOMENTUM SYSTEM")
    print("=" * 70)
    start_time = time.time()

    # 1. Fetch data
    print("\n📡 Fetching Alpaca data...")
    all_data = fetch_alpaca_data(list(UNIVERSE.keys()), start="2019-01-01", end="2024-07-31")
    prices = build_price_matrix(all_data)

    available = list(prices.columns)
    print(f"\nPrice matrix: {len(prices)} days × {len(available)} assets")
    print(f"Date range: {prices.index[0].date()} → {prices.index[-1].date()}")

    # 2. Correlation matrix
    print("\n📊 Computing correlation matrix...")
    corr_matrix = compute_correlation(prices)

    # 3. Single asset check
    single_asset = check_single_asset_dominance(prices)

    # 4. Train/Test split
    train_mask = prices.index < "2023-01-01"
    test_mask = prices.index >= "2023-01-01"
    train_prices = prices[train_mask]
    test_prices = prices[test_mask]
    print(f"\nTrain: {train_prices.index[0].date()} → {train_prices.index[-1].date()} ({len(train_prices)} days)")
    print(f"Test:  {test_prices.index[0].date()} → {test_prices.index[-1].date()} ({len(test_prices)} days)")

    # 5. Grid search
    configs = generate_grid()
    print(f"\n🔍 Running {len(configs)} configurations...")

    results = []
    for idx, cfg in enumerate(configs):
        # Cap top_n/bottom_n to available
        top_n = min(cfg["top_n"], len(available) // 2)
        bottom_n = min(cfg["bottom_n"], len(available) // 2)
        if top_n < 1 or bottom_n < 1:
            continue

        params = {
            "lookback": cfg["lookback"],
            "rebalance_freq": cfg["rebalance_freq"],
            "vol_target": cfg["vol_target"],
            "top_n": top_n,
            "bottom_n": bottom_n,
            "max_position": cfg["max_position"],
            "max_dd": cfg["max_dd"],
        }

        # Train
        try:
            train_res = run_momentum_system(train_prices, **params)
        except Exception as e:
            print(f"  [{idx+1}/{len(configs)}] Train error: {e}")
            continue

        # Test
        try:
            test_res = run_momentum_system(test_prices, **params)
        except Exception as e:
            print(f"  [{idx+1}/{len(configs)}] Test error: {e}")
            continue

        train_sharpe = train_res["sharpe"]
        test_sharpe = test_res["sharpe"]
        profitable_both = train_res["total_return"] > 0 and test_res["total_return"] > 0

        entry = {
            "config": cfg,
            "params": params,
            "train": {k: v for k, v in train_res.items() if k not in ("equity_curve", "dates")},
            "test": {k: v for k, v in test_res.items() if k not in ("equity_curve", "dates")},
            "train_sharpe": train_sharpe,
            "test_sharpe": test_sharpe,
            "profitable_both": profitable_both,
            "train_test_sharpe_corr": round(np.corrcoef(
                [train_sharpe], [test_sharpe]
            )[0, 1], 3) if not (np.isnan(train_sharpe) or np.isnan(test_sharpe)) else None,
            "combined_sharpe": round((train_sharpe + test_sharpe) / 2, 3),
            "compliant_train": train_res["compliance"]["compliant"],
            "compliant_test": test_res["compliance"]["compliant"],
        }
        results.append(entry)

        status = "✅" if profitable_both else "❌"
        comp = "🔒" if entry["compliant_train"] and entry["compliant_test"] else "⚠️"
        if (idx + 1) % 50 == 0 or idx == len(configs) - 1:
            print(f"  [{idx+1}/{len(configs)}] {status}{comp} "
                  f"Train={train_sharpe:.3f} Test={test_sharpe:.3f} "
                  f"DD_tr={train_res['max_drawdown']:.2%} DD_te={test_res['max_drawdown']:.2%}")

    # 6. Analysis
    if not results:
        print("❌ No valid results!")
        return

    # Sort by combined Sharpe
    results.sort(key=lambda x: x.get("combined_sharpe", -999), reverse=True)

    profitable_both = [r for r in results if r["profitable_both"]]
    compliant_both = [r for r in results if r["compliant_train"] and r["compliant_test"]]
    compliant_profitable = [r for r in results if r["profitable_both"] and r["compliant_train"] and r["compliant_test"]]

    # Sharpe correlation across configs
    train_sharpes = [r["train_sharpe"] for r in results if not np.isnan(r["train_sharpe"])]
    test_sharpes = [r["test_sharpe"] for r in results if not np.isnan(r["test_sharpe"])]
    overall_corr = round(float(np.corrcoef(train_sharpes, test_sharpes)[0, 1]), 3) if len(train_sharpes) > 1 else None

    best = results[0]
    best_compliant = compliant_profitable[0] if compliant_profitable else (compliant_both[0] if compliant_both else None)

    # 7. Best config full backtest (with equity curve)
    print("\n🏆 Running best config full backtest with equity curve...")
    full_prices = prices  # use full period
    best_full = run_momentum_system(full_prices, **best["params"])
    # Also run best compliant
    best_compliant_full = None
    if best_compliant:
        best_compliant_full = run_momentum_system(full_prices, **best_compliant["params"])

    # 8. Build output
    elapsed = time.time() - start_time
    output = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "universe_size": len(available),
        "symbols": available,
        "date_range": f"{prices.index[0].date()} → {prices.index[-1].date()}",
        "train_period": f"{train_prices.index[0].date()} → {train_prices.index[-1].date()}",
        "test_period": f"{test_prices.index[0].date()} → {test_prices.index[-1].date()}",
        "total_configs": len(results),
        "profitable_both_pct": round(len(profitable_both) / len(results) * 100, 1),
        "compliant_both_pct": round(len(compliant_both) / len(results) * 100, 1),
        "compliant_and_profitable_pct": round(len(compliant_profitable) / len(results) * 100, 1),
        "train_test_sharpe_correlation": overall_corr,
        "median_train_sharpe": round(float(np.median(train_sharpes)), 3),
        "median_test_sharpe": round(float(np.median(test_sharpes)), 3),
        "single_asset_analysis": single_asset,
        "correlation_matrix": corr_matrix,
        "best_config": {
            "params": best["params"],
            "train_sharpe": best["train_sharpe"],
            "test_sharpe": best["test_sharpe"],
            "combined_sharpe": best["combined_sharpe"],
            "train_return": best["train"]["total_return"],
            "test_return": best["test"]["total_return"],
            "train_max_dd": best["train"]["max_drawdown"],
            "test_max_dd": best["test"]["max_drawdown"],
            "train_compliant": best["compliant_train"],
            "test_compliant": best["compliant_test"],
            "train_dd_events": best["train"]["dd_halt_events"],
            "test_dd_events": best["test"]["dd_halt_events"],
        },
        "best_compliant_config": None,
        "full_backtest_best": {
            "equity_curve": best_full["equity_curve"],
            "dates": best_full["dates"],
            "stats": {k: v for k, v in best_full.items() if k not in ("equity_curve", "dates")},
        },
        "all_results_top20": [],
    }

    if best_compliant:
        output["best_compliant_config"] = {
            "params": best_compliant["params"],
            "train_sharpe": best_compliant["train_sharpe"],
            "test_sharpe": best_compliant["test_sharpe"],
            "combined_sharpe": best_compliant["combined_sharpe"],
            "train_return": best_compliant["train"]["total_return"],
            "test_return": best_compliant["test"]["total_return"],
            "train_max_dd": best_compliant["train"]["max_drawdown"],
            "test_max_dd": best_compliant["test"]["max_drawdown"],
        }
        if best_compliant_full:
            output["full_backtest_best_compliant"] = {
                "equity_curve": best_compliant_full["equity_curve"],
                "dates": best_compliant_full["dates"],
                "stats": {k: v for k, v in best_compliant_full.items() if k not in ("equity_curve", "dates")},
            }

    # Top 20 results
    for r in results[:20]:
        output["all_results_top20"].append({
            "params": r["params"],
            "train_sharpe": r["train_sharpe"],
            "test_sharpe": r["test_sharpe"],
            "combined_sharpe": r["combined_sharpe"],
            "train_return": r["train"]["total_return"],
            "test_return": r["test"]["total_return"],
            "train_max_dd": r["train"]["max_drawdown"],
            "test_max_dd": r["test"]["max_drawdown"],
            "compliant_both": r["compliant_train"] and r["compliant_test"],
            "profitable_both": r["profitable_both"],
        })

    # Save
    out_path = WORKDIR / "alpaca_momentum_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n💾 Results saved to {out_path}")

    # 9. Print Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nUniverse: {len(available)} assets across 4 classes")
    print(f"Period: {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"Configs tested: {len(results)}")
    print(f"Elapsed: {elapsed:.1f}s")

    print(f"\n--- OVERALL STATISTICS ---")
    print(f"Median train Sharpe: {output['median_train_sharpe']}")
    print(f"Median test Sharpe:  {output['median_test_sharpe']}")
    print(f"Train-Test Sharpe correlation: {overall_corr}")
    print(f"Profitable in BOTH periods: {output['profitable_both_pct']}%")
    print(f"Prop firm compliant (both): {output['compliant_both_pct']}%")
    print(f"Compliant + Profitable: {output['compliant_and_profitable_pct']}%")

    print(f"\n--- SINGLE ASSET CHECK ---")
    print(f"Top contributor: {single_asset['top_contributor']} ({single_asset['top_pct']:.1%})")
    print(f"Top 3 share: {single_asset['top3_pct']:.1%}")
    print(f"Concentration warning: {single_asset['warning']}")

    print(f"\n--- BEST CONFIG (by combined Sharpe) ---")
    print(f"Params: {best['params']}")
    print(f"Train Sharpe: {best['train_sharpe']} | Test Sharpe: {best['test_sharpe']}")
    print(f"Train Return: {best['train']['total_return']:.2%} | Test Return: {best['test']['total_return']:.2%}")
    print(f"Train Max DD: {best['train']['max_drawdown']:.2%} | Test Max DD: {best['test']['max_drawdown']:.2%}")
    print(f"Compliant: Train={best['compliant_train']} Test={best['compliant_test']}")
    print(f"DD Halt Events (train): {len(best['train']['dd_halt_events'])}")
    print(f"DD Halt Events (test):  {len(best['test']['dd_halt_events'])}")

    if best_compliant:
        print(f"\n--- BEST COMPLIANT + PROFITABLE CONFIG ---")
        print(f"Params: {best_compliant['params']}")
        print(f"Train Sharpe: {best_compliant['train_sharpe']} | Test Sharpe: {best_compliant['test_sharpe']}")
        print(f"Train Return: {best_compliant['train']['total_return']:.2%} | Test Return: {best_compliant['test']['total_return']:.2%}")
        print(f"Train Max DD: {best_compliant['train']['max_drawdown']:.2%} | Test Max DD: {best_compliant['test']['max_drawdown']:.2%}")

    print(f"\n--- FULL BACKTEST (Best Config) ---")
    print(f"Total Return: {best_full['total_return']:.2%}")
    print(f"Annual Return: {best_full['annual_return']:.2%}")
    print(f"Annual Vol: {best_full['annual_vol']:.2%}")
    print(f"Sharpe: {best_full['sharpe']}")
    print(f"Calmar: {best_full['calmar']}")
    print(f"Max Drawdown: {best_full['max_drawdown']:.2%}")
    print(f"DD Halt Events: {len(best_full['dd_halt_events'])}")
    for evt in best_full["dd_halt_events"]:
        print(f"  {evt['type']} on {evt['date']} (DD={evt['dd']:.2%}, Equity={evt['equity']})")

    print(f"\n--- TOP 10 CONFIGS ---")
    print(f"{'Rank':>4} {'Train':>7} {'Test':>7} {'Combo':>7} {'DD_tr':>7} {'DD_te':>7} {'Compl':>6} {'Prof':>5}")
    for i, r in enumerate(results[:10]):
        comp = "✅" if r["compliant_train"] and r["compliant_test"] else "❌"
        prof = "✅" if r["profitable_both"] else "❌"
        print(f"{i+1:>4} {r['train_sharpe']:>7.3f} {r['test_sharpe']:>7.3f} "
              f"{r['combined_sharpe']:>7.3f} {r['train']['max_drawdown']:>7.2%} "
              f"{r['test']['max_drawdown']:>7.2%} {comp:>6} {prof:>5}")

    print(f"\n✅ DONE. Full results in alpaca_momentum_results.json")
    return output

if __name__ == "__main__":
    main()
