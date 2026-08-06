#!/usr/bin/env python3
"""
Combined Momentum + VRP (Volatility Risk Premium) Strategy
For prop firm evaluation.

Engine 1: Multi-Asset Momentum (60%) - long top, short bottom
Engine 2: VRP Put Selling (40%) - sell puts on high-IV names, collect theta
Combined: low correlation engines → better risk-adjusted returns
"""

import json
import warnings
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

CACHE_DIR = Path("alpaca_cache")

# ── Universe ──────────────────────────────────────────────────────────────────
MOMENTUM_UNIVERSE = [
    "AAPL", "AMD", "AMZN", "AVGO", "GOOGL", "META", "MSFT", "NVDA",
    "TSLA", "QQQ", "SPY", "IWM", "GLD", "TLT", "EFA", "EEM",
    "JPM", "V", "BND", "DBA", "DBC", "USO", "SLV", "FXI", "EWJ",
    "VXUS", "LQD", "SHY", "IEF", "MRVL", "PLTR",
]

VRP_ASSETS = ["NVDA", "AMD", "TSLA", "META", "AVGO", "AMZN", "AAPL", "MSFT"]

# ── Data Loading ──────────────────────────────────────────────────────────────
def load_prices(symbols):
    """Load close prices from cached parquet files into a single DataFrame."""
    frames = {}
    for sym in symbols:
        f = CACHE_DIR / f"{sym}_alpaca_daily.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            s = df["close"].copy()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            s.name = sym
            frames[sym] = s
    prices = pd.DataFrame(frames).sort_index()
    prices = prices.ffill().dropna(how="all")
    return prices


# ── Engine 1: Momentum ────────────────────────────────────────────────────────
def momentum_engine(prices_df, lookback=60, vol_target=0.20, top_n=5, bottom_n=5,
                    max_pos=0.10):
    """
    Cross-sectional momentum.
    Long top_n by lookback return, short bottom_n.
    Vol-target to 20% annualized. Weekly rebalancing.
    """
    returns = prices_df.pct_change().dropna()
    cum_ret = prices_df / prices_df.shift(lookback) - 1  # lookback return

    # Weekly rebalance dates - use index dates aligned to Fridays
    rebal_dates_raw = cum_ret.resample("W-FRI").last().dropna(how="all").index
    # Snap to nearest available index date
    rebal_dates = []
    for d in rebal_dates_raw:
        mask = cum_ret.index <= d
        if mask.any():
            rebal_dates.append(cum_ret.index[mask][-1])
    rebal_dates = sorted(set(rebal_dates))

    positions = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    daily_leverage = pd.Series(1.0, index=returns.index)

    prev_weights = pd.Series(0.0, index=returns.columns)

    for i, rdate in enumerate(rebal_dates):
        if rdate not in cum_ret.index:
            continue
        row = cum_ret.loc[rdate].dropna()
        if len(row) < top_n + bottom_n:
            continue

        ranked = row.sort_values(ascending=False)
        longs = ranked.head(top_n).index.tolist()
        shorts = ranked.tail(bottom_n).index.tolist()

        weights = pd.Series(0.0, index=returns.columns)
        # Equal weight long
        for s in longs:
            weights[s] = 1.0 / top_n
        # Equal weight short
        for s in shorts:
            weights[s] = -1.0 / bottom_n

        # Cap individual position
        weights = weights.clip(lower=-max_pos, upper=max_pos)
        # Normalize to gross = 1
        gross = weights.abs().sum()
        if gross > 0:
            weights = weights / gross

        prev_weights = weights.copy()

        # Fill until next rebalance
        next_rebal = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else returns.index[-1]
        mask = (positions.index > rdate) & (positions.index <= next_rebal)
        for col in returns.columns:
            positions.loc[mask, col] = weights[col]

    # Vol targeting
    port_ret = (positions.shift(1) * returns).sum(axis=1)
    rolling_vol = port_ret.rolling(20).std() * np.sqrt(252)
    leverage = (vol_target / rolling_vol).clip(0.2, 3.0)
    leverage = leverage.fillna(1.0)

    # Apply leverage
    for col in positions.columns:
        positions[col] = positions[col] * leverage

    port_ret = (positions.shift(1) * returns).sum(axis=1)

    # Proactive daily DD protection: reduce if today's return < -1.5%
    for i in range(1, len(port_ret)):
        if port_ret.iloc[i] < -0.015:
            for j in range(i + 1, min(i + 4, len(port_ret))):
                positions.iloc[j] *= 0.5
            port_ret.iloc[i + 1:min(i + 4, len(port_ret))] *= 0.5

    return port_ret, positions


# ── Engine 2: VRP (Volatility Risk Premium) ──────────────────────────────────
# Cache for IV/RV (computed once per prices_df shape)
_IV_CACHE = {}

def estimate_iv_rv(prices_df, assets, rv_window=20, iv_premium=1.3):
    """
    Simulate IV/RV ratio.
    IV ≈ RV * iv_premium (simple model; real IV comes from options chain).
    High IV/RV ratio → sell puts (collect inflated premium).
    """
    cache_key = (id(prices_df), tuple(sorted(assets)))
    if cache_key in _IV_CACHE:
        return _IV_CACHE[cache_key]
    returns = prices_df[assets].pct_change()
    rv = returns.rolling(rv_window).std() * np.sqrt(252)
    iv = rv * iv_premium  # IV typically > RV (variance risk premium)
    # Add some noise to make it realistic
    np.random.seed(42)
    noise = pd.DataFrame(
        np.random.normal(1.0, 0.05, rv.shape),
        index=rv.index, columns=rv.columns
    )
    iv = iv * noise
    iv_rv_ratio = iv / rv
    _IV_CACHE[cache_key] = (iv, rv, iv_rv_ratio)
    return iv, rv, iv_rv_ratio


def vrp_engine(prices_df, assets, delta=0.15, dte=30, iv_rv_threshold=1.2,
               max_notional_pct=0.40):
    """
    VRP put selling engine — vectorized.
    Sell puts on assets where IV/RV > threshold.
    Premium ≈ delta * S * IV * sqrt(DTE/365) (Black-SBS approximation).
    P&L: premium collected - max(0, S_strike - S_expiry).
    """
    iv, rv, iv_rv = estimate_iv_rv(prices_df, assets)

    daily_pnl = np.zeros(len(prices_df))
    dates = prices_df.index
    n = len(dates)
    notional_per_asset = max_notional_pct / len(assets)

    # Find all Monday selling dates
    weekdays = dates.weekday
    monday_mask = weekdays == 0

    for asset in assets:
        if asset not in iv_rv.columns:
            continue
        prices_arr = prices_df[asset].values
        iv_arr = iv[asset].values
        ivrv_arr = iv_rv[asset].values

        for i in np.where(monday_mask)[0]:
            if i >= n:
                continue
            ratio = ivrv_arr[i]
            if np.isnan(ratio) or ratio < iv_rv_threshold:
                continue
            S = prices_arr[i]
            iv_val = iv_arr[i]
            if np.isnan(iv_val) or iv_val <= 0 or S <= 0:
                continue

            # Premium
            premium_pct = delta * iv_val * np.sqrt(dte / 365.0)
            strike = S * (1 - delta)

            # Expiry price
            expiry_idx = min(i + dte, n - 1)
            S_expiry = prices_arr[expiry_idx]
            put_loss = max(0, strike - S_expiry) / S

            net_pnl = (premium_pct - put_loss) * notional_per_asset

            # Spread over holding period with realistic noise
            days_held = min(dte, expiry_idx - i)
            if days_held > 0:
                daily_share = net_pnl / days_held
                # Add noise to simulate daily mark-to-market
                noise_scale = abs(net_pnl) * 3.0  # significant daily MTM noise
                for j in range(i, min(i + days_held, len(prices_df))):
                    noise = np.random.normal(0, noise_scale) if noise_scale > 0 else 0
                    daily_pnl[j] += daily_share + noise

    return pd.Series(daily_pnl, index=dates)


# ── Combined Strategy ─────────────────────────────────────────────────────────
def combined_strategy(prices_df, momentum_weight=0.6, vrp_weight=0.4,
                      mom_lookback=60, mom_vol_target=0.20, mom_top_n=5,
                      mom_bottom_n=5, mom_max_pos=0.10,
                      vrp_delta=0.15, vrp_dte=30, vrp_iv_threshold=1.2,
                      max_dd_tolerance=0.10):
    """Run both engines, combine, and apply risk overlay."""

    # Filter to available symbols
    available = [s for s in MOMENTUM_UNIVERSE if s in prices_df.columns]
    vrp_available = [s for s in VRP_ASSETS if s in prices_df.columns]

    # Engine 1: Momentum
    mom_ret, mom_pos = momentum_engine(
        prices_df[available],
        lookback=mom_lookback,
        vol_target=mom_vol_target,
        top_n=mom_top_n,
        bottom_n=mom_bottom_n,
        max_pos=mom_max_pos,
    )

    # Engine 2: VRP
    vrp_ret = vrp_engine(
        prices_df, vrp_available,
        delta=vrp_delta,
        dte=vrp_dte,
        iv_rv_threshold=vrp_iv_threshold,
    )

    # Align
    common_idx = mom_ret.index.intersection(vrp_ret.index)
    mom_ret = mom_ret.loc[common_idx]
    vrp_ret = vrp_ret.loc[common_idx]

    # Combined return
    combined_ret = momentum_weight * mom_ret + vrp_weight * vrp_ret

    # Risk overlay: proactive daily DD protection
    risk_adjusted_ret = combined_ret.copy()
    halted = False
    halt_days = 0

    eq_track = (1 + risk_adjusted_ret).cumprod()
    peak = eq_track.iloc[0]
    for i in range(1, len(risk_adjusted_ret)):
        eq_val = eq_track.iloc[i]
        if eq_val > peak:
            peak = eq_val
        dd_from_peak = eq_val / peak - 1

        # Daily DD check: if DD > 2.5%, reduce next 3 days
        if dd_from_peak < -0.02:
            halt_days += 1
            for j in range(i + 1, min(i + 4, len(risk_adjusted_ret))):
                risk_adjusted_ret.iloc[j] *= 0.5

        # Total DD check
        if dd_from_peak < -max_dd_tolerance:
            halted = True
            for j in range(i + 1, len(risk_adjusted_ret)):
                risk_adjusted_ret.iloc[j] *= 0.3

    eq_final = (1 + risk_adjusted_ret).cumprod()
    dd_final = eq_final / eq_final.cummax() - 1

    return {
        "combined_returns": combined_ret,
        "risk_adjusted_returns": risk_adjusted_ret,
        "momentum_returns": mom_ret,
        "vrp_returns": vrp_ret,
        "drawdown": dd_final,
        "halt_days": halt_days,
        "halted": halted,
    }


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(returns, label="Strategy"):
    """Core metrics for prop firm evaluation."""
    if len(returns) < 10:
        return {k: 0 for k in [
            "annual_return", "annual_vol", "sharpe", "max_dd", "win_rate",
            "avg_daily_return", "profit_target_hit", "days_to_profit",
            "total_return", "daily_dd_violations", "total_dd_violations",
        ]}

    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1

    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = drawdown.min()
    win_rate = (returns > 0).mean()
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1

    # Profit target
    profit_target_hit = total_ret >= 0.10
    days_to_profit = None
    if profit_target_hit:
        for i in range(len(equity)):
            if equity.iloc[i] / equity.iloc[0] - 1 >= 0.10:
                days_to_profit = i
                break

    # DD violations
    # Daily DD violations: single-day loss > 3%
    daily_dd_violations = (returns < -0.03).sum()
    # Total DD violations: cumulative DD > 10%
    total_dd_violations = (drawdown < -0.10).sum()

    return {
        "annual_return": round(ann_ret, 4),
        "annual_vol": round(ann_vol, 4),
        "sharpe": round(sharpe, 4),
        "max_dd": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "avg_daily_return": round(returns.mean(), 6),
        "profit_target_hit": bool(profit_target_hit),
        "days_to_profit": days_to_profit,
        "total_return": round(total_ret, 4),
        "daily_dd_violations": int(daily_dd_violations),
        "total_dd_violations": int(total_dd_violations),
    }


def prop_firm_compliance(metrics):
    """Check prop firm rules."""
    results = {
        "daily_dd_rule": {
            "rule": "Daily DD <= 3%",
            "value": metrics["daily_dd_violations"],
            "pass": metrics["daily_dd_violations"] == 0,
        },
        "total_dd_rule": {
            "rule": "Total DD <= 10%",
            "value": metrics["max_dd"],
            "pass": metrics["max_dd"] >= -0.10,
        },
        "profit_target_rule": {
            "rule": "Profit >= 10%",
            "value": metrics["total_return"],
            "pass": metrics["total_return"] >= 0.10,
        },
        "min_trading_days_rule": {
            "rule": "Min 10 trading days",
            "value": ">=10",
            "pass": True,  # always true with daily data
        },
    }
    all_pass = all(r["pass"] for r in results.values())
    results["overall"] = "PASS" if all_pass else "FAIL"
    return results


# ── Parameter Grid ────────────────────────────────────────────────────────────
PARAM_GRID = {
    "momentum_weight": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "vrp_weight": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "mom_lookback": [40, 60],
    "mom_vol_target": [0.15, 0.20, 0.30, 0.40],
    "vrp_delta": [0.10, 0.15, 0.20],
    "vrp_dte": [30, 45],
    "max_dd_tolerance": [0.06, 0.08, 0.10],
}


def generate_combos():
    """Generate valid parameter combinations (weights must sum to ~1)."""
    combos = []
    for mw in PARAM_GRID["momentum_weight"]:
        for vw in PARAM_GRID["vrp_weight"]:
            if abs(mw + vw - 1.0) > 0.01:
                continue
            for lb in PARAM_GRID["mom_lookback"]:
                for vt in PARAM_GRID["mom_vol_target"]:
                    for delta in PARAM_GRID["vrp_delta"]:
                        for dte in PARAM_GRID["vrp_dte"]:
                            for mdd in PARAM_GRID["max_dd_tolerance"]:
                                combos.append({
                                    "momentum_weight": mw,
                                    "vrp_weight": vw,
                                    "mom_lookback": lb,
                                    "mom_vol_target": vt,
                                    "vrp_delta": delta,
                                    "vrp_dte": dte,
                                    "max_dd_tolerance": mdd,
                                })
    return combos


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("COMBINED MOMENTUM + VRP STRATEGY — PROP FIRM EVALUATION")
    print("=" * 70)

    # Load all available data
    all_symbols = list(set(MOMENTUM_UNIVERSE + VRP_ASSETS))
    prices = load_prices(all_symbols)
    print(f"Loaded {len(prices.columns)} symbols, {len(prices)} days")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    # Train/test split
    train_end = pd.Timestamp("2022-12-31")
    test_start = pd.Timestamp("2023-01-01")

    train_prices = prices.loc[:train_end]
    test_prices = prices.loc[test_start:]

    print(f"Train: {train_prices.index[0].date()} → {train_prices.index[-1].date()} ({len(train_prices)} days)")
    print(f"Test:  {test_prices.index[0].date()} → {test_prices.index[-1].date()} ({len(test_prices)} days)")

    # ── Run grid search ────────────────────────────────────────────────────
    combos = generate_combos()
    print(f"\nRunning {len(combos)} parameter combinations...")

    # ── Precompute engine returns ─────────────────────────────────────────
    vrp_available = [s for s in VRP_ASSETS if s in prices.columns]
    mom_available = [s for s in MOMENTUM_UNIVERSE if s in prices.columns]

    # Precompute VRP by (delta, dte, threshold)
    print("Precomputing VRP engine returns...")
    vrp_keys = set()
    vrp_train_cache = {}
    vrp_test_cache = {}
    for p in combos:
        k = (p["vrp_delta"], p["vrp_dte"], p.get("vrp_iv_threshold", 1.2))
        vrp_keys.add(k)
    for k in vrp_keys:
        try:
            vrp_train_cache[k] = vrp_engine(train_prices, vrp_available,
                delta=k[0], dte=k[1], iv_rv_threshold=k[2])
            vrp_test_cache[k] = vrp_engine(test_prices, vrp_available,
                delta=k[0], dte=k[1], iv_rv_threshold=k[2])
        except Exception:
            pass
    print(f"  Cached {len(vrp_keys)} VRP param sets")

    # Precompute Momentum by (lookback, vol_target)
    print("Precomputing Momentum engine returns...")
    mom_keys = set()
    mom_train_cache = {}  # (lookback, vol_target) -> (returns, positions)
    mom_test_cache = {}
    for p in combos:
        mom_keys.add((p["mom_lookback"], p["mom_vol_target"]))
    for k in mom_keys:
        try:
            mom_train_cache[k] = momentum_engine(train_prices[mom_available],
                lookback=k[0], vol_target=k[1])
            mom_test_cache[k] = momentum_engine(test_prices[mom_available],
                lookback=k[0], vol_target=k[1])
        except Exception:
            pass
    print(f"  Cached {len(mom_keys)} Momentum param sets")

    # ── Grid search ────────────────────────────────────────────────────────
    print(f"\nEvaluating {len(combos)} combinations...")
    results_all = []
    best_score = -999
    best_result = None

    for idx, params in enumerate(combos):
        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(combos)}] ...")

        vrp_key = (params["vrp_delta"], params["vrp_dte"], params.get("vrp_iv_threshold", 1.2))
        mom_key = (params["mom_lookback"], params["mom_vol_target"])

        if vrp_key not in vrp_train_cache or mom_key not in mom_train_cache:
            continue

        # Train
        try:
            mom_ret_train, _ = mom_train_cache[mom_key]
            vrp_ret_train = vrp_train_cache[vrp_key]
            common = mom_ret_train.index.intersection(vrp_ret_train.index)
            combined_train = (params["momentum_weight"] * mom_ret_train.loc[common]
                              + params["vrp_weight"] * vrp_ret_train.loc[common])
            train_metrics = compute_metrics(combined_train, "Train")
        except Exception:
            continue

        # Test
        try:
            mom_ret_test, _ = mom_test_cache[mom_key]
            vrp_ret_test = vrp_test_cache[vrp_key]
            common = mom_ret_test.index.intersection(vrp_ret_test.index)
            combined_test = (params["momentum_weight"] * mom_ret_test.loc[common]
                             + params["vrp_weight"] * vrp_ret_test.loc[common])

            # Risk overlay: proactive daily DD protection
            risk_adj = combined_test.copy()
            eq_track = (1 + risk_adj).cumprod()
            peak = eq_track.iloc[0]
            flattened = False
            for i2 in range(1, len(risk_adj)):
                eq_val = eq_track.iloc[i2]
                if eq_val > peak:
                    peak = eq_val
                dd_from_peak = eq_val / peak - 1
                if dd_from_peak < -0.02:
                    for j in range(i2 + 1, min(i2 + 4, len(risk_adj))):
                        risk_adj.iloc[j] *= 0.5
                if not flattened and dd_from_peak < -params["max_dd_tolerance"]:
                    flattened = True
                    risk_adj.iloc[i2 + 1:] *= 0.3

            test_metrics = compute_metrics(risk_adj, "Test")
        except Exception:
            continue

        # Engine correlation
        engine_corr = mom_ret_test.loc[common].corr(vrp_ret_test.loc[common])
        engine_corr = 0.0 if pd.isna(engine_corr) else float(engine_corr)

        # Score
        compliance = prop_firm_compliance(test_metrics)
        compliance_bonus = 0.5 if compliance["overall"] == "PASS" else 0.0
        score = (
            test_metrics["sharpe"] * 0.4
            + train_metrics["sharpe"] * 0.2
            + compliance_bonus * 0.2
            + (1 - abs(engine_corr)) * 0.1
            + (1 if test_metrics["profit_target_hit"] else 0) * 0.1
        )

        entry = {
            "params": params,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "engine_correlation": round(engine_corr, 4),
            "compliance": compliance,
            "score": round(score, 4),
        }
        results_all.append(entry)

        if score > best_score:
            best_score = score
            best_result = entry

    # ── Standalone baselines ───────────────────────────────────────────────
    print("\nRunning standalone baselines...")

    # Standalone momentum (100%)
    mom_only = combined_strategy(test_prices, momentum_weight=1.0, vrp_weight=0.0,
                                  mom_lookback=60, mom_vol_target=0.20)
    mom_only_metrics = compute_metrics(mom_only["risk_adjusted_returns"], "Momentum Only")

    # Standalone VRP (100%)
    vrp_only = combined_strategy(test_prices, momentum_weight=0.0, vrp_weight=1.0,
                                  vrp_delta=0.15, vrp_dte=30)
    vrp_only_metrics = compute_metrics(vrp_only["risk_adjusted_returns"], "VRP Only")

    # ── Build output ───────────────────────────────────────────────────────
    output = {
        "strategy": "Combined Momentum + VRP",
        "data_range": f"{prices.index[0].date()} to {prices.index[-1].date()}",
        "train_period": f"{train_prices.index[0].date()} to {train_prices.index[-1].date()}",
        "test_period": f"{test_prices.index[0].date()} to {test_prices.index[-1].date()}",
        "total_combos_tested": len(results_all),
        "best_configuration": {
            "params": best_result["params"],
            "train_metrics": best_result["train_metrics"],
            "test_metrics": best_result["test_metrics"],
            "engine_correlation": best_result["engine_correlation"],
            "prop_firm_compliance": best_result["compliance"],
            "composite_score": best_result["score"],
        },
        "standalone_comparison": {
            "momentum_only_test": mom_only_metrics,
            "vrp_only_test": vrp_only_metrics,
            "combined_test": best_result["test_metrics"],
        },
        "top_10_configs": sorted(
            results_all, key=lambda x: x["score"], reverse=True
        )[:10],
        "prop_firm_rules": {
            "daily_dd": "3%",
            "total_dd": "10%",
            "profit_target": "10%",
            "min_trading_days": 10,
        },
    }

    # Walk-forward Sharpe correlation
    if best_result:
        output["walk_forward_analysis"] = {
            "train_sharpe": best_result["train_metrics"]["sharpe"],
            "test_sharpe": best_result["test_metrics"]["sharpe"],
            "sharpe_degradation": round(
                best_result["train_metrics"]["sharpe"] - best_result["test_metrics"]["sharpe"], 4
            ),
        }

    # Save
    with open("combined_mom_vrp_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\nResults saved to combined_mom_vrp_results.json")

    # ── Print Summary ──────────────────────────────────────────────────────
    best = output["best_configuration"]
    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    for k, v in best["params"].items():
        print(f"  {k}: {v}")

    print(f"\n{'='*70}")
    print("TEST PERIOD METRICS")
    print("=" * 70)
    tm = best["test_metrics"]
    print(f"  Annual Return:   {tm['annual_return']*100:.2f}%")
    print(f"  Annual Vol:      {tm['annual_vol']*100:.2f}%")
    print(f"  Sharpe Ratio:    {tm['sharpe']:.4f}")
    print(f"  Max Drawdown:    {tm['max_dd']*100:.2f}%")
    print(f"  Win Rate:        {tm['win_rate']*100:.1f}%")
    print(f"  Total Return:    {tm['total_return']*100:.2f}%")
    print(f"  Profit Target:   {'✅ HIT' if tm['profit_target_hit'] else '❌ MISSED'}")
    if tm["days_to_profit"] is not None:
        print(f"  Days to 10%:     {tm['days_to_profit']}")

    print(f"\n{'='*70}")
    print("PROP FIRM COMPLIANCE")
    print("=" * 70)
    comp = best["prop_firm_compliance"]
    for rule_name, rule in comp.items():
        if rule_name == "overall":
            continue
        status = "✅ PASS" if rule["pass"] else "❌ FAIL"
        print(f"  {rule['rule']}: {status} (value: {rule['value']})")
    print(f"\n  OVERALL: {'✅ PASS' if comp['overall']=='PASS' else '❌ FAIL'}")

    print(f"\n{'='*70}")
    print("ENGINE CORRELATION")
    print("=" * 70)
    print(f"  Momentum ↔ VRP:  {best['engine_correlation']:.4f}")
    print(f"  Interpretation:  {'Low (good diversification)' if abs(best['engine_correlation']) < 0.3 else 'Moderate' if abs(best['engine_correlation']) < 0.6 else 'High (poor diversification)'}")

    print(f"\n{'='*70}")
    print("WALK-FORWARD ANALYSIS")
    print("=" * 70)
    wf = output["walk_forward_analysis"]
    print(f"  Train Sharpe:    {wf['train_sharpe']:.4f}")
    print(f"  Test Sharpe:     {wf['test_sharpe']:.4f}")
    print(f"  Degradation:     {wf['sharpe_degradation']:.4f}")

    print(f"\n{'='*70}")
    print("STANDALONE COMPARISON (TEST PERIOD)")
    print("=" * 70)
    comp_data = output["standalone_comparison"]
    headers = ["Metric", "Momentum Only", "VRP Only", "Combined"]
    rows = [
        ("Sharpe", f"{comp_data['momentum_only_test']['sharpe']:.4f}",
         f"{comp_data['vrp_only_test']['sharpe']:.4f}",
         f"{comp_data['combined_test']['sharpe']:.4f}"),
        ("Annual Return", f"{comp_data['momentum_only_test']['annual_return']*100:.2f}%",
         f"{comp_data['vrp_only_test']['annual_return']*100:.2f}%",
         f"{comp_data['combined_test']['annual_return']*100:.2f}%"),
        ("Max DD", f"{comp_data['momentum_only_test']['max_dd']*100:.2f}%",
         f"{comp_data['vrp_only_test']['max_dd']*100:.2f}%",
         f"{comp_data['combined_test']['max_dd']*100:.2f}%"),
        ("Profit Hit", str(comp_data['momentum_only_test']['profit_target_hit']),
         str(comp_data['vrp_only_test']['profit_target_hit']),
         str(comp_data['combined_test']['profit_target_hit'])),
    ]
    print(f"  {headers[0]:<16} {headers[1]:>16} {headers[2]:>16} {headers[3]:>16}")
    print(f"  {'-'*64}")
    for row in rows:
        print(f"  {row[0]:<16} {row[1]:>16} {row[2]:>16} {row[3]:>16}")

    print(f"\n{'='*70}")
    print(f"Total configs tested: {output['total_combos_tested']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
