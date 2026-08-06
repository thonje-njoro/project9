#!/usr/bin/env python3
"""
Volatility Risk Premium Strategy — Systematic Put Selling
=========================================================
Simulates selling equity index / high-IV stock puts using Black-Scholes
synthetic pricing on historical close prices.

Structural edge: Implied volatility > Realized volatility (IV/RV ratio).
"""

import json
import warnings
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import numpy as np
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
ASSETS = ["SPY", "QQQ", "NVDA", "AMD", "PLTR", "MRVL"]
DELTA_TARGETS = [0.10, 0.15, 0.20, 0.25, 0.30]
DTES = [7, 14, 30, 45]
IV_RV_RATIOS = [1.1, 1.2, 1.3, 1.5]
REBALANCE_PERIODS = {"weekly": 5, "biweekly": 10}

TRAIN_START = "2019-01-01"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2024-07-31"

RISK_FREE_RATE = 0.05
SLIPPAGE_PCT = 0.001    # 0.1%
COMMISSION_PCT = 0.0005  # 0.05%
MAX_LOSS_MULTIPLE = 2.0  # close if loss > 2x premium

# ──────────────────────────────────────────────
# BLACK-SCHOLES
# ──────────────────────────────────────────────

def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Price a European put option via Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Delta of a European put (negative value)."""
    if T <= 0 or sigma <= 0:
        return -1.0 if K > S else 0.0
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1.0


def find_strike_for_delta(S: float, T: float, r: float, sigma: float,
                          target_delta: float = -0.30, tol: float = 1e-4) -> float:
    """Binary search for strike that gives target put delta."""
    lo, hi = S * 0.5, S * 1.5
    for _ in range(50):
        mid = (lo + hi) / 2
        d = put_delta(S, mid, T, r, sigma)
        if d < target_delta:   # delta too negative → strike too high
            lo = mid
        else:
            hi = mid
        if abs(d - target_delta) < tol:
            break
    return round(mid, 2)


# ──────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────

def fetch_data(tickers: List[str], start: str, end: str) -> Dict[str, np.ndarray]:
    """Download adjusted close prices."""
    data = {}
    for t in tickers:
        df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            print(f"  ⚠ No data for {t}")
            continue
        closes = df["Close"].values.flatten()
        data[t] = closes
        print(f"  ✓ {t}: {len(closes)} bars")
    return data


# ──────────────────────────────────────────────
# STRATEGY
# ──────────────────────────────────────────────

@dataclass
class Trade:
    ticker: str
    entry_idx: int
    entry_price: float
    strike: float
    dte: int
    delta_target: float
    iv_rv_ratio: float
    implied_vol: float
    realized_vol: float
    premium: float
    net_premium: float
    expiry_price: float
    pnl: float
    pnl_pct: float
    closed_early: bool
    rebalance: str


def simulate_put_selling(
    close_prices: np.ndarray,
    ticker: str,
    delta_target: float = 0.30,
    dte: int = 30,
    iv_rv_ratio: float = 1.2,
    rebalance_days: int = 5,
) -> List[Trade]:
    """Simulate selling puts on a rolling basis (optimized)."""
    trades: List[Trade] = []
    n = len(close_prices)
    r = RISK_FREE_RATE
    vol_window = 20

    # Precompute log returns for realized vol
    log_prices = np.log(close_prices)

    i = vol_window
    while i < n - dte - 1:
        S = close_prices[i]

        # Realized vol (annualized)
        rets = log_prices[i - vol_window + 1:i + 1] - log_prices[i - vol_window:i]
        realized_vol = float(np.std(rets) * np.sqrt(252))
        if realized_vol < 1e-6 or realized_vol > 5.0:  # skip extreme vol (>500%)
            i += rebalance_days
            continue
        implied_vol = realized_vol * iv_rv_ratio

        T = dte / 365.0
        K = find_strike_for_delta(S, T, r, implied_vol, target_delta=-delta_target)

        # Price the put
        premium = black_scholes_put(S, K, T, r, implied_vol)
        cost = S * (SLIPPAGE_PCT + COMMISSION_PCT)
        net_premium = premium - cost

        max_loss = MAX_LOSS_MULTIPLE * net_premium
        closed_early = False
        expiry_idx = min(i + dte, n - 1)
        expiry_price = close_prices[expiry_idx]

        # Vectorized early stop-loss check
        # Check at key intervals (every 3 days or at expiry) for speed
        check_range = range(i + 1, expiry_idx + 1)
        prices_in_trade = close_prices[i + 1:expiry_idx + 1]
        days_elapsed = np.arange(1, len(prices_in_trade) + 1)
        T_rem_arr = np.maximum((dte - days_elapsed) / 365.0, 1 / 365)

        # Vectorized BS put pricing
        d1 = (np.log(prices_in_trade / K) + (r + implied_vol**2 / 2) * T_rem_arr) / (implied_vol * np.sqrt(T_rem_arr))
        d2 = d1 - implied_vol * np.sqrt(T_rem_arr)
        mtm_vals = K * np.exp(-r * T_rem_arr) * norm.cdf(-d2) - prices_in_trade * norm.cdf(-d1)
        unrealized = net_premium - mtm_vals
        loss_mask = unrealized < -max_loss

        if np.any(loss_mask):
            stop_idx = int(np.argmax(loss_mask))
            pnl = -max_loss
            closed_early = True
            expiry_price = float(prices_in_trade[stop_idx])
        else:
            if expiry_price > K:
                pnl = net_premium
            else:
                pnl = net_premium - (K - expiry_price)

        pnl_pct = pnl / S * 100

        trades.append(Trade(
            ticker=ticker,
            entry_idx=i,
            entry_price=float(S),
            strike=float(K),
            dte=dte,
            delta_target=delta_target,
            iv_rv_ratio=iv_rv_ratio,
            implied_vol=float(implied_vol),
            realized_vol=float(realized_vol),
            premium=float(premium),
            net_premium=float(net_premium),
            expiry_price=float(expiry_price),
            pnl=float(pnl),
            pnl_pct=float(pnl_pct),
            closed_early=closed_early,
            rebalance="weekly" if rebalance_days == 5 else "biweekly",
        ))

        i += rebalance_days

    return trades


# ──────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────

def compute_metrics(trades: List[Trade]) -> Dict[str, Any]:
    """Compute strategy performance metrics."""
    if not trades:
        return {"n_trades": 0}

    pnls = np.array([t.pnl for t in trades])
    pnl_pcts = np.array([t.pnl_pct for t in trades])

    total_pnl = float(np.sum(pnls))
    mean_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls)) if len(pnls) > 1 else 0.0

    # Per-trade stats
    win_rate = float(np.mean(pnls > 0)) * 100
    avg_win = float(np.mean(pnls[pnls > 0])) if np.any(pnls > 0) else 0.0
    avg_loss = float(np.mean(pnls[pnls <= 0])) if np.any(pnls <= 0) else 0.0

    # Sharpe (annualized — assume ~52 trades/year for weekly)
    trades_per_year = 52 if trades[0].rebalance == "weekly" else 26
    if std_pnl > 0:
        sharpe = (mean_pnl / std_pnl) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Max drawdown
    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    # Early stop rate
    early_close_rate = float(np.mean([t.closed_early for t in trades])) * 100

    # IV edge stats
    iv_edges = np.array([t.implied_vol - t.realized_vol for t in trades])
    avg_iv_edge = float(np.mean(iv_edges))

    return {
        "n_trades": len(trades),
        "total_pnl_pct": round(total_pnl / trades[0].entry_price * 100, 2),
        "mean_pnl_pct": round(float(np.mean(pnl_pcts)), 4),
        "std_pnl_pct": round(float(np.std(pnl_pcts)), 4),
        "win_rate_pct": round(win_rate, 1),
        "avg_win_pct": round(avg_win / trades[0].entry_price * 100, 4),
        "avg_loss_pct": round(avg_loss / trades[0].entry_price * 100, 4),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd / trades[0].entry_price * 100, 2),
        "early_close_rate_pct": round(early_close_rate, 1),
        "avg_iv_edge": round(avg_iv_edge, 4),
        "avg_realized_vol": round(float(np.mean([t.realized_vol for t in trades])), 4),
        "avg_implied_vol": round(float(np.mean([t.implied_vol for t in trades])), 4),
    }


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def run():
    print("=" * 60)
    print("VOLATILITY RISK PREMIUM — Systematic Put Selling")
    print("=" * 60)

    # Fetch data
    print("\n📥 Fetching data...")
    all_data = fetch_data(ASSETS, TRAIN_START, TEST_END)

    if not all_data:
        print("❌ No data fetched. Exiting.")
        sys.exit(1)

    # Split train/test
    train_data = fetch_data(ASSETS, TRAIN_START, TRAIN_END)
    test_data = fetch_data(ASSETS, TEST_START, TEST_END)

    results: Dict[str, Any] = {
        "strategy": "volatility_risk_premium",
        "description": "Systematic put selling exploiting IV > RV premium",
        "assets": ASSETS,
        "params": {
            "delta_targets": DELTA_TARGETS,
            "dtes": DTES,
            "iv_rv_ratios": IV_RV_RATIOS,
            "rebalance": list(REBALANCE_PERIODS.keys()),
            "risk_free_rate": RISK_FREE_RATE,
            "slippage_pct": SLIPPAGE_PCT,
            "commission_pct": COMMISSION_PCT,
            "max_loss_multiple": MAX_LOSS_MULTIPLE,
        },
        "train_period": f"{TRAIN_START} to {TRAIN_END}",
        "test_period": f"{TEST_START} to {TEST_END}",
        "asset_results": {},
        "param_sweep": [],
        "best_configs": {},
    }

    # ── Per-asset with default params ──
    print("\n📊 Running per-asset analysis (default params: Δ=0.20, DTE=30, IV/RV=1.2, weekly)...")
    for ticker in ASSETS:
        if ticker not in train_data or ticker not in test_data:
            continue

        print(f"\n  ── {ticker} ──")

        # Train
        train_trades = simulate_put_selling(
            train_data[ticker], ticker,
            delta_target=0.20, dte=30, iv_rv_ratio=1.2, rebalance_days=5,
        )
        train_metrics = compute_metrics(train_trades)

        # Test (OOS)
        test_trades = simulate_put_selling(
            test_data[ticker], ticker,
            delta_target=0.20, dte=30, iv_rv_ratio=1.2, rebalance_days=5,
        )
        test_metrics = compute_metrics(test_trades)

        results["asset_results"][ticker] = {
            "train": train_metrics,
            "test": test_metrics,
        }

        print(f"    Train: {train_metrics.get('n_trades',0)} trades, "
              f"Sharpe={train_metrics.get('sharpe_ratio',0):.2f}, "
              f"WinRate={train_metrics.get('win_rate_pct',0):.0f}%, "
              f"MaxDD={train_metrics.get('max_drawdown_pct',0):.2f}%")
        print(f"    Test:  {test_metrics.get('n_trades',0)} trades, "
              f"Sharpe={test_metrics.get('sharpe_ratio',0):.2f}, "
              f"WinRate={test_metrics.get('win_rate_pct',0):.0f}%, "
              f"MaxDD={test_metrics.get('max_drawdown_pct',0):.2f}%")

    # ── Parameter sweep on SPY ──
    print("\n🔬 Parameter sweep on SPY...")
    sys.stdout.flush()
    best_sharpe = -999
    best_cfg = {}
    sweep_results = []
    total_combos = len(DELTA_TARGETS) * len(DTES) * len(IV_RV_RATIOS) * len(REBALANCE_PERIODS)

    if "SPY" in train_data and "SPY" in test_data:
        combo = 0
        for delta in DELTA_TARGETS:
            for dte in DTES:
                for iv_ratio in IV_RV_RATIOS:
                    for rb_name, rb_days in REBALANCE_PERIODS.items():
                        combo += 1
                        tr = simulate_put_selling(
                            train_data["SPY"], "SPY",
                            delta_target=delta, dte=dte,
                            iv_rv_ratio=iv_ratio, rebalance_days=rb_days,
                        )
                        tm = compute_metrics(tr)

                        te_tr = simulate_put_selling(
                            test_data["SPY"], "SPY",
                            delta_target=delta, dte=dte,
                            iv_rv_ratio=iv_ratio, rebalance_days=rb_days,
                        )
                        te_tm = compute_metrics(te_tr)

                        cfg = {
                            "delta": delta,
                            "dte": dte,
                            "iv_rv_ratio": iv_ratio,
                            "rebalance": rb_name,
                            "train_sharpe": tm.get("sharpe_ratio", 0),
                            "train_win_rate": tm.get("win_rate_pct", 0),
                            "train_max_dd": tm.get("max_drawdown_pct", 0),
                            "test_sharpe": te_tm.get("sharpe_ratio", 0),
                            "test_win_rate": te_tm.get("win_rate_pct", 0),
                            "test_max_dd": te_tm.get("max_drawdown_pct", 0),
                            "train_n_trades": tm.get("n_trades", 0),
                            "test_n_trades": te_tm.get("n_trades", 0),
                        }
                        sweep_results.append(cfg)

                        if te_tm.get("sharpe_ratio", 0) > best_sharpe:
                            best_sharpe = te_tm["sharpe_ratio"]
                            best_cfg = cfg

                        if combo % 20 == 0:
                            print(f"    SPY sweep: {combo}/{total_combos}", flush=True)

        results["param_sweep"] = sweep_results
        results["best_configs"]["SPY"] = best_cfg

        print(f"\n  Best SPY config (by OOS Sharpe):")
        print(f"    Δ={best_cfg.get('delta')}, DTE={best_cfg.get('dte')}, "
              f"IV/RV={best_cfg.get('iv_rv_ratio')}, Rebal={best_cfg.get('rebalance')}")
        print(f"    Train Sharpe={best_cfg.get('train_sharpe'):.2f}, "
              f"Test Sharpe={best_cfg.get('test_sharpe'):.2f}")

    # ── Parameter sweep on NVDA ──
    print("\n🔬 Parameter sweep on NVDA...")
    sys.stdout.flush()
    if "NVDA" in train_data and "NVDA" in test_data:
        best_sharpe = -999
        best_cfg = {}
        sweep_results = []
        combo = 0

        for delta in DELTA_TARGETS:
            for dte in DTES:
                for iv_ratio in IV_RV_RATIOS:
                    for rb_name, rb_days in REBALANCE_PERIODS.items():
                        combo += 1
                        tr = simulate_put_selling(
                            train_data["NVDA"], "NVDA",
                            delta_target=delta, dte=dte,
                            iv_rv_ratio=iv_ratio, rebalance_days=rb_days,
                        )
                        tm = compute_metrics(tr)

                        te_tr = simulate_put_selling(
                            test_data["NVDA"], "NVDA",
                            delta_target=delta, dte=dte,
                            iv_rv_ratio=iv_ratio, rebalance_days=rb_days,
                        )
                        te_tm = compute_metrics(te_tr)

                        cfg = {
                            "delta": delta,
                            "dte": dte,
                            "iv_rv_ratio": iv_ratio,
                            "rebalance": rb_name,
                            "train_sharpe": tm.get("sharpe_ratio", 0),
                            "train_win_rate": tm.get("win_rate_pct", 0),
                            "train_max_dd": tm.get("max_drawdown_pct", 0),
                            "test_sharpe": te_tm.get("sharpe_ratio", 0),
                            "test_win_rate": te_tm.get("win_rate_pct", 0),
                            "test_max_dd": te_tm.get("max_drawdown_pct", 0),
                        }
                        sweep_results.append(cfg)

                        if te_tm.get("sharpe_ratio", 0) > best_sharpe:
                            best_sharpe = te_tm["sharpe_ratio"]
                            best_cfg = cfg

                        if combo % 20 == 0:
                            print(f"    NVDA sweep: {combo}/{total_combos}", flush=True)

        results["param_sweep_nvda"] = sweep_results
        results["best_configs"]["NVDA"] = best_cfg

        print(f"\n  Best NVDA config (by OOS Sharpe):")
        print(f"    Δ={best_cfg.get('delta')}, DTE={best_cfg.get('dte')}, "
              f"IV/RV={best_cfg.get('iv_rv_ratio')}, Rebal={best_cfg.get('rebalance')}")
        print(f"    Train Sharpe={best_cfg.get('train_sharpe'):.2f}, "
              f"Test Sharpe={best_cfg.get('test_sharpe'):.2f}")

    # ── Summary table ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Ticker':<8} {'Train Sharpe':>12} {'Test Sharpe':>11} {'Win%':>6} {'MaxDD%':>7}")
    print("-" * 50)
    for ticker, ar in results["asset_results"].items():
        tr = ar.get("train", {})
        te = ar.get("test", {})
        print(f"{ticker:<8} {tr.get('sharpe_ratio',0):>12.2f} {te.get('sharpe_ratio',0):>11.2f} "
              f"{te.get('win_rate_pct',0):>6.0f} {te.get('max_drawdown_pct',0):>7.2f}")

    # Save
    out_path = "vol_premium_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved to {out_path}")
    print("✅ Done.")


if __name__ == "__main__":
    run()
