#!/usr/bin/env python3
"""
Portfolio Assessment - Combines multiple strategy results into a unified portfolio.
Reads backtest results, builds combined portfolio, runs risk & prop firm compliance.
"""

import json
import os
import sys
import time
import math
from datetime import datetime
from pathlib import Path
from collections import defaultdict

WORKSPACE = Path("/home/work/.openclaw/workspace")

RESULT_FILES = {
    "backtest_v3": WORKSPACE / "backtest_v3_results.json",
    "pltr_gap_fill": WORKSPACE / "pltr_gap_fill_results.json",
    "regime_detection": WORKSPACE / "regime_detection_results.json",
    "xagusd_trend": WORKSPACE / "xagusd_trend_results.json",
}

OUTPUT_JSON = WORKSPACE / "portfolio_assessment.json"
OUTPUT_MD = WORKSPACE / "PORTFOLIO_REPORT.md"

# ─── Data Structures ───────────────────────────────────────────────────────

STRATEGY_SCHEMAS = {
    "backtest_v3": {
        "description": "Multi-timeframe momentum strategy",
        "asset_class": "equities",
        "typical_win_rate": 0.55,
        "typical_pf": 1.5,
        "typical_sharpe": 1.2,
    },
    "pltr_gap_fill": {
        "description": "Gap-fill mean reversion on PLTR",
        "asset_class": "equities",
        "typical_win_rate": 0.60,
        "typical_pf": 1.4,
        "typical_sharpe": 1.0,
    },
    "regime_detection": {
        "description": "Regime-adaptive allocation strategy",
        "asset_class": "multi-asset",
        "typical_win_rate": 0.52,
        "typical_pf": 1.6,
        "typical_sharpe": 1.3,
    },
    "xagusd_trend": {
        "description": "Silver trend-following strategy",
        "asset_class": "metals",
        "typical_win_rate": 0.48,
        "typical_pf": 1.8,
        "typical_sharpe": 1.1,
    },
}


# ─── Portfolio Class ───────────────────────────────────────────────────────

class Portfolio:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}  # symbol -> {entry, size, stop, strategy}
        self.daily_pnl = []
        self.trade_log = []
        self.max_positions = 3
        self.risk_per_trade = 0.01   # 1%
        self.daily_dd_limit = 0.03   # 3%
        self.total_dd_limit = 0.10   # 10%
        self.peak_capital = initial_capital

    def can_trade(self):
        if len(self.positions) >= self.max_positions:
            return False, "max_positions"
        if self.daily_loss_exceeded():
            return False, "daily_dd"
        if self.total_dd_exceeded():
            return False, "total_dd"
        return True, "ok"

    def daily_loss_exceeded(self):
        if not self.daily_pnl:
            return False
        last_day = self.daily_pnl[-1]
        return (last_day / self.capital) < -self.daily_dd_limit

    def total_dd_exceeded(self):
        drawdown = (self.peak_capital - self.capital) / self.peak_capital
        return drawdown > self.total_dd_limit

    def calculate_position_size(self, entry, stop):
        risk_per_unit = abs(entry - stop)
        if risk_per_unit == 0:
            return 0
        risk_amount = self.capital * self.risk_per_trade
        return risk_amount / risk_per_unit

    def record_trade(self, pnl, strategy, symbol):
        self.capital += pnl
        self.peak_capital = max(self.peak_capital, self.capital)
        self.trade_log.append({
            "strategy": strategy,
            "symbol": symbol,
            "pnl": pnl,
            "capital_after": self.capital,
            "timestamp": datetime.now().isoformat(),
        })

    def record_daily_pnl(self, pnl):
        self.daily_pnl.append(pnl)

    def get_drawdown_series(self):
        if not self.trade_log:
            return []
        running = self.initial_capital
        peak = running
        dd = []
        for t in self.trade_log:
            running = t["capital_after"]
            peak = max(peak, running)
            dd.append((peak - running) / peak)
        return dd


# ─── Prop Firm Compliance ──────────────────────────────────────────────────

def calc_win_rate(trades):
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    return wins / len(trades)

def calc_profit_factor(trades):
    gross_profit = sum(t["pnl"] for t in trades if t.get("pnl", 0) > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t.get("pnl", 0) < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss

def calc_sharpe(daily_returns, risk_free=0.0):
    if len(daily_returns) < 2:
        return 0.0
    mean_r = sum(daily_returns) / len(daily_returns)
    var = sum((r - mean_r) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0:
        return 0.0
    return (mean_r - risk_free) / std * math.sqrt(252)

def calc_sortino(daily_returns, risk_free=0.0):
    if len(daily_returns) < 2:
        return 0.0
    mean_r = sum(daily_returns) / len(daily_returns)
    downside = [r for r in daily_returns if r < risk_free]
    if not downside:
        return float("inf") if mean_r > risk_free else 0.0
    dd_var = sum((r - risk_free) ** 2 for r in downside) / len(downside)
    dd_std = math.sqrt(dd_var)
    if dd_std == 0:
        return 0.0
    return (mean_r - risk_free) / dd_std * math.sqrt(252)

def calc_max_drawdown(equity_curve):
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        peak = max(peak, val)
        dd = (peak - val) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
    return max_dd

def check_prop_firm(portfolio, daily_pnls=None):
    """Comprehensive prop firm rule check."""
    trades = portfolio.trade_log
    capital = portfolio.initial_capital

    daily_returns = []
    if daily_pnls:
        daily_returns = [p / capital for p in daily_pnls]
    elif trades:
        # approximate from trades
        prev = capital
        for t in trades:
            daily_returns.append((t["capital_after"] - prev) / prev)
            prev = t["capital_after"]

    equity_curve = [capital] + [t["capital_after"] for t in trades] if trades else [capital]
    cum_pnl = portfolio.capital - capital
    total_return = cum_pnl / capital

    # Daily drawdown check
    daily_dd_ok = True
    if daily_pnls:
        daily_dd_ok = all(p > -capital * 0.03 for p in daily_pnls)
    elif trades:
        running = capital
        peak = running
        for t in trades:
            running = t["capital_after"]
            peak = max(peak, running)
            dd = (peak - running) / peak
            if dd > 0.03:
                daily_dd_ok = False
                break

    # Cumulative drawdown
    max_dd = calc_max_drawdown(equity_curve)
    total_dd_ok = max_dd <= 0.10

    # Profit target
    profit_target_met = total_return >= 0.10

    # Trading days
    num_days = len(daily_pnls) if daily_pnls else len(trades)
    min_days_ok = num_days >= 10

    win_rate = calc_win_rate(trades) if trades else 0.0
    profit_factor = calc_profit_factor(trades) if trades else 0.0
    sharpe = calc_sharpe(daily_returns) if daily_returns else 0.0
    sortino = calc_sortino(daily_returns) if daily_returns else 0.0

    checks = {
        "daily_dd_ok": daily_dd_ok,
        "daily_dd_limit": "3%",
        "total_dd_ok": total_dd_ok,
        "total_dd_limit": "10%",
        "max_drawdown_pct": round(max_dd * 100, 2),
        "profit_target_met": profit_target_met,
        "profit_target": "10%",
        "total_return_pct": round(total_return * 100, 2),
        "min_trading_days_ok": min_days_ok,
        "min_trading_days": 10,
        "actual_trading_days": num_days,
        "win_rate_ok": win_rate > 0.45,
        "win_rate": round(win_rate, 4),
        "profit_factor_ok": profit_factor > 1.2,
        "profit_factor": round(profit_factor, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "all_pass": all([
            daily_dd_ok, total_dd_ok, profit_target_met,
            min_days_ok, win_rate > 0.45, profit_factor > 1.2,
        ]),
    }
    return checks


# ─── Correlation & Diversification ─────────────────────────────────────────

def calc_correlation(series_a, series_b):
    """Pearson correlation between two return series."""
    n = min(len(series_a), len(series_b))
    if n < 2:
        return 0.0
    a, b = series_a[:n], series_b[:n]
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n)) / (n - 1)
    var_a = sum((x - mean_a) ** 2 for x in a) / (n - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (n - 1)
    std_a = math.sqrt(var_a) if var_a > 0 else 0
    std_b = math.sqrt(var_b) if var_b > 0 else 0
    if std_a == 0 or std_b == 0:
        return 0.0
    return cov / (std_a * std_b)

def build_correlation_matrix(strategy_returns):
    """Build pairwise correlation matrix between strategies."""
    names = list(strategy_returns.keys())
    matrix = {}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i <= j:
                corr = calc_correlation(strategy_returns[a], strategy_returns[b])
                matrix[(a, b)] = round(corr, 4)
                matrix[(b, a)] = round(corr, 4)
    return matrix, names

def diversification_benefit(strategy_returns, weights):
    """Calculate diversification ratio: 1 - portfolio_vol / weighted_sum_vols."""
    names = list(strategy_returns.keys())
    # Portfolio returns (weighted sum)
    n = min(len(strategy_returns[n]) for n in names)
    port_returns = []
    for i in range(n):
        r = sum(weights.get(nm, 0) * strategy_returns[nm][i] for nm in names)
        port_returns.append(r)

    port_vol = _std(port_returns)
    weighted_vol_sum = sum(
        weights.get(nm, 0) * _std(strategy_returns[nm][:n]) for nm in names
    )
    if weighted_vol_sum == 0:
        return 0.0
    return 1 - port_vol / weighted_vol_sum

def _std(series):
    if len(series) < 2:
        return 0.0
    m = sum(series) / len(series)
    v = sum((x - m) ** 2 for x in series) / (len(series) - 1)
    return math.sqrt(v)


# ─── Allocation Engine ─────────────────────────────────────────────────────

def compute_allocations(strategy_meta, corr_matrix, strategy_names, max_corr=0.7):
    """
    Compute allocation weights. Penalize highly correlated strategies.
    Equal-weight base, reduce if corr > threshold.
    """
    n = len(strategy_names)
    if n == 0:
        return {}

    # Score each strategy by Sharpe (use typical if no real data)
    scores = {}
    for nm in strategy_names:
        meta = strategy_meta.get(nm, {})
        scores[nm] = meta.get("sharpe", meta.get("typical_sharpe", 1.0))

    # Equal-weight base
    base_w = {nm: 1.0 / n for nm in strategy_names}

    # Find highly correlated pairs and reduce the lower-scored one
    adjusted = dict(base_w)
    penalized = set()
    for i, a in enumerate(strategy_names):
        for j, b in enumerate(strategy_names):
            if i >= j:
                continue
            corr = corr_matrix.get((a, b), 0)
            if corr > max_corr:
                loser = a if scores[a] < scores[b] else b
                if loser not in penalized:
                    adjusted[loser] *= 0.5
                    penalized.add(loser)

    # Re-normalize
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: round(v / total, 4) for k, v in adjusted.items()}
    return adjusted


# ─── Result File Readers ───────────────────────────────────────────────────

def extract_strategy_results(name, data):
    """Normalize different result formats into a common structure."""
    result = {
        "name": name,
        "description": STRATEGY_SCHEMAS.get(name, {}).get("description", ""),
        "available": True,
        "raw": data,
    }

    if not isinstance(data, dict):
        return result

    # ── Regime detection format: summary is a list of symbol stats ──
    if "summary" in data and isinstance(data["summary"], list) and "strategy_results" in data:
        # Extract best sharpe across all symbols as the strategy's sharpe
        summaries = data["summary"]
        best_sharpe = max((s.get("best_sharpe", -999) for s in summaries), default=-999)
        best_pnl = sum(s.get("best_pnl", 0) for s in summaries)
        total_trades = sum(s.get("baseline_trades", 0) for s in summaries)
        best_improvement = max((s.get("sharpe_improvement", 0) for s in summaries), default=0)

        result["sharpe"] = best_sharpe if best_sharpe > -999 else None
        result["total_return"] = best_pnl / 100000 if best_pnl != 0 else None
        result["total_return_dollar"] = best_pnl
        result["num_trades"] = total_trades
        result["symbols"] = [s.get("symbol") for s in summaries]
        result["best_filters"] = {s.get("symbol"): s.get("best_filter") for s in summaries}
        result["sharpe_improvement"] = best_improvement
        result["daily_returns"] = []
        result["equity_curve"] = []
        result["win_rate"] = None
        result["profit_factor"] = None
        result["max_drawdown"] = None
        return result

    # ── Standard flat/nested formats ──
    metrics = data if "total_return" in data or "win_rate" in data else {}
    for key in ["results", "metrics", "summary", "stats", "performance"]:
        if key in data and isinstance(data[key], dict):
            metrics = {**metrics, **data[key]}

    result["total_return"] = metrics.get("total_return", metrics.get("return", metrics.get("total_pnl_pct", None)))
    result["win_rate"] = metrics.get("win_rate", metrics.get("winrate", None))
    result["profit_factor"] = metrics.get("profit_factor", metrics.get("pf", None))
    result["sharpe"] = metrics.get("sharpe", metrics.get("sharpe_ratio", None))
    result["max_drawdown"] = metrics.get("max_drawdown", metrics.get("max_dd", metrics.get("mdd", None)))
    result["num_trades"] = metrics.get("num_trades", metrics.get("total_trades", metrics.get("trades", None)))
    result["daily_returns"] = metrics.get("daily_returns", metrics.get("returns", []))
    result["equity_curve"] = metrics.get("equity_curve", metrics.get("equity", []))

    if result["total_return"] is not None and isinstance(result["total_return"], (int, float)):
        if abs(result["total_return"]) > 10:
            result["total_return_dollar"] = result["total_return"]
            result["total_return"] = result["total_return"] / 100000

    return result


def load_results():
    """Load all available result files."""
    loaded = {}
    missing = []
    for name, path in RESULT_FILES.items():
        if path.exists():
            try:
                with open(path) as f:
                    raw = json.load(f)
                loaded[name] = extract_strategy_results(name, raw)
                print(f"  ✓ Loaded {name} from {path.name}")
            except Exception as e:
                print(f"  ✗ Error reading {name}: {e}")
                missing.append(name)
        else:
            missing.append(name)
            print(f"  ○ {name}: {path.name} not found")
    return loaded, missing


# ─── Simulation (when no real data) ────────────────────────────────────────

def simulate_strategy_returns(name, num_days=60, seed=None):
    """Generate synthetic returns for framework/demo purposes."""
    import random
    if seed is not None:
        random.seed(seed)
    meta = STRATEGY_SCHEMAS.get(name, {})
    # Base parameters
    daily_mean = 0.001  # ~25% annual
    daily_std = 0.015   # ~24% annual vol
    returns = []
    for _ in range(num_days):
        r = random.gauss(daily_mean, daily_std)
        returns.append(round(r, 6))
    return returns


def generate_demo_trades(name, returns, capital=100000):
    """Convert returns into trade-like records."""
    trades = []
    running = capital
    for i, r in enumerate(returns):
        pnl = running * r
        running += pnl
        trades.append({
            "strategy": name,
            "symbol": name.upper(),
            "pnl": round(pnl, 2),
            "capital_after": round(running, 2),
            "day": i + 1,
            "timestamp": f"2026-{(i // 30) + 6:02d}-{(i % 30) + 1:02d}",
        })
    return trades, round(running, 2)


# ─── Main Assessment ───────────────────────────────────────────────────────

def run_assessment():
    print("=" * 60)
    print("PORTFOLIO ASSESSMENT ENGINE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    # 1. Load results
    print("\n[1/6] Loading strategy results...")
    loaded, missing = load_results()

    # Check if any loaded strategy has actual return series data
    has_return_data = any(
        len(v.get("daily_returns", [])) > 0 or len(v.get("equity_curve", [])) > 0
        for v in loaded.values()
    )
    use_demo = len(loaded) == 0 or not has_return_data
    if use_demo:
        if len(loaded) > 0:
            print("\n  ⚠ Loaded files lack daily return series. Supplementing with demo data.")
        else:
            print("\n  ⚠ No result files found.")
        print("  Generating synthetic return data for all 4 strategies...\n")
        for name in STRATEGY_SCHEMAS:
            if name in loaded and loaded[name].get("daily_returns"):
                continue  # already has real return data
            rets = simulate_strategy_returns(name, num_days=60, seed=hash(name) % 10000)
            existing = loaded.get(name, {})
            loaded[name] = {
                "name": name,
                "description": STRATEGY_SCHEMAS[name]["description"],
                "available": True,
                "simulated": True,
                "daily_returns": rets,
                "total_return": existing.get("total_return") or sum(rets),
                "win_rate": existing.get("win_rate") or STRATEGY_SCHEMAS[name]["typical_win_rate"],
                "profit_factor": existing.get("profit_factor") or STRATEGY_SCHEMAS[name]["typical_pf"],
                "sharpe": existing.get("sharpe") or STRATEGY_SCHEMAS[name]["typical_sharpe"],
                "max_drawdown": existing.get("max_drawdown") or 0.05,
                "num_trades": existing.get("num_trades") or len(rets),
                "symbols": existing.get("symbols"),
                "best_filters": existing.get("best_filters"),
                "sharpe_improvement": existing.get("sharpe_improvement"),
                "total_return_dollar": existing.get("total_return_dollar"),
            }

    strategy_names = list(loaded.keys())
    print(f"\n  Active strategies: {len(strategy_names)}")
    for nm in strategy_names:
        s = loaded[nm]
        sim = " [SIMULATED]" if s.get("simulated") else ""
        print(f"    • {nm}: {s.get('description', 'N/A')}{sim}")

    # 2. Build strategy return series
    print("\n[2/6] Building return series...")
    strategy_returns = {}
    strategy_meta = {}
    for nm in strategy_names:
        s = loaded[nm]
        rets = s.get("daily_returns", [])
        if not rets and s.get("equity_curve"):
            ec = s["equity_curve"]
            rets = [(ec[i] - ec[i-1]) / ec[i-1] if ec[i-1] != 0 else 0 for i in range(1, len(ec))]
        strategy_returns[nm] = rets
        strategy_meta[nm] = {
            "sharpe": s.get("sharpe", STRATEGY_SCHEMAS.get(nm, {}).get("typical_sharpe", 1.0)),
            "total_return": s.get("total_return", 0),
            "win_rate": s.get("win_rate", 0),
            "profit_factor": s.get("profit_factor", 0),
            "max_drawdown": s.get("max_drawdown", 0),
        }
        print(f"    {nm}: {len(rets)} data points")

    # 3. Correlation analysis
    print("\n[3/6] Correlation analysis...")
    corr_matrix, names = build_correlation_matrix(strategy_returns)
    print("    Correlation matrix:")
    header = "    " + " " * 20 + "  ".join(f"{n[:12]:>12}" for n in names)
    print(header)
    for a in names:
        row = f"    {a:<20}" + "  ".join(f"{corr_matrix.get((a,b), 0):>12.4f}" for b in names)
        print(row)

    high_corr_pairs = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i < j:
                c = corr_matrix.get((a, b), 0)
                if abs(c) > 0.7:
                    high_corr_pairs.append((a, b, c))

    if high_corr_pairs:
        print(f"\n    ⚠ High correlation pairs (>{0.7}):")
        for a, b, c in high_corr_pairs:
            print(f"      {a} ↔ {b}: {c:.4f}")
    else:
        print(f"\n    ✓ No pairs exceed 0.7 correlation threshold")

    # 4. Portfolio allocation
    print("\n[4/6] Computing portfolio allocation...")
    allocations = compute_allocations(strategy_meta, corr_matrix, names)
    print("    Weights:")
    for nm, w in sorted(allocations.items(), key=lambda x: -x[1]):
        print(f"      {nm}: {w:.2%}")

    # Diversification benefit
    div_benefit = diversification_benefit(strategy_returns, allocations)
    print(f"\n    Diversification benefit: {div_benefit:.2%}")

    # 5. Build portfolio & simulate
    print("\n[5/6] Simulating combined portfolio...")
    portfolio = Portfolio(initial_capital=100000)

    # Combine returns weighted by allocation
    min_len = min(len(v) for v in strategy_returns.values()) if strategy_returns else 0
    portfolio_daily = []
    for i in range(min_len):
        combined = sum(
            allocations.get(nm, 0) * strategy_returns[nm][i]
            for nm in strategy_names
        )
        portfolio_daily.append(combined)

    # Generate portfolio equity curve
    equity = [portfolio.initial_capital]
    for r in portfolio_daily:
        equity.append(equity[-1] * (1 + r))

    # Record as trades
    for i, r in enumerate(portfolio_daily):
        pnl = equity[i] * r
        portfolio.record_trade(pnl, "combined", "PORTFOLIO")
        portfolio.record_daily_pnl(pnl)

    final_capital = equity[-1]
    total_return = (final_capital - portfolio.initial_capital) / portfolio.initial_capital

    port_sharpe = calc_sharpe(portfolio_daily)
    port_sortino = calc_sortino(portfolio_daily)
    port_max_dd = calc_max_drawdown(equity)
    port_win_rate = calc_win_rate(portfolio.trade_log)
    port_pf = calc_profit_factor(portfolio.trade_log)

    print(f"    Final capital:    ${final_capital:,.2f}")
    print(f"    Total return:     {total_return:.2%}")
    print(f"    Sharpe ratio:     {port_sharpe:.4f}")
    print(f"    Sortino ratio:    {port_sortino:.4f}")
    print(f"    Max drawdown:     {port_max_dd:.2%}")
    print(f"    Win rate:         {port_win_rate:.2%}")
    print(f"    Profit factor:    {port_pf:.4f}")

    # 6. Prop firm compliance
    print("\n[6/6] Prop firm compliance check...")
    pf_checks = check_prop_firm(portfolio, portfolio.daily_pnl)
    for k, v in pf_checks.items():
        if isinstance(v, bool):
            status = "✓ PASS" if v else "✗ FAIL"
            print(f"    {k}: {status}")
        else:
            print(f"    {k}: {v}")

    # ─── Build output ──────────────────────────────────────────────────

    # Individual strategy summaries
    strategy_summaries = {}
    for nm in strategy_names:
        s = loaded[nm]
        strategy_summaries[nm] = {
            "description": s.get("description", ""),
            "simulated": s.get("simulated", False),
            "total_return_pct": round((s.get("total_return", 0) or 0) * 100, 2),
            "win_rate": round(s.get("win_rate", 0) or 0, 4),
            "profit_factor": round(s.get("profit_factor", 0) or 0, 4),
            "sharpe": round(s.get("sharpe", 0) or 0, 4),
            "max_drawdown_pct": round((s.get("max_drawdown", 0) or 0) * 100, 2),
            "num_trades": s.get("num_trades", 0),
            "data_points": len(strategy_returns.get(nm, [])),
        }

    assessment = {
        "timestamp": datetime.now().isoformat(),
        "mode": "demo" if use_demo else "live",
        "initial_capital": portfolio.initial_capital,
        "strategies": strategy_summaries,
        "correlation_matrix": {f"{a}|{b}": corr_matrix.get((a, b), 0) for a in names for b in names},
        "high_correlation_pairs": [{"a": a, "b": b, "corr": c} for a, b, c in high_corr_pairs],
        "allocations": allocations,
        "diversification_benefit_pct": round(div_benefit * 100, 2),
        "portfolio": {
            "final_capital": round(final_capital, 2),
            "total_return_pct": round(total_return * 100, 2),
            "sharpe_ratio": round(port_sharpe, 4),
            "sortino_ratio": round(port_sortino, 4),
            "max_drawdown_pct": round(port_max_dd * 100, 2),
            "win_rate": round(port_win_rate, 4),
            "profit_factor": round(port_pf, 4),
            "num_trades": len(portfolio.trade_log),
            "daily_pnl_count": len(portfolio.daily_pnl),
        },
        "prop_firm_compliance": pf_checks,
        "risk_params": {
            "max_positions": portfolio.max_positions,
            "risk_per_trade": portfolio.risk_per_trade,
            "daily_dd_limit": portfolio.daily_dd_limit,
            "total_dd_limit": portfolio.total_dd_limit,
        },
    }

    # Save JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(assessment, f, indent=2, default=str)
    print(f"\n✓ Saved {OUTPUT_JSON}")

    # Generate markdown report
    generate_report(assessment, loaded, missing, corr_matrix, names)
    print(f"✓ Saved {OUTPUT_MD}")

    print("\n" + "=" * 60)
    print("ASSESSMENT COMPLETE")
    print("=" * 60)
    return assessment


def generate_report(assessment, loaded, missing, corr_matrix, names):
    """Generate PORTFOLIO_REPORT.md."""
    p = assessment["portfolio"]
    pf = assessment["prop_firm_compliance"]
    mode = assessment["mode"]

    lines = [
        "# Portfolio Assessment Report",
        "",
        f"**Generated:** {assessment['timestamp']}",
        f"**Mode:** {'Demo (synthetic data)' if mode == 'demo' else 'Live results'}",
        f"**Initial Capital:** ${assessment['initial_capital']:,.0f}",
        "",
        "---",
        "",
        "## Strategy Summary",
        "",
        "| Strategy | Return | Win Rate | PF | Sharpe | Max DD | Trades |",
        "|----------|--------|----------|-----|--------|--------|--------|",
    ]

    for nm, s in assessment["strategies"].items():
        sim = " *" if s.get("simulated") else ""
        lines.append(
            f"| {nm}{sim} | {s['total_return_pct']:.2f}% | {s['win_rate']:.2%} | "
            f"{s['profit_factor']:.2f} | {s['sharpe']:.2f} | {s['max_drawdown_pct']:.2f}% | {s['num_trades']} |"
        )

    if mode == "demo":
        lines.append("")
        lines.append("*Simulated data — awaiting real backtest results*")

    lines += [
        "",
        "## Correlation Matrix",
        "",
    ]

    header = "| | " + " | ".join(n[:12] for n in names) + " |"
    sep = "|---|" + "|".join("-------" for _ in names) + "|"
    lines += [header, sep]
    for a in names:
        row = f"| **{a[:12]}** | " + " | ".join(
            f"{corr_matrix.get((a, b), 0):.3f}" for b in names
        ) + " |"
        lines.append(row)

    if assessment["high_correlation_pairs"]:
        lines += [
            "",
            "### ⚠ High Correlation Pairs",
            "",
        ]
        for pair in assessment["high_correlation_pairs"]:
            lines.append(f"- {pair['a']} ↔ {pair['b']}: **{pair['corr']:.4f}**")

    lines += [
        "",
        "## Portfolio Allocation",
        "",
    ]
    for nm, w in sorted(assessment["allocations"].items(), key=lambda x: -x[1]):
        bar = "█" * int(w * 20)
        lines.append(f"- **{nm}**: {w:.2%} {bar}")

    lines += [
        "",
        f"**Diversification Benefit:** {assessment['diversification_benefit_pct']:.2f}%",
        "",
        "---",
        "",
        "## Portfolio Performance",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Final Capital | ${p['final_capital']:,.2f} |",
        f"| Total Return | {p['total_return_pct']:.2f}% |",
        f"| Sharpe Ratio | {p['sharpe_ratio']:.4f} |",
        f"| Sortino Ratio | {p['sortino_ratio']:.4f} |",
        f"| Max Drawdown | {p['max_drawdown_pct']:.2f}% |",
        f"| Win Rate | {p['win_rate']:.2%} |",
        f"| Profit Factor | {p['profit_factor']:.4f} |",
        f"| Total Trades | {p['num_trades']} |",
        "",
        "---",
        "",
        "## Prop Firm Compliance",
        "",
        "| Rule | Status | Value |",
        "|------|--------|-------|",
    ]

    rule_map = {
        "daily_dd_ok": ("Daily DD < 3%", pf.get("daily_dd_ok")),
        "total_dd_ok": ("Total DD < 10%", pf.get("total_dd_ok")),
        "profit_target_met": ("Profit Target > 10%", pf.get("profit_target_met")),
        "min_trading_days_ok": ("Min 10 Trading Days", pf.get("min_trading_days_ok")),
        "win_rate_ok": ("Win Rate > 45%", pf.get("win_rate_ok")),
        "profit_factor_ok": ("PF > 1.2", pf.get("profit_factor_ok")),
    }

    for key, (label, status) in rule_map.items():
        icon = "✅" if status else "❌"
        val = pf.get(key.replace("_ok", "").replace("met", ""), "")
        lines.append(f"| {label} | {icon} | {val} |")

    all_pass = pf.get("all_pass", False)
    lines += [
        "",
        f"### Overall: {'✅ ALL CHECKS PASSED' if all_pass else '❌ SOME CHECKS FAILED'}",
        "",
        "---",
        "",
        "## Risk Parameters",
        "",
        f"- Max Positions: {assessment['risk_params']['max_positions']}",
        f"- Risk Per Trade: {assessment['risk_params']['risk_per_trade']:.1%}",
        f"- Daily Drawdown Limit: {assessment['risk_params']['daily_dd_limit']:.1%}",
        f"- Total Drawdown Limit: {assessment['risk_params']['total_dd_limit']:.1%}",
        "",
    ]

    if missing:
        lines += [
            "## Missing Strategy Files",
            "",
        ]
        for m in missing:
            lines.append(f"- `{RESULT_FILES[m].name}` — not found")
        lines += [
            "",
            "Run the individual strategy scripts to generate these files.",
            "This report will auto-update on next run.",
            "",
        ]

    lines += [
        "---",
        "",
        f"*Report generated by portfolio_assessment.py*",
    ]

    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run_assessment()
