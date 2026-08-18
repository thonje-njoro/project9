#!/usr/bin/env python3
"""
Full Validation Pipeline for Top Strategies
- Monte Carlo simulation (2000 runs)
- Out-of-sample test (2024-2026)
- Portfolio construction with correlation analysis
- Final prop-firm readiness check
"""

import os
import json
import sys
import numpy as np
import requests
from datetime import datetime, timedelta
from collections import defaultdict

# =============================================================================
# LSE API CONFIG
# =============================================================================
API_KEY = os.environ.get("LSE_API_KEY", "")
if not API_KEY:
    env_path = os.path.join(os.path.dirname(__file__), "backtest", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("LSE_API_KEY="):
                    API_KEY = line.split("=", 1)[1].strip()
                    break

BASE_URL = "https://api.londonstrategicedge.com/vault/candles"
HEADERS = {"x-api-key": API_KEY}
COST = 0.001  # 0.1% per trade

# =============================================================================
# TOP STRATEGIES TO VALIDATE (from multi-instrument test)
# =============================================================================
STRATEGIES = [
    # 3/3 walk-forward winners
    {"name": "NVDA/EMA(10/30)", "symbol": "NVDA", "type": "ema", "params": {"fast": 10, "slow": 30}},
    {"name": "AMD/Momentum(50)", "symbol": "AMD", "type": "momentum", "params": {"lookback": 50}},
    {"name": "SPY/EMA(10/30)", "symbol": "SPY", "type": "ema", "params": {"fast": 10, "slow": 30}},
    {"name": "NVDA/Momentum(20)", "symbol": "NVDA", "type": "momentum", "params": {"lookback": 20}},
    {"name": "AMD/Momentum(20)", "symbol": "AMD", "type": "momentum", "params": {"lookback": 20}},
    {"name": "NVDA/Momentum(50)", "symbol": "NVDA", "type": "momentum", "params": {"lookback": 50}},
    # 2/3 high quality
    {"name": "AAPL/Donchian(50,rr=3)", "symbol": "AAPL", "type": "donchian", "params": {"lookback": 50, "rr": 3.0}},
    {"name": "NZDUSD/Bollinger(20,2.0)", "symbol": "NZD/USD", "type": "bollinger_mr", "params": {"lookback": 20, "width": 2.0}},
    {"name": "AUDUSD/RSI(14,30/70)", "symbol": "AUD/USD", "type": "rsi_mr", "params": {"lookback": 14, "oversold": 30, "overbought": 70}},
    {"name": "MSFT/RSI(14,30/70)", "symbol": "MSFT", "type": "rsi_mr", "params": {"lookback": 14, "oversold": 30, "overbought": 70}},
]


def fetch_lse(symbol, start_date, end_date):
    """Fetch daily candles from LSE API with pagination."""
    lse_symbol = symbol.replace("/", "")
    lse_symbol = lse_symbol.replace("USD", "/USD") if "USD" in symbol and "/" not in symbol else symbol
    lse_symbol = symbol  # pass as-is since they're already in correct format

    all_bars = []
    current_start = start_date
    max_pages = 20

    for _ in range(max_pages):
        params = {"symbol": lse_symbol, "start": current_start, "limit": 5000}
        try:
            r = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  API error for {lse_symbol}: {e}")
            break

        if not data or not isinstance(data, list):
            break

        new_bars = 0
        for row in data:
            ts = row.get("ts", "")
            if not ts:
                continue
            ts_date = ts[:10]
            if ts_date > end_date:
                continue
            if ts_date >= start_date:
                all_bars.append({
                    "date": ts_date,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0)),
                })
                new_bars += 1

        if new_bars == 0 or len(data) < 100:
            break

        last_ts = data[-1]["ts"][:10]
        next_date = (datetime.strptime(last_ts, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        if next_date <= current_start:
            break
        current_start = next_date

    # Deduplicate
    seen = set()
    unique = []
    for bar in all_bars:
        if bar["date"] not in seen:
            seen.add(bar["date"])
            unique.append(bar)
    unique.sort(key=lambda x: x["date"])
    return unique


def compute_ema(closes, period):
    """Compute EMA."""
    ema = [closes[0]]
    mult = 2.0 / (period + 1)
    for i in range(1, len(closes)):
        ema.append(closes[i] * mult + ema[-1] * (1 - mult))
    return np.array(ema)


def compute_rsi(closes, period=14):
    """Compute RSI."""
    rsi = np.full(len(closes), 50.0)
    if len(closes) < period + 1:
        return rsi
    deltas = np.diff(closes)
    for i in range(period, len(closes)):
        gains = deltas[max(0, i - period):i]
        up = np.mean(gains[gains > 0]) if np.any(gains > 0) else 0.0
        down = -np.mean(gains[gains < 0]) if np.any(gains < 0) else 1e-10
        rs = up / down if down > 0 else 100.0
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def compute_atr(highs, lows, closes, period=14):
    """Compute ATR."""
    atr = np.zeros(len(closes))
    tr = np.zeros(len(closes))
    for i in range(1, len(closes)):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    for i in range(period, len(closes)):
        atr[i] = np.mean(tr[max(0, i-period+1):i+1])
    return atr


def run_strategy(bars, strat):
    """Run a strategy and return trades."""
    if len(bars) < 60:
        return []

    closes = np.array([b["close"] for b in bars])
    highs = np.array([b["high"] for b in bars])
    lows = np.array([b["low"] for b in bars])
    dates = [b["date"] for b in bars]
    stype = strat["type"]
    params = strat["params"]

    trades = []
    in_trade = False
    entry_price = 0
    entry_date = ""
    stop_price = 0
    direction = 1

    for i in range(60, len(closes)):
        price = closes[i]

        if in_trade:
            # Check stop
            if direction == 1 and lows[i] <= stop_price:
                exit_price = stop_price
                ret = (exit_price / entry_price - 1) * direction
                trades.append({"entry": entry_date, "exit": dates[i], "return": ret - COST})
                in_trade = False
                continue
            elif direction == -1 and highs[i] >= stop_price:
                exit_price = stop_price
                ret = (exit_price / entry_price - 1) * direction
                trades.append({"entry": entry_date, "exit": dates[i], "return": ret - COST})
                in_trade = False
                continue

            # Trailing stop update
            atr = compute_atr(highs[:i+1], lows[:i+1], closes[:i+1], 14)
            if direction == 1:
                new_stop = price - 2.0 * atr[i]
                stop_price = max(stop_price, new_stop)
            else:
                new_stop = price + 2.0 * atr[i]
                stop_price = min(stop_price, new_stop)

        if not in_trade:
            if stype == "ema":
                ema_fast = compute_ema(closes[:i+1], params["fast"])
                ema_slow = compute_ema(closes[:i+1], params["slow"])
                if ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]:
                    direction = 1
                    entry_price = price
                    entry_date = dates[i]
                    atr = compute_atr(highs[:i+1], lows[:i+1], closes[:i+1], 14)
                    stop_price = price - 2.0 * atr[i]
                    in_trade = True
                elif ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]:
                    direction = -1
                    entry_price = price
                    entry_date = dates[i]
                    atr = compute_atr(highs[:i+1], lows[:i+1], closes[:i+1], 14)
                    stop_price = price + 2.0 * atr[i]
                    in_trade = True

            elif stype == "momentum":
                lb = params["lookback"]
                if i >= lb:
                    ret = price / closes[i - lb] - 1
                    atr = compute_atr(highs[:i+1], lows[:i+1], closes[:i+1], 14)
                    if ret > 0.02:  # 2% threshold
                        direction = 1
                        entry_price = price
                        entry_date = dates[i]
                        stop_price = price - 2.0 * atr[i]
                        in_trade = True
                    elif ret < -0.02:
                        direction = -1
                        entry_price = price
                        entry_date = dates[i]
                        stop_price = price + 2.0 * atr[i]
                        in_trade = True

            elif stype == "donchian":
                lb = params["lookback"]
                rr = params.get("rr", 3.0)
                if i >= lb:
                    upper = np.max(highs[i-lb:i])
                    lower = np.min(lows[i-lb:i])
                    atr = compute_atr(highs[:i+1], lows[:i+1], closes[:i+1], 14)
                    if price > upper:
                        direction = 1
                        entry_price = price
                        entry_date = dates[i]
                        stop_price = price - 2.0 * atr[i]
                        in_trade = True
                    elif price < lower:
                        direction = -1
                        entry_price = price
                        entry_date = dates[i]
                        stop_price = price + 2.0 * atr[i]
                        in_trade = True

            elif stype == "bollinger_mr":
                lb = params["lookback"]
                width = params.get("width", 2.0)
                if i >= lb:
                    sma = np.mean(closes[i-lb:i])
                    std = np.std(closes[i-lb:i])
                    lower_band = sma - width * std
                    upper_band = sma + width * std
                    atr = compute_atr(highs[:i+1], lows[:i+1], closes[:i+1], 14)
                    if price < lower_band:
                        direction = 1
                        entry_price = price
                        entry_date = dates[i]
                        stop_price = price - 2.0 * atr[i]
                        in_trade = True
                    elif price > upper_band:
                        direction = -1
                        entry_price = price
                        entry_date = dates[i]
                        stop_price = price + 2.0 * atr[i]
                        in_trade = True

            elif stype == "rsi_mr":
                lb = params.get("lookback", 14)
                oversold = params.get("oversold", 30)
                overbought = params.get("overbought", 70)
                rsi = compute_rsi(closes[:i+1], lb)
                atr = compute_atr(highs[:i+1], lows[:i+1], closes[:i+1], 14)
                if rsi[i] < oversold and i > 0 and rsi[i-1] >= oversold:
                    direction = 1
                    entry_price = price
                    entry_date = dates[i]
                    stop_price = price - 2.0 * atr[i]
                    in_trade = True
                elif rsi[i] > overbought and i > 0 and rsi[i-1] <= overbought:
                    direction = -1
                    entry_price = price
                    entry_date = dates[i]
                    stop_price = price + 2.0 * atr[i]
                    in_trade = True

    # Close any open trade at end
    if in_trade:
        ret = (closes[-1] / entry_price - 1) * direction
        trades.append({"entry": entry_date, "exit": dates[-1], "return": ret - COST})

    return trades


def compute_metrics(trades):
    """Compute strategy metrics from trade list."""
    if len(trades) < 3:
        return {"trades": len(trades), "wr": 0, "pf": 0, "sharpe": 0, "max_dd": 100, "ret": 0}

    returns = [t["return"] for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    win_rate = len(wins) / len(returns) * 100 if returns else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-10
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0

    mean_ret = np.mean(returns)
    std_ret = np.std(returns) if len(returns) > 1 else 1e-10
    sharpe = mean_ret / std_ret * np.sqrt(252 / max(1, len(returns) / 5)) if std_ret > 0 else 0

    cumulative = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_dd = np.max(drawdown) * 100 if len(drawdown) > 0 else 0

    total_return = sum(returns) * 100

    return {
        "trades": len(trades),
        "wr": win_rate,
        "pf": profit_factor,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "ret": total_return,
    }


def monte_carlo(trades, n_sims=2000):
    """Run Monte Carlo simulation by shuffling trade order."""
    returns = [t["return"] for t in trades]
    if len(returns) < 5:
        return {"sharpe_med": 0, "sharpe_lo": 0, "sharpe_hi": 0,
                "dd_med": 100, "dd_99": 100, "survival": 0, "p_profit": 0}

    sharpes = []
    max_dds = []
    finals = []

    for _ in range(n_sims):
        shuffled = np.random.permutation(returns)
        cum = np.cumsum(shuffled)

        mean_r = np.mean(shuffled)
        std_r = np.std(shuffled)
        s = mean_r / std_r * np.sqrt(252) if std_r > 0 else 0
        sharpes.append(s)

        peak = np.maximum.accumulate(cum)
        dd = np.max(peak - cum) * 100
        max_dds.append(dd)
        finals.append(cum[-1])

    survival = sum(1 for dd in max_dds if dd < 20) / n_sims * 100
    p_profit = sum(1 for f in finals if f > 0) / n_sims * 100

    return {
        "sharpe_med": np.median(sharpes),
        "sharpe_lo": np.percentile(sharpes, 5),
        "sharpe_hi": np.percentile(sharpes, 95),
        "dd_med": np.median(max_dds),
        "dd_99": np.percentile(max_dds, 99),
        "survival": survival,
        "p_profit": p_profit,
    }


def split_windows(bars, n_windows=3):
    """Split bars into N equal windows."""
    chunk = len(bars) // n_windows
    windows = []
    for i in range(n_windows):
        start = i * chunk
        end = (i + 1) * chunk if i < n_windows - 1 else len(bars)
        windows.append(bars[start:end])
    return windows


def main():
    print("=" * 80)
    print("  FULL VALIDATION PIPELINE — Top Strategies on Real LSE Data")
    print("=" * 80)
    print()

    if not API_KEY:
        print("ERROR: No LSE_API_KEY found. Set in backtest/.env or environment.")
        sys.exit(1)

    # =========================================================================
    # PHASE 1: Fetch data
    # =========================================================================
    print("PHASE 1: Fetching daily data from LSE API...")
    print("-" * 60)

    cache = {}
    symbols = list(set(s["symbol"] for s in STRATEGIES))

    for sym in symbols:
        print(f"  Fetching {sym} (2015-01-01 to 2026-08-19)...", end=" ")
        bars = fetch_lse(sym, "2015-01-01", "2026-08-19")
        cache[sym] = bars
        print(f"{len(bars)} bars ({bars[0]['date'] if bars else 'N/A'} to {bars[-1]['date'] if bars else 'N/A'})")

    # =========================================================================
    # PHASE 2: In-sample validation (2015-2024)
    # =========================================================================
    print()
    print("PHASE 2: In-sample backtest (2015-01-01 to 2023-12-31)")
    print("-" * 60)

    results = {}
    for strat in STRATEGIES:
        sym = strat["symbol"]
        bars = [b for b in cache[sym] if "2015-01-01" <= b["date"] <= "2023-12-31"]
        trades = run_strategy(bars, strat)
        metrics = compute_metrics(trades)
        results[strat["name"]] = {"metrics": metrics, "trades": trades}

        print(f"  {strat['name']:30s}  T={metrics['trades']:3d}  WR={metrics['wr']:5.1f}%  "
              f"PF={metrics['pf']:5.2f}  Sh={metrics['sharpe']:7.3f}  "
              f"DD={metrics['max_dd']:5.1f}%  Ret={metrics['ret']:7.1f}%")

    # =========================================================================
    # PHASE 3: Walk-forward validation
    # =========================================================================
    print()
    print("PHASE 3: Walk-forward validation (3 windows)")
    print("-" * 60)

    wf_results = {}
    for strat in STRATEGIES:
        sym = strat["symbol"]
        bars_is = [b for b in cache[sym] if "2015-01-01" <= b["date"] <= "2023-12-31"]
        windows = split_windows(bars_is, 3)

        wf_pass = 0
        wf_details = []
        for wi, wbars in enumerate(windows):
            w_trades = run_strategy(wbars, strat)
            w_metrics = compute_metrics(w_trades)
            passed = w_metrics["sharpe"] > 0 and w_metrics["pf"] > 1.0
            if passed:
                wf_pass += 1
            wf_details.append(w_metrics)

        wf_results[strat["name"]] = {"pass": wf_pass, "details": wf_details}
        pass_str = f"{wf_pass}/3"
        star = "⭐" if wf_pass >= 2 else "  "

        print(f"  {star} {strat['name']:30s}  WF={pass_str}  "
              f"W1: Sh={wf_details[0]['sharpe']:6.2f}  "
              f"W2: Sh={wf_details[1]['sharpe']:6.2f}  "
              f"W3: Sh={wf_details[2]['sharpe']:6.2f}")

    # =========================================================================
    # PHASE 4: Out-of-sample test (2024-2026)
    # =========================================================================
    print()
    print("PHASE 4: Out-of-sample test (2024-01-01 to 2026-08-19)")
    print("-" * 60)

    oos_results = {}
    for strat in STRATEGIES:
        sym = strat["symbol"]
        bars_oos = [b for b in cache[sym] if b["date"] >= "2024-01-01"]
        trades = run_strategy(bars_oos, strat)
        metrics = compute_metrics(trades)
        oos_results[strat["name"]] = metrics

        verdict = "✅" if metrics["sharpe"] > 0 and metrics["pf"] > 1.0 else "❌"
        print(f"  {verdict} {strat['name']:30s}  T={metrics['trades']:3d}  WR={metrics['wr']:5.1f}%  "
              f"PF={metrics['pf']:5.2f}  Sh={metrics['sharpe']:7.3f}  Ret={metrics['ret']:7.1f}%")

    # =========================================================================
    # PHASE 5: Monte Carlo simulation
    # =========================================================================
    print()
    print("PHASE 5: Monte Carlo simulation (2000 runs)")
    print("-" * 60)

    mc_results = {}
    for strat in STRATEGIES:
        name = strat["name"]
        trades = results[name]["trades"]
        if len(trades) < 10:
            print(f"  SKIP {name} (only {len(trades)} trades)")
            continue

        mc = monte_carlo(trades, 2000)
        mc_results[name] = mc

        print(f"  {name:30s}  Sh_med={mc['sharpe_med']:6.2f}  "
              f"DD_med={mc['dd_med']:5.1f}%  DD_99={mc['dd_99']:5.1f}%  "
              f"P(profit)={mc['p_profit']:5.1f}%  Survival={mc['survival']:5.1f}%")

    # =========================================================================
    # PHASE 6: Portfolio construction
    # =========================================================================
    print()
    print("PHASE 6: Portfolio — correlation matrix & combined metrics")
    print("-" * 60)

    # Get monthly returns for each strategy (in-sample)
    portfolio_strats = []
    for strat in STRATEGIES:
        name = strat["name"]
        m = results[name]["metrics"]
        wf = wf_results[name]
        oos = oos_results[name]

        # Filter: WF >= 2/3, OOS Sharpe > 0, IS PF > 1.2
        if wf["pass"] >= 2 and oos["sharpe"] > 0 and m["pf"] > 1.2:
            portfolio_strats.append(strat)

    if not portfolio_strats:
        print("  No strategies pass all filters for portfolio construction.")
        print("  Relaxing filters to WF >= 2/3 only...")
        for strat in STRATEGIES:
            name = strat["name"]
            wf = wf_results[name]
            if wf["pass"] >= 2:
                portfolio_strats.append(strat)

    if portfolio_strats:
        print(f"\n  Portfolio members ({len(portfolio_strats)} strategies):")
        for s in portfolio_strats:
            m = results[s["name"]]["metrics"]
            wf = wf_results[s["name"]]
            oos = oos_results[s["name"]]
            print(f"    {s['name']:30s}  IS_PF={m['pf']:.2f}  WF={wf['pass']}/3  OOS_Sh={oos['sharpe']:.2f}")

        # Compute monthly returns for correlation
        monthly_rets = {}
        for strat in portfolio_strats:
            sym = strat["symbol"]
            bars = cache[sym]
            # Group trades by month
            month_rets = defaultdict(float)
            for t in results[strat["name"]]["trades"]:
                month = t["entry"][:7]  # YYYY-MM
                month_rets[month] += t["return"]
            monthly_rets[strat["name"]] = month_rets

        # Build correlation matrix
        if len(portfolio_strats) > 1:
            print(f"\n  Correlation matrix:")
            names = [s["name"] for s in portfolio_strats]
            # Get common months
            all_months = set()
            for name in names:
                all_months.update(monthly_rets[name].keys())
            common_months = sorted(all_months)

            # Build return arrays
            ret_arrays = {}
            for name in names:
                ret_arrays[name] = [monthly_rets[name].get(m, 0.0) for m in common_months]

            # Compute correlations
            n = len(names)
            print(f"  {'':30s}", end="")
            for name in names:
                print(f"  {name[:10]:>10s}", end="")
            print()

            corrs = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i == j:
                        corrs[i][j] = 1.0
                    else:
                        a = np.array(ret_arrays[names[i]])
                        b = np.array(ret_arrays[names[j]])
                        if np.std(a) > 0 and np.std(b) > 0:
                            corrs[i][j] = np.corrcoef(a, b)[0][1]
                        else:
                            corrs[i][j] = 0.0

            for i in range(n):
                print(f"  {names[i][:28]:30s}", end="")
                for j in range(n):
                    print(f"  {corrs[i][j]:10.2f}", end="")
                print()

            # Combined portfolio metrics
            # Equal-weight portfolio monthly returns
            portfolio_monthly = []
            for mi, m in enumerate(common_months):
                avg_ret = np.mean([ret_arrays[name][mi] for name in names])
                portfolio_monthly.append(avg_ret)

            p_mean = np.mean(portfolio_monthly)
            p_std = np.std(portfolio_monthly) if len(portfolio_monthly) > 1 else 1e-10
            p_sharpe = p_mean / p_std * np.sqrt(12) if p_std > 0 else 0
            p_cum = np.cumsum(portfolio_monthly)
            p_peak = np.maximum.accumulate(p_cum)
            p_dd = np.max(p_peak - p_cum) * 100 if len(p_cum) > 0 else 0

            print(f"\n  Combined portfolio (equal weight):")
            print(f"    Monthly Sharpe: {p_sharpe:.2f}")
            print(f"    Total return: {p_cum[-1]*100:.1f}%")
            print(f"    Max drawdown: {p_dd:.1f}%")
            print(f"    Avg correlation: {np.mean(corrs[np.triu_indices(n, k=1)]):.2f}")

    # =========================================================================
    # FINAL VERDICT
    # =========================================================================
    print()
    print("=" * 80)
    print("  FINAL VERDICT")
    print("=" * 80)

    approved = []
    for strat in STRATEGIES:
        name = strat["name"]
        m = results[name]["metrics"]
        wf = wf_results[name]
        oos = oos_results[name]
        mc = mc_results.get(name, {})

        # Criteria: WF >= 2/3, OOS Sharpe > 0, OOS PF > 1.0, MC survival > 60%
        if (wf["pass"] >= 2 and
            oos["sharpe"] > 0 and
            oos["pf"] > 1.0 and
            mc.get("survival", 0) > 60):
            approved.append(strat)

    if approved:
        print(f"\n  ✅ APPROVED FOR PAPER TRADING ({len(approved)} strategies):")
        for s in approved:
            name = s["name"]
            m = results[name]["metrics"]
            wf = wf_results[name]
            oos = oos_results[name]
            mc = mc_results.get(name, {})
            print(f"    {name:30s}  IS_PF={m['pf']:.2f}  WF={wf['pass']}/3  "
                  f"OOS_Sh={oos['sharpe']:.2f}  OOS_PF={oos['pf']:.2f}  "
                  f"MC_surv={mc.get('survival', 0):.0f}%")
    else:
        print("\n  ❌ NO STRATEGIES APPROVED")
        print("  All candidates failed at least one validation criterion.")

        # Show closest candidates
        print("\n  Closest candidates:")
        for strat in STRATEGIES:
            name = strat["name"]
            m = results[name]["metrics"]
            wf = wf_results[name]
            oos = oos_results[name]
            mc = mc_results.get(name, {})
            issues = []
            if wf["pass"] < 2:
                issues.append(f"WF={wf['pass']}/3")
            if oos["sharpe"] <= 0:
                issues.append(f"OOS_Sh={oos['sharpe']:.2f}")
            if oos["pf"] <= 1.0:
                issues.append(f"OOS_PF={oos['pf']:.2f}")
            if mc.get("survival", 0) <= 60:
                issues.append(f"MC_surv={mc.get('survival', 0):.0f}%")
            print(f"    {name:30s}  Issues: {', '.join(issues)}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
