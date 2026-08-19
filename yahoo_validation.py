#!/usr/bin/env python3
"""
Multi-Instrument Strategy Test — Yahoo Finance + Alpha Vantage Data
Uses Yahoo Finance (free, 20+ years) as primary, Alpha Vantage as fallback.
Tests equities and forex with full walk-forward validation.
"""

import os
import json
import sys
import numpy as np
import requests
from datetime import datetime, timedelta
from collections import defaultdict

# =============================================================================
# CONFIG
# =============================================================================
COST = 0.001  # 0.1% per trade

# Alpha Vantage API key (free tier: 25 requests/day)
ALPHA_VANTAGE_KEY = "demo"  # Replace with real key if available

# Symbols to test
SYMBOLS = {
    # Equities (Yahoo Finance tickers)
    "SPY": {"yahoo": "SPY", "alpha": "SPY", "type": "equity"},
    "QQQ": {"yahoo": "QQQ", "alpha": "QQQ", "type": "equity"},
    "NVDA": {"yahoo": "NVDA", "alpha": "NVDA", "type": "equity"},
    "AMD": {"yahoo": "AMD", "alpha": "AMD", "type": "equity"},
    "AAPL": {"yahoo": "AAPL", "alpha": "AAPL", "type": "equity"},
    "MSFT": {"yahoo": "MSFT", "alpha": "MSFT", "type": "equity"},
    # Forex (Yahoo Finance format: EURUSD=X)
    "EUR/USD": {"yahoo": "EURUSD=X", "alpha": "EURUSD", "type": "forex"},
    "GBP/USD": {"yahoo": "GBPUSD=X", "alpha": "GBPUSD", "type": "forex"},
    "USD/JPY": {"yahoo": "USDJPY=X", "alpha": "USDJPY", "type": "forex"},
    "AUD/USD": {"yahoo": "AUDUSD=X", "alpha": "AUDUSD", "type": "forex"},
    "NZD/USD": {"yahoo": "NZDUSD=X", "alpha": "NZDUSD", "type": "forex"},
    # Commodities
    "XAU/USD": {"yahoo": "GC=F", "alpha": "XAUUSD", "type": "commodity"},
    "XAG/USD": {"yahoo": "SI=F", "alpha": "XAGUSD", "type": "commodity"},
    "USOIL": {"yahoo": "CL=F", "alpha": "WTI", "type": "commodity"},
}

# Strategies to test
STRATEGIES = [
    # Trend following
    {"name": "EMA(10/30)", "type": "ema", "params": {"fast": 10, "slow": 30}},
    {"name": "EMA(20/50)", "type": "ema", "params": {"fast": 20, "slow": 50}},
    {"name": "EMA(50/200)", "type": "ema", "params": {"fast": 50, "slow": 200}},
    # Momentum
    {"name": "Momentum(20)", "type": "momentum", "params": {"lookback": 20}},
    {"name": "Momentum(50)", "type": "momentum", "params": {"lookback": 50}},
    # Donchian breakout
    {"name": "Donchian(10,rr=2)", "type": "donchian", "params": {"lookback": 10, "rr": 2.0}},
    {"name": "Donchian(20,rr=2)", "type": "donchian", "params": {"lookback": 20, "rr": 2.0}},
    {"name": "Donchian(20,rr=3)", "type": "donchian", "params": {"lookback": 20, "rr": 3.0}},
    {"name": "Donchian(50,rr=3)", "type": "donchian", "params": {"lookback": 50, "rr": 3.0}},
    # Mean reversion
    {"name": "RSI(14,30/70)", "type": "rsi_mr", "params": {"lookback": 14, "oversold": 30, "overbought": 70}},
    {"name": "Bollinger(20,2.0)", "type": "bollinger_mr", "params": {"lookback": 20, "width": 2.0}},
]


# =============================================================================
# DATA FETCHING — YAHOO FINANCE (no API key needed)
# =============================================================================
def fetch_yahoo(ticker, start_date, end_date):
    """Fetch daily data from Yahoo Finance via yfinance or direct API."""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df is None or df.empty:
            return []

        bars = []
        for idx, row in df.iterrows():
            date_str = str(idx.date()) if hasattr(idx, 'date') else str(idx)[:10]
            bars.append({
                "date": date_str,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row.get("Volume", 0)),
            })
        return bars
    except ImportError:
        pass

    # Fallback: direct Yahoo Finance API (no library needed)
    try:
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {
            "period1": start_ts,
            "period2": end_ts,
            "interval": "1d",
            "events": "history",
        }
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()

        result = data.get("chart", {}).get("result", [])
        if not result:
            return []

        timestamps = result[0].get("timestamp", [])
        quote = result[0].get("indicators", {}).get("quote", [{}])[0]

        bars = []
        for i, ts in enumerate(timestamps):
            dt = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            o = quote.get("open", [None])[i]
            h = quote.get("high", [None])[i]
            l = quote.get("low", [None])[i]
            c = quote.get("close", [None])[i]
            v = quote.get("volume", [0])[i]
            if all(x is not None for x in [o, h, l, c]):
                bars.append({
                    "date": dt,
                    "open": float(o), "high": float(h),
                    "low": float(l), "close": float(c),
                    "volume": float(v or 0),
                })
        return bars
    except Exception as e:
        print(f"    Yahoo direct API failed: {e}")
        return []


# =============================================================================
# DATA FETCHING — ALPHA VANTAGE (free tier: 25 requests/day)
# =============================================================================
def fetch_alpha_vantage(symbol, is_forex=False):
    """Fetch daily data from Alpha Vantage."""
    try:
        if is_forex:
            url = "https://www.alphavantage.co/query"
            from_curr = symbol[:3]
            to_curr = symbol[3:]
            params = {
                "function": "FX_DAILY",
                "from_symbol": from_curr,
                "to_symbol": to_curr,
                "outputsize": "full",
                "apikey": ALPHA_VANTAGE_KEY,
            }
        else:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": "full",
                "apikey": ALPHA_VANTAGE_KEY,
            }

        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        # Find the time series key
        ts_key = None
        for key in data.keys():
            if "Time Series" in key or "time series" in key:
                ts_key = key
                break

        if not ts_key:
            return []

        bars = []
        for date_str, values in data[ts_key].items():
            bars.append({
                "date": date_str,
                "open": float(values.get("1. open", 0)),
                "high": float(values.get("2. high", 0)),
                "low": float(values.get("3. low", 0)),
                "close": float(values.get("4. close", 0)),
                "volume": float(values.get("5. volume", 0)),
            })

        bars.sort(key=lambda x: x["date"])
        return bars
    except Exception as e:
        print(f"    Alpha Vantage failed: {e}")
        return []


# =============================================================================
# STRATEGY IMPLEMENTATIONS
# =============================================================================
def compute_ema(closes, period):
    ema = [closes[0]]
    mult = 2.0 / (period + 1)
    for i in range(1, len(closes)):
        ema.append(closes[i] * mult + ema[-1] * (1 - mult))
    return np.array(ema)


def compute_rsi(closes, period=14):
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
    atr = np.zeros(len(closes))
    tr = np.zeros(len(closes))
    for i in range(1, len(closes)):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    for i in range(period, len(closes)):
        atr[i] = np.mean(tr[max(0, i-period+1):i+1])
    return atr


def run_strategy(bars, strat):
    """Run a strategy and return trades."""
    if len(bars) < 250:  # Need at least 1 year of daily data
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
    bars_in_trade = 0

    for i in range(200, len(closes)):
        price = closes[i]

        if in_trade:
            bars_in_trade += 1

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

            # Max hold period: 60 days
            if bars_in_trade >= 60:
                ret = (price / entry_price - 1) * direction
                trades.append({"entry": entry_date, "exit": dates[i], "return": ret - COST})
                in_trade = False
                continue

            # Trailing stop
            atr = compute_atr(highs[:i+1], lows[:i+1], closes[:i+1], 14)
            if direction == 1:
                new_stop = price - 2.0 * atr[i]
                stop_price = max(stop_price, new_stop)
            else:
                new_stop = price + 2.0 * atr[i]
                stop_price = min(stop_price, new_stop)

        if not in_trade:
            atr = compute_atr(highs[:i+1], lows[:i+1], closes[:i+1], 14)

            if stype == "ema":
                ema_fast = compute_ema(closes[:i+1], params["fast"])
                ema_slow = compute_ema(closes[:i+1], params["slow"])
                if ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]:
                    direction = 1
                    entry_price = price
                    entry_date = dates[i]
                    stop_price = price - 2.0 * atr[i]
                    in_trade = True
                    bars_in_trade = 0
                elif ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]:
                    direction = -1
                    entry_price = price
                    entry_date = dates[i]
                    stop_price = price + 2.0 * atr[i]
                    in_trade = True
                    bars_in_trade = 0

            elif stype == "momentum":
                lb = params["lookback"]
                if i >= lb:
                    ret = price / closes[i - lb] - 1
                    if ret > 0.05:  # 5% threshold for daily
                        direction = 1
                        entry_price = price
                        entry_date = dates[i]
                        stop_price = price - 2.0 * atr[i]
                        in_trade = True
                        bars_in_trade = 0
                    elif ret < -0.05:
                        direction = -1
                        entry_price = price
                        entry_date = dates[i]
                        stop_price = price + 2.0 * atr[i]
                        in_trade = True
                        bars_in_trade = 0

            elif stype == "donchian":
                lb = params["lookback"]
                rr = params.get("rr", 3.0)
                if i >= lb:
                    upper = np.max(highs[i-lb:i])
                    lower = np.min(lows[i-lb:i])
                    if price > upper:
                        direction = 1
                        entry_price = price
                        entry_date = dates[i]
                        stop_price = price - 2.0 * atr[i]
                        in_trade = True
                        bars_in_trade = 0
                    elif price < lower:
                        direction = -1
                        entry_price = price
                        entry_date = dates[i]
                        stop_price = price + 2.0 * atr[i]
                        in_trade = True
                        bars_in_trade = 0

            elif stype == "bollinger_mr":
                lb = params["lookback"]
                width = params.get("width", 2.0)
                if i >= lb:
                    sma = np.mean(closes[i-lb:i])
                    std = np.std(closes[i-lb:i])
                    lower_band = sma - width * std
                    upper_band = sma + width * std
                    if price < lower_band:
                        direction = 1
                        entry_price = price
                        entry_date = dates[i]
                        stop_price = price - 2.0 * atr[i]
                        in_trade = True
                        bars_in_trade = 0
                    elif price > upper_band:
                        direction = -1
                        entry_price = price
                        entry_date = dates[i]
                        stop_price = price + 2.0 * atr[i]
                        in_trade = True
                        bars_in_trade = 0

            elif stype == "rsi_mr":
                lb = params.get("lookback", 14)
                oversold = params.get("oversold", 30)
                overbought = params.get("overbought", 70)
                rsi = compute_rsi(closes[:i+1], lb)
                if rsi[i] < oversold and i > 0 and rsi[i-1] >= oversold:
                    direction = 1
                    entry_price = price
                    entry_date = dates[i]
                    stop_price = price - 2.0 * atr[i]
                    in_trade = True
                    bars_in_trade = 0
                elif rsi[i] > overbought and i > 0 and rsi[i-1] <= overbought:
                    direction = -1
                    entry_price = price
                    entry_date = dates[i]
                    stop_price = price + 2.0 * atr[i]
                    in_trade = True
                    bars_in_trade = 0

    # Close open trade at end
    if in_trade:
        ret = (closes[-1] / entry_price - 1) * direction
        trades.append({"entry": entry_date, "exit": dates[-1], "return": ret - COST})

    return trades


# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(trades):
    if len(trades) < 3:
        return {"trades": len(trades), "wr": 0, "pf": 0, "sharpe": 0, "max_dd": 100, "ret": 0}

    returns = [t["return"] for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    win_rate = len(wins) / len(returns) * 100
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-10
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0

    mean_ret = np.mean(returns)
    std_ret = np.std(returns) if len(returns) > 1 else 1e-10
    sharpe = mean_ret / std_ret * np.sqrt(252 / max(1, len(returns) / 12)) if std_ret > 0 else 0

    cumulative = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_dd = np.max(drawdown) * 100 if len(drawdown) > 0 else 0

    return {
        "trades": len(returns),
        "wr": win_rate,
        "pf": profit_factor,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "ret": sum(returns) * 100,
    }


def monte_carlo(trades, n_sims=2000):
    returns = [t["return"] for t in trades]
    if len(returns) < 10:
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
    chunk = len(bars) // n_windows
    windows = []
    for i in range(n_windows):
        start = i * chunk
        end = (i + 1) * chunk if i < n_windows - 1 else len(bars)
        windows.append(bars[start:end])
    return windows


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 80)
    print("  MULTI-INSTRUMENT STRATEGY TEST — Yahoo Finance + Alpha Vantage")
    print("=" * 80)
    print()
    print(f"Symbols: {len(SYMBOLS)} | Strategies: {len(STRATEGIES)} | "
          f"Total combos: {len(SYMBOLS) * len(STRATEGIES)}")
    print()

    # =========================================================================
    # PHASE 1: Fetch data
    # =========================================================================
    print("PHASE 1: Fetching daily data...")
    print("-" * 60)

    cache = {}
    for sym_name, sym_info in SYMBOLS.items():
        print(f"  {sym_name} ({sym_info['yahoo']})...", end=" ")

        # Try Yahoo Finance first
        bars = fetch_yahoo(sym_info["yahoo"], "2005-01-01", "2026-08-19")

        # Fallback to Alpha Vantage
        if len(bars) < 500:
            print(f"Yahoo: {len(bars)} bars, trying Alpha Vantage...", end=" ")
            av_bars = fetch_alpha_vantage(
                sym_info["alpha"],
                is_forex=(sym_info["type"] == "forex")
            )
            if len(av_bars) > len(bars):
                bars = av_bars

        cache[sym_name] = bars
        if bars:
            print(f"{len(bars)} bars ({bars[0]['date']} to {bars[-1]['date']})")
        else:
            print("NO DATA")

    # =========================================================================
    # PHASE 2: Full backtest (2005-2023)
    # =========================================================================
    print()
    print("PHASE 2: In-sample backtest (2005-01-01 to 2023-12-31)")
    print("-" * 60)
    print(f"{'Symbol':12s} {'Strategy':25s} {'T':>4s} {'WR%':>6s} {'PF':>6s} "
          f"{'Sharpe':>8s} {'DD%':>6s} {'Ret%':>8s} {'WF':>4s}")
    print("-" * 80)

    results = {}
    for sym_name, sym_info in SYMBOLS.items():
        bars_is = [b for b in cache[sym_name] if "2005-01-01" <= b["date"] <= "2023-12-31"]

        if len(bars_is) < 250:
            continue

        for strat in STRATEGIES:
            name = f"{sym_name}/{strat['name']}"
            trades = run_strategy(bars_is, strat)
            metrics = compute_metrics(trades)

            # Walk-forward
            windows = split_windows(bars_is, 3)
            wf_pass = 0
            for wbars in windows:
                w_trades = run_strategy(wbars, strat)
                w_metrics = compute_metrics(w_trades)
                if w_metrics["sharpe"] > 0 and w_metrics["pf"] > 1.0:
                    wf_pass += 1

            results[name] = {"metrics": metrics, "trades": trades, "wf": wf_pass, "strat": strat, "symbol": sym_name}

            star = "⭐" if wf_pass >= 2 and metrics["pf"] > 1.2 else "  "
            print(f"{star}{sym_name:12s} {strat['name']:25s} {metrics['trades']:4d} "
                  f"{metrics['wr']:6.1f} {metrics['pf']:6.2f} {metrics['sharpe']:8.3f} "
                  f"{metrics['max_dd']:6.1f} {metrics['ret']:8.1f} {wf_pass}/3")

    # =========================================================================
    # PHASE 3: Strong signals
    # =========================================================================
    print()
    print("=" * 80)
    strong = [(name, data) for name, data in results.items()
              if data["wf"] >= 2 and data["metrics"]["pf"] > 1.2 and data["metrics"]["trades"] >= 20]
    strong.sort(key=lambda x: x[1]["metrics"]["sharpe"], reverse=True)

    if strong:
        print(f"  STRONG SIGNALS ({len(strong)} strategies with WF>=2/3, PF>1.2, T>=20):")
        print("-" * 80)
        print(f"  {'Strategy':35s} {'T':>4s} {'WR%':>6s} {'PF':>6s} "
              f"{'Sharpe':>8s} {'DD%':>6s} {'Ret%':>8s} {'WF':>4s}")
        print("-" * 80)

        for name, data in strong[:20]:
            m = data["metrics"]
            print(f"  {name:35s} {m['trades']:4d} {m['wr']:6.1f} {m['pf']:6.2f} "
                  f"{m['sharpe']:8.3f} {m['max_dd']:6.1f} {m['ret']:8.1f} {data['wf']}/3")
    else:
        print("  ❌ NO STRONG SIGNALS FOUND (WF>=2/3, PF>1.2, T>=20)")

    # =========================================================================
    # PHASE 4: Monte Carlo on top candidates
    # =========================================================================
    print()
    print("=" * 80)
    print("  MONTE CARLO — Top Candidates (2000 simulations)")
    print("-" * 80)

    top_candidates = strong[:10] if strong else [(name, data) for name, data in
                                                  sorted(results.items(), key=lambda x: x[1]["metrics"]["sharpe"], reverse=True)[:5]]

    for name, data in top_candidates:
        if data["metrics"]["trades"] < 10:
            continue
        mc = monte_carlo(data["trades"], 2000)
        print(f"  {name:35s} Sh_med={mc['sharpe_med']:6.2f}  DD_med={mc['dd_med']:5.1f}%  "
              f"DD_99={mc['dd_99']:5.1f}%  P(profit)={mc['p_profit']:5.1f}%  "
              f"Survival={mc['survival']:5.1f}%")

    # =========================================================================
    # PHASE 5: Portfolio construction
    # =========================================================================
    print()
    print("=" * 80)
    print("  PORTFOLIO — Correlation & Combined Metrics")
    print("-" * 80)

    if strong:
        # Filter for portfolio: WF>=2/3, PF>1.2, T>=20, MC survival>60%
        portfolio_members = []
        for name, data in strong:
            if data["metrics"]["trades"] >= 20:
                mc = monte_carlo(data["trades"], 500)
                if mc["survival"] > 60:
                    portfolio_members.append((name, data, mc))

        if len(portfolio_members) >= 2:
            print(f"\n  Portfolio members ({len(portfolio_members)} strategies):")
            for name, data, mc in portfolio_members:
                m = data["metrics"]
                print(f"    {name:35s}  PF={m['pf']:.2f}  WF={data['wf']}/3  "
                      f"Sh={m['sharpe']:.2f}  Surv={mc['survival']:.0f}%")

            # Correlation matrix
            names = [n for n, _, _ in portfolio_members]
            monthly_rets = {}
            for name, data, _ in portfolio_members:
                month_rets = defaultdict(float)
                for t in data["trades"]:
                    month = t["entry"][:7]
                    month_rets[month] += t["return"]
                monthly_rets[name] = month_rets

            all_months = sorted(set().union(*[set(mr.keys()) for mr in monthly_rets.values()]))

            n = len(names)
            corrs = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i == j:
                        corrs[i][j] = 1.0
                    else:
                        a = np.array([monthly_rets[names[i]].get(m, 0.0) for m in all_months])
                        b = np.array([monthly_rets[names[j]].get(m, 0.0) for m in all_months])
                        if np.std(a) > 0 and np.std(b) > 0:
                            corrs[i][j] = np.corrcoef(a, b)[0][1]

            print(f"\n  Correlation matrix:")
            print(f"  {'':35s}", end="")
            for name in names:
                print(f"  {name[:10]:>10s}", end="")
            print()
            for i in range(n):
                print(f"  {names[i][:33]:35s}", end="")
                for j in range(n):
                    print(f"  {corrs[i][j]:10.2f}", end="")
                print()

            # Combined portfolio
            portfolio_monthly = []
            for m in all_months:
                avg_ret = np.mean([monthly_rets[name].get(m, 0.0) for name in names])
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
        else:
            print(f"  Only {len(portfolio_members)} strategies pass filters. Need >= 2 for portfolio.")
    else:
        print("  No strong signals to build portfolio from.")

    # =========================================================================
    # FINAL VERDICT
    # =========================================================================
    print()
    print("=" * 80)
    print("  FINAL VERDICT")
    print("=" * 80)

    approved = []
    for name, data in strong:
        if data["metrics"]["trades"] >= 20:
            mc = monte_carlo(data["trades"], 500)
            if mc["survival"] > 60:
                approved.append((name, data, mc))

    if approved:
        print(f"\n  ✅ APPROVED FOR PAPER TRADING ({len(approved)} strategies):")
        for name, data, mc in approved:
            m = data["metrics"]
            print(f"    {name:35s}  PF={m['pf']:.2f}  WF={data['wf']}/3  "
                  f"Sh={m['sharpe']:.2f}  DD={m['max_dd']:.1f}%  "
                  f"Surv={mc['survival']:.0f}%  P(profit)={mc['p_profit']:.0f}%")
    else:
        print("\n  ❌ NO STRATEGIES APPROVED")
        print("  Closest candidates:")
        for name, data in sorted(results.items(), key=lambda x: x[1]["metrics"]["sharpe"], reverse=True)[:5]:
            m = data["metrics"]
            if m["trades"] >= 5:
                print(f"    {name:35s}  Sh={m['sharpe']:.2f}  PF={m['pf']:.2f}  "
                      f"WF={data['wf']}/3  T={m['trades']}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
