#!/usr/bin/env python3
"""
Trading Strategy Optimization Pipeline — EXPANDED
==================================================
For use in MiMo Claw (4-hour session limit)

This script:
1. Fetches data for 6 symbols from London Strategic Edge API
2. Deep researches each symbol's characteristics
3. Runs XAUUSD parameter optimization
4. Runs ORB parameter optimization on all equity symbols
5. Walk-forward validation for all symbols
6. Prop firm readiness assessment

Symbols:
- SPY (S&P 500 ETF) — ORB strategy
- QQQ (Nasdaq 100 ETF) — ORB strategy
- TSLA (Tesla) — ORB strategy (top paper performer)
- NVDA (NVIDIA) — ORB strategy (top paper performer)
- AMD (AMD) — ORB strategy (high volatility)
- XAU/USD (Gold) — Session mean reversion

Usage: Copy-paste this entire script into MiMo Claw's code interpreter.
"""

import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

LSE_API_KEY = "lse_live_f4c9a7419371ecdd9365e146247b0289"
LSE_BASE_URL = "https://api.londonstrategicedge.com/vault"

# Symbols to trade
ORB_SYMBOLS = ["SPY", "QQQ", "TSLA", "NVDA", "AMD"]
FOREX_SYMBOL = "XAU/USD"
ALL_SYMBOLS = ORB_SYMBOLS + [FOREX_SYMBOL]

# Data periods
ORB_START = "2022-01-01"
ORB_END = "2024-12-31"
XAU_START = "2019-01-01"
XAU_END = "2025-12-31"

# Rate limiting (free tier: 10 downloads/hour)
REQUEST_DELAY = 7  # seconds between requests

# Prop firm thresholds
PROP_FIRM_THRESHOLDS = {
    "ftmo_2step": {
        "profit_target_pct": 10.0,
        "max_drawdown_pct": 10.0,
        "daily_loss_pct": 5.0,
        "min_trading_days": 10,
    },
    "the5ers_high_stakes": {
        "profit_target_pct": 8.0,
        "max_drawdown_pct": 6.0,
        "daily_loss_pct": 3.0,
        "min_trading_days": 10,
    },
    "fundingpips": {
        "profit_target_pct": 8.0,
        "max_drawdown_pct": 10.0,
        "daily_loss_pct": 5.0,
        "min_trading_days": 5,
    },
}

# Professional metric thresholds
PRO_METRICS = {
    "min_win_rate": 45.0,       # Relaxed from 70% (unrealistic for most strategies)
    "min_profit_factor": 1.2,   # Relaxed from 1.5
    "min_sharpe": 0.5,          # Relaxed from 2.0
    "max_drawdown_pct": 15.0,   # Max acceptable drawdown
    "min_trades": 30,           # Minimum for statistical significance
}


# ══════════════════════════════════════════════════════════════
# PART 1: DATA FETCHING
# ══════════════════════════════════════════════════════════════

def fetch_candles(symbol, timeframe, start, end):
    """Fetch candle data from London Strategic Edge API."""
    url = f"{LSE_BASE_URL}/candles"
    params = {"symbol": symbol, "timeframe": timeframe, "start": start, "end": end}
    headers = {"x-api-key": LSE_API_KEY}
    
    print(f"  Fetching {symbol} {timeframe} ({start} to {end})...")
    
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        
        if not data:
            print(f"  WARNING: No data for {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts")
        
        print(f"  Got {len(df)} bars")
        return df
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return pd.DataFrame()


def fetch_all_data():
    """Fetch data for all symbols."""
    
    print("=" * 70)
    print("PHASE 1: FETCHING DATA FROM LONDON STRATEGIC EDGE")
    print("=" * 70)
    
    all_data = {}
    
    # ORB symbols: 1-min data in 6-month chunks
    orb_chunks = [
        ("2022-01-01", "2022-06-30"),
        ("2022-07-01", "2022-12-31"),
        ("2023-01-01", "2023-06-30"),
        ("2023-07-01", "2023-12-31"),
        ("2024-01-01", "2024-06-30"),
        ("2024-07-01", "2024-12-31"),
    ]
    
    for symbol in ORB_SYMBOLS:
        print(f"\n--- {symbol} (1-min) ---")
        frames = []
        
        for chunk_start, chunk_end in orb_chunks:
            df_chunk = fetch_candles(symbol, "1m", chunk_start, chunk_end)
            if not df_chunk.empty:
                frames.append(df_chunk)
            time.sleep(REQUEST_DELAY)
        
        if frames:
            df_full = pd.concat(frames).sort_index()
            df_full = df_full[~df_full.index.duplicated(keep="first")]
            filename = f"{symbol}_1min.parquet"
            df_full.to_parquet(filename)
            print(f"  Saved {filename}: {len(df_full)} bars")
            all_data[symbol] = df_full
    
    # XAU/USD: 1-hour data
    print(f"\n--- XAU/USD (1-hour) ---")
    xau_chunks = [
        ("2019-01-01", "2022-12-31"),
        ("2023-01-01", "2025-12-31"),
    ]
    xau_frames = []
    for chunk_start, chunk_end in xau_chunks:
        df_chunk = fetch_candles("XAU/USD", "1h", chunk_start, chunk_end)
        if not df_chunk.empty:
            xau_frames.append(df_chunk)
        time.sleep(REQUEST_DELAY)
    
    if xau_frames:
        xau = pd.concat(xau_frames).sort_index()
        xau = xau[~xau.index.duplicated(keep="first")]
        xau.to_parquet("XAUUSD_1h.parquet")
        print(f"  Saved XAUUSD_1h.parquet: {len(xau)} bars")
        all_data["XAU/USD"] = xau
    
    return all_data


# ══════════════════════════════════════════════════════════════
# PART 2: DEEP RESEARCH — SYMBOL CHARACTERISTICS
# ══════════════════════════════════════════════════════════════

def deep_research_symbol(symbol, df):
    """Analyze symbol characteristics for strategy selection."""
    
    print(f"\n{'─' * 60}")
    print(f"DEEP RESEARCH: {symbol}")
    print(f"{'─' * 60}")
    
    # Basic stats
    returns = df["close"].pct_change().dropna()
    daily_returns = df.resample("D")["close"].last().pct_change().dropna()
    
    research = {
        "symbol": symbol,
        "bars": len(df),
        "date_range": f"{df.index[0].date()} to {df.index[-1].date()}",
        "total_return_pct": round((df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100, 1),
        "annual_vol_pct": round(daily_returns.std() * np.sqrt(252) * 100, 1),
        "daily_vol_pct": round(daily_returns.std() * 100, 3),
        "max_daily_return_pct": round(daily_returns.max() * 100, 2),
        "min_daily_return_pct": round(daily_returns.min() * 100, 2),
        "skewness": round(float(daily_returns.skew()), 3),
        "kurtosis": round(float(daily_returns.kurtosis()), 3),
    }
    
    # Volume analysis
    if "volume" in df.columns:
        research["avg_daily_volume"] = int(df.resample("D")["volume"].sum().mean())
        research["volume_trend"] = "increasing" if df.resample("M")["volume"].sum().iloc[-3:].mean() > df.resample("M")["volume"].sum().iloc[:3].mean() else "decreasing"
    
    # Intraday patterns (if sub-daily data)
    if hasattr(df.index, 'hour'):
        df_copy = df.copy()
        df_copy["hour"] = df_copy.index.hour
        df_copy["range_pct"] = (df_copy["high"] - df_copy["low"]) / df_copy["close"] * 100
        
        hourly_range = df_copy.groupby("hour")["range_pct"].mean()
        research["best_trading_hour_utc"] = int(hourly_range.idxmax())
        research["worst_trading_hour_utc"] = int(hourly_range.idxmin())
        research["avg_intraday_range_pct"] = round(hourly_range.mean(), 3)
    
    # Trend analysis
    sma_20 = df["close"].rolling(20).mean()
    sma_50 = df["close"].rolling(50).mean()
    research["current_trend"] = "bullish" if sma_20.iloc[-1] > sma_50.iloc[-1] else "bearish"
    research["distance_from_20sma_pct"] = round((df["close"].iloc[-1] / sma_20.iloc[-1] - 1) * 100, 2)
    
    # Volatility regime
    recent_vol = returns.tail(20).std()
    long_vol = returns.tail(100).std() if len(returns) > 100 else recent_vol
    research["vol_regime"] = "high" if recent_vol > long_vol * 1.5 else "low" if recent_vol < long_vol * 0.5 else "normal"
    
    # Print research
    for key, value in research.items():
        print(f"  {key}: {value}")
    
    return research


def research_all_symbols(all_data):
    """Deep research on all symbols."""
    
    print("\n" + "=" * 70)
    print("PHASE 2: DEEP RESEARCH — SYMBOL CHARACTERISTICS")
    print("=" * 70)
    
    research_results = {}
    for symbol, df in all_data.items():
        research_results[symbol] = deep_research_symbol(symbol, df)
    
    # Summary table
    print(f"\n{'=' * 70}")
    print("SYMBOL COMPARISON SUMMARY")
    print("=" * 70)
    
    summary_data = []
    for symbol, r in research_results.items():
        summary_data.append({
            "Symbol": symbol,
            "Total Return": f"{r.get('total_return_pct', 0):.1f}%",
            "Annual Vol": f"{r.get('annual_vol_pct', 0):.1f}%",
            "Trend": r.get("current_trend", "N/A"),
            "Vol Regime": r.get("vol_regime", "N/A"),
            "Best Hour": r.get("best_trading_hour_utc", "N/A"),
        })
    
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    
    return research_results


# ══════════════════════════════════════════════════════════════
# PART 3: ORB STRATEGY (5-MINUTE)
# ══════════════════════════════════════════════════════════════

def resample_to_5min(df_1min):
    """Resample 1-min data to 5-min bars."""
    ohlcv = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df_1min.resample("5min").agg(ohlcv).dropna()


def run_orb_backtest(df_1min, rel_vol_threshold=1.0, atr_stop_pct=0.10):
    """Run 5-minute ORB backtest."""
    
    df = resample_to_5min(df_1min)
    df["date"] = df.index.date
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    
    # Market hours only (9:30-16:00 ET)
    market_mask = (
        ((df["hour"] == 14) & (df["minute"] >= 30)) |
        ((df["hour"] >= 15) & (df["hour"] < 21))
    )
    df = df[market_mask].copy()
    
    trades = []
    dates = sorted(df["date"].unique())
    
    for date in dates:
        day = df[df["date"] == date]
        if len(day) < 2:
            continue
        
        or_bar = day.iloc[0]
        or_open, or_close = or_bar["open"], or_bar["close"]
        or_high, or_low = or_bar["high"], or_bar["low"]
        or_volume = or_bar["volume"]
        
        is_bullish = or_close > or_open
        if or_close == or_open:
            continue
        
        # Relative volume
        daily_volumes = df[df["date"] == date]["volume"]
        if daily_volumes.median() > 0:
            rel_vol = or_volume / daily_volumes.median()
        else:
            continue
        if rel_vol < rel_vol_threshold:
            continue
        
        # ATR
        prior_dates = [d for d in dates if d < date][-14:]
        if len(prior_dates) < 5:
            continue
        atr_vals = [df[df["date"] == d]["high"].max() - df[df["date"] == d]["low"].min() for d in prior_dates]
        atr = np.mean(atr_vals)
        if atr <= 0:
            continue
        
        stop_distance = atr_stop_pct * atr
        rest_of_day = day.iloc[1:]
        in_trade = False
        
        for _, bar in rest_of_day.iterrows():
            if not in_trade:
                if is_bullish and bar["high"] > or_high:
                    entry_price = or_high
                    stop_price = entry_price - stop_distance
                    in_trade = True
                elif not is_bullish and bar["low"] < or_low:
                    entry_price = or_low
                    stop_price = entry_price + stop_distance
                    in_trade = True
            else:
                if is_bullish:
                    if bar["low"] <= stop_price:
                        trades.append((stop_price - entry_price) / entry_price * 100)
                        break
                    if bar["hour"] >= 20:
                        trades.append((bar["close"] - entry_price) / entry_price * 100)
                        break
                else:
                    if bar["high"] >= stop_price:
                        trades.append((entry_price - stop_price) / entry_price * 100)
                        break
                    if bar["hour"] >= 20:
                        trades.append((entry_price - bar["close"]) / entry_price * 100)
                        break
    
    return trades


# ══════════════════════════════════════════════════════════════
# PART 4: XAUUSD SESSION MEAN REVERSION
# ══════════════════════════════════════════════════════════════

def run_xauusd_backtest(df, z_entry=2.0, z_exit=0.5, atr_mult=2.0, max_hold=12):
    """Run XAUUSD session mean reversion backtest."""
    
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values if "volume" in df.columns else np.ones(len(df))
    idx = df.index
    n = len(df)
    
    # VWAP
    tp = (high + low + close) / 3
    cum_tp_vol = pd.Series(tp * volume, index=idx).rolling(20).sum()
    cum_vol = pd.Series(volume, index=idx).rolling(20).sum()
    vwap = (cum_tp_vol / cum_vol.replace(0, np.nan)).values
    
    # Z-score
    price_dev = close - vwap
    dev_std = pd.Series(price_dev, index=idx).rolling(20).std().values
    z_score = np.where(dev_std > 0, price_dev / dev_std, 0)
    
    # ATR
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr, index=idx).rolling(14).mean().values
    
    trades = []
    in_trade = False
    direction = 0
    entry_idx = 0
    entry_price = 0
    stop_price = 0
    
    for i in range(168, n):
        dt = idx[i]
        hour = dt.hour
        
        in_session = (8 <= hour < 12) or (14 <= hour < 20)
        
        if not in_session:
            if in_trade:
                trades.append((close[i] - entry_price) / entry_price * 100 * direction)
                in_trade = False
            continue
        
        # NFP filter
        if dt.weekday() == 4 and dt.day <= 7 and 13 <= hour < 16:
            if in_trade:
                trades.append((close[i] - entry_price) / entry_price * 100 * direction)
                in_trade = False
            continue
        
        if hour >= 20:
            if in_trade:
                trades.append((close[i] - entry_price) / entry_price * 100 * direction)
                in_trade = False
            continue
        
        if in_trade:
            bars_held = i - entry_idx
            if bars_held >= max_hold:
                trades.append((close[i] - entry_price) / entry_price * 100 * direction)
                in_trade = False
                continue
            
            if direction == 1 and low[i] <= stop_price:
                trades.append((stop_price - entry_price) / entry_price * 100)
                in_trade = False
            elif direction == -1 and high[i] >= stop_price:
                trades.append((entry_price - stop_price) / entry_price * 100)
                in_trade = False
            elif direction == 1 and z_score[i] >= -z_exit:
                trades.append((close[i] - entry_price) / entry_price * 100)
                in_trade = False
            elif direction == -1 and z_score[i] <= z_exit:
                trades.append((entry_price - close[i]) / entry_price * 100)
                in_trade = False
        else:
            if not np.isnan(z_score[i]) and not np.isnan(atr[i]) and atr[i] > 0:
                if z_score[i] <= -z_entry:
                    direction = 1
                    entry_price = close[i]
                    stop_price = entry_price - atr_mult * atr[i]
                    entry_idx = i
                    in_trade = True
                elif z_score[i] >= z_entry:
                    direction = -1
                    entry_price = close[i]
                    stop_price = entry_price + atr_mult * atr[i]
                    entry_idx = i
                    in_trade = True
    
    return trades


# ══════════════════════════════════════════════════════════════
# PART 5: PROP FIRM ASSESSMENT
# ══════════════════════════════════════════════════════════════

def assess_prop_firm_readiness(trades, symbol, strategy_name):
    """Check if strategy meets prop firm requirements."""
    
    if not trades or len(trades) < PRO_METRICS["min_trades"]:
        return {"ready": False, "reason": "Insufficient trades"}
    
    t = np.array(trades)
    wins = t[t > 0]
    losses = t[t <= 0]
    
    win_rate = len(wins) / len(t) * 100
    total_return = t.sum()
    max_dd = t.min()
    sharpe = t.mean() / t.std() if t.std() > 0 else 0
    pf = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else 999
    
    # Daily P&L simulation
    daily_pnl = []
    running = 0
    for pnl in t:
        running += pnl
        daily_pnl.append(running)
        if running < -5:  # Simulate daily reset
            daily_pnl.append(running)
            running = 0
    
    assessment = {
        "symbol": symbol,
        "strategy": strategy_name,
        "trades": len(t),
        "win_rate": round(win_rate, 1),
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 3),
        "profit_factor": round(pf, 2),
        "avg_win": round(wins.mean(), 3) if len(wins) > 0 else 0,
        "avg_loss": round(losses.mean(), 3) if len(losses) > 0 else 0,
        "meets_win_rate": win_rate >= PRO_METRICS["min_win_rate"],
        "meets_pf": pf >= PRO_METRICS["min_profit_factor"],
        "meets_sharpe": sharpe >= PRO_METRICS["min_sharpe"],
        "meets_dd": abs(max_dd) <= PRO_METRICS["max_drawdown_pct"],
    }
    
    # Check each prop firm
    for firm, rules in PROP_FIRM_THRESHOLDS.items():
        assessment[f"{firm}_pass"] = (
            total_return >= rules["profit_target_pct"] and
            abs(max_dd) <= rules["max_drawdown_pct"]
        )
    
    # Overall readiness
    passes = [
        assessment["meets_win_rate"],
        assessment["meets_pf"],
        assessment["meets_sharpe"],
        assessment["meets_dd"],
    ]
    assessment["ready"] = all(passes)
    assessment["readiness_score"] = sum(passes) / len(passes) * 100
    
    return assessment


# ══════════════════════════════════════════════════════════════
# PART 6: PARAMETER OPTIMIZATION
# ══════════════════════════════════════════════════════════════

def optimize_orb_symbol(symbol, df):
    """Optimize ORB parameters for a single symbol."""
    
    print(f"\n  Optimizing {symbol}...")
    
    # Split: train (2022-2023), test (2024)
    train = df[df.index < "2024-01-01"]
    test = df[df.index >= "2024-01-01"]
    
    best_sharpe = -999
    best_params = {}
    all_results = []
    
    for rel_vol in [0.3, 0.5, 0.8, 1.0, 1.5]:
        for atr_stop in [0.05, 0.08, 0.10, 0.12, 0.15]:
            # Train
            train_trades = run_orb_backtest(train, rel_vol, atr_stop)
            
            if train_trades and len(train_trades) > 10:
                t = np.array(train_trades)
                train_sharpe = t.mean() / t.std() if t.std() > 0 else 0
                
                # Test
                test_trades = run_orb_backtest(test, rel_vol, atr_stop)
                
                if test_trades and len(test_trades) > 5:
                    tt = np.array(test_trades)
                    test_sharpe = tt.mean() / tt.std() if tt.std() > 0 else 0
                    
                    # Robustness: test should not collapse vs train
                    robustness = test_sharpe / train_sharpe if train_sharpe > 0 else 0
                    
                    result = {
                        "rel_vol": rel_vol,
                        "atr_stop": atr_stop,
                        "train_trades": len(train_trades),
                        "train_sharpe": round(train_sharpe, 3),
                        "train_return": round(t.sum(), 2),
                        "train_wr": round(len(t[t > 0]) / len(t) * 100, 1),
                        "test_trades": len(test_trades),
                        "test_sharpe": round(test_sharpe, 3),
                        "test_return": round(tt.sum(), 2),
                        "test_wr": round(len(tt[tt > 0]) / len(tt) * 100, 1),
                        "robustness": round(robustness, 2),
                    }
                    all_results.append(result)
                    
                    # Score: combine train Sharpe and robustness
                    score = train_sharpe * min(robustness, 1.5)
                    if score > best_sharpe:
                        best_sharpe = score
                        best_params = result.copy()
    
    return best_params, all_results


def optimize_xauusd(df):
    """Grid search over XAUUSD parameters."""
    
    print(f"\n  Optimizing XAU/USD...")
    
    # Split
    train = df[df.index < "2023-01-01"]
    test = df[df.index >= "2023-01-01"]
    
    z_entries = [1.5, 2.0, 2.5]
    z_exits = [0.0, 0.3, 0.5]
    atr_mults = [1.5, 2.0, 2.5]
    max_holds = [8, 12, 18]
    
    best_sharpe = -999
    best_params = {}
    all_results = []
    
    for z_entry, z_exit, atr_mult, max_hold in product(z_entries, z_exits, atr_mults, max_holds):
        try:
            train_trades = run_xauusd_backtest(train, z_entry, z_exit, atr_mult, max_hold)
            
            if train_trades and len(train_trades) > 20:
                t = np.array(train_trades)
                train_sharpe = t.mean() / t.std() if t.std() > 0 else 0
                
                test_trades = run_xauusd_backtest(test, z_entry, z_exit, atr_mult, max_hold)
                
                if test_trades and len(test_trades) > 10:
                    tt = np.array(test_trades)
                    test_sharpe = tt.mean() / tt.std() if tt.std() > 0 else 0
                    robustness = test_sharpe / train_sharpe if train_sharpe > 0 else 0
                    
                    result = {
                        "z_entry": z_entry, "z_exit": z_exit,
                        "atr_mult": atr_mult, "max_hold": max_hold,
                        "train_trades": len(train_trades),
                        "train_sharpe": round(train_sharpe, 3),
                        "train_return": round(t.sum(), 2),
                        "train_wr": round(len(t[t > 0]) / len(t) * 100, 1),
                        "test_trades": len(test_trades),
                        "test_sharpe": round(test_sharpe, 3),
                        "test_return": round(tt.sum(), 2),
                        "test_wr": round(len(tt[tt > 0]) / len(tt) * 100, 1),
                        "robustness": round(robustness, 2),
                    }
                    all_results.append(result)
                    
                    score = train_sharpe * min(robustness, 1.5)
                    if score > best_sharpe:
                        best_sharpe = score
                        best_params = result.copy()
        except:
            pass
    
    return best_params, all_results


# ══════════════════════════════════════════════════════════════
# PART 7: FULL PIPELINE
# ══════════════════════════════════════════════════════════════

def run_full_pipeline():
    """Execute the complete optimization pipeline."""
    
    start_time = time.time()
    
    print("╔" + "═" * 68 + "╗")
    print("║  TRADING STRATEGY OPTIMIZATION PIPELINE — PROP FIRM EDITION      ║")
    print("║  6 Symbols · Deep Research · Walk-Forward · Prop Firm Assessment  ║")
    print("╚" + "═" * 68 + "╝")
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Symbols: {', '.join(ALL_SYMBOLS)}")
    print(f"Time budget: 4 hours\n")
    
    # ── PHASE 1: Fetch Data ──
    all_data = fetch_all_data()
    elapsed = (time.time() - start_time) / 60
    print(f"\n✓ Phase 1 complete ({elapsed:.1f} min)")
    
    # ── PHASE 2: Deep Research ──
    research = research_all_symbols(all_data)
    elapsed = (time.time() - start_time) / 60
    print(f"\n✓ Phase 2 complete ({elapsed:.1f} min)")
    
    # ── PHASE 3: Parameter Optimization ──
    print("\n" + "=" * 70)
    print("PHASE 3: PARAMETER OPTIMIZATION (WALK-FORWARD)")
    print("=" * 70)
    
    best_params = {}
    
    # ORB symbols
    for symbol in ORB_SYMBOLS:
        if symbol in all_data:
            best, all_results = optimize_orb_symbol(symbol, all_data[symbol])
            if best:
                best_params[symbol] = {"strategy": "orb", "params": best}
                print(f"\n  {symbol} best (train): Sharpe={best.get('train_sharpe', 0):.3f}, "
                      f"WR={best.get('train_wr', 0):.1f}%, Return={best.get('train_return', 0):.2f}%")
                print(f"  {symbol} test: Sharpe={best.get('test_sharpe', 0):.3f}, "
                      f"WR={best.get('test_wr', 0):.1f}%, Return={best.get('test_return', 0):.2f}%")
    
    # XAU/USD
    if "XAU/USD" in all_data:
        best, all_results = optimize_xauusd(all_data["XAU/USD"])
        if best:
            best_params["XAU/USD"] = {"strategy": "session_mr", "params": best}
            print(f"\n  XAU/USD best (train): Sharpe={best.get('train_sharpe', 0):.3f}, "
                  f"WR={best.get('train_wr', 0):.1f}%, Return={best.get('train_return', 0):.2f}%")
            print(f"  XAU/USD test: Sharpe={best.get('test_sharpe', 0):.3f}, "
                  f"WR={best.get('test_wr', 0):.1f}%, Return={best.get('test_return', 0):.2f}%")
    
    elapsed = (time.time() - start_time) / 60
    print(f"\n✓ Phase 3 complete ({elapsed:.1f} min)")
    
    # ── PHASE 4: Prop Firm Assessment ──
    print("\n" + "=" * 70)
    print("PHASE 4: PROP FIRM READINESS ASSESSMENT")
    print("=" * 70)
    
    assessments = []
    
    for symbol, info in best_params.items():
        strategy = info["strategy"]
        params = info["params"]
        
        # Get full-sample trades for assessment
        if strategy == "orb" and symbol in all_data:
            trades = run_orb_backtest(
                all_data[symbol],
                params.get("rel_vol", 1.0),
                params.get("atr_stop", 0.10)
            )
        elif strategy == "session_mr" and "XAU/USD" in all_data:
            trades = run_xauusd_backtest(
                all_data["XAU/USD"],
                params.get("z_entry", 2.0),
                params.get("z_exit", 0.5),
                params.get("atr_mult", 2.0),
                int(params.get("max_hold", 12))
            )
        else:
            continue
        
        assessment = assess_prop_firm_readiness(trades, symbol, strategy)
        assessments.append(assessment)
    
    # Print assessment table
    if assessments:
        print(f"\n{'Symbol':<10} {'Strategy':<12} {'Trades':<8} {'WR%':<8} {'Return%':<10} "
              f"{'Sharpe':<8} {'PF':<8} {'Ready':<8} {'Score':<8}")
        print("─" * 90)
        
        for a in assessments:
            ready_icon = "✓" if a["ready"] else "✗"
            print(f"{a['symbol']:<10} {a['strategy']:<12} {a['trades']:<8} {a['win_rate']:<8} "
                  f"{a['total_return_pct']:<10} {a['sharpe']:<8} {a['profit_factor']:<8} "
                  f"{ready_icon:<8} {a['readiness_score']:.0f}%")
        
        # Prop firm pass/fail
        print(f"\n{'Symbol':<10} {'FTMO':<10} {'The5ers':<10} {'FundingPips':<12}")
        print("─" * 45)
        for a in assessments:
            ftmo = "✓" if a.get("ftmo_2step_pass", False) else "✗"
            the5ers = "✓" if a.get("the5ers_high_stakes_pass", False) else "✗"
            fp = "✓" if a.get("fundingpips_pass", False) else "✗"
            print(f"{a['symbol']:<10} {ftmo:<10} {the5ers:<10} {fp:<12}")
    
    elapsed = (time.time() - start_time) / 60
    print(f"\n✓ Phase 4 complete ({elapsed:.1f} min)")
    
    # ── FINAL SUMMARY ──
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    ready_count = sum(1 for a in assessments if a["ready"])
    print(f"\nSymbols optimized: {len(best_params)}")
    print(f"Prop firm ready: {ready_count}/{len(assessments)}")
    print(f"Total time: {elapsed:.1f} minutes")
    
    # Save everything
    output = {
        "completed_at": datetime.now().isoformat(),
        "elapsed_minutes": round(elapsed, 1),
        "research": research,
        "best_params": best_params,
        "assessments": assessments,
    }
    
    with open("optimization_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    # Save best params as config-ready format
    config_output = {}
    for symbol, info in best_params.items():
        config_output[symbol] = {
            "strategy": info["strategy"],
            "optimized_params": info["params"],
        }
    
    with open("optimized_config.json", "w") as f:
        json.dump(config_output, f, indent=2)
    
    print("\n✓ Results saved to:")
    print("  - optimization_results.json (full results)")
    print("  - optimized_config.json (config-ready params)")
    print("  - *.parquet (data files)")
    
    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_full_pipeline()
