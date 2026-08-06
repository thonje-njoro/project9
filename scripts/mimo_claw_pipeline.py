#!/usr/bin/env python3
"""
Trading Strategy Optimization Pipeline
=======================================
For use in MiMo Claw (4-hour session limit)

This script:
1. Fetches SPY/QQQ 1-min data from London Strategic Edge API (2022-2024)
2. Runs XAUUSD parameter optimization
3. Performs walk-forward validation

Usage: Copy-paste this entire script into MiMo Claw's code interpreter.
"""

import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

LSE_API_KEY = "lse_live_f4c9a7419371ecdd9365e146247b0289"
LSE_BASE_URL = "https://api.londonstrategicedge.com/vault"

# Data fetching
SYMBOLS = ["SPY", "QQQ"]
TIMEFRAME = "1m"
START_DATE = "2022-01-01"
END_DATE = "2024-12-31"

# Rate limiting (free tier: 10 downloads/hour)
REQUEST_DELAY = 7  # seconds between requests (stays under limit)

# ══════════════════════════════════════════════════════════════
# PART 1: DATA FETCHING FROM LSE API
# ══════════════════════════════════════════════════════════════

def fetch_candles(symbol, timeframe, start, end):
    """Fetch candle data from London Strategic Edge API."""
    url = f"{LSE_BASE_URL}/candles"
    params = {
        "symbol": symbol,
        "timeframe": timeframe,
        "start": start,
        "end": end,
    }
    headers = {"x-api-key": LSE_API_KEY}
    
    print(f"  Fetching {symbol} {timeframe} ({start} to {end})...")
    
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        
        if not data:
            print(f"  WARNING: No data returned for {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts")
        df = df.rename(columns={"symbol": "ticker"})
        
        print(f"  Got {len(df)} bars ({df.index[0]} to {df.index[-1]})")
        return df
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return pd.DataFrame()


def fetch_all_data():
    """Fetch SPY and QQQ 1-min data in 6-month chunks (respects rate limits)."""
    
    print("=" * 70)
    print("PHASE 1: FETCHING DATA FROM LONDON STRATEGIC EDGE")
    print("=" * 70)
    
    all_data = {}
    
    # Define 6-month chunks to stay under 1M row limit
    chunks = [
        ("2022-01-01", "2022-06-30"),
        ("2022-07-01", "2022-12-31"),
        ("2023-01-01", "2023-06-30"),
        ("2023-07-01", "2023-12-31"),
        ("2024-01-01", "2024-06-30"),
        ("2024-07-01", "2024-12-31"),
    ]
    
    for symbol in SYMBOLS:
        print(f"\n--- {symbol} ---")
        frames = []
        
        for chunk_start, chunk_end in chunks:
            df_chunk = fetch_candles(symbol, TIMEFRAME, chunk_start, chunk_end)
            if not df_chunk.empty:
                frames.append(df_chunk)
            
            # Rate limit: wait between requests
            print(f"  Waiting {REQUEST_DELAY}s (rate limit)...")
            time.sleep(REQUEST_DELAY)
        
        if frames:
            df_full = pd.concat(frames).sort_index()
            df_full = df_full[~df_full.index.duplicated(keep="first")]
            
            # Save to parquet
            filename = f"{symbol}_1min_2022_2024.parquet"
            df_full.to_parquet(filename)
            print(f"\n  Saved {filename}: {len(df_full)} bars")
            all_data[symbol] = df_full
        else:
            print(f"  No data fetched for {symbol}!")
    
    return all_data


# ══════════════════════════════════════════════════════════════
# PART 2: ORB STRATEGY (5-MINUTE)
# ══════════════════════════════════════════════════════════════

def resample_to_5min(df_1min):
    """Resample 1-min data to 5-min bars."""
    ohlcv = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    df_5min = df_1min.resample("5min").agg(ohlcv).dropna()
    return df_5min


def run_orb_backtest(df_1min, rel_vol_threshold=1.0, atr_stop_pct=0.10):
    """Run 5-minute ORB backtest on 1-min data."""
    
    df = resample_to_5min(df_1min)
    df["date"] = df.index.date
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    
    # Filter to market hours only (9:30-16:00 ET = 14:30-21:00 UTC)
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
        
        # First 5-min bar (14:30-14:35 UTC = 9:30-9:35 ET)
        or_bars = day[day.index[0]:day.index[0] + timedelta(minutes=5)]
        if len(or_bars) == 0:
            continue
        
        or_bar = or_bars.iloc[0]
        or_open = or_bar["open"]
        or_close = or_bar["close"]
        or_high = or_bar["high"]
        or_low = or_bar["low"]
        or_volume = or_bar["volume"]
        
        # Direction
        is_bullish = or_close > or_open
        if or_close == or_open:
            continue  # Doji - skip
        
        # Relative volume check (simplified: compare to median)
        daily_volumes = df[df["date"] == date]["volume"]
        if daily_volumes.median() > 0:
            rel_vol = or_volume / daily_volumes.median()
        else:
            continue
        
        if rel_vol < rel_vol_threshold:
            continue
        
        # ATR (using prior 14 days)
        prior_dates = [d for d in dates if d < date][-14:]
        if len(prior_dates) < 5:
            continue
        
        atr_vals = []
        for pd_date in prior_dates:
            pd_day = df[df["date"] == pd_date]
            if len(pd_day) > 0:
                daily_range = pd_day["high"].max() - pd_day["low"].min()
                atr_vals.append(daily_range)
        
        atr = np.mean(atr_vals) if atr_vals else 0
        if atr <= 0:
            continue
        
        stop_distance = atr_stop_pct * atr
        
        # Scan for entry
        rest_of_day = day.iloc[1:]  # After OR bar
        in_trade = False
        entry_price = 0
        stop_price = 0
        
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
                # Check exit
                if is_bullish:
                    if bar["low"] <= stop_price:
                        pnl = (stop_price - entry_price) / entry_price * 100
                        trades.append(pnl)
                        break
                    if bar["hour"] >= 20:  # EOD
                        pnl = (bar["close"] - entry_price) / entry_price * 100
                        trades.append(pnl)
                        break
                else:
                    if bar["high"] >= stop_price:
                        pnl = (entry_price - stop_price) / entry_price * 100
                        trades.append(pnl)
                        break
                    if bar["hour"] >= 20:
                        pnl = (entry_price - bar["close"]) / entry_price * 100
                        trades.append(pnl)
                        break
    
    return trades


# ══════════════════════════════════════════════════════════════
# PART 3: XAUUSD SESSION MEAN REVERSION
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
    tp_vol = tp * volume
    cum_tp_vol = pd.Series(tp_vol, index=idx).rolling(20).sum()
    cum_vol = pd.Series(volume, index=idx).rolling(20).sum()
    vwap = (cum_tp_vol / cum_vol.replace(0, np.nan)).values
    
    # Z-score
    price_dev = close - vwap
    dev_std = pd.Series(price_dev, index=idx).rolling(20).std().values
    z_score = np.where(dev_std > 0, price_dev / dev_std, 0)
    
    # ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
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
        
        # Session filter: London 8-12, NY 14-20
        in_session = (8 <= hour < 12) or (14 <= hour < 20)
        
        if not in_session:
            if in_trade:
                pnl = (close[i] - entry_price) / entry_price * 100 * direction
                trades.append(pnl)
                in_trade = False
            continue
        
        # News filter: NFP (1st Friday), skip 13-16 UTC
        if dt.weekday() == 4 and dt.day <= 7 and 13 <= hour < 16:
            if in_trade:
                pnl = (close[i] - entry_price) / entry_price * 100 * direction
                trades.append(pnl)
                in_trade = False
            continue
        
        # Close before rollover
        if hour >= 20:
            if in_trade:
                pnl = (close[i] - entry_price) / entry_price * 100 * direction
                trades.append(pnl)
                in_trade = False
            continue
        
        if in_trade:
            bars_held = i - entry_idx
            
            # Time stop
            if bars_held >= max_hold:
                pnl = (close[i] - entry_price) / entry_price * 100 * direction
                trades.append(pnl)
                in_trade = False
                continue
            
            # Stop loss
            if direction == 1 and low[i] <= stop_price:
                pnl = (stop_price - entry_price) / entry_price * 100
                trades.append(pnl)
                in_trade = False
            elif direction == -1 and high[i] >= stop_price:
                pnl = (entry_price - stop_price) / entry_price * 100
                trades.append(pnl)
                in_trade = False
            # Take profit (z_exit)
            elif direction == 1 and z_score[i] >= -z_exit:
                pnl = (close[i] - entry_price) / entry_price * 100
                trades.append(pnl)
                in_trade = False
            elif direction == -1 and z_score[i] <= z_exit:
                pnl = (entry_price - close[i]) / entry_price * 100
                trades.append(pnl)
                in_trade = False
        else:
            # Entry
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
# PART 4: PARAMETER OPTIMIZATION
# ══════════════════════════════════════════════════════════════

def optimize_xauusd(df):
    """Grid search over XAUUSD parameters."""
    
    print("\n" + "=" * 70)
    print("PHASE 2: XAUUSD PARAMETER OPTIMIZATION")
    print("=" * 70)
    
    # Parameter grid
    z_entries = [1.5, 2.0, 2.5]
    z_exits = [0.0, 0.3, 0.5]
    atr_mults = [1.5, 2.0, 2.5]
    max_holds = [8, 12, 18]
    
    total = len(z_entries) * len(z_exits) * len(atr_mults) * len(max_holds)
    print(f"Testing {total} parameter combinations...")
    
    results = []
    count = 0
    
    for z_entry, z_exit, atr_mult, max_hold in product(z_entries, z_exits, atr_mults, max_holds):
        count += 1
        
        try:
            trades = run_xauusd_backtest(df, z_entry, z_exit, atr_mult, max_hold)
            
            if trades and len(trades) > 20:
                t = np.array(trades)
                sharpe = t.mean() / t.std() if t.std() > 0 else 0
                wr = len(t[t > 0]) / len(t) * 100
                pf = abs(t[t > 0].sum() / t[t <= 0].sum()) if t[t <= 0].sum() != 0 else 999
                
                results.append({
                    "z_entry": z_entry,
                    "z_exit": z_exit,
                    "atr_mult": atr_mult,
                    "max_hold": max_hold,
                    "trades": len(t),
                    "win_rate": round(wr, 1),
                    "total_return": round(t.sum(), 2),
                    "sharpe": round(sharpe, 3),
                    "profit_factor": round(pf, 2),
                    "avg_return": round(t.mean(), 4),
                    "max_loss": round(t.min(), 2),
                })
        except Exception as e:
            pass
        
        if count % 10 == 0:
            print(f"  Progress: {count}/{total}")
    
    if results:
        df_results = pd.DataFrame(results).sort_values("sharpe", ascending=False)
        
        print(f"\n{'=' * 70}")
        print("TOP 10 PARAMETER COMBINATIONS (by Sharpe)")
        print("=" * 70)
        print(df_results.head(10).to_string(index=False))
        
        best = df_results.iloc[0]
        print(f"\n{'=' * 70}")
        print("BEST PARAMETERS")
        print("=" * 70)
        print(f"  z_entry={best['z_entry']}, z_exit={best['z_exit']}")
        print(f"  atr_mult={best['atr_mult']}, max_hold={best['max_hold']}")
        print(f"  Trades={best['trades']}, WR={best['win_rate']}%")
        print(f"  Total return={best['total_return']}%")
        print(f"  Sharpe={best['sharpe']}, PF={best['profit_factor']}")
        
        return df_results, best
    
    return pd.DataFrame(), None


# ══════════════════════════════════════════════════════════════
# PART 5: WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════

def walk_forward_validate(df, best_params):
    """Validate best params on out-of-sample data."""
    
    print("\n" + "=" * 70)
    print("PHASE 3: WALK-FORWARD VALIDATION")
    print("=" * 70)
    
    # Split: 2019-2022 train, 2023-2025 test
    train = df[df.index < "2023-01-01"]
    test = df[df.index >= "2023-01-01"]
    
    z_entry = best_params["z_entry"]
    z_exit = best_params["z_exit"]
    atr_mult = best_params["atr_mult"]
    max_hold = int(best_params["max_hold"])
    
    for label, data in [("Train (2019-2022)", train), ("Test (2023-2025)", test)]:
        trades = run_xauusd_backtest(data, z_entry, z_exit, atr_mult, max_hold)
        
        if trades:
            t = np.array(trades)
            sharpe = t.mean() / t.std() if t.std() > 0 else 0
            wr = len(t[t > 0]) / len(t) * 100
            pf = abs(t[t > 0].sum() / t[t <= 0].sum()) if t[t <= 0].sum() != 0 else 0
            
            print(f"\n{label}:")
            print(f"  Trades={len(t)}, WR={wr:.1f}%")
            print(f"  Total return={t.sum():.2f}%")
            print(f"  Sharpe={sharpe:.3f}, PF={pf:.2f}")
            print(f"  Max loss={t.min():.2f}%")
        else:
            print(f"\n{label}: No trades!")


# ══════════════════════════════════════════════════════════════
# PART 6: ORB PARAMETER SWEEP
# ══════════════════════════════════════════════════════════════

def optimize_orb(all_data):
    """Sweep ORB parameters on SPY/QQQ."""
    
    print("\n" + "=" * 70)
    print("PHASE 4: ORB PARAMETER OPTIMIZATION")
    print("=" * 70)
    
    for symbol in ["SPY", "QQQ"]:
        if symbol not in all_data:
            continue
        
        print(f"\n--- {symbol} ---")
        df = all_data[symbol]
        
        # Split train/test
        train = df[df.index < "2024-01-01"]
        test = df[df.index >= "2024-01-01"]
        
        for label, data in [("Train (2022-2023)", train), ("Test (2024)", test)]:
            best_sharpe = -999
            best_result = None
            
            for rel_vol in [0.5, 0.8, 1.0, 1.5]:
                for atr_stop in [0.08, 0.10, 0.12, 0.15]:
                    trades = run_orb_backtest(data, rel_vol, atr_stop)
                    
                    if trades and len(trades) > 10:
                        t = np.array(trades)
                        sharpe = t.mean() / t.std() if t.std() > 0 else 0
                        
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_result = {
                                "rel_vol": rel_vol,
                                "atr_stop": atr_stop,
                                "trades": len(t),
                                "win_rate": len(t[t > 0]) / len(t) * 100,
                                "total_return": t.sum(),
                                "sharpe": sharpe,
                            }
            
            if best_result:
                print(f"\n  {label} (best):")
                print(f"    rel_vol={best_result['rel_vol']}, atr_stop={best_result['atr_stop']}")
                print(f"    Trades={best_result['trades']}, WR={best_result['win_rate']:.1f}%")
                print(f"    Return={best_result['total_return']:.2f}%, Sharpe={best_result['sharpe']:.3f}")
            else:
                print(f"\n  {label}: No valid results")


# ══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    
    print("╔" + "═" * 68 + "╗")
    print("║  TRADING STRATEGY OPTIMIZATION PIPELINE                          ║")
    print("║  London Strategic Edge API + XAUUSD + ORB                        ║")
    print("╚" + "═" * 68 + "╝")
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Time budget: 4 hours\n")
    
    # PHASE 1: Fetch data
    all_data = fetch_all_data()
    
    elapsed = (time.time() - start_time) / 60
    print(f"\nPhase 1 complete. Elapsed: {elapsed:.1f} min")
    
    # PHASE 2: XAUUSD optimization
    # Load existing XAUUSD data (you'll need to provide this or fetch it)
    print("\n--- Loading XAUUSD data ---")
    try:
        # Try to fetch XAUUSD from LSE
        xau_frames = []
        for chunk_start, chunk_end in [("2019-01-01", "2022-12-31"), ("2023-01-01", "2025-12-31")]:
            df_chunk = fetch_candles("XAU/USD", "1h", chunk_start, chunk_end)
            if not df_chunk.empty:
                xau_frames.append(df_chunk)
            time.sleep(REQUEST_DELAY)
        
        if xau_frames:
            xau = pd.concat(xau_frames).sort_index()
            xau = xau[~xau.index.duplicated(keep="first")]
            xau.to_parquet("XAUUSD_1h.parquet")
            print(f"  XAUUSD: {len(xau)} bars")
        else:
            print("  WARNING: Could not fetch XAUUSD data")
            xau = None
    except Exception as e:
        print(f"  XAUUSD fetch error: {e}")
        xau = None
    
    if xau is not None and len(xau) > 1000:
        results_df, best_params = optimize_xauusd(xau)
        
        if best_params is not None:
            walk_forward_validate(xau, best_params)
    
    elapsed = (time.time() - start_time) / 60
    print(f"\nPhase 2 complete. Elapsed: {elapsed:.1f} min")
    
    # PHASE 3: ORB optimization
    if all_data:
        optimize_orb(all_data)
    
    elapsed = (time.time() - start_time) / 60
    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETE. Total time: {elapsed:.1f} min")
    print(f"{'=' * 70}")
    
    # Save summary
    summary = {
        "completed_at": datetime.now().isoformat(),
        "elapsed_minutes": round(elapsed, 1),
        "data_fetched": list(all_data.keys()),
    }
    
    with open("optimization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nResults saved to:")
    print("  - SPY_1min_2022_2024.parquet")
    print("  - QQQ_1min_2022_2024.parquet")
    print("  - XAUUSD_1h.parquet")
    print("  - optimization_summary.json")


if __name__ == "__main__":
    main()
