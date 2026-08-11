#!/usr/bin/env python3
"""
XPT/USD Regime Filter: detect when mean-reversion works vs. doesn't,
and add regime-aware filtering to Bollinger/Z-Score/RSI strategies.
"""

import sys, os, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_data import fetch_all_timeframes
from strategies import _make_signals, _trend_filter
from backtest import run_backtest


# ============================================================
# REGIME DETECTION METHODS
# ============================================================

def compute_regime_indicators(df):
    """Compute various regime indicators on daily data."""
    df = df.copy()
    
    # 1. ATR-based volatility regime
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))
    )
    df['atr_20'] = df['tr'].rolling(20).mean()
    df['atr_pct'] = df['atr_20'] / df['close'] * 100
    df['vol_percentile'] = df['atr_pct'].rolling(252).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5, raw=False)
    
    # 2. ADX trend strength
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    atr_14 = df['tr'].rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14.replace(0, np.nan))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    df['adx'] = dx.rolling(14).mean()
    
    # 3. Bollinger Bandwidth (volatility squeeze detection)
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
    df['bb_width_pctile'] = df['bb_width'].rolling(252).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5, raw=False)
    
    # 4. Price relative to VWAP
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap_20'] = (typical_price * df['volume'].fillna(0)).rolling(20).sum() / df['volume'].fillna(0).rolling(20).sum()
    df['price_vs_vwap'] = (df['close'] - df['vwap_20']) / df['vwap_20'] * 100
    
    # 5. Consecutive direction (mean reversion tendency)
    df['direction'] = np.sign(df['close'] - df['close'].shift(1))
    df['consec_same'] = 0
    count = 0
    for i in range(1, len(df)):
        if df['direction'].iloc[i] == df['direction'].iloc[i-1] and df['direction'].iloc[i] != 0:
            count += 1
        else:
            count = 0
        df.iloc[i, df.columns.get_loc('consec_same')] = count
    
    # 6. Realized vol vs implied vol proxy (BB width as % of price)
    df['realized_vol_20'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
    df['realized_vol_60'] = df['close'].pct_change().rolling(60).std() * np.sqrt(252) * 100
    df['vol_term_structure'] = df['realized_vol_20'] / df['realized_vol_60'].replace(0, np.nan)
    
    # 7. Distance from 200-day SMA (trend regime)
    df['sma_200'] = df['close'].rolling(200).mean()
    df['dist_from_sma200'] = (df['close'] - df['sma_200']) / df['sma_200'] * 100
    
    return df


def classify_regime(row):
    """Classify market regime based on multiple indicators."""
    regimes = []
    
    # Volatility regime
    vol_pct = row.get('vol_percentile', 0.5)
    if vol_pct < 0.25:
        regimes.append('low_vol')
    elif vol_pct > 0.75:
        regimes.append('high_vol')
    else:
        regimes.append('mid_vol')
    
    # Trend regime
    adx = row.get('adx', 25)
    if adx > 30:
        regimes.append('strong_trend')
    elif adx > 20:
        regimes.append('weak_trend')
    else:
        regimes.append('range')
    
    # Mean reversion tendency
    consec = row.get('consec_same', 0)
    if consec >= 4:
        regimes.append('trending')
    else:
        regimes.append('mean_reverting')
    
    # Distance from SMA200
    dist = row.get('dist_from_sma200', 0)
    if abs(dist) > 15:
        regimes.append('extended')
    elif abs(dist) > 8:
        regimes.append('moderate_trend')
    else:
        regimes.append('near_mean')
    
    # BB squeeze
    bb_pct = row.get('bb_width_pctile', 0.5)
    if bb_pct < 0.2:
        regimes.append('squeeze')
    elif bb_pct > 0.8:
        regimes.append('expanded')
    
    return '_'.join(regimes)


# ============================================================
# REGIME-AWARE STRATEGIES
# ============================================================

def regime_filtered_bollinger(df, vol_threshold=0.75, adx_threshold=25, period=20, std_dev=2.0):
    """Bollinger Reversion with regime filter: skip trades in unfavorable regimes."""
    df = compute_regime_indicators(df)
    
    df['bb_mid'] = df['close'].rolling(period).mean()
    df['bb_std'] = df['close'].rolling(period).std()
    df['bb_upper'] = df['bb_mid'] + std_dev * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - std_dev * df['bb_std']
    
    # Regime filter: only trade when volatility is not extreme AND not in strong trend
    regime_ok = (
        (df['vol_percentile'].fillna(0.5) < vol_threshold) &  # Not extreme vol
        (df['adx'].fillna(25) < adx_threshold) &  # Not strong trend
        (df['consec_same'].fillna(0) < 5)  # Not in extended run
    )
    
    entries_long = (df['close'] < df['bb_lower']) & (df['close'].shift(1) >= df['bb_lower'].shift(1)) & regime_ok
    entries_short = (df['close'] > df['bb_upper']) & (df['close'].shift(1) <= df['bb_upper'].shift(1)) & regime_ok
    exit_long = df['close'] > df['bb_mid']
    exit_short = df['close'] < df['bb_mid']
    
    return _make_signals(df, entries_long, entries_short, exit_long, exit_short)


def regime_filtered_rsi(df, vol_threshold=0.75, adx_threshold=25, period=14, oversold=30, overbought=70, exit_mid=50):
    """RSI with regime filter."""
    df = compute_regime_indicators(df)
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    regime_ok = (
        (df['vol_percentile'].fillna(0.5) < vol_threshold) &
        (df['adx'].fillna(25) < adx_threshold) &
        (df['consec_same'].fillna(0) < 5)
    )
    
    entries_long = (df['rsi'] < oversold) & (df['rsi'].shift(1) >= oversold) & regime_ok
    entries_short = (df['rsi'] > overbought) & (df['rsi'].shift(1) <= overbought) & regime_ok
    exit_long = df['rsi'] > exit_mid
    exit_short = df['rsi'] < exit_mid
    
    return _make_signals(df, entries_long, entries_short, exit_long, exit_short)


def regime_filtered_zscore(df, vol_threshold=0.75, adx_threshold=25, period=20, z_entry=2.0, z_exit=0.0):
    """Z-Score with regime filter."""
    df = compute_regime_indicators(df)
    
    df['mean'] = df['close'].rolling(period).mean()
    df['std'] = df['close'].rolling(period).std()
    df['zscore'] = (df['close'] - df['mean']) / df['std'].replace(0, np.nan)
    
    regime_ok = (
        (df['vol_percentile'].fillna(0.5) < vol_threshold) &
        (df['adx'].fillna(25) < adx_threshold) &
        (df['consec_same'].fillna(0) < 5)
    )
    
    entries_long = (df['zscore'] < -z_entry) & (df['zscore'].shift(1) >= -z_entry) & regime_ok
    entries_short = (df['zscore'] > z_entry) & (df['zscore'].shift(1) <= z_entry) & regime_ok
    exit_long = df['zscore'] > z_exit
    exit_short = df['zscore'] < z_exit
    
    return _make_signals(df, entries_long, entries_short, exit_long, exit_short)


# Original strategies (no filter) for comparison
def raw_bollinger(df, period=20, std_dev=2.0):
    df = df.copy()
    df['bb_mid'] = df['close'].rolling(period).mean()
    df['bb_std'] = df['close'].rolling(period).std()
    df['bb_upper'] = df['bb_mid'] + std_dev * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - std_dev * df['bb_std']
    entries_long = (df['close'] < df['bb_lower']) & (df['close'].shift(1) >= df['bb_lower'].shift(1))
    entries_short = (df['close'] > df['bb_upper']) & (df['close'].shift(1) <= df['bb_upper'].shift(1))
    exit_long = df['close'] > df['bb_mid']
    exit_short = df['close'] < df['bb_mid']
    return _make_signals(df, entries_long, entries_short, exit_long, exit_short)


def raw_rsi(df, period=14, oversold=30, overbought=70, exit_mid=50):
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    entries_long = (df['rsi'] < oversold) & (df['rsi'].shift(1) >= oversold)
    entries_short = (df['rsi'] > overbought) & (df['rsi'].shift(1) <= overbought)
    exit_long = df['rsi'] > exit_mid
    exit_short = df['rsi'] < exit_mid
    return _make_signals(df, entries_long, entries_short, exit_long, exit_short)


def raw_zscore(df, period=20, z_entry=2.0, z_exit=0.0):
    df = df.copy()
    df['mean'] = df['close'].rolling(period).mean()
    df['std'] = df['close'].rolling(period).std()
    df['zscore'] = (df['close'] - df['mean']) / df['std'].replace(0, np.nan)
    entries_long = (df['zscore'] < -z_entry) & (df['zscore'].shift(1) >= -z_entry)
    entries_short = (df['zscore'] > z_entry) & (df['zscore'].shift(1) <= z_entry)
    exit_long = df['zscore'] > z_exit
    exit_short = df['zscore'] < z_exit
    return _make_signals(df, entries_long, entries_short, exit_long, exit_short)


# ============================================================
# MAIN
# ============================================================

def run_comparison(df, strategy_name, raw_fn, filtered_fn, raw_params=None, filtered_params=None):
    """Compare raw vs regime-filtered strategy."""
    raw_params = raw_params or {}
    filtered_params = filtered_params or {}
    
    # Raw
    try:
        raw_signals = raw_fn(df, **raw_params)
        raw_result = run_backtest(df, raw_signals, f"{strategy_name} (Raw)", "XPT/USD", "1d")
    except Exception as e:
        raw_result = None
        print(f"  Raw {strategy_name}: ERROR {e}")
    
    # Filtered
    try:
        filtered_signals = filtered_fn(df, **filtered_params)
        filtered_result = run_backtest(df, filtered_signals, f"{strategy_name} (Regime)", "XPT/USD", "1d")
    except Exception as e:
        filtered_result = None
        print(f"  Filtered {strategy_name}: ERROR {e}")
    
    return raw_result, filtered_result


def main():
    print("="*70)
    print("  XPT/USD REGIME FILTER ANALYSIS")
    print("="*70)
    
    # Fetch data
    print("\n[1/3] Fetching XPT/USD data...")
    data = fetch_all_timeframes("XPT/USD", start="2021-01-01", end="2026-08-10")
    df = data.get("1d")
    if df is None or df.empty:
        print("ERROR: No daily data")
        return
    
    print(f"  Daily data: {len(df)} rows")
    
    # Analyze regimes
    print("\n[2/3] Analyzing market regimes...")
    df_regime = compute_regime_indicators(df)
    
    # Show regime distribution
    df_regime['regime'] = df_regime.apply(classify_regime, axis=1)
    regime_counts = df_regime['regime'].value_counts().head(10)
    print(f"\n  Top 10 regime combinations:")
    for regime, count in regime_counts.items():
        pct = count / len(df_regime) * 100
        print(f"    {regime}: {count} days ({pct:.1f}%)")
    
    # Compare raw vs filtered strategies
    print("\n[3/3] Comparing Raw vs Regime-Filtered strategies...")
    
    strategies = [
        ("Bollinger Reversion", raw_bollinger, regime_filtered_bollinger,
         {"period": 20, "std_dev": 2.0}, {"period": 20, "std_dev": 2.0}),
        ("RSI Oversold/Overbought", raw_rsi, regime_filtered_rsi,
         {"period": 14, "oversold": 30, "overbought": 70, "exit_mid": 50},
         {"period": 14, "oversold": 30, "overbought": 70, "exit_mid": 50}),
        ("Z-Score Reversion", raw_zscore, regime_filtered_zscore,
         {"period": 20, "z_entry": 2.0, "z_exit": 0.0},
         {"period": 20, "z_entry": 2.0, "z_exit": 0.0}),
    ]
    
    all_results = []
    
    for name, raw_fn, filtered_fn, raw_p, filt_p in strategies:
        print(f"\n  {name}:")
        raw, filtered = run_comparison(df, name, raw_fn, filtered_fn, raw_p, filt_p)
        
        if raw:
            meets_raw = "✅" if raw.meets_criteria() else "❌"
            print(f"    RAW:      WR={raw.win_rate*100:.1f}% PF={raw.profit_factor:.2f} SR={raw.sharpe_ratio:.2f} DD={raw.max_drawdown:.1f}% T={raw.total_trades} {meets_raw}")
            all_results.append({
                "strategy": f"{name} (Raw)", "params": raw_p,
                "win_rate": round(raw.win_rate*100, 2), "profit_factor": round(raw.profit_factor, 3),
                "sharpe": round(raw.sharpe_ratio, 3), "max_dd": round(raw.max_drawdown, 2),
                "trades": raw.total_trades, "meets": raw.meets_criteria(),
            })
        
        if filtered:
            meets_filt = "✅" if filtered.meets_criteria() else "❌"
            print(f"    FILTERED: WR={filtered.win_rate*100:.1f}% PF={filtered.profit_factor:.2f} SR={filtered.sharpe_ratio:.2f} DD={filtered.max_drawdown:.1f}% T={filtered.total_trades} {meets_filt}")
            all_results.append({
                "strategy": f"{name} (Regime)", "params": filt_p,
                "win_rate": round(filtered.win_rate*100, 2), "profit_factor": round(filtered.profit_factor, 3),
                "sharpe": round(filtered.sharpe_ratio, 3), "max_dd": round(filtered.max_drawdown, 2),
                "trades": filtered.total_trades, "meets": filtered.meets_criteria(),
            })
            
            if raw and filtered:
                # Improvement analysis
                wr_diff = (filtered.win_rate - raw.win_rate) * 100
                pf_diff = filtered.profit_factor - raw.profit_factor
                sr_diff = filtered.sharpe_ratio - raw.sharpe_ratio
                dd_diff = filtered.max_drawdown - raw.max_drawdown
                trade_diff = filtered.total_trades - raw.total_trades
                
                print(f"    DELTA:    WR {wr_diff:+.1f}% | PF {pf_diff:+.2f} | SR {sr_diff:+.2f} | DD {dd_diff:+.1f}% | Trades {trade_diff:+d}")
                
                if filtered.meets_criteria() and not raw.meets_criteria():
                    print(f"    🎯 REGIME FILTER TURNED A LOSER INTO A WINNER!")
                elif raw.meets_criteria() and not filtered.meets_criteria():
                    print(f"    ⚠️ Regime filter degraded a passing strategy")
                elif filtered.sharpe_ratio > raw.sharpe_ratio:
                    print(f"    ✅ Regime filter improved risk-adjusted returns")
                else:
                    print(f"    ➖ Regime filter did not help")
    
    # Test different regime thresholds
    print(f"\n  PARAMETER SWEPT REGIME FILTERS:")
    print(f"  {'='*60}")
    
    threshold_results = []
    for vol_thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for adx_thresh in [20, 25, 30, 35]:
            try:
                signals = regime_filtered_bollinger(df, vol_threshold=vol_thresh, adx_threshold=adx_thresh)
                result = run_backtest(df, signals, "Bollinger Regime", "XPT/USD", "1d")
                if result.total_trades >= 10:
                    threshold_results.append({
                        'vol_thresh': vol_thresh, 'adx_thresh': adx_thresh,
                        'win_rate': round(result.win_rate*100, 2),
                        'profit_factor': round(result.profit_factor, 3),
                        'sharpe': round(result.sharpe_ratio, 3),
                        'max_dd': round(result.max_drawdown, 2),
                        'trades': result.total_trades,
                        'meets': result.meets_criteria(),
                    })
            except:
                pass
    
    # Sort by Sharpe
    threshold_results.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  Top 10 regime filter parameter combos (by Sharpe):")
    for i, r in enumerate(threshold_results[:10]):
        status = "✅" if r['meets'] else "❌"
        print(f"    {i+1}. vol<{r['vol_thresh']}, adx<{r['adx_thresh']}: WR={r['win_rate']}% PF={r['profit_factor']} SR={r['sharpe']} DD={r['max_dd']}% T={r['trades']} {status}")
    
    passing = [r for r in threshold_results if r['meets']]
    print(f"\n  {len(passing)}/{len(threshold_results)} regime filter combos pass all criteria")
    
    # Save
    os.makedirs("results", exist_ok=True)
    output = {
        "type": "regime_filter",
        "symbol": "XPT/USD",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "strategy_comparison": all_results,
        "regime_threshold_sweep": threshold_results,
        "passing_combos": len(passing),
        "total_combos": len(threshold_results),
    }
    outpath = "results/XPT_regime_filter.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {outpath}")
    
    # Verdict
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")
    if passing:
        best = passing[0]
        print(f"  🎯 REGIME FILTER WORKS!")
        print(f"  Best combo: vol<{best['vol_thresh']}, adx<{best['adx_thresh']}")
        print(f"  WR={best['win_rate']}% PF={best['profit_factor']} SR={best['sharpe']} DD={best['max_dd']}% T={best['trades']}")
    else:
        print(f"  ⚠️ No regime filter combo passes all criteria for XPT/USD")
        if threshold_results:
            best = threshold_results[0]
            print(f"  Best attempt: vol<{best['vol_thresh']}, adx<{best['adx_thresh']}")
            print(f"  WR={best['win_rate']}% PF={best['profit_factor']} SR={best['sharpe']} DD={best['max_dd']}% T={best['trades']}")


if __name__ == "__main__":
    main()
