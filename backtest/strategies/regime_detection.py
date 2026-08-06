#!/usr/bin/env python3
"""
Regime Detection System for ORB Strategy
Tests ORB strategy with multiple regime filters across NVDA, AMD, PLTR, MRVL.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

# ============================================================
# Configuration
# ============================================================
SYMBOLS = ['NVDA', 'AMD', 'PLTR', 'MRVL']
OR_BARS = 12           # Opening range: first 12 bars (60 min)
ATR_PERIOD = 14
TRAIL_ATR_MULT = 1.5
COST_ENTRY_SLIP = 0.003   # 0.3%
COST_EXIT_SLIP = 0.002    # 0.2%
COST_COMMISSION = 0.001   # 0.1% round-trip
TOTAL_COST = COST_ENTRY_SLIP + COST_EXIT_SLIP + COST_COMMISSION  # 0.6%

# ============================================================
# Regime Detection Functions
# ============================================================

def compute_upfraction(df, window=63):
    """UpFraction: % positive daily returns in trailing window."""
    daily_close = df['close'].resample('D').last().dropna()
    daily_ret = daily_close.pct_change()
    up_frac = daily_ret.rolling(window, min_periods=max(20, window//3)).apply(
        lambda x: (x > 0).mean(), raw=True
    )
    # Reindex back to intraday - forward fill
    up_frac_intra = up_frac.reindex(df.index, method='ffill')
    return up_frac_intra


def compute_sma_slope(df, sma_period=50, slope_period=10):
    """SMA slope: pct change of SMA over slope_period bars."""
    sma = df['close'].rolling(sma_period, min_periods=sma_period//2).mean()
    slope = sma.diff(slope_period) / sma.shift(slope_period) * 100
    return slope


def compute_adx(df, period=14):
    """Average Directional Index."""
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return adx


def compute_vol_regime(df, vol_window=20, ma_window=100):
    """Volatility regime: compare rolling vol to its moving average."""
    ret = df['close'].pct_change()
    vol = ret.rolling(vol_window, min_periods=vol_window//2).std()
    vol_ma = vol.rolling(ma_window, min_periods=ma_window//2).mean()
    return vol, vol_ma


def compute_atr(df, period=14):
    """Average True Range."""
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/period, adjust=False).mean()


# ============================================================
# Regime Filter Functions
# ============================================================

def regime_upfraction_long(up_frac, threshold=0.55):
    """Allow longs only when up_frac > threshold."""
    return up_frac > threshold

def regime_upfraction_short(up_frac, threshold=0.55):
    """Allow shorts only when up_frac < (1 - threshold)."""
    return up_frac < (1 - threshold)

def regime_sma_long(sma_slope, threshold=0.5):
    return sma_slope > threshold

def regime_sma_short(sma_slope, threshold=0.5):
    return sma_slope < -threshold

def regime_adx(adx, threshold=20):
    """ADX filter: allow trades only when trending."""
    return adx > threshold

def regime_vol_low(vol, vol_ma):
    """Low vol regime."""
    return vol < vol_ma

def regime_vol_high(vol, vol_ma):
    """High vol regime."""
    return vol > vol_ma


# ============================================================
# ORB Strategy Engine
# ============================================================

def run_orb_strategy(df, regime_filter_long=None, regime_filter_short=None, label="baseline"):
    """
    Run ORB strategy on 5-min data.
    
    Parameters:
    -----------
    df : DataFrame with columns [open, high, low, close, volume], datetime index
    regime_filter_long : Series of bool (same index), True = allow long
    regime_filter_short : Series of bool (same index), True = allow short
    label : str, name for this variant
    
    Returns:
    --------
    dict of performance metrics
    """
    df = df.copy()
    df['date'] = df.index.date
    df['time'] = df.index.time
    atr = compute_atr(df, ATR_PERIOD)
    df['atr'] = atr

    trades = []
    dates = sorted(df['date'].unique())

    for date in dates:
        day_data = df[df['date'] == date]
        if len(day_data) < OR_BARS + 2:
            continue

        # Opening range
        or_data = day_data.iloc[:OR_BARS]
        or_high = or_data['high'].max()
        or_low = or_data['low'].min()
        or_range = or_high - or_low
        if or_range <= 0:
            continue

        # Rest of the day after OR
        post_or = day_data.iloc[OR_BARS:]
        if len(post_or) == 0:
            continue

        # Check regime filters at OR close time
        or_end_idx = or_data.index[-1]
        
        allow_long = True
        allow_short = True
        
        if regime_filter_long is not None:
            if or_end_idx in regime_filter_long.index:
                allow_long = bool(regime_filter_long.loc[or_end_idx])
            else:
                allow_long = False
                
        if regime_filter_short is not None:
            if or_end_idx in regime_filter_short.index:
                allow_short = bool(regime_filter_short.loc[or_end_idx])
            else:
                allow_short = False

        if not allow_long and not allow_short:
            continue

        # Simulate trading through the day
        position = None  # 'long', 'short', or None
        entry_price = None
        stop_price = None
        trail_activated = False
        best_price = None
        entry_time = None

        for idx, row in post_or.iterrows():
            bar_atr = df.loc[idx, 'atr'] if idx in df.index else or_range * 0.1
            if pd.isna(bar_atr) or bar_atr <= 0:
                bar_atr = or_range * 0.1

            if position is None:
                # Check for breakout
                if allow_long and row['close'] > or_high:
                    # Enter long at next bar's open (look-ahead safe: use current bar close as signal,
                    # enter at this bar's close with slippage — conservative)
                    entry_price = row['close'] * (1 + COST_ENTRY_SLIP)
                    stop_price = or_low
                    position = 'long'
                    trail_activated = False
                    best_price = entry_price
                    entry_time = idx
                elif allow_short and row['close'] < or_low:
                    entry_price = row['close'] * (1 - COST_ENTRY_SLIP)
                    stop_price = or_high
                    position = 'short'
                    trail_activated = False
                    best_price = entry_price
                    entry_time = idx

            elif position == 'long':
                best_price = max(best_price, row['high'])
                pnl_r = (best_price - entry_price) / (entry_price - stop_price) if (entry_price - stop_price) > 0 else 0

                # Trail after 1R profit
                if not trail_activated and pnl_r >= 1.0:
                    trail_activated = True
                if trail_activated:
                    new_stop = row['high'] - TRAIL_ATR_MULT * bar_atr
                    stop_price = max(stop_price, new_stop)

                # Check stop
                if row['low'] <= stop_price:
                    exit_price = stop_price * (1 - COST_EXIT_SLIP)
                    pnl_pct = (exit_price - entry_price) / entry_price - COST_COMMISSION
                    trades.append({
                        'date': str(date), 'direction': 'long',
                        'entry': round(entry_price, 4), 'exit': round(exit_price, 4),
                        'pnl_pct': round(pnl_pct * 100, 4),
                        'entry_time': str(entry_time), 'exit_time': str(idx),
                        'label': label
                    })
                    position = None

            elif position == 'short':
                best_price = min(best_price, row['low'])
                pnl_r = (entry_price - best_price) / (stop_price - entry_price) if (stop_price - entry_price) > 0 else 0

                if not trail_activated and pnl_r >= 1.0:
                    trail_activated = True
                if trail_activated:
                    new_stop = row['low'] + TRAIL_ATR_MULT * bar_atr
                    stop_price = min(stop_price, new_stop)

                if row['high'] >= stop_price:
                    exit_price = stop_price * (1 + COST_EXIT_SLIP)
                    pnl_pct = (entry_price - exit_price) / entry_price - COST_COMMISSION
                    trades.append({
                        'date': str(date), 'direction': 'short',
                        'entry': round(entry_price, 4), 'exit': round(exit_price, 4),
                        'pnl_pct': round(pnl_pct * 100, 4),
                        'entry_time': str(entry_time), 'exit_time': str(idx),
                        'label': label
                    })
                    position = None

        # Close any open position at end of day
        if position is not None:
            last_row = post_or.iloc[-1]
            if position == 'long':
                exit_price = last_row['close'] * (1 - COST_EXIT_SLIP)
                pnl_pct = (exit_price - entry_price) / entry_price - COST_COMMISSION
            else:
                exit_price = last_row['close'] * (1 + COST_EXIT_SLIP)
                pnl_pct = (entry_price - exit_price) / entry_price - COST_COMMISSION
            trades.append({
                'date': str(date), 'direction': position,
                'entry': round(entry_price, 4), 'exit': round(exit_price, 4),
                'pnl_pct': round(pnl_pct * 100, 4),
                'entry_time': str(entry_time), 'exit_time': str(post_or.index[-1]),
                'label': label
            })

    return trades, compute_metrics(trades, label)


def compute_metrics(trades, label):
    """Compute strategy performance metrics."""
    if not trades:
        return {
            'label': label, 'total_trades': 0, 'win_rate': 0,
            'avg_pnl_pct': 0, 'total_pnl_pct': 0, 'max_drawdown_pct': 0,
            'sharpe': 0, 'profit_factor': 0, 'avg_win_pct': 0,
            'avg_loss_pct': 0, 'long_trades': 0, 'short_trades': 0,
            'long_win_rate': 0, 'short_win_rate': 0
        }

    pnls = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    longs = [t for t in trades if t['direction'] == 'long']
    shorts = [t for t in trades if t['direction'] == 'short']
    long_wins = [t for t in longs if t['pnl_pct'] > 0]
    short_wins = [t for t in shorts if t['pnl_pct'] > 0]

    # Equity curve for drawdown
    equity = np.cumsum(pnls)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max
    max_dd = abs(min(drawdown)) if len(drawdown) > 0 else 0

    pnl_arr = np.array(pnls)
    sharpe = (pnl_arr.mean() / pnl_arr.std() * np.sqrt(252)) if pnl_arr.std() > 0 else 0

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss

    return {
        'label': label,
        'total_trades': len(trades),
        'win_rate': round(len(wins) / len(trades) * 100, 2) if trades else 0,
        'avg_pnl_pct': round(np.mean(pnls), 4),
        'total_pnl_pct': round(sum(pnls), 2),
        'max_drawdown_pct': round(max_dd, 2),
        'sharpe': round(sharpe, 3),
        'profit_factor': round(profit_factor, 3),
        'avg_win_pct': round(np.mean(wins), 4) if wins else 0,
        'avg_loss_pct': round(np.mean(losses), 4) if losses else 0,
        'long_trades': len(longs),
        'short_trades': len(shorts),
        'long_win_rate': round(len(long_wins) / len(longs) * 100, 2) if longs else 0,
        'short_win_rate': round(len(short_wins) / len(shorts) * 100, 2) if shorts else 0,
    }


# ============================================================
# Main Execution
# ============================================================

def main():
    results = {}
    summary = []

    for symbol in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"Processing {symbol}")
        print(f"{'='*60}")

        df = pd.read_parquet(f'{symbol}_5min.parquet')
        df = df.sort_index()
        
        # Compute regime indicators
        print(f"  Computing regime indicators...")
        up_frac = compute_upfraction(df)
        sma_slope = compute_sma_slope(df)
        adx = compute_adx(df)
        vol, vol_ma = compute_vol_regime(df)

        # Build filter masks
        filt_uf_long = regime_upfraction_long(up_frac)
        filt_uf_short = regime_upfraction_short(up_frac)
        filt_sma_long = regime_sma_long(sma_slope)
        filt_sma_short = regime_sma_short(sma_slope)
        filt_adx = regime_adx(adx)
        # Combined: UpFraction + ADX
        filt_uf_adx_long = filt_uf_long & filt_adx
        filt_uf_adx_short = filt_uf_short & filt_adx

        symbol_results = {}

        # Test matrix
        configs = [
            ("1_baseline",          None,              None),
            ("2_upfraction",        filt_uf_long,      filt_uf_short),
            ("3_sma_slope",         filt_sma_long,     filt_sma_short),
            ("4_adx",               filt_adx,          filt_adx),
            ("5_upfraction_adx",    filt_uf_adx_long,  filt_uf_adx_short),
        ]

        for config_label, f_long, f_short in configs:
            print(f"  Running: {config_label}...", end=" ")
            trades, metrics = run_orb_strategy(df, f_long, f_short, f"{symbol}_{config_label}")
            symbol_results[config_label] = metrics
            print(f"Trades={metrics['total_trades']}, WR={metrics['win_rate']}%, "
                  f"PnL={metrics['total_pnl_pct']}%, Sharpe={metrics['sharpe']}, "
                  f"PF={metrics['profit_factor']}")

        results[symbol] = symbol_results

        # Summary row
        base = symbol_results["1_baseline"]
        best_config = max(symbol_results.items(), key=lambda x: x[1].get('sharpe', 0))
        summary.append({
            'symbol': symbol,
            'baseline_trades': base['total_trades'],
            'baseline_sharpe': base['sharpe'],
            'baseline_pnl': base['total_pnl_pct'],
            'best_filter': best_config[0],
            'best_sharpe': best_config[1]['sharpe'],
            'best_pnl': best_config[1]['total_pnl_pct'],
            'sharpe_improvement': round(best_config[1]['sharpe'] - base['sharpe'], 3)
        })

    # Regime indicator statistics
    print(f"\n{'='*60}")
    print("REGIME INDICATOR SUMMARY")
    print(f"{'='*60}")
    regime_stats = {}
    for symbol in SYMBOLS:
        df = pd.read_parquet(f'{symbol}_5min.parquet').sort_index()
        uf = compute_upfraction(df).dropna()
        sma = compute_sma_slope(df).dropna()
        adx = compute_adx(df).dropna()
        vol, vol_ma = compute_vol_regime(df)
        vol_ratio = (vol / vol_ma).dropna()

        stats = {
            'upfraction': {
                'mean': round(float(uf.mean()), 3),
                'std': round(float(uf.std()), 3),
                'pct_bullish': round(float((uf > 0.55).mean() * 100), 1),
                'pct_bearish': round(float((uf < 0.45).mean() * 100), 1),
            },
            'sma_slope': {
                'mean': round(float(sma.mean()), 3),
                'pct_trending_up': round(float((sma > 0.5).mean() * 100), 1),
                'pct_trending_down': round(float((sma < -0.5).mean() * 100), 1),
            },
            'adx': {
                'mean': round(float(adx.mean()), 1),
                'pct_trending': round(float((adx > 20).mean() * 100), 1),
            },
            'vol_ratio': {
                'mean': round(float(vol_ratio.mean()), 3),
                'pct_high_vol': round(float((vol_ratio > 1).mean() * 100), 1),
            }
        }
        regime_stats[symbol] = stats
        print(f"\n  {symbol}:")
        print(f"    UpFraction: mean={stats['upfraction']['mean']}, "
              f"bull={stats['upfraction']['pct_bullish']}%, bear={stats['upfraction']['pct_bearish']}%")
        print(f"    SMA Slope:  mean={stats['sma_slope']['mean']}%, "
              f"up={stats['sma_slope']['pct_trending_up']}%, down={stats['sma_slope']['pct_trending_down']}%")
        print(f"    ADX:        mean={stats['adx']['mean']}, trending={stats['adx']['pct_trending']}%")
        print(f"    Vol Ratio:  mean={stats['vol_ratio']['mean']}, high={stats['vol_ratio']['pct_high_vol']}%")

    # Final summary
    print(f"\n{'='*60}")
    print("STRATEGY COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Symbol':<8} {'Baseline':>10} {'Best Filter':>18} {'Sharpe Δ':>10}")
    print("-" * 50)
    for s in summary:
        print(f"{s['symbol']:<8} {s['baseline_sharpe']:>10.3f} {s['best_filter']:>18} {s['sharpe_improvement']:>+10.3f}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'or_bars': OR_BARS,
            'atr_period': ATR_PERIOD,
            'trail_atr_mult': TRAIL_ATR_MULT,
            'total_cost_pct': TOTAL_COST * 100,
        },
        'regime_statistics': regime_stats,
        'strategy_results': results,
        'summary': summary
    }

    with open('regime_detection_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n✅ Results saved to regime_detection_results.json")
    return output


if __name__ == '__main__':
    main()
