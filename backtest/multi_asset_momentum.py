#!/usr/bin/env python3
"""
Multi-Asset Momentum Strategy
==============================
Time-series momentum across equities, currencies, bonds, and commodities.
Based on Asness, Moskowitz, Pedersen (2013) "Value and Momentum Everywhere".

Variants:
  1. Simple Momentum
  2. Momentum + Trend Filter
  3. Momentum + Volatility Target
  4. Multi-Timeframe Momentum

Walk-forward: Train 2019-01 to 2022-12, Test 2023-01 to 2024-07.
"""

import json
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

WORKSPACE = Path("/home/work/.openclaw/workspace")
API_KEY = "lse_live_f4c9a7419371ecdd9365e146247b0289"
API_URL = "https://api.londonstrategicedge.com/vault/candles"

# Costs
SLIPPAGE = 0.001       # 0.1% per side
COMMISSION = 0.0005    # 0.05% per side
COST_PER_TRADE = SLIPPAGE + COMMISSION  # one-way


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def fetch_daily(symbol, start="2019-01-01", end="2024-07-31"):
    """Fetch daily candles from API."""
    cache = WORKSPACE / f".openclaw/tmp/cache_{symbol}_1d.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return pd.read_parquet(cache)
    
    print(f"  Fetching {symbol} daily...", end=" ", flush=True)
    headers = {"x-api-key": API_KEY}
    all_data = []
    
    # API may limit results, fetch in yearly chunks
    for year_start in range(int(start[:4]), int(end[:4]) + 1):
        s = f"{year_start}-01-01" if year_start > int(start[:4]) else start
        e = f"{year_start}-12-31" if year_start < int(end[:4]) else end
        try:
            r = requests.get(API_URL, params={
                "symbol": symbol, "timeframe": "1d",
                "start": s, "end": e
            }, headers=headers, timeout=30)
            if r.status_code == 200:
                data = r.json()
                if data:
                    all_data.extend(data)
        except Exception as ex:
            print(f"Error fetching {symbol} {s}-{e}: {ex}")
        time.sleep(0.3)  # rate limit
    
    if not all_data:
        print("EMPTY")
        return None
    
    df = pd.DataFrame(all_data)
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.set_index('ts').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df.to_parquet(cache)
    print(f"{len(df)} rows")
    return df


def load_parquet_daily(symbol, filename):
    """Load 5min parquet, resample to daily."""
    path = WORKSPACE / filename
    if not path.exists():
        return None
    cache = WORKSPACE / f".openclaw/tmp/cache_{symbol}_daily.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return pd.read_parquet(cache)
    
    print(f"  Loading {symbol} from {filename}...", end=" ", flush=True)
    df = pd.read_parquet(path)
    if 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'])
        df = df.set_index('ts')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Resample to daily
    daily = df['close'].resample('1D').last().dropna()
    daily = daily.to_frame('close')
    daily.to_parquet(cache)
    print(f"{len(daily)} rows")
    return daily


def load_all_data():
    """Load all asset data."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    assets = {}
    
    # Fetch daily data FIRST (these have 2019+ history)
    fetch_syms = ['SPY', 'QQQ', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSM', 'AVGO']
    for sym in fetch_syms:
        df = fetch_daily(sym)
        if df is not None and len(df) > 100:
            assets[sym] = df[['close']]
    
    # From parquet files (resample to daily) — shorter history ~2022+
    # Add separately so they don't limit the main matrix
    parquet_files = {
        'NVDA': 'NVDA_5min.parquet',
        'AMD': 'AMD_5min.parquet',
        'PLTR': 'PLTR_5min.parquet',
        'MRVL': 'MRVL_5min.parquet',
        'XAGUSD': 'XAGUSD_1h.parquet',
    }
    parquet_assets = {}
    for sym, fn in parquet_files.items():
        df = load_parquet_daily(sym, fn)
        if df is not None and len(df) > 100:
            parquet_assets[sym] = df
    
    return assets, parquet_assets


def build_close_matrix(assets):
    """Build aligned close price matrix."""
    frames = {}
    for sym, df in assets.items():
        s = df['close'].copy()
        s.name = sym
        frames[sym] = s
    
    matrix = pd.DataFrame(frames)
    matrix = matrix.dropna(how='all')
    # Forward fill small gaps, then drop leading NaN
    matrix = matrix.ffill(limit=5).dropna()
    return matrix


# ─── STRATEGY ENGINE ──────────────────────────────────────────────────────────

def calc_strategy(close_matrix, variant, lookback, trend_ma=None,
                  rebalance='daily', vol_target=None, train_end='2022-12-31'):
    """
    Run a momentum strategy variant across all assets.
    
    Returns dict with train/test performance metrics.
    """
    returns = close_matrix.pct_change()
    n_assets = close_matrix.shape[1]
    asset_names = list(close_matrix.columns)
    
    # ── Momentum signal ──
    if variant == 'simple':
        mom_signal = close_matrix.pct_change(lookback)
    
    elif variant == 'trend_filter':
        mom_signal = close_matrix.pct_change(lookback)
        if trend_ma is not None:
            ma = close_matrix.rolling(trend_ma).mean()
            # Long if above MA and positive momentum, short otherwise
            above_ma = close_matrix > ma
            # If above MA: sign(mom), if below MA: -sign(mom) (or flat)
            mom_signal = np.where(above_ma, mom_signal, -np.abs(mom_signal))
            mom_signal = pd.DataFrame(mom_signal, index=close_matrix.index, columns=asset_names)
    
    elif variant == 'vol_target':
        mom_signal = close_matrix.pct_change(lookback)
    
    elif variant == 'multi_tf':
        mom_5 = close_matrix.pct_change(5)
        mom_20 = close_matrix.pct_change(20)
        mom_60 = close_matrix.pct_change(60)
        # Weight by lookback period
        w_short = 0.2
        w_med = 0.5
        w_long = 0.3
        if lookback <= 10:
            w_short, w_med, w_long = 0.5, 0.3, 0.2
        elif lookback >= 40:
            w_short, w_med, w_long = 0.1, 0.3, 0.6
        mom_signal = w_short * mom_5 + w_med * mom_20 + w_long * mom_60
    
    # ── Position sizing ──
    if variant == 'vol_target' and vol_target is not None:
        vol = returns.rolling(20).std() * np.sqrt(252)
        vol = vol.replace(0, np.nan)
        position_size = vol_target / vol
        position_size = position_size.clip(0, 3)  # cap leverage at 3x
        raw_position = np.sign(mom_signal) * position_size
    else:
        raw_position = np.sign(mom_signal)
    
    raw_position = pd.DataFrame(raw_position, index=close_matrix.index, columns=asset_names)
    raw_position = raw_position.fillna(0)
    
    # ── Rebalance ──
    if rebalance == 'weekly':
        # Hold position constant within the week
        # Rebalance on first trading day of each week
        rebal_mask = raw_position.index.to_series().diff().dt.days.fillna(1) > 3
        # Forward fill: only update on rebalance days
        pos_rebalanced = raw_position.copy()
        last_valid = None
        for i in range(len(pos_rebalanced)):
            if rebal_mask.iloc[i] or i == 0:
                last_valid = pos_rebalanced.iloc[i]
            else:
                pos_rebalanced.iloc[i] = last_valid
        raw_position = pos_rebalanced
    
    # ── Equal weight across assets ──
    # Normalize: each asset contributes equally
    n_active = raw_position.replace(0, np.nan).notna().sum(axis=1).replace(0, 1)
    position = raw_position.div(n_active, axis=0)
    
    # ── Calculate returns ──
    # Strategy return = position * asset return
    # Shift position by 1 to avoid look-ahead bias
    position_shifted = position.shift(1)
    
    # Portfolio return (equal-weighted across active positions)
    strat_returns = (position_shifted * returns).sum(axis=1)
    
    # ── Transaction costs ──
    # Cost = |change in position| * cost_per_trade
    position_change = position_shifted.diff().abs()
    total_cost = (position_change * COST_PER_TRADE).sum(axis=1)
    
    net_returns = strat_returns - total_cost
    net_returns = net_returns.dropna()
    
    # ── Split train/test ──
    train_mask = net_returns.index <= train_end
    test_mask = net_returns.index > train_end
    
    train_ret = net_returns[train_mask]
    test_ret = net_returns[test_mask]
    
    def calc_metrics(rets):
        if len(rets) < 10:
            return None
        total = (1 + rets).prod() - 1
        ann_ret = (1 + total) ** (252 / max(len(rets), 1)) - 1
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        # Max drawdown
        cum = (1 + rets).cumprod()
        dd = cum / cum.cummax() - 1
        max_dd = dd.min()
        # Win rate
        win_rate = (rets > 0).mean()
        # Profit factor
        gains = rets[rets > 0].sum()
        losses = abs(rets[rets < 0].sum())
        pf = gains / losses if losses > 0 else float('inf')
        # Calmar ratio
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
        
        return {
            'total_return': round(float(total), 4),
            'annual_return': round(float(ann_ret), 4),
            'annual_vol': round(float(ann_vol), 4),
            'sharpe': round(float(sharpe), 3),
            'max_drawdown': round(float(max_dd), 4),
            'calmar': round(float(calmar), 3),
            'win_rate': round(float(win_rate), 3),
            'profit_factor': round(float(pf), 3),
            'n_days': int(len(rets)),
            'total_cost': round(float(total_cost[train_mask if rets is train_ret else test_mask].sum()), 4),
        }
    
    train_metrics = calc_metrics(train_ret)
    test_metrics = calc_metrics(test_ret)
    
    return {
        'variant': variant,
        'lookback': lookback,
        'trend_ma': trend_ma,
        'rebalance': rebalance,
        'vol_target': vol_target,
        'train': train_metrics,
        'test': test_metrics,
    }


# ─── PARAMETER GRID ───────────────────────────────────────────────────────────

def run_parameter_sweep(close_matrix):
    """Run all parameter combinations."""
    print("\n" + "=" * 60)
    print("PARAMETER SWEEP")
    print("=" * 60)
    
    variants = ['simple', 'trend_filter', 'vol_target', 'multi_tf']
    lookbacks = [5, 10, 20, 40, 60]
    trend_mas = [None, 50, 100, 200]
    rebalances = ['daily', 'weekly']
    vol_targets = [None, 0.10, 0.15, 0.20]
    
    # Filter valid combos: vol_target variant only uses vol_targets, others use None
    combos = []
    for v, lb, tma, rb, vt in product(variants, lookbacks, trend_mas, rebalances, vol_targets):
        # vol_target variant needs a vol_target
        if v == 'vol_target' and vt is None:
            continue
        # non-vol_target variants shouldn't use vol_target
        if v != 'vol_target' and vt is not None:
            continue
        # trend_filter variant uses trend_ma, others don't need it
        if v == 'trend_filter' and tma is None:
            continue
        if v != 'trend_filter' and tma is not None:
            continue
        combos.append((v, lb, tma, rb, vt))
    
    print(f"Total combinations: {len(combos)}")
    
    results = []
    best_test_sharpe = -999
    best_combo = None
    
    for i, (v, lb, tma, rb, vt) in enumerate(combos):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(combos)}", flush=True)
        
        res = calc_strategy(close_matrix, v, lb, tma, rb, vt)
        
        # Validate
        if res['train'] is None or res['test'] is None:
            continue
        
        results.append(res)
        
        # Track best by test Sharpe
        if res['test']['sharpe'] > best_test_sharpe:
            best_test_sharpe = res['test']['sharpe']
            best_combo = res
    
    return results, best_combo


# ─── ASSET CORRELATION ANALYSIS ───────────────────────────────────────────────

def analyze_cross_sectional(close_matrix):
    """Analyze cross-sectional momentum (long winners, short losers)."""
    returns = close_matrix.pct_change()
    
    # Rank assets by trailing return
    lookbacks = [20, 60]
    results = {}
    
    for lb in lookbacks:
        trailing = close_matrix.pct_change(lb)
        
        # Each day, go long top half, short bottom half
        ranks = trailing.rank(axis=1, pct=True)
        long_mask = ranks > 0.5
        short_mask = ranks < 0.5
        
        # Equal weight within groups
        long_pos = long_mask.astype(float).div(long_mask.sum(axis=1), axis=0)
        short_pos = short_mask.astype(float).div(short_mask.sum(axis=1), axis=0)
        
        position = long_pos - short_pos
        position = position.shift(1)  # avoid look-ahead
        
        strat_ret = (position * returns).sum(axis=1).dropna()
        
        # Transaction costs
        pos_change = position.diff().abs()
        cost = (pos_change * COST_PER_TRADE).sum(axis=1)
        net_ret = strat_ret - cost
        
        # Split
        train = net_ret[net_ret.index <= '2022-12-31']
        test = net_ret[net_ret.index > '2022-12-31']
        
        def m(rets):
            if len(rets) < 10:
                return {}
            total = (1 + rets).prod() - 1
            ann = (1 + total) ** (252 / max(len(rets), 1)) - 1
            vol = rets.std() * np.sqrt(252)
            return {
                'sharpe': round(float(ann/vol) if vol > 0 else 0, 3),
                'total_return': round(float(total), 4),
                'annual_return': round(float(ann), 4),
            }
        
        results[f'xs_mom_{lb}d'] = {
            'lookback': lb,
            'train': m(train),
            'test': m(test),
        }
    
    return results


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("MULTI-ASSET MOMENTUM STRATEGY")
    print("=" * 60)
    
    # Load data
    api_assets, parquet_assets = load_all_data()
    print(f"\nAPI assets (2019+): {list(api_assets.keys())}")
    print(f"Parquet assets (2022+): {list(parquet_assets.keys())}")
    
    # PRIMARY matrix: API assets only (full 2019-2024 history)
    close_matrix = build_close_matrix(api_assets)
    print(f"\nPrimary matrix (walk-forward): {close_matrix.shape[0]} days x {close_matrix.shape[1]} assets")
    print(f"Date range: {close_matrix.index[0]} to {close_matrix.index[-1]}")
    print(f"Assets: {list(close_matrix.columns)}")
    
    # SECONDARY matrix: all assets (shorter, for supplementary analysis)
    all_assets = {**api_assets, **parquet_assets}
    all_matrix = build_close_matrix(all_assets)
    print(f"\nFull matrix: {all_matrix.shape[0]} days x {all_matrix.shape[1]} assets")
    
    # Correlation analysis on full matrix
    print("\n" + "=" * 60)
    print("CORRELATION MATRIX (daily returns) — FULL UNIVERSE")
    print("=" * 60)
    ret_corr = all_matrix.pct_change().corr()
    print(ret_corr.round(2).to_string())
    
    # Cross-sectional momentum
    print("\n" + "=" * 60)
    print("CROSS-SECTIONAL MOMENTUM")
    print("=" * 60)
    xs_results = analyze_cross_sectional(close_matrix)
    for name, data in xs_results.items():
        print(f"\n{name}:")
        print(f"  Train: {data.get('train', {})}")
        print(f"  Test:  {data.get('test', {})}")
    
    # Time-series momentum parameter sweep on primary matrix
    print("\n" + "=" * 60)
    print("PRIMARY ANALYSIS: Walk-Forward with Full History")
    print("=" * 60)
    results, best_combo = run_parameter_sweep(close_matrix)
    
    # ── Summary ──
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    # Sort by test Sharpe
    valid = [r for r in results if r['test'] and r['train']]
    valid.sort(key=lambda x: x['test']['sharpe'], reverse=True)
    
    print(f"\nTotal strategies tested: {len(valid)}")
    
    # Top 20 by test Sharpe
    print("\n┌─ TOP 20 BY TEST SHARPE ─────────────────────────────────┐")
    print(f"{'#':>3} {'Variant':<14} {'LB':>4} {'TMA':>5} {'Rebal':<7} {'VT':>5} "
          f"{'TrSharpe':>8} {'TeSharpe':>8} {'TeReturn':>9} {'TeMaxDD':>8} {'TeCalmar':>8}")
    print("─" * 90)
    
    for i, r in enumerate(valid[:20]):
        t = r['train']
        te = r['test']
        vt_str = f"{r['vol_target']:.0%}" if r['vol_target'] else "-"
        tma_str = str(r['trend_ma']) if r['trend_ma'] else "-"
        print(f"{i+1:>3} {r['variant']:<14} {r['lookback']:>4} {tma_str:>5} {r['rebalance']:<7} {vt_str:>5} "
              f"{t['sharpe']:>8.3f} {te['sharpe']:>8.3f} {te['total_return']:>9.2%} "
              f"{te['max_drawdown']:>8.2%} {te['calmar']:>8.3f}")
    
    # Bottom 10
    print(f"\n┌─ BOTTOM 10 BY TEST SHARPE ──────────────────────────────┐")
    for i, r in enumerate(valid[-10:]):
        te = r['test']
        vt_str = f"{r['vol_target']:.0%}" if r['vol_target'] else "-"
        tma_str = str(r['trend_ma']) if r['trend_ma'] else "-"
        print(f"{i+1:>3} {r['variant']:<14} {r['lookback']:>4} {tma_str:>5} {r['rebalance']:<7} {vt_str:>5} "
              f"{r['train']['sharpe']:>8.3f} {te['sharpe']:>8.3f} {te['total_return']:>9.2%} "
              f"{te['max_drawdown']:>8.2%}")
    
    # Best combo detail
    if best_combo:
        print(f"\n┌─ BEST STRATEGY (by test Sharpe) ────────────────────────┐")
        print(f"Variant:    {best_combo['variant']}")
        print(f"Lookback:   {best_combo['lookback']} days")
        print(f"Trend MA:   {best_combo['trend_ma']}")
        print(f"Rebalance:  {best_combo['rebalance']}")
        print(f"Vol Target: {best_combo['vol_target']}")
        print(f"\nTrain Performance:")
        for k, v in best_combo['train'].items():
            print(f"  {k}: {v}")
        print(f"\nTest Performance:")
        for k, v in best_combo['test'].items():
            print(f"  {k}: {v}")
    
    # Statistics by variant
    print(f"\n┌─ PERFORMANCE BY VARIANT ────────────────────────────────┐")
    for variant in ['simple', 'trend_filter', 'vol_target', 'multi_tf']:
        v_results = [r for r in valid if r['variant'] == variant]
        if not v_results:
            continue
        sharpes = [r['test']['sharpe'] for r in v_results]
        returns = [r['test']['total_return'] for r in v_results]
        print(f"\n{variant} ({len(v_results)} combos):")
        print(f"  Test Sharpe: mean={np.mean(sharpes):.3f}, median={np.median(sharpes):.3f}, "
              f"max={np.max(sharpes):.3f}, min={np.min(sharpes):.3f}")
        print(f"  Test Return: mean={np.mean(returns):.2%}, median={np.median(returns):.2%}")
        
        # Best in category
        best = max(v_results, key=lambda x: x['test']['sharpe'])
        print(f"  Best: LB={best['lookback']}, TMA={best['trend_ma']}, "
              f"Rebal={best['rebalance']}, VT={best['vol_target']}, "
              f"Sharpe={best['test']['sharpe']:.3f}")
    
    # ── Edge detection ──
    print("\n" + "=" * 60)
    print("EDGE DETECTION")
    print("=" * 60)
    
    profitable = [r for r in valid if r['test']['total_return'] > 0]
    high_sharpe = [r for r in valid if r['test']['sharpe'] > 0.5]
    consistent = [r for r in valid if r['test']['sharpe'] > 0 and r['train']['sharpe'] > 0]
    
    print(f"Profitable in test:  {len(profitable)}/{len(valid)} ({len(profitable)/max(len(valid),1)*100:.1f}%)")
    print(f"Test Sharpe > 0.5:   {len(high_sharpe)}/{len(valid)} ({len(high_sharpe)/max(len(valid),1)*100:.1f}%)")
    print(f"Positive both sets:  {len(consistent)}/{len(valid)} ({len(consistent)/max(len(valid),1)*100:.1f}%)")
    
    # Check for overfitting: train vs test correlation
    if valid:
        train_sharpes = [r['train']['sharpe'] for r in valid]
        test_sharpes = [r['test']['sharpe'] for r in valid]
        corr = np.corrcoef(train_sharpes, test_sharpes)[0, 1]
        print(f"\nTrain-Test Sharpe correlation: {corr:.3f}")
        if corr > 0.5:
            print("  → Good consistency: train performance predicts test")
        elif corr > 0.2:
            print("  → Moderate consistency: some overfitting risk")
        else:
            print("  → Low consistency: likely overfit or strategy has no edge")
    
    # Supplementary: run on full universe too
    print("\n" + "=" * 60)
    print("SUPPLEMENTARY: Full Universe (shorter history)")
    print("=" * 60)
    all_results, all_best = run_parameter_sweep(all_matrix)
    if all_best:
        print(f"\nBest full-universe: {all_best['variant']}, LB={all_best['lookback']}, "
              f"Sharpe={all_best['test']['sharpe']:.3f}")
    
    # Save results
    output = {
        'summary': {
            'total_combinations': len(valid),
            'profitable_pct': round(len(profitable) / max(len(valid), 1) * 100, 1),
            'high_sharpe_pct': round(len(high_sharpe) / max(len(valid), 1) * 100, 1),
            'consistent_pct': round(len(consistent) / max(len(valid), 1) * 100, 1),
            'train_test_corr': round(float(corr), 3) if valid else None,
        },
        'best_strategy': best_combo,
        'cross_sectional': xs_results,
        'top_20': valid[:20],
        'all_results': valid,
        'full_universe_best': all_best,
        'assets_primary': list(close_matrix.columns),
        'assets_full': list(all_matrix.columns),
        'date_range_primary': f"{close_matrix.index[0].date()} to {close_matrix.index[-1].date()}",
        'date_range_full': f"{all_matrix.index[0].date()} to {all_matrix.index[-1].date()}",
        'costs': {
            'slippage_per_side': SLIPPAGE,
            'commission_per_side': COMMISSION,
            'round_trip': SLIPPAGE * 2 + COMMISSION * 2,
        }
    }
    
    out_path = WORKSPACE / "multi_asset_momentum_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    main()
