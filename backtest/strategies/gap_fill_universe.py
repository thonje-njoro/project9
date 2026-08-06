#!/usr/bin/env python3
"""
Gap Fill Strategy Universe Test - OPTIMIZED
Tests across 5 symbols: PLTR, AMD, NVDA, TSLA, QQQ
Walk-forward: train before 2023-07-01, test after
"""

import pandas as pd
import numpy as np
import json
import time
import os
import sys
from itertools import product
from datetime import datetime, timedelta

# ============================================================
# DATA LOADING
# ============================================================

def load_or_resample(symbol):
    """Load 5min parquet, or resample from 1min."""
    path5 = f"{symbol}_5min.parquet"
    if os.path.exists(path5):
        df = pd.read_parquet(path5)
        if len(df) > 1000:
            df.index.name = 'ts'
            print(f"  Loaded {symbol}_5min.parquet: {len(df)} rows")
            return df
    
    path1 = f"{symbol}_1min.parquet"
    if os.path.exists(path1):
        df = pd.read_parquet(path1)
        df.index.name = 'ts'
        agg = {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
        if 'symbol' in df.columns:
            agg['symbol'] = 'first'
        resampled = df.resample('5min').agg(agg).dropna(subset=['open'])
        print(f"  Resampled {symbol} from 1min: {len(resampled)} rows")
        return resampled
    
    print(f"  [!] No data for {symbol}")
    return None

def fetch_5min_from_api(symbol):
    """Fetch 5min data from London Strategic Edge API."""
    try:
        import requests
    except ImportError:
        return None
    
    api_key = None
    with open('mimo_claw_pipeline_expanded.py', 'r') as f:
        for line in f:
            if 'LSE_API_KEY' in line and 'lse_live' in line:
                api_key = line.split('"')[1]
                break
    if not api_key:
        return None
    
    url = "https://api.londonstrategicedge.com/vault/candles"
    headers = {"x-api-key": api_key}
    chunks = [
        ("2022-01-01","2022-03-31"),("2022-04-01","2022-06-30"),
        ("2022-07-01","2022-09-30"),("2022-10-01","2022-12-31"),
        ("2023-01-01","2023-03-31"),("2023-04-01","2023-06-30"),
        ("2023-07-01","2023-09-30"),("2023-10-01","2023-12-31"),
        ("2024-01-01","2024-03-31"),("2024-04-01","2024-07-31"),
    ]
    all_dfs = []
    for i, (cs, ce) in enumerate(chunks):
        print(f"  Fetching {symbol} {i+1}/{len(chunks)}: {cs}..{ce}")
        try:
            resp = requests.get(url, params={"symbol":symbol,"timeframe":"5m","start":cs,"end":ce}, headers=headers, timeout=60)
            if resp.status_code != 200:
                print(f"  HTTP {resp.status_code}")
                continue
            data = resp.json()
            records = data if isinstance(data, list) else data.get('data', data.get('candles', data.get('results', [])))
            if records:
                chunk_df = pd.DataFrame(records)
                all_dfs.append(chunk_df)
                print(f"  Got {len(chunk_df)} rows")
        except Exception as e:
            print(f"  Error: {e}")
        if i < len(chunks)-1:
            time.sleep(7)
    
    if not all_dfs:
        return None
    
    df = pd.concat(all_dfs, ignore_index=True)
    # Find timestamp column
    for col in ['ts','timestamp','time','date','datetime','t']:
        if col in df.columns:
            df['ts'] = pd.to_datetime(df[col])
            break
    df = df.set_index('ts').sort_index()
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ['o','open']: col_map[c]='open'
        elif cl in ['h','high']: col_map[c]='high'
        elif cl in ['l','low']: col_map[c]='low'
        elif cl in ['c','close']: col_map[c]='close'
        elif cl in ['v','vol','volume']: col_map[c]='volume'
    df = df.rename(columns=col_map)
    df['symbol'] = symbol
    df.to_parquet(f"{symbol}_5min.parquet")
    print(f"  Cached {len(df)} rows")
    return df

# ============================================================
# PRECOMPUTE DAILY STRUCTURE (once per symbol)
# ============================================================

def precompute_daily(df, split_date='2023-07-01'):
    """
    Precompute daily gap info and bar arrays for fast strategy evaluation.
    Returns dict with precomputed arrays that can be reused across configs.
    """
    df = df.sort_index().copy()
    df['date'] = df.index.date
    df['time'] = df.index.time
    
    # Detect timezone: if earliest bar time is >= 13:00, likely UTC
    sample_times = sorted(set(df['time']))[:5]
    earliest = min(sample_times)
    if earliest.hour >= 13:
        # UTC timestamps: market hours 14:30-21:00 UTC = 9:30-16:00 ET
        market_open = pd.Timestamp('14:30').time()
        market_close = pd.Timestamp('21:00').time()
    else:
        # ET timestamps
        market_open = pd.Timestamp('09:30').time()
        market_close = pd.Timestamp('16:00').time()
    
    mask = (df['time'] >= market_open) & (df['time'] < market_close)
    dfm = df[mask].copy()
    
    if len(dfm) < 200:
        return None
    
    # Compute ATR(14) on daily timeframe using 5min bars
    # ~78 bars per day, so 14 days = 1092 bars
    high = dfm['high'].values
    low = dfm['low'].values
    close = dfm['close'].values
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr_period = 14 * 78  # 14 trading days
    atr = pd.Series(tr, index=dfm.index).rolling(atr_period, min_periods=78).mean().values
    dfm['atr'] = atr
    
    # Daily aggregates
    daily = dfm.groupby('date').agg(
        day_open=('open','first'),
        day_close=('close','last'),
        first_bar_open=('open','first'),
        second_bar_open=('open', lambda x: x.iloc[1] if len(x)>1 else np.nan),
        second_bar_idx=('open', lambda x: x.index[1] if len(x)>1 else None),
    )
    
    # Previous day close
    daily['prev_close'] = daily['day_close'].shift(1)
    daily['gap'] = daily['day_open'] - daily['prev_close']
    daily['gap_pct'] = daily['gap'] / daily['prev_close'] * 100
    daily = daily.dropna(subset=['prev_close'])
    
    trading_days = sorted(daily.index)
    
    # Build per-day bar data as arrays for fast access
    day_bars = {}
    for d in trading_days:
        bars = dfm[dfm['date'] == d]
        if len(bars) >= 2:
            day_bars[d] = {
                'open': bars['open'].values,
                'high': bars['high'].values,
                'low': bars['low'].values,
                'close': bars['close'].values,
                'atr': bars['atr'].values,
                'times': bars['time'].values,
                'index': bars.index,
            }
    
    # Split train/test
    split = pd.Timestamp(split_date).date()
    train_days = [d for d in trading_days if d < split]
    test_days = [d for d in trading_days if d >= split]
    
    return {
        'daily': daily,
        'day_bars': day_bars,
        'train_days': train_days,
        'test_days': test_days,
        'trading_days': trading_days,
        'dfm': dfm,  # for multi-day lookups
        'market_open': market_open,
    }

# ============================================================
# FAST GAP FILL SIMULATION
# ============================================================

def simulate_gap_fill_fast(precomp, days, min_gap_pct, stop_method, stop_mult, 
                           max_hold_days, session_filter):
    """
    Fast gap fill simulation using precomputed data.
    Returns list of trade dicts.
    """
    daily = precomp['daily']
    day_bars = precomp['day_bars']
    trading_days = precomp['trading_days']
    dfm = precomp['dfm']
    td_arr = np.array(trading_days)
    
    trades = []
    cost_pct = 0.30  # total round-trip cost
    
    # Two-hour cutoff for session filter
    mkt_open = precomp.get('market_open', pd.Timestamp('09:30').time())
    # Compute 2h after market open
    mkt_open_dt = pd.Timestamp.combine(pd.Timestamp.today().date(), mkt_open)
    two_hour_cutoff = (mkt_open_dt + pd.Timedelta(hours=2)).time()
    
    for day in days:
        if day not in day_bars:
            continue
        
        row = daily.loc[day]
        gap_pct = row['gap_pct']
        
        if abs(gap_pct) < min_gap_pct:
            continue
        
        bars = day_bars[day]
        if len(bars['open']) < 2:
            continue
        
        # Entry at second bar's open (first bar after market open)
        entry_price = bars['open'][1]
        entry_time = bars['index'][1]
        
        if gap_pct < -min_gap_pct:
            direction = 1  # Long (gap down, expect fill up)
            target_price = row['prev_close']
        else:
            direction = -1  # Short (gap up, expect fill down)
            target_price = row['prev_close']
        
        # Compute stop
        atr_val = bars['atr'][1]
        if stop_method == "atr" and not np.isnan(atr_val) and atr_val > 0:
            if direction == 1:
                stop_price = entry_price - stop_mult * atr_val
            else:
                stop_price = entry_price + stop_mult * atr_val
        else:
            if direction == 1:
                stop_price = entry_price * (1 - stop_mult / 100)
            else:
                stop_price = entry_price * (1 + stop_mult / 100)
        
        # Build array of future bars to check (from bar index 2 onwards on entry day, then subsequent days)
        exit_price = None
        exit_reason = None
        exit_time = None
        
        day_idx = np.searchsorted(td_arr, day)
        
        # Collect bars to scan
        bars_to_scan = []
        
        # Remaining bars on entry day (from index 2)
        if session_filter == "first_2h":
            for bi in range(2, len(bars['open'])):
                if bars['times'][bi] <= two_hour_cutoff:
                    bars_to_scan.append(bi)
                else:
                    break
        else:
            for bi in range(2, len(bars['open'])):
                bars_to_scan.append(bi)
        
        # If max_hold_days > 1, add bars from subsequent days
        if max_hold_days > 1 and session_filter != "first_2h":
            for offset in range(1, max_hold_days):
                next_idx = day_idx + offset
                if next_idx >= len(trading_days):
                    break
                next_day = trading_days[next_idx]
                if next_day in day_bars:
                    nb = day_bars[next_day]
                    # Check up to end of day (or all bars)
                    for bi in range(len(nb['open'])):
                        bars_to_scan.append(('next', next_day, bi))
        
        # Simulate
        for bi in bars_to_scan:
            if isinstance(bi, tuple):
                _, bday, bidx = bi
                bar_h = day_bars[bday]['high'][bidx]
                bar_l = day_bars[bday]['low'][bidx]
                bar_c = day_bars[bday]['close'][bidx]
                bar_time = day_bars[bday]['index'][bidx]
            else:
                bar_h = bars['high'][bi]
                bar_l = bars['low'][bi]
                bar_c = bars['close'][bi]
                bar_time = bars['index'][bi]
            
            if direction == 1:  # Long
                if bar_l <= stop_price:
                    exit_price = stop_price
                    exit_reason = 'stop'
                    exit_time = bar_time
                    break
                if bar_h >= target_price:
                    exit_price = target_price
                    exit_reason = 'target'
                    exit_time = bar_time
                    break
            else:  # Short
                if bar_h >= stop_price:
                    exit_price = stop_price
                    exit_reason = 'stop'
                    exit_time = bar_time
                    break
                if bar_l <= target_price:
                    exit_price = target_price
                    exit_reason = 'target'
                    exit_time = bar_time
                    break
        
        # If no exit found, use last scanned bar's close or entry day last bar
        if exit_price is None:
            if bars_to_scan:
                last_bi = bars_to_scan[-1]
                if isinstance(last_bi, tuple):
                    _, bday, bidx = last_bi
                    exit_price = day_bars[bday]['close'][bidx]
                    exit_time = day_bars[bday]['index'][bidx]
                else:
                    exit_price = bars['close'][last_bi]
                    exit_time = bars['index'][last_bi]
                exit_reason = 'timeout'
            else:
                continue
        
        # P&L
        if direction == 1:
            raw_ret = (exit_price - entry_price) / entry_price * 100
        else:
            raw_ret = (entry_price - exit_price) / entry_price * 100
        
        net_ret = raw_ret - cost_pct
        
        trades.append({
            'date': str(day),
            'direction': 'long' if direction==1 else 'short',
            'gap_pct': round(gap_pct, 3),
            'entry': round(entry_price, 4),
            'exit': round(exit_price, 4),
            'exit_reason': exit_reason,
            'raw_ret': round(raw_ret, 4),
            'net_ret': round(net_ret, 4),
        })
    
    return trades

def compute_metrics(trades):
    """Fast metrics computation."""
    if not trades:
        return {'trades':0,'win_rate':0,'avg_return':0,'total_return':0,
                'sharpe':0,'profit_factor':0,'max_drawdown':0,'gap_fill_rate':0}
    
    rets = np.array([t['net_ret'] for t in trades])
    n = len(rets)
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    
    wr = len(wins)/n*100
    avg_ret = float(np.mean(rets))
    total_ret = float(np.sum(rets))
    
    std = np.std(rets)
    sharpe = float(np.mean(rets)/std*np.sqrt(252)) if std > 0 else 0
    
    gp = float(np.sum(wins)) if len(wins) > 0 else 0
    gl = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0001
    pf = gp / gl if gl > 0 else 999
    
    cum = np.cumsum(rets)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0
    
    fills = sum(1 for t in trades if t['exit_reason']=='target')
    fill_rate = fills/n*100
    
    return {
        'trades': n,
        'win_rate': round(wr, 1),
        'avg_return': round(avg_ret, 4),
        'total_return': round(total_ret, 2),
        'sharpe': round(sharpe, 3),
        'profit_factor': round(pf, 3),
        'max_drawdown': round(max_dd, 2),
        'gap_fill_rate': round(fill_rate, 1),
    }

# ============================================================
# PARAMETER GRID SEARCH
# ============================================================

def run_search(precomp, symbol):
    """Run full grid search with walk-forward validation."""
    train_days = precomp['train_days']
    test_days = precomp['test_days']
    print(f"  {symbol}: {len(train_days)} train days, {len(test_days)} test days")
    
    gap_vals = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    stop_methods = ["atr", "fixed"]
    atr_mults = [1.0, 1.5, 2.0, 2.5, 3.0]
    fixed_mults = [0.5, 1.0, 1.5, 2.0, 3.0]
    hold_days_list = [1, 2, 3, 5]
    sessions = ["all", "first_2h"]
    
    results = []
    total = 0
    
    for gap, sm, hd, sf in product(gap_vals, stop_methods, hold_days_list, sessions):
        mults = atr_mults if sm == "atr" else fixed_mults
        for mult in mults:
            total += 1
            
            train_trades = simulate_gap_fill_fast(precomp, train_days, gap, sm, mult, hd, sf)
            train_m = compute_metrics(train_trades)
            
            test_trades = simulate_gap_fill_fast(precomp, test_days, gap, sm, mult, hd, sf)
            test_m = compute_metrics(test_trades)
            
            if test_m['trades'] < 50:
                continue
            
            wf = test_m['sharpe']*0.5 + (test_m['win_rate']/100)*0.3 + (test_m['profit_factor']/3)*0.2
            
            results.append({
                'config': {'min_gap_pct':gap,'stop_method':sm,'stop_mult':mult,
                           'max_hold_days':hd,'session_filter':sf},
                'train': train_m,
                'test': test_m,
                'wf_score': round(wf, 4),
            })
    
    results.sort(key=lambda x: x['wf_score'], reverse=True)
    print(f"  {symbol}: {total} configs tested, {len(results)} with >=50 test trades")
    return results

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("GAP FILL STRATEGY - UNIVERSE TEST (5 SYMBOLS)")
    print("=" * 70)
    
    symbols = ['PLTR', 'AMD', 'NVDA', 'TSLA', 'QQQ']
    data = {}
    
    for sym in symbols:
        print(f"\n=== Loading {sym} ===")
        df = load_or_resample(sym)
        if df is None or len(df) < 200:
            # Try API fetch
            print(f"  Trying API fetch for {sym}...")
            df = fetch_5min_from_api(sym)
        if df is not None and len(df) > 200:
            data[sym] = df
        else:
            print(f"  [!] FAILED: no data for {sym}")
    
    print(f"\n{'='*70}")
    print(f"Data loaded: {list(data.keys())}")
    print(f"{'='*70}")
    
    # Precompute daily structure for each symbol
    precomps = {}
    for sym, df in data.items():
        print(f"\nPrecomputing {sym}...")
        pc = precompute_daily(df)
        if pc:
            precomps[sym] = pc
            print(f"  Train days: {len(pc['train_days'])}, Test days: {len(pc['test_days'])}")
    
    # Run parameter search
    all_results = {}
    summaries = []
    
    for sym in precomps:
        print(f"\n{'='*70}")
        print(f"PARAMETER SEARCH: {sym}")
        print(f"{'='*70}")
        
        results = run_search(precomps[sym], sym)
        all_results[sym] = results
        
        if results:
            best = results[0]
            c = best['config']
            t = best['test']
            tr = best['train']
            passes = (t['sharpe'] > 0.5 and t['win_rate'] > 55 and t['profit_factor'] > 1.2)
            
            print(f"\n  BEST for {sym}: gap>={c['min_gap_pct']}%, stop={c['stop_method']}({c['stop_mult']}), "
                  f"hold={c['max_hold_days']}d, session={c['session_filter']}")
            print(f"  Train: WR={tr['win_rate']}%, Sharpe={tr['sharpe']}, PF={tr['profit_factor']}, N={tr['trades']}")
            print(f"  Test:  WR={t['win_rate']}%, Sharpe={t['sharpe']}, PF={t['profit_factor']}, N={t['trades']}")
            print(f"  Fill Rate: {t['gap_fill_rate']}%, WF={best['wf_score']}, PF Pass={'✓' if passes else '✗'}")
            
            summaries.append({
                'symbol': sym, 'best_config': c,
                'train': tr, 'test': t,
                'wf_score': best['wf_score'], 'passes_prop_firm': passes,
            })
        else:
            print(f"\n  {sym}: NO viable configs (all <50 test trades)")
            summaries.append({
                'symbol': sym, 'best_config': None, 'train': None, 'test': None,
                'wf_score': 0, 'passes_prop_firm': False,
            })
    
    # Combined portfolio
    print(f"\n{'='*70}")
    print("COMBINED PORTFOLIO (Best Config per Symbol, Test Period)")
    print(f"{'='*70}")
    
    combined_trades = []
    for sym, results in all_results.items():
        if results:
            best_cfg = results[0]['config']
            test_days = precomps[sym]['test_days']
            trades = simulate_gap_fill_fast(precomps[sym], test_days, **best_cfg)
            for t in trades:
                t['symbol'] = sym
            combined_trades.extend(trades)
    
    cm = compute_metrics(combined_trades)
    
    # Summary table
    print(f"\n{'Symbol':<8} {'Trades':>7} {'WR%':>6} {'Sharpe':>7} {'PF':>6} {'MaxDD%':>7} {'FillR%':>7} {'PF✓':>5} {'WF':>8}")
    print("-" * 68)
    
    for s in summaries:
        if s['test']:
            t = s['test']
            mk = "✓" if s['passes_prop_firm'] else "✗"
            print(f"{s['symbol']:<8} {t['trades']:>7} {t['win_rate']:>6.1f} {t['sharpe']:>7.3f} "
                  f"{t['profit_factor']:>6.2f} {t['max_drawdown']:>7.2f} {t['gap_fill_rate']:>7.1f} "
                  f"{mk:>5} {s['wf_score']:>8.4f}")
        else:
            print(f"{s['symbol']:<8} {'N/A':>7}")
    
    print("-" * 68)
    print(f"{'TOTAL':<8} {cm['trades']:>7} {cm['win_rate']:>6.1f} {cm['sharpe']:>7.3f} "
          f"{cm['profit_factor']:>6.2f} {cm['max_drawdown']:>7.2f} {cm['gap_fill_rate']:>7.1f}")
    
    print(f"\n  Combined: {cm['trades']} trades, WR={cm['win_rate']}%, Sharpe={cm['sharpe']}, "
          f"PF={cm['profit_factor']}, MaxDD={cm['max_drawdown']}%, TotalRet={cm['total_return']}%")
    
    # Top 3 per symbol
    print(f"\n{'='*70}")
    print("TOP 3 CONFIGS PER SYMBOL")
    print(f"{'='*70}")
    
    for sym, results in all_results.items():
        print(f"\n--- {sym} ---")
        if not results:
            print("  No viable configs")
            continue
        for rank, r in enumerate(results[:3], 1):
            c = r['config']
            t = r['test']
            print(f"  #{rank}: gap>={c['min_gap_pct']}%, stop={c['stop_method']}({c['stop_mult']}), "
                  f"hold={c['max_hold_days']}d, session={c['session_filter']}")
            print(f"       WR={t['win_rate']}%, Sharpe={t['sharpe']}, PF={t['profit_factor']}, "
                  f"N={t['trades']}, FillR={t['gap_fill_rate']}%, WF={r['wf_score']}")
    
    # Save results
    output = {
        'run_timestamp': datetime.now().isoformat(),
        'split_date': '2023-07-01',
        'costs': {'entry_slip':0.15,'exit_slip':0.10,'commission':0.05,'total':0.30},
        'min_test_trades': 50,
        'symbol_summaries': summaries,
        'combined': cm,
        'per_symbol': {},
    }
    for sym, results in all_results.items():
        output['per_symbol'][sym] = {
            'configs_with_50plus': len(results),
            'top_10': results[:10],
        }
    
    with open('gap_fill_universe_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to gap_fill_universe_results.json")
    print("DONE")

if __name__ == '__main__':
    main()
