#!/usr/bin/env python3
"""
PLTR Gap Fill Strategy — Backtester with Walk-Forward Validation

Strategy: Fade overnight gaps that exceed a threshold.
- Gap Down (> threshold): BUY at open, target = previous close (gap fill)
- Gap Up (> threshold): SELL at open, target = previous close (gap fill)
Realistic costs: entry slippage 0.3%, exit slippage 0.2%, commission 0.1% RT
"""

import json
import numpy as np
import pandas as pd
from itertools import product

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_FILE = "PLTR_5min.parquet"
RESULTS_FILE = "pltr_gap_fill_results.json"
TRAIN_CUTOFF = "2023-07-01"

ENTRY_SLIPPAGE = 0.003
EXIT_SLIPPAGE  = 0.002
COMMISSION     = 0.001

GAP_THRESHOLDS  = [0.3, 0.5, 0.8, 1.0, 1.5]
STOP_ATR_MULTS  = [1.0, 1.5, 2.0]
STOP_FIXED_PCTS = [1.0, 2.0, 3.0]
MAX_HOLD_DAYS   = [1, 2, 3, 5]
SESSION_FILTERS = ["all_day", "first_2h"]

ATR_PERIOD = 14
FIRST_2H_BARS = 24  # 24 * 5min = 2h


# ─── Data Loading & Preparation ─────────────────────────────────────────────

def load_and_prepare():
    df = pd.read_parquet(DATA_FILE)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df['date'] = df.index.date

    # Pre-group by date for fast lookup
    date_groups = {}
    for d, grp in df.groupby('date'):
        date_groups[d] = grp

    # Daily OHLC
    daily = df.groupby('date').agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
    )
    daily.index = pd.to_datetime(daily.index)

    # ATR
    daily['prev_close'] = daily['close'].shift(1)
    daily['tr'] = np.maximum(
        daily['high'] - daily['low'],
        np.maximum(
            abs(daily['high'] - daily['prev_close']),
            abs(daily['low'] - daily['prev_close'])
        )
    )
    daily['atr'] = daily['tr'].rolling(ATR_PERIOD, min_periods=1).mean()

    # Gaps
    daily = daily.dropna(subset=['prev_close'])
    daily['gap'] = daily['open'] - daily['prev_close']
    daily['gap_pct'] = daily['gap'] / daily['prev_close'] * 100

    # Sorted date list for multi-day holds
    sorted_dates = sorted(date_groups.keys())
    date_to_idx = {d: i for i, d in enumerate(sorted_dates)}

    return df, daily, date_groups, sorted_dates, date_to_idx


# ─── Core Backtester (optimized) ────────────────────────────────────────────

def _simulate_trade(date_groups, sorted_dates, date_to_idx,
                    start_date_idx, direction, entry_price, target, stop,
                    max_hold_days, session_filter):
    """Simulate one trade across up to max_hold_days."""
    cost = ENTRY_SLIPPAGE + EXIT_SLIPPAGE + COMMISSION
    exited = False

    for d_off in range(max_hold_days):
        idx = start_date_idx + d_off
        if idx >= len(sorted_dates):
            break

        d = sorted_dates[idx]
        bars = date_groups[d]
        if d_off > 0:
            bars = bars  # full session for continuation days

        # Choose slice
        if d_off == 0 and session_filter == "first_2h":
            bars = bars.iloc[:FIRST_2H_BARS]
        elif d_off > 0:
            pass  # full day

        if len(bars) < 2:
            # Update entry for next day
            if d_off > 0 and len(bars) > 0:
                entry_price = bars.iloc[-1]['close']
            continue

        # For continuation days, entry = previous day's close
        if d_off > 0:
            prev_d = sorted_dates[idx - 1]
            prev_bars = date_groups[prev_d]
            if len(prev_bars) > 0:
                entry_price = prev_bars.iloc[-1]['close']

        # Iterate bars (skip first = entry bar)
        arr_low = bars['low'].values[1:]
        arr_high = bars['high'].values[1:]
        arr_close = bars['close'].values

        for i in range(len(arr_low)):
            if direction == 'long':
                if arr_low[i] <= stop:
                    pnl = (stop - entry_price) / entry_price - cost
                    return pnl, 'stop', d_off + 1
                if arr_high[i] >= target:
                    pnl = (target - entry_price) / entry_price - cost
                    return pnl, 'target', d_off + 1
            else:
                if arr_high[i] >= stop:
                    pnl = (entry_price - stop) / entry_price - cost
                    return pnl, 'stop', d_off + 1
                if arr_low[i] <= target:
                    pnl = (entry_price - target) / entry_price - cost
                    return pnl, 'target', d_off + 1

    # Max hold exit
    last_idx = min(start_date_idx + max_hold_days - 1, len(sorted_dates) - 1)
    last_d = sorted_dates[last_idx]
    last_bars = date_groups[last_d]
    if len(last_bars) > 0:
        last_close = last_bars.iloc[-1]['close']
        if direction == 'long':
            pnl = (last_close - entry_price) / entry_price - cost
        else:
            pnl = (entry_price - last_close) / entry_price - cost
        return pnl, 'max_hold', max_hold_days

    return 0.0, 'no_trade', 0


def run_strategy(date_groups, sorted_dates, date_to_idx,
                 daily_gaps, gap_threshold, stop_method, stop_param,
                 max_hold_days, session_filter):
    trades = []

    for dt, row in daily_gaps.iterrows():
        gap_pct = row['gap_pct']
        atr_val = row.get('atr', np.nan)
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        if gap_pct < -gap_threshold:
            direction = 'long'
            entry_price = row['open'] * (1 + ENTRY_SLIPPAGE)
            target = row['prev_close']
            if stop_method == 'atr':
                stop = row['open'] - stop_param * atr_val
            else:
                stop = row['open'] * (1 - stop_param / 100)
        elif gap_pct > gap_threshold:
            direction = 'short'
            entry_price = row['open'] * (1 - ENTRY_SLIPPAGE)
            target = row['prev_close']
            if stop_method == 'atr':
                stop = row['open'] + stop_param * atr_val
            else:
                stop = row['open'] * (1 + stop_param / 100)
        else:
            continue

        day_date = dt.date() if hasattr(dt, 'date') else dt
        if day_date not in date_to_idx:
            continue

        pnl, exit_reason, hold = _simulate_trade(
            date_groups, sorted_dates, date_to_idx,
            date_to_idx[day_date], direction, entry_price, target, stop,
            max_hold_days, session_filter
        )

        trades.append({
            'date': str(day_date),
            'direction': direction,
            'gap_pct': gap_pct,
            'pnl': pnl,
            'exit': exit_reason,
            'hold_days': hold,
        })

    return trades


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(trades, total_gaps_at_threshold):
    if not trades:
        return {
            'trade_count': 0, 'win_rate': 0, 'profit_factor': 0,
            'avg_pnl': 0, 'total_pnl': 0, 'sharpe': 0,
            'max_drawdown': 0, 'gap_fill_rate': 0,
            'avg_win': 0, 'avg_loss': 0,
        }

    pnls = np.array([t['pnl'] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    win_rate = len(wins) / len(pnls)
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-8
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    std = pnls.std()
    sharpe = (pnls.mean() / std * np.sqrt(252)) if std > 0 else 0

    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    max_dd = (peak - cum).max()

    filled = sum(1 for t in trades if t['exit'] == 'target')
    gap_fill_rate = filled / total_gaps_at_threshold if total_gaps_at_threshold > 0 else 0

    return {
        'trade_count': int(len(trades)),
        'win_rate': round(float(win_rate), 4),
        'profit_factor': round(float(profit_factor), 4),
        'avg_pnl': round(float(pnls.mean()), 6),
        'total_pnl': round(float(pnls.sum()), 6),
        'sharpe': round(float(sharpe), 4),
        'max_drawdown': round(float(max_dd), 6),
        'gap_fill_rate': round(float(gap_fill_rate), 4),
        'avg_win': round(float(wins.mean()), 6) if len(wins) > 0 else 0,
        'avg_loss': round(float(losses.mean()), 6) if len(losses) > 0 else 0,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Loading and preparing data...")
    df, daily, date_groups, sorted_dates, date_to_idx = load_and_prepare()
    print(f"  {len(df)} bars, {len(daily)} trading days")
    print(f"  Range: {df.index.min()} to {df.index.max()}")

    # Split
    train_daily = daily[daily.index < TRAIN_CUTOFF]
    test_daily = daily[daily.index >= TRAIN_CUTOFF]
    print(f"  Train: {len(train_daily)} days | Test: {len(test_daily)} days")

    # Pre-compute gap counts per threshold for fill rate
    def count_gaps(dg, thresh):
        return int(((dg['gap_pct'] < -thresh) | (dg['gap_pct'] > thresh)).sum())

    # ─── Parameter Sweep ─────────────────────────────────────────────────
    combos = []
    for g, h, sf in product(GAP_THRESHOLDS, MAX_HOLD_DAYS, SESSION_FILTERS):
        for m in STOP_ATR_MULTS:
            combos.append(('atr', g, m, h, sf))
        for p in STOP_FIXED_PCTS:
            combos.append(('fixed', g, p, h, sf))

    total = len(combos)
    print(f"\nRunning {total} parameter combinations...")

    all_results = []
    for idx, (stop_method, gap_thresh, stop_param, max_hold, sess) in enumerate(combos):
        if (idx + 1) % 60 == 0 or idx == 0 or idx == total - 1:
            print(f"  [{idx+1}/{total}] gap={gap_thresh}% {stop_method}({stop_param}) hold={max_hold}d {sess}")

        train_trades = run_strategy(date_groups, sorted_dates, date_to_idx,
                                    train_daily, gap_thresh, stop_method, stop_param, max_hold, sess)
        test_trades = run_strategy(date_groups, sorted_dates, date_to_idx,
                                   test_daily, gap_thresh, stop_method, stop_param, max_hold, sess)

        train_total_gaps = count_gaps(train_daily, gap_thresh)
        test_total_gaps = count_gaps(test_daily, gap_thresh)

        train_m = compute_metrics(train_trades, train_total_gaps)
        test_m = compute_metrics(test_trades, test_total_gaps)

        all_results.append({
            'params': {
                'gap_threshold': gap_thresh,
                'stop_method': stop_method,
                'stop_param': stop_param,
                'max_hold_days': max_hold,
                'session_filter': sess,
            },
            'train': train_m,
            'test': test_m,
        })

    # ─── Rank & Find Best ────────────────────────────────────────────────
    valid = [r for r in all_results if r['train']['trade_count'] >= 10]

    def best_of(lst, key, split='test', cap=100):
        if not lst:
            return None
        return max(lst, key=lambda r: r[split].get(key, 0) if r[split].get(key, 0) < cap else -999)

    best_train_sharpe = best_of(valid, 'sharpe', 'train')
    best_test_sharpe = best_of(valid, 'sharpe', 'test')
    best_test_pf = best_of(valid, 'profit_factor', 'test')
    best_test_wr = best_of(valid, 'win_rate', 'test')

    top10_train = sorted(valid, key=lambda r: r['train']['sharpe'], reverse=True)[:10]

    # Gap stats
    gap_stats = {}
    for t in GAP_THRESHOLDS:
        gd = daily[daily['gap_pct'] < -t]
        gu = daily[daily['gap_pct'] > t]
        gap_stats[f'{t}%'] = {
            'total': len(gd) + len(gu),
            'down': len(gd), 'up': len(gu),
            'avg_down_pct': round(float(gd['gap_pct'].mean()), 4) if len(gd) else 0,
            'avg_up_pct': round(float(gu['gap_pct'].mean()), 4) if len(gu) else 0,
        }

    # Summary
    if valid:
        tr_s = [r['train']['sharpe'] for r in valid]
        te_s = [r['test']['sharpe'] for r in valid]
        summary = {
            'total_configs': total,
            'valid_configs': len(valid),
            'train_sharpe_range': [round(min(tr_s), 4), round(max(tr_s), 4)],
            'test_sharpe_range': [round(min(te_s), 4), round(max(te_s), 4)],
            'mean_train_sharpe': round(float(np.mean(tr_s)), 4),
            'mean_test_sharpe': round(float(np.mean(te_s)), 4),
        }
    else:
        summary = {}

    # Degradation analysis: train→test sharpe drop
    degradation = []
    for r in valid:
        drop = r['train']['sharpe'] - r['test']['sharpe']
        degradation.append({
            'params': r['params'],
            'train_sharpe': r['train']['sharpe'],
            'test_sharpe': r['test']['sharpe'],
            'drop': round(drop, 4),
        })
    degradation.sort(key=lambda x: x['drop'])

    # ─── Assemble & Save ─────────────────────────────────────────────────
    results = {
        'strategy': 'PLTR_Gap_Fill',
        'data_range': f"{df.index.min()} to {df.index.max()}",
        'train_cutoff': TRAIN_CUTOFF,
        'costs': {
            'entry_slippage': ENTRY_SLIPPAGE,
            'exit_slippage': EXIT_SLIPPAGE,
            'round_trip_commission': COMMISSION,
        },
        'summary': summary,
        'gap_statistics': gap_stats,
        'best_configs': {
            'best_train_sharpe': best_train_sharpe,
            'best_test_sharpe': best_test_sharpe,
            'best_test_profit_factor': best_test_pf,
            'best_test_win_rate': best_test_wr,
        },
        'top10_by_train_sharpe': top10_train,
        'least_degradation': degradation[:5],
        'most_degradation': degradation[-5:],
        'all_configs': all_results,
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Results saved to {RESULTS_FILE}")

    # ─── Print Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  PLTR GAP FILL STRATEGY — RESULTS")
    print("=" * 72)

    if summary:
        print(f"\nConfigs: {summary['valid_configs']}/{summary['total_configs']} valid (≥10 trades)")
        print(f"Train Sharpe: {summary['train_sharpe_range']}  (mean {summary['mean_train_sharpe']})")
        print(f"Test  Sharpe: {summary['test_sharpe_range']}  (mean {summary['mean_test_sharpe']})")

    print("\n── Gap Statistics ──")
    for k, v in gap_stats.items():
        print(f"  ≥{k:>4s} gap: {v['total']:>3d} days  (↓{v['down']:>2d} avg {v['avg_down_pct']:>+6.2f}%  ↑{v['up']:>2d} avg {v['avg_up_pct']:>+6.2f}%)")

    for label, cfg in [("Best Train Sharpe", best_train_sharpe),
                        ("Best Test Sharpe (Oracle)", best_test_sharpe),
                        ("Best Test PF", best_test_pf),
                        ("Best Test Win Rate", best_test_wr)]:
        if cfg:
            p = cfg['params']
            t, v = cfg['train'], cfg['test']
            print(f"\n── {label} ──")
            print(f"  Params: gap≥{p['gap_threshold']}%  stop={p['stop_method']}({p['stop_param']})  hold≤{p['max_hold_days']}d  session={p['session_filter']}")
            print(f"  TRAIN → trades={t['trade_count']:>3d}  WR={t['win_rate']:.1%}  PF={t['profit_factor']:.2f}  Sharpe={t['sharpe']:>+.3f}  MaxDD={t['max_drawdown']:.4f}  GapFill={t['gap_fill_rate']:.1%}")
            print(f"  TEST  → trades={v['trade_count']:>3d}  WR={v['win_rate']:.1%}  PF={v['profit_factor']:.2f}  Sharpe={v['sharpe']:>+.3f}  MaxDD={v['max_drawdown']:.4f}  GapFill={v['gap_fill_rate']:.1%}")

    print("\n── Top 5 by Train Sharpe (Walk-Forward) ──")
    for i, r in enumerate(top10_train[:5]):
        p = r['params']
        print(f"  #{i+1} gap≥{p['gap_threshold']}% {p['stop_method']}({p['stop_param']}) hold≤{p['max_hold_days']}d {p['session_filter']}")
        print(f"      Train: Sharpe={r['train']['sharpe']:>+.3f} WR={r['train']['win_rate']:.1%} | Test: Sharpe={r['test']['sharpe']:>+.3f} WR={r['test']['win_rate']:.1%}")

    print("\n── Walk-Forward Stability (least degradation) ──")
    for d in degradation[:5]:
        p = d['params']
        print(f"  {p['gap_threshold']}% {p['stop_method']}({p['stop_param']}) {p['max_hold_days']}d {p['session_filter']}: {d['train_sharpe']:>+.3f} → {d['test_sharpe']:>+.3f} (Δ{d['drop']:+.3f})")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
