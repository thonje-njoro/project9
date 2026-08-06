#!/usr/bin/env python3
"""
VRP (Volatility Risk Premium) Options Selling Strategy
======================================================
Systematic put selling exploiting IV > RV structural edge.
Tested across multiple assets for prop firm evaluation compliance.
"""

import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from itertools import product
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── Black-Scholes ───────────────────────────────────────────────
def black_scholes_put(S, K, T, r, sigma):
    """European put price."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def put_delta(S, K, T, r, sigma):
    """Put delta (negative)."""
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1


def find_strike_for_delta(S, T, r, sigma, target_delta):
    """Analytical strike for target put delta (no iteration)."""
    # Put delta = N(d1) - 1, so |target_delta| = 1 - N(d1) => d1 = N^{-1}(1 - |target_delta|)
    d1 = norm.ppf(1 - abs(target_delta))
    # K = S * exp(-(d1 * sigma * sqrt(T) - (r + sigma^2/2) * T))
    K = S * np.exp(-(d1 * sigma * np.sqrt(T) - (r + sigma**2 / 2) * T))
    return K


# ─── Core Simulation ─────────────────────────────────────────────
def simulate_vrp(prices, dates, delta=0.15, dte=30, iv_rv_ratio=1.2,
                 weekly=True, max_loss_mult=2.0, capital=100000):
    """
    Sell puts systematically.
    Returns list of trade dicts and equity curve.
    """
    trades = []
    step = 5 if weekly else 10
    r = 0.05
    equity = capital
    peak_equity = capital
    max_dd = 0
    daily_dd_breaches = 0
    total_dd_breached = False
    equity_curve = []

    for i in range(max(20, dte), len(prices) - dte, step):
        S = prices[i]
        # Realized vol from prior 20 days
        ret_window = np.diff(np.log(prices[max(0, i - 20):i + 1]))
        realized_vol = np.std(ret_window) * np.sqrt(252)
        implied_vol = realized_vol * iv_rv_ratio

        T = dte / 365
        K = find_strike_for_delta(S, T, r, implied_vol, delta)
        premium = black_scholes_put(S, K, T, r, implied_vol)

        # P&L at expiration
        expiry_idx = min(i + dte, len(prices) - 1)
        expiry_price = prices[expiry_idx]

        # Max loss cap: max_loss_mult * premium
        raw_pnl = premium if expiry_price > K else premium - (K - expiry_price)
        pnl = max(raw_pnl, -max_loss_mult * premium)

        # Position sizing: max 25% of capital per asset
        notional = min(S * 100, equity * 0.25)  # per contract ~100 shares
        contracts = max(1, int(notional / (S * 100)))
        pnl_dollar = pnl * 100 * contracts  # options are per 100 shares

        equity += pnl_dollar
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity
        max_dd = max(max_dd, dd)

        trade = {
            'entry_idx': int(i),
            'entry_date': str(dates[i].date()) if hasattr(dates[i], 'date') else str(dates[i]),
            'entry_price': round(float(S), 2),
            'strike': round(float(K), 2),
            'delta': round(float(delta), 3),
            'dte': int(dte),
            'implied_vol': round(float(implied_vol), 4),
            'realized_vol': round(float(realized_vol), 4),
            'iv_rv_ratio': round(float(iv_rv_ratio), 2),
            'premium': round(float(premium), 4),
            'premium_pct': round(float(premium / S * 100), 4),
            'expiry_price': round(float(expiry_price), 2),
            'pnl_raw': round(float(raw_pnl), 4),
            'pnl_capped': round(float(pnl), 4),
            'pnl_pct': round(float(pnl / S * 100), 4),
            'pnl_dollar': round(float(pnl_dollar), 2),
            'equity': round(float(equity), 2),
            'contracts': int(contracts),
        }
        trades.append(trade)
        equity_curve.append({'idx': int(i), 'equity': float(equity), 'drawdown': float(dd)})

    return trades, equity_curve, {
        'final_equity': round(float(equity), 2),
        'total_return_pct': round(float((equity / capital - 1) * 100), 2),
        'max_drawdown_pct': round(float(max_dd * 100), 2),
        'num_trades': len(trades),
        'win_rate': round(float(sum(1 for t in trades if t['pnl_capped'] > 0) / max(1, len(trades)) * 100), 1),
        'avg_premium_pct': round(float(np.mean([t['premium_pct'] for t in trades])), 4) if trades else 0,
        'avg_pnl_pct': round(float(np.mean([t['pnl_pct'] for t in trades])), 4) if trades else 0,
    }


# ─── Prop Firm Compliance ───────────────────────────────────────
def check_prop_firm(equity_curve, stats, capital=100000,
                    daily_dd_limit=0.03, total_dd_limit=0.10,
                    profit_target_pct=0.08, min_trading_days=0):
    """Check standard prop firm rules."""
    violations = []
    eqs = [e['equity'] for e in equity_curve]

    # Total DD
    peak = capital
    max_dd = 0
    for eq in eqs:
        peak = max(peak, eq)
        dd = (peak - eq) / peak
        max_dd = max(max_dd, dd)

    passed_dd_total = max_dd <= total_dd_limit
    if not passed_dd_total:
        violations.append(f"Total DD {max_dd*100:.1f}% > {total_dd_limit*100:.0f}% limit")

    # Profit target
    total_return = (eqs[-1] / capital - 1) if eqs else 0
    passed_profit = total_return >= profit_target_pct
    if not passed_profit:
        violations.append(f"Return {total_return*100:.1f}% < {profit_target_pct*100:.0f}% target")

    # Win rate (soft check)
    win_rate = stats['win_rate'] / 100

    return {
        'passed_total_dd': passed_dd_total,
        'passed_profit_target': passed_profit,
        'max_drawdown_pct': round(max_dd * 100, 2),
        'total_return_pct': round(total_return * 100, 2),
        'violations': violations,
        'overall_pass': passed_dd_total and passed_profit,
    }


# ─── Sharpe Calculation ─────────────────────────────────────────
def calc_sharpe(trades, periods_per_year=52):
    """Annualized Sharpe from trade returns."""
    if len(trades) < 2:
        return 0.0
    returns = [t['pnl_pct'] / 100 for t in trades]
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1)
    if std_r == 0:
        return 0.0
    return round(float((mean_r / std_r) * np.sqrt(periods_per_year)), 3)


def calc_tail_risk(trades):
    """Tail risk metrics."""
    pnls = [t['pnl_pct'] for t in trades]
    if len(pnls) < 5:
        return {}
    sorted_pnls = sorted(pnls)
    return {
        'worst_trade_pct': round(float(sorted_pnls[0]), 4),
        '5th_percentile_pct': round(float(np.percentile(pnls, 5)), 4),
        '10th_percentile_pct': round(float(np.percentile(pnls, 10)), 4),
        'skewness': round(float(pd.Series(pnls).skew()), 3),
        'kurtosis': round(float(pd.Series(pnls).kurtosis()), 3),
        'avg_loss_pct': round(float(np.mean([p for p in pnls if p < 0])), 4) if any(p < 0 for p in pnls) else 0,
        'max_consecutive_losses': max_consecutive(pnls, negative=True),
    }


def max_consecutive(values, negative=True):
    """Max consecutive losses."""
    max_c = 0
    current = 0
    for v in values:
        if (negative and v < 0) or (not negative and v > 0):
            current += 1
            max_c = max(max_c, current)
        else:
            current = 0
    return max_c


# ─── Data Loading ────────────────────────────────────────────────
def load_data(symbols):
    """Load daily parquet files."""
    all_data = {}
    for sym in symbols:
        for pattern in [f"{sym}_daily.parquet", f"{sym}_alpaca_daily.parquet"]:
            try:
                df = pd.read_parquet(pattern)
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                all_data[sym] = {
                    'dates': df.index,
                    'close': df['close'].values.astype(float),
                }
                break
            except Exception:
                continue
    return all_data


# ─── Portfolio Simulation ────────────────────────────────────────
def run_portfolio(all_data, assets, delta, dte, iv_rv_ratio, weekly,
                  capital=100000, train_end='2022-12-31', test_start='2023-01-01'):
    """Run VRP on multiple assets, split train/test."""
    per_asset = {}
    all_test_trades = []

    for asset in assets:
        if asset not in all_data:
            continue
        dates = all_data[asset]['dates']
        prices = all_data[asset]['close']

        # Train period
        train_mask = dates <= pd.Timestamp(train_end)
        if train_mask.sum() < 50:
            continue
        train_prices = prices[train_mask]
        train_dates = dates[train_mask]

        # Test period
        test_mask = dates >= pd.Timestamp(test_start)
        if test_mask.sum() < 50:
            continue
        test_prices = prices[test_mask]
        test_dates = dates[test_mask]

        # Full run
        for label, p, d in [('train', train_prices, train_dates),
                             ('test', test_prices, test_dates)]:
            trades, eq_curve, stats = simulate_vrp(
                p, d, delta=delta, dte=dte, iv_rv_ratio=iv_rv_ratio,
                weekly=weekly, capital=capital
            )
            if asset not in per_asset:
                per_asset[asset] = {}
            per_asset[asset][label] = {
                'stats': stats,
                'trades': trades,
                'equity_curve': eq_curve,
                'sharpe': calc_sharpe(trades),
                'tail_risk': calc_tail_risk(trades),
            }
            if label == 'test':
                all_test_trades.extend(trades)

    # Portfolio combined (test period)
    if all_test_trades:
        all_test_trades.sort(key=lambda t: t['entry_idx'])
        # Aggregate: sum P&L dollar, rebuild equity curve
        portfolio_equity = capital
        portfolio_eq_curve = []
        peak = capital
        for t in all_test_trades:
            portfolio_equity += t['pnl_dollar']
            peak = max(peak, portfolio_equity)
            dd = (peak - portfolio_equity) / peak
            portfolio_eq_curve.append({'equity': portfolio_equity, 'drawdown': dd})

        portfolio_stats = {
            'final_equity': round(portfolio_equity, 2),
            'total_return_pct': round((portfolio_equity / capital - 1) * 100, 2),
            'max_drawdown_pct': round(max(e['drawdown'] for e in portfolio_eq_curve) * 100, 2) if portfolio_eq_curve else 0,
            'num_trades': len(all_test_trades),
            'win_rate': round(sum(1 for t in all_test_trades if t['pnl_capped'] > 0) / max(1, len(all_test_trades)) * 100, 1),
            'avg_premium_pct': round(np.mean([t['premium_pct'] for t in all_test_trades]), 4),
            'avg_pnl_pct': round(np.mean([t['pnl_pct'] for t in all_test_trades]), 4),
        }
        portfolio_sharpe = calc_sharpe(all_test_trades)
        portfolio_tail = calc_tail_risk(all_test_trades)
        portfolio_prop = check_prop_firm(portfolio_eq_curve, portfolio_stats, capital)
    else:
        portfolio_stats = {}
        portfolio_sharpe = 0
        portfolio_tail = {}
        portfolio_prop = {'overall_pass': False}

    return per_asset, {
        'stats': portfolio_stats,
        'sharpe': portfolio_sharpe,
        'tail_risk': portfolio_tail,
        'prop_firm': portfolio_prop,
    }


# ─── Main ────────────────────────────────────────────────────────
def main():
    ASSETS = ['SPY', 'QQQ', 'NVDA', 'AMD', 'TSLA', 'META']
    CAPITAL = 100000
    TRAIN_END = '2022-12-31'
    TEST_START = '2023-01-01'

    DELTAS = [0.10, 0.15, 0.20, 0.25]
    DTES = [7, 14, 30, 45]
    IV_RV_RATIOS = [1.1, 1.2, 1.3, 1.5]
    REBALANCE = [True, False]  # weekly=True, biweekly=False

    print("=" * 80)
    print("VRP OPTIONS SELLING STRATEGY - PROP FIRM EVALUATION")
    print("=" * 80)

    # Load data
    all_data = load_data(ASSETS)
    available = [a for a in ASSETS if a in all_data]
    print(f"\nLoaded data for: {available}")
    for a in available:
        d = all_data[a]
        print(f"  {a}: {d['dates'][0].date()} to {d['dates'][-1].date()} ({len(d['close'])} bars)")

    # Parameter grid search
    configs = list(product(DELTAS, DTES, IV_RV_RATIOS, REBALANCE))
    print(f"\nTotal configurations to test: {len(configs)}")
    print(f"Assets per portfolio: {available}")
    print(f"Train: up to {TRAIN_END} | Test: {TEST_START} onwards")
    print(f"Capital: ${CAPITAL:,}")
    print()

    # Pre-select asset subsets for portfolio (test a few combinations)
    if len(available) >= 6:
        asset_subsets = [
            available,                          # All
            ['SPY', 'QQQ'],                     # Conservative
            ['NVDA', 'AMD', 'TSLA', 'META'],    # High IV
        ]
    else:
        asset_subsets = [available]

    all_results = []
    best_score = -999
    best_result = None

    print("Running grid search...")
    for idx, (delta, dte, iv_rr, weekly) in enumerate(configs):
        if idx % 20 == 0:
            print(f"  [{idx+1}/{len(configs)}] delta={delta}, dte={dte}, iv_rv={iv_rr}, {'weekly' if weekly else 'biweekly'}")

        for assets_subset in asset_subsets:
            per_asset, portfolio = run_portfolio(
                all_data, assets_subset, delta, dte, iv_rr, weekly,
                CAPITAL, TRAIN_END, TEST_START
            )

            if not portfolio['stats']:
                continue

            # Score: Sharpe * return / max_dd (higher is better)
            sharpe = portfolio['sharpe']
            ret = portfolio['stats'].get('total_return_pct', 0)
            dd = max(portfolio['stats'].get('max_drawdown_pct', 1), 0.1)
            prop_pass = portfolio['prop_firm'].get('overall_pass', False)

            # Penalize if prop firm fails
            score = sharpe * ret / dd
            if prop_pass:
                score += 10  # Bonus for passing

            result = {
                'config': {
                    'delta': delta,
                    'dte': dte,
                    'iv_rv_ratio': iv_rr,
                    'rebalance': 'weekly' if weekly else 'biweekly',
                    'assets': assets_subset,
                },
                'portfolio': portfolio,
                'per_asset': {a: {
                    'train': per_asset[a].get('train', {}).get('stats', {}),
                    'train_sharpe': per_asset[a].get('train', {}).get('sharpe', 0),
                    'test': per_asset[a].get('test', {}).get('stats', {}),
                    'test_sharpe': per_asset[a].get('test', {}).get('sharpe', 0),
                    'test_tail_risk': per_asset[a].get('test', {}).get('tail_risk', {}),
                } for a in per_asset if 'test' in per_asset[a]},
                'score': round(score, 3),
            }
            all_results.append(result)

            if score > best_score:
                best_score = score
                best_result = result

    # Sort by score
    all_results.sort(key=lambda x: x['score'], reverse=True)
    top_10 = all_results[:10]

    # ─── Output ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 80)

    for i, r in enumerate(top_10):
        c = r['config']
        p = r['portfolio']['stats']
        pf = r['portfolio']['prop_firm']
        print(f"\n#{i+1} [Score: {r['score']}]")
        print(f"  Config: delta={c['delta']}, dte={c['dte']}, iv_rv={c['iv_rv_ratio']}, {c['rebalance']}")
        print(f"  Assets: {c['assets']}")
        print(f"  Return: {p.get('total_return_pct', 0):.1f}% | MaxDD: {p.get('max_drawdown_pct', 0):.1f}% | "
              f"Sharpe: {r['portfolio']['sharpe']:.2f} | WinRate: {p.get('win_rate', 0):.1f}%")
        print(f"  Prop Firm: {'✅ PASS' if pf.get('overall_pass') else '❌ FAIL'} - {pf.get('violations', [])}")

    # ─── Best Configuration Detail ──────────────────────────────
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION DETAIL")
    print("=" * 80)
    if best_result:
        c = best_result['config']
        print(f"\nConfig: delta={c['delta']}, dte={c['dte']}, iv_rv={c['iv_rv_ratio']}, {c['rebalance']}")
        print(f"Assets: {c['assets']}")

        print("\nPer-Asset Results (Test Period):")
        for asset, data in best_result['per_asset'].items():
            t = data.get('test', {})
            tr = data.get('test_tail_risk', {})
            print(f"\n  {asset}:")
            print(f"    Return: {t.get('total_return_pct', 0):.2f}% | MaxDD: {t.get('max_drawdown_pct', 0):.2f}%")
            print(f"    Sharpe: {data.get('test_sharpe', 0):.2f} | Trades: {t.get('num_trades', 0)} | WinRate: {t.get('win_rate', 0):.1f}%")
            print(f"    Avg Premium: {t.get('avg_premium_pct', 0):.3f}% | Avg PnL: {t.get('avg_pnl_pct', 0):.3f}%")
            if tr:
                print(f"    Tail Risk: Worst={tr.get('worst_trade_pct', 0):.2f}%, "
                      f"5th%={tr.get('5th_percentile_pct', 0):.2f}%, "
                      f"Skew={tr.get('skewness', 0):.2f}, "
                      f"MaxConsecLosses={tr.get('max_consecutive_losses', 0)}")

        print(f"\nPortfolio Combined (Test Period):")
        p = best_result['portfolio']['stats']
        pt = best_result['portfolio']['tail_risk']
        pf = best_result['portfolio']['prop_firm']
        print(f"  Final Equity: ${p.get('final_equity', 0):,.2f}")
        print(f"  Total Return: {p.get('total_return_pct', 0):.2f}%")
        print(f"  Max Drawdown: {p.get('max_drawdown_pct', 0):.2f}%")
        print(f"  Sharpe Ratio: {best_result['portfolio']['sharpe']:.3f}")
        print(f"  Win Rate: {p.get('win_rate', 0):.1f}%")
        print(f"  Num Trades: {p.get('num_trades', 0)}")
        if pt:
            print(f"  Tail Risk: Worst={pt.get('worst_trade_pct', 0):.2f}%, "
                  f"Skew={pt.get('skewness', 0):.2f}, "
                  f"Kurt={pt.get('kurtosis', 0):.2f}")

        print(f"\n  Prop Firm Compliance: {'✅ PASS' if pf.get('overall_pass') else '❌ FAIL'}")
        print(f"    Total DD: {pf.get('max_drawdown_pct', 0):.2f}% {'✅' if pf.get('passed_total_dd') else '❌'}")
        print(f"    Profit Target: {pf.get('total_return_pct', 0):.2f}% {'✅' if pf.get('passed_profit_target') else '❌'}")
        if pf.get('violations'):
            for v in pf['violations']:
                print(f"    ⚠️ {v}")

    # ─── Prop Firm Compliance Summary ───────────────────────────
    print("\n" + "=" * 80)
    print("PROP FIRM COMPLIANCE SUMMARY")
    print("=" * 80)
    passing = [r for r in all_results if r['portfolio']['prop_firm'].get('overall_pass')]
    print(f"\nTotal configs tested: {len(all_results)}")
    print(f"Passing prop firm rules: {len(passing)}")
    if passing:
        print("\nPassing configurations:")
        for r in passing[:5]:
            c = r['config']
            p = r['portfolio']['stats']
            print(f"  delta={c['delta']}, dte={c['dte']}, iv_rv={c['iv_rv_ratio']}, {c['rebalance']}, "
                  f"assets={c['assets']}")
            print(f"    Return: {p.get('total_return_pct', 0):.1f}% | DD: {p.get('max_drawdown_pct', 0):.1f}% | "
                  f"Sharpe: {r['portfolio']['sharpe']:.2f}")

    # ─── Risk Metrics ───────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RISK METRICS (BEST CONFIG)")
    print("=" * 80)
    if best_result and best_result['portfolio']['tail_risk']:
        pt = best_result['portfolio']['tail_risk']
        print(f"  Worst Single Trade: {pt.get('worst_trade_pct', 0):.2f}%")
        print(f"  5th Percentile: {pt.get('5th_percentile_pct', 0):.2f}%")
        print(f"  10th Percentile: {pt.get('10th_percentile_pct', 0):.2f}%")
        print(f"  Skewness: {pt.get('skewness', 0):.3f}")
        print(f"  Kurtosis: {pt.get('kurtosis', 0):.3f}")
        print(f"  Avg Loss: {pt.get('avg_loss_pct', 0):.2f}%")
        print(f"  Max Consecutive Losses: {pt.get('max_consecutive_losses', 0)}")

    # ─── Save Results ───────────────────────────────────────────
    output = {
        'run_timestamp': datetime.now().isoformat(),
        'parameters': {
            'assets': ASSETS,
            'capital': CAPITAL,
            'train_end': TRAIN_END,
            'test_start': TEST_START,
            'deltas': DELTAS,
            'dtes': DTES,
            'iv_rv_ratios': IV_RV_RATIOS,
            'total_configs': len(configs),
        },
        'best_config': best_result,
        'top_10': top_10,
        'all_passing': passing if passing else [],
        'summary': {
            'total_configs_tested': len(all_results),
            'configs_passing_prop': len(passing),
            'best_score': round(best_score, 3),
        },
    }

    with open('vrp_options_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✅ Results saved to vrp_options_results.json")


if __name__ == '__main__':
    main()
