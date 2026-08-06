"""
PROP FIRM CHALLENGE POSITION SIZING MATH
=========================================
Correct Monte Carlo simulation using the actual strategy's edge.

Key insight: The strategy has modest edge (PF=2.02, WR=53.72%) but with
very small per-trade returns because ATR × max_exposure cap constrains
position sizes. For a challenge we can increase these limits.

The per-trade PnL formula:
    PnL = position_value * trade_return_fraction

Where trade_return_fraction is from the strategy's distribution (e.g.,
winners avg +0.185%, losers avg -0.108% of position value).
"""

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')


def compute_trade_stats_from_backtest():
    """
    Compute the actual per-trade return distribution from the combined backtest.
    
    From summary.csv:
      COMBINED: 121 trades, 53.72% WR, PF=1.994, 1.084% total return
      Config: 2% risk/trade, 6% max exposure, 3 concurrent
      Capital: $50k
    
    Actual per-trade PnL figures:
      - Total PnL: $50k × 1.084% = $542
      - Avg PnL: $542 / 121 = $4.48
      - With max_exposure_pct=6%, avg position value ≈ $3,000
      - Winner avg return on position: $16.73 / $3,000 = 0.558%
      - Loser avg return on position: $9.74 / $3,000 = 0.325%
    
    For the MC simulation, we use RETURN ON POSITION directly.
    Position value is computed as: min(risk_dollars/ATR, max_exposure_dollars)
    """
    return {
        'n_trades': 121,
        'capital': 50000,
        'total_return_pct': 1.084,
        'win_rate': 0.5372,
        'profit_factor': 1.994,
        'avg_win_return': 0.00558,     # 0.558% return on position (winners)
        'avg_loss_return': -0.00325,   # -0.325% return on position (losers)
        'avg_position_value': 3000,    # Capped by max_exposure=6%
    }


def build_trade_return_distribution(
    n_trades: int = 121,
    win_rate: float = 0.5372,
    avg_win_ret: float = 0.00558,
    avg_loss_ret: float = -0.00325,
    std_win: float = 0.004,
    std_loss: float = 0.0025,
    seed: int = 42
) -> np.ndarray:
    """
    Build synthetic trade return distribution (RETURN ON POSITION, not on capital).
    These are the vectorbt 'Return' column values.
    """
    rng = np.random.default_rng(seed)
    
    n_winners = int(n_trades * win_rate)
    n_losers = n_trades - n_winners
    
    # Log-normal-like distribution for winners, left-skewed for losers
    winners = np.abs(rng.normal(avg_win_ret, std_win, n_winners))
    winners = np.clip(winners, 0.0001, None)  # min 0.01%
    
    losers = -np.abs(rng.normal(abs(avg_loss_ret), std_loss, n_losers))
    losers = np.clip(losers, None, -0.0001)
    
    all_returns = np.concatenate([winners, losers])
    rng.shuffle(all_returns)
    
    actual_wr = (all_returns > 0).mean()
    w = all_returns[all_returns > 0]
    l = all_returns[all_returns < 0]
    actual_pf = w.sum() / abs(l.sum()) if len(l) > 0 and l.sum() != 0 else 0
    
    print(f"  Trade return distribution (RETURN ON POSITION):", flush=True)
    print(f"    {len(all_returns)} trades, WR={actual_wr:.1%}, PF={actual_pf:.2f}", flush=True)
    print(f"    Winners: mean={w.mean():.4f}, std={w.std():.4f}" if len(w) > 0 else "    Winners: N/A", flush=True)
    print(f"    Losers: mean={l.mean():.4f}, std={l.std():.4f}" if len(l) > 0 else "    Losers: N/A", flush=True)
    
    return all_returns


def simulate_challenge(
    trade_position_returns: np.ndarray,
    max_exposure_pct: float = 0.06,
    max_concurrent: int = 3,
    max_daily_loss_pct: float = 0.04,
    max_total_loss_pct: float = 0.10,
    profit_target_pct: float = 0.10,
    max_trading_days: int = 22,
    initial_capital: float = 50_000,
    daily_trade_prob: float = 0.6,
    n_simulations: int = 20000,
    seed: int = 42,
):
    """
    Monte Carlo simulation of prop firm challenge.
    
    CORRECT MODEL:
    - trade_position_returns: distribution of per-trade returns on POSITION
    - position_value = capital * max_exposure_pct (user sets this)
    - PnL per trade = position_value * trade_return
    - max_exposure_pct is the USER-CHOSEN exposure per trade
    
    We assume the user overrides the ATR cap and just uses fixed exposure.
    """
    n_returns = len(trade_position_returns)
    rng = np.random.default_rng(seed)
    
    passed = np.zeros(n_simulations, dtype=bool)
    failed_daily = np.zeros(n_simulations, dtype=bool)
    failed_max = np.zeros(n_simulations, dtype=bool)
    final_capital = np.full(n_simulations, float(initial_capital))
    max_dds = np.zeros(n_simulations)
    
    position_value = initial_capital * max_exposure_pct  # fixed per trade
    
    for sim in range(n_simulations):
        capital = float(initial_capital)
        peak = capital
        
        for day in range(max_trading_days):
            if rng.random() > daily_trade_prob:
                continue
            
            day_n_trades = rng.integers(1, max_concurrent + 1)
            day_pnl = 0.0
            
            for _ in range(day_n_trades):
                r = trade_position_returns[rng.integers(0, n_returns)]
                trade_pnl = position_value * r
                day_pnl += trade_pnl
            
            # Check daily loss limit (4% of day-start capital)
            if day_pnl < -capital * max_daily_loss_pct:
                failed_daily[sim] = True
                break
            
            capital += day_pnl
            if capital > peak:
                peak = capital
            
            # Check trailing max drawdown (10%)
            current_dd = max(0, (peak - capital) / peak)
            if current_dd > max_total_loss_pct:
                failed_max[sim] = True
                break
            
            # Check profit target
            if (capital - initial_capital) / initial_capital >= profit_target_pct:
                passed[sim] = True
                break
        
        final_capital[sim] = capital
        max_dds[sim] = max(0, (peak - capital) / peak) if capital > 0 else 1.0
    
    total_return_pct = (final_capital - initial_capital) / initial_capital * 100
    
    return {
        'max_exposure_pct': max_exposure_pct,
        'max_concurrent': max_concurrent,
        'position_value_pct': max_exposure_pct * 100,
        'n_simulations': n_simulations,
        'pass_rate': passed.mean(),
        'daily_dd_breach_prob': failed_daily.mean(),
        'max_dd_breach_prob': failed_max.mean(),
        'timeout_rate': (1 - passed - failed_daily - failed_max).mean(),
        'avg_return_pct': total_return_pct.mean(),
        'med_return_pct': np.median(total_return_pct),
        'p10_return_pct': np.percentile(total_return_pct, 10),
        'p90_return_pct': np.percentile(total_return_pct, 90),
        'prob_5pct_plus': (total_return_pct >= 5).mean(),
        'prob_10pct_plus': (total_return_pct >= 10).mean(),
        'prob_neg_return': (total_return_pct < 0).mean(),
        'avg_max_drawdown': max_dds.mean(),
        'p95_max_drawdown': np.percentile(max_dds, 95),
        'p99_max_drawdown': np.percentile(max_dds, 99),
    }


def run_and_report(trade_returns, max_exposure, concurrent, sims=20000, label=""):
    """Run simulation and print results."""
    t0 = time.time()
    print(f"  [{label}] exposure={max_exposure*100:.1f}% conc={concurrent} "
          f"({sims} sims)...", end=" ", flush=True)
    
    res = simulate_challenge(
        trade_returns,
        max_exposure_pct=max_exposure,
        max_concurrent=concurrent,
        n_simulations=sims,
    )
    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s)", flush=True)
    print(f"    Pass rate (10% target):  {res['pass_rate']:6.1%}", flush=True)
    print(f"    Daily DD breach:         {res['daily_dd_breach_prob']:6.1%}", flush=True)
    print(f"    Max DD breach:           {res['max_dd_breach_prob']:6.1%}", flush=True)
    print(f"    Avg return:              {res['avg_return_pct']:6.2f}%", flush=True)
    print(f"    Median return:           {res['med_return_pct']:6.2f}%", flush=True)
    print(f"    P(return >= 10%):        {res['prob_10pct_plus']:6.1%}", flush=True)
    print(f"    P(return >= 5%):         {res['prob_5pct_plus']:6.1%}", flush=True)
    print(f"    P(return < 0%):          {res['prob_neg_return']:6.1%}", flush=True)
    print(f"    P90 return:              {res['p90_return_pct']:6.2f}%", flush=True)
    print(f"    P10 return:              {res['p10_return_pct']:6.2f}%", flush=True)
    print(f"    Avg max DD:              {res['avg_max_drawdown']:6.1%}", flush=True)
    print(f"    P95 max DD:              {res['p95_max_drawdown']:6.1%}", flush=True)
    return res


def main():
    print("=" * 72, flush=True)
    print("PROP FIRM CHALLENGE POSITION SIZING ANALYSIS", flush=True)
    print("=" * 72, flush=True)
    print(flush=True)
    print("Challenge: $50k FTMO-style", flush=True)
    print("  Profit target: 10% ($5,000) in 22 trading days", flush=True)
    print("  Daily loss limit: 4% ($2,000)", flush=True)
    print("  Max loss limit: 10% ($5,000)", flush=True)
    print(flush=True)
    print("System: 6-instrument portfolio (kalman_trend + cper_gld_ratio)", flush=True)
    print("  Backtest PF: 2.02 | Win rate: 53.72% | 121 trades", flush=True)
    print(flush=True)
    
    # Build trade return distribution (fractional return ON POSITION)
    stats = compute_trade_stats_from_backtest()
    print(f"Backtest-derived trade stats:", flush=True)
    print(f"  Winners: avg return on position = {stats['avg_win_return']:.4f} ({stats['avg_win_return']*100:.2f}%)", flush=True)
    print(f"  Losers:  avg return on position = {stats['avg_loss_return']:.4f} ({stats['avg_loss_return']*100:.2f}%)", flush=True)
    print(f"  Win rate: {stats['win_rate']:.1%}, PF: {stats['profit_factor']:.2f}", flush=True)
    print(flush=True)
    
    trade_returns = build_trade_return_distribution(
        n_trades=stats['n_trades'],
        win_rate=stats['win_rate'],
        avg_win_ret=stats['avg_win_return'],
        avg_loss_ret=stats['avg_loss_return'],
    )
    print(flush=True)
    
    all_results = []
    
    # ================================================================
    # QUESTION 1: max_exposure=12% (4% * 3 concurrent), 3 concurrent
    # ================================================================
    print("=" * 72, flush=True)
    print("QUESTION 1: 12% exposure/trade, 3 concurrent", flush=True)
    print("  (max_risk_per_trade_pct=4%, max_exposure_pct=0.12, 3 concurrent)", flush=True)
    print("=" * 72, flush=True)
    r = run_and_report(trade_returns, 0.12, 3, 20000, "Q1")
    all_results.append(r)
    print(flush=True)
    
    # ================================================================
    # QUESTION 2: 4%/5 concurrent (original pre-conservative config)
    # ================================================================
    print("=" * 72, flush=True)
    print("QUESTION 2: 4% exposure/trade, 5 concurrent (original config)", flush=True)
    print("=" * 72, flush=True)
    r = run_and_report(trade_returns, 0.04, 5, 20000, "Q2")
    all_results.append(r)
    print(flush=True)
    
    # ================================================================
    # QUESTION 3: Find risk level where P(return >= 10%) > 25%
    # ================================================================
    print("=" * 72, flush=True)
    print("QUESTION 3: Exposure where P(return>=10%) > 25%", flush=True)
    print("=" * 72, flush=True)
    
    threshold_conc3 = None
    threshold_conc5 = None
    
    for exp in [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
        for conc in [3, 5]:
            r = simulate_challenge(trade_returns, exp, conc, n_simulations=10000)
            all_results.append(r)
            p10 = r['prob_10pct_plus']
            if p10 > 0.01:
                print(f"  exposure={exp*100:.1f}% conc={conc}: "
                      f"P(10%+)={p10:.1%} pass={r['pass_rate']:.1%} "
                      f"dailyDD={r['daily_dd_breach_prob']:.1%} "
                      f"avgR={r['avg_return_pct']:.2f}%", flush=True)
            if conc == 3 and threshold_conc3 is None and p10 > 0.25:
                threshold_conc3 = (exp, r)
            if conc == 5 and threshold_conc5 is None and p10 > 0.25:
                threshold_conc5 = (exp, r)
    
    print(flush=True)
    if threshold_conc3:
        exp, rr = threshold_conc3
        print(f"  >>> [CONC=3] P(10%+) > 25% at exposure >= {exp*100:.0f}%", flush=True)
        print(f"      At {exp*100:.0f}%: pass={rr['pass_rate']:.1%}, dailyDD={rr['daily_dd_breach_prob']:.1%}", flush=True)
    if threshold_conc5:
        exp, rr = threshold_conc5
        print(f"  >>> [CONC=5] P(10%+) > 25% at exposure >= {exp*100:.0f}%", flush=True)
        print(f"      At {exp*100:.0f}%: pass={rr['pass_rate']:.1%}, dailyDD={rr['daily_dd_breach_prob']:.1%}", flush=True)
    print(flush=True)
    
    # ================================================================
    # QUESTIONS 4 & 5: Full grid
    # ================================================================
    print("=" * 72, flush=True)
    print("QUESTIONS 4 & 5: Full exposure/concurrent grid", flush=True)
    print("=" * 72, flush=True)
    
    exposures = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    concurrents = [1, 2, 3, 4, 5]
    
    for exp in exposures:
        for conc in concurrents:
            already = [x for x in all_results 
                      if abs(x['max_exposure_pct'] - exp) < 0.001 and x['max_concurrent'] == conc]
            if not already:
                r = simulate_challenge(trade_returns, exp, conc, n_simulations=10000)
                all_results.append(r)
    
    print(flush=True)
    
    # ================================================================
    # FINAL RESULTS
    # ================================================================
    df = pd.DataFrame(all_results)
    df = df.drop_duplicates(subset=['max_exposure_pct', 'max_concurrent'])
    df = df.sort_values(['max_exposure_pct', 'max_concurrent'])
    
    out_path = '/home/admin1/project9/backtest/results/prop_firm_sizing_results.csv'
    df.to_csv(out_path, index=False)
    
    print("=" * 100, flush=True)
    print("FULL RESULTS TABLE", flush=True)
    print("=" * 100, flush=True)
    show_cols = ['max_exposure_pct', 'max_concurrent', 'pass_rate', 'daily_dd_breach_prob',
                 'max_dd_breach_prob', 'avg_return_pct', 'med_return_pct',
                 'prob_10pct_plus', 'prob_5pct_plus', 'prob_neg_return',
                 'p95_max_drawdown', 'avg_max_drawdown']
    # Rename for display
    df_display = df[show_cols].copy()
    df_display['max_exposure_pct'] = df_display['max_exposure_pct'] * 100
    print(df_display.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
    print(flush=True)
    
    # Q4: Expected attempts
    print("=" * 100, flush=True)
    print("Q4: EXPECTED ATTEMPTS TO PASS", flush=True)
    print("=" * 100, flush=True)
    print(f"  {'Exposure':>9s} {'Conc':>5s} {'PassRate':>9s} {'ExpAttempts':>13s} {'DailyDD':>8s} {'AvgRet%':>8s} {'P10+':>6s}", flush=True)
    print(f"  {'-'*9} {'-'*5} {'-'*9} {'-'*13} {'-'*8} {'-'*8} {'-'*6}", flush=True)
    for _, r in df.sort_values(['max_exposure_pct', 'max_concurrent']).iterrows():
        exp_a = 1/r['pass_rate'] if r['pass_rate'] > 0 else float('inf')
        print(f"  {r['max_exposure_pct']*100:7.1f}% {r['max_concurrent']:5d} "
              f"{r['pass_rate']:8.1%} "
              f"{exp_a:13.1f} "
              f"{r['daily_dd_breach_prob']:7.1%} "
              f"{r['avg_return_pct']:7.2f}% "
              f"{r['prob_10pct_plus']:5.1%}", flush=True)
    print(flush=True)
    
    # Q5
    print("=" * 100, flush=True)
    print("Q5: EXPOSURE WHERE EXPECTED RETURN ≈ 10% AND DD BREACH < 50%", flush=True)
    print("=" * 100, flush=True)
    
    candidates = df[(df['avg_return_pct'] >= 8) & (df['avg_return_pct'] <= 12)]
    if len(candidates):
        safe = candidates[candidates['daily_dd_breach_prob'] < 0.50]
        if len(safe):
            best = safe.loc[safe['avg_return_pct'].sub(10).abs().idxmin()]
            print("Configs meeting criteria:", flush=True)
            for _, r in safe.iterrows():
                print(f"  exposure={r['max_exposure_pct']*100:5.1f}% conc={r['max_concurrent']}: "
                      f"avg_ret={r['avg_return_pct']:6.2f}% "
                      f"dailyDD={r['daily_dd_breach_prob']:5.1%} "
                      f"pass={r['pass_rate']:5.1%}", flush=True)
            print(flush=True)
            print(f"BEST MATCH: exposure={best['max_exposure_pct']*100:.1f}% "
                  f"conc={best['max_concurrent']}: "
                  f"avg_ret={best['avg_return_pct']:.2f}% "
                  f"dailyDD={best['daily_dd_breach_prob']:.1%} "
                  f"pass_rate={best['pass_rate']:.1%}", flush=True)
        else:
            print("All 8-12% return configs have DD > 50%:", flush=True)
            for _, r in candidates.iterrows():
                print(f"  exposure={r['max_exposure_pct']*100:5.1f}% conc={r['max_concurrent']}: "
                      f"avg_ret={r['avg_return_pct']:6.2f}% "
                      f"dailyDD={r['daily_dd_breach_prob']:5.1%}", flush=True)
    print(flush=True)
    
    # Risk scaling analysis
    print("=" * 100, flush=True)
    print("RISK SCALING ANALYSIS (non-linearity)", flush=True)
    print("=" * 100, flush=True)
    print(f"  {'Exposure':>9s} {'Conc':>5s} {'AvgRet%':>8s} {'RetLinear':>9s} {'RetActual/Lin':>13s} {'DailyDD':>8s} {'DD_mult':>8s}", flush=True)
    print(f"  {'-'*9} {'-'*5} {'-'*8} {'-'*9} {'-'*13} {'-'*8} {'-'*8}", flush=True)
    
    base = df[(df['max_exposure_pct'] == 0.06) & (df['max_concurrent'] == 3)]
    base_ret = base['avg_return_pct'].values[0] if len(base) else 0.088
    base_dd = base['daily_dd_breach_prob'].values[0] if len(base) else 0.001
    
    for _, r in df[df['max_concurrent'] == 3].sort_values('max_exposure_pct').iterrows():
        scale = r['max_exposure_pct'] / 0.06
        linear_ret = base_ret * scale
        actual_ret = r['avg_return_pct']
        ret_ratio = actual_ret / linear_ret if linear_ret > 0 else 0
        dd_ratio = r['daily_dd_breach_prob'] / base_dd if base_dd > 0 else 0
        print(f"  {r['max_exposure_pct']*100:7.1f}% {r['max_concurrent']:5d} "
              f"{actual_ret:7.2f}% "
              f"{linear_ret:8.2f}% "
              f"{ret_ratio:12.2f}x "
              f"{r['daily_dd_breach_prob']:7.1%} "
              f"{dd_ratio:7.1f}x", flush=True)
    print(flush=True)
    
    print(f"Results saved to {out_path}", flush=True)


if __name__ == '__main__':
    main()
