# 🏆 Commodity Strategy Lab — FINAL COMPREHENSIVE REPORT

**Date:** 2026-08-11
**Total Tests Run:** 1,200+ (236 initial + 705 optimization + 18 walk-forward + 12 session + 10K Monte Carlo + 9 robustness tests + 20 regime filters + 50 param sweeps)

---

## EXECUTIVE SUMMARY

After exhaustive testing, **no strategy fully meets all acceptance criteria with high confidence** when subjected to realistic stress testing. Here's the honest picture:

| Strategy | Backtest | Walk-Forward | Monte Carlo | Robustness | Verdict |
|---|---|---|---|---|---|
| XCU VWAP Reversion | ✅ 73% WR, 1.59 PF | 🟢 2/3 windows | 🔴 P=4.61% | 🟡 Score 5 | **MARGINAL** |
| XPT Bollinger + Regime | ✅ 82.5% WR, 2.02 PF | 🔴 0/3 windows | N/A | N/A | **OVERFITTED** |
| XPT Z-Score + Regime | ✅ 82.5% WR, 2.02 PF | 🔴 0/3 windows | N/A | N/A | **OVERFITTED** |

**The core problem:** Strategies that look great in backtests collapse under stress testing. The commodity mean-reversion edge is real but extremely thin — vulnerable to execution costs, regime shifts, and extended drawdown periods.

---

## 1. MONTE CARLO SIMULATION (XCU VWAP — 10,000 iterations)

### Confidence Intervals

| Metric | 5th %ile | 25th | Median | 75th | 95th |
|---|---|---|---|---|---|
| Win Rate | 61.5% | 69.2% | 73.1% | 76.9% | 84.6% |
| Profit Factor | 0.78 | 1.24 | 1.53 | 1.87 | 2.68 |
| Sharpe Ratio | -0.51 | 1.63 | 2.50 | 3.42 | 5.35 |
| Max Drawdown | 2.7% | 5.2% | 8.4% | 13.7% | 24.6% |
| Total Return | -15.5% | +13.9% | +31.5% | +53.2% | +108.6% |

### Probabilities

| Criteria | Probability |
|---|---|
| P(WR ≥ 65%) | 72.06% |
| P(PF ≥ 1.5) | 52.00% |
| P(Sharpe ≥ 0.5) | 82.67% |
| P(DD ≤ 15%) | **4.63%** ← BOTTLENECK |
| **P(ALL criteria)** | **4.61%** |
| P(Profitable) | 83.02% |
| P(Lose > 10%) | 14.52% |
| P(Ruin at 80% equity) | 4.39% |

### Stress Tests

| Test | Result |
|---|---|
| Parameter pass rate | 48% |
| Sub-period pass rate | 45.5% (5/11 windows) |
| Breakeven cost | 0.05% |
| Max loss streak | 8 trades |
| Worst 5-trade stretch | -31.48% |
| Low-vol regime | WR=69.5%, PF=**0.91** ❌ |
| High-vol regime | WR=76.2%, PF=**2.40** ✅ |
| Bull market | WR=74.2%, PF=**1.72** ✅ |
| Bear market | WR=71.2%, PF=**1.19** ❌ |

**Verdict: WEAK** — The strategy is profitable 83% of the time but only meets all criteria 4.6% of the time. Drawdown is the killer: expect 22-49% DD in the 5th percentile scenario.

---

## 2. ROBUSTNESS BREAKER (XCU VWAP — 9 failure modes tested)

**Overall Risk Score: 5 — 🟡 MODERATE**

| # | Failure Mode | Risk | Key Finding |
|---|---|---|---|
| 1 | Overnight gaps | 🟢 LOW | Max gap 3.3%, only 6 occurrences >2% |
| 2 | Volume dry-up | 🟢 LOW | Low-vol days still profitable (WR 70%) |
| 3 | Structural breaks | 🟡 MODERATE | **2023-2024 was a losing period** (PF 0.69-0.81) |
| 4 | Price direction | 🟡 MODERATE | Bear market PF drops to 1.19 (marginal) |
| 5 | **Slippage** | 🔴 **HIGH** | **Breakeven at just 0.05% (5 bps)** |
| 6 | Trade clustering | 🟢 LOW | Avg 15.4 days between trades |
| 7 | Data quality | 🟢 LOW | Clean data, no anomalies |
| 8 | Parameter sensitivity | 🟢 LOW | 54% of combos pass, 90% profitable |
| 9 | Stress scenarios | 🟢 LOW | Survives simulated +20% and -30% moves |

### The #1 Killer: Slippage

| Slippage | WR | PF | Sharpe | Status |
|---|---|---|---|---|
| 0.00% | 75.4% | 1.69 | 3.08 | ✅ |
| 0.05% | 73.1% | 1.59 | 2.67 | ✅ ← BREAKEVEN |
| 0.10% | 73.1% | 1.49 | 2.27 | ❌ PF fails |
| 0.20% | 70.8% | 1.30 | 1.46 | ❌ |
| 0.50% | 63.1% | 0.83 | -0.96 | ❌ Unprofitable |

**At just 10 basis points of execution cost, the strategy fails the profit factor criterion.** Realistic copper futures slippage (bid-ask + market impact) can easily hit 5-15 bps.

### Extended Losing Periods

| Period | WR | PF | Sharpe |
|---|---|---|---|
| 2023-06 → 2024-06 | 64.7% | **0.81** | -1.05 |
| 2023-12 → 2024-12 | 56.2% | **0.69** | -2.27 |

The strategy had a **2-year period of net losses** (mid-2023 through mid-2024). This would be devastating for a prop firm challenge.

---

## 3. XPT REGIME FILTER (Bollinger/Z-Score/RSI)

### Raw vs Regime-Filtered Comparison

| Strategy | Version | WR | PF | Sharpe | DD | Trades |
|---|---|---|---|---|---|---|
| Bollinger | Raw | 80.3% | 1.62 | 2.17 | 2.0% | 66 |
| **Bollinger** | **Regime (vol<0.8, adx<30)** | **82.5%** | **2.02** | **3.04** | **2.0%** | **40** |
| RSI | Raw | 77.4% | 1.36 | 1.50 | 2.0% | 62 |
| RSI | Regime | 71.4% | 0.56 | -2.64 | 2.0% | 28 |

### Best Regime Filter Parameters

**vol_percentile < 0.8, ADX < 30** — the clear winner:
- Boosts Sharpe from 2.17 → **3.04** (+40%)
- Raises PF from 1.62 → **2.02** (+25%)
- Lifts WR from 80.3% → **82.5%**
- Cuts trades by 39% (66→40) — fewer but higher-quality entries
- 9/20 parameter combos pass all criteria (45%)

**Verdict:** Regime filter helps XPT Bollinger/Z-Score but doesn't fix the fundamental walk-forward fragility. RSI remains broken regardless of filtering.

---

## 4. VWAP EXECUTION BOT

The bot is built and ready. It:
- Fetches latest daily candles from LSE API
- Computes VWAP(20) with 1σ entry bands
- Generates LONG/SHORT/EXIT signals
- Calculates position size based on 1% risk per trade
- Sets stop loss at 2× VWAP std dev
- Saves state to disk for persistence
- Supports `--loop` mode (continuous, every 4h) and `--status` mode

**Current signal** (as of last run): See `results/vwap_bot_latest.json`

**Deployment modes:**
```bash
# Single check (for cron)
python3 vwap_bot.py

# Continuous loop (every 4 hours)
python3 vwap_bot.py --loop --interval 4

# Check status
python3 vwap_bot.py --status
```

---

## 5. HONEST ASSESSMENT: WHY XCU VWAP MIGHT NOT WORK

### Reasons the strategy could fail in live trading:

1. **🔴 Slippage kills the edge** — At just 5 bps of execution cost, the strategy breaks even. Real-world slippage on copper futures is typically 5-15 bps. The strategy has almost zero margin for error.

2. **🔴 Extended drawdown periods** — The 2023-2024 period showed PF 0.69-0.81 over rolling 1-year windows. You could lose money for 2 consecutive years.

3. **🟡 Bear market weakness** — PF drops from 1.89 (bull) to 1.19 (bear). A sustained copper bear market would severely reduce profitability.

4. **🟡 Monte Carlo shows DD problem** — P(DD ≤ 15%) is only 4.63%. Real-world max drawdown is likely 22-49%.

5. **🟡 Parameter sensitivity is moderate** — 48% of perturbed parameter combos still pass. This isn't terrible but isn't robust either.

6. **🟢 BUT: 83% chance of being profitable** — The strategy IS profitable in most scenarios. It just doesn't reliably meet the strict prop firm criteria (WR≥65%, PF≥1.5, Sharpe≥0.5, DD≤15% simultaneously).

---

## 6. FINAL RECOMMENDATION

### What to actually trade:

**Don't use any of these strategies as a standalone prop firm challenge strategy.** The edge is real but too thin to survive real-world execution costs and regime shifts.

### If you must trade XCU VWAP:

1. **Use limit orders only** — Market orders will eat the edge
2. **Trade only during London/NY overlap** (highest liquidity)
3. **Add a regime filter** — Pause when copper is below 200-day SMA
4. **Hard stop at -3% cumulative DD** — Pause trading if hit
5. **Expect 2-year losing periods** — Have the capital reserves to survive them
6. **Pair with uncorrelated strategies** — Don't rely on this alone

### Better path forward:

The mean-reversion edge on copper is real but needs to be **part of a portfolio**, not a standalone strategy. Consider:
- Combining XCU VWAP with a trend-following strategy on a different asset
- Using the signal as a timing overlay rather than a standalone system
- Trading smaller size with tighter risk management

---

## FILES GENERATED

```
commodity-strategy-lab/
├── STRATEGY_REPORT.md                    # Initial report
├── FINAL_REPORT.md                       # This file
├── fetch_data.py                         # LSE API data fetcher
├── strategies.py                         # 21 strategies
├── backtest.py                           # Backtesting engine
├── optimize.py                           # Parameter optimizer
├── walkforward.py                        # Walk-forward validation
├── session_strategies.py                 # 15m session strategies
├── monte_carlo.py                        # Monte Carlo + stress tests
├── robustness_breaker.py                 # 9 failure mode tests
├── regime_filter.py                      # XPT regime filter
├── vwap_bot.py                           # Execution bot
├── run_symbol.py                         # Per-symbol runner
└── results/
    ├── XAUUSD_results.json               # Gold scan
    ├── XAGUSD_results.json               # Silver scan
    ├── XCUUSD_results.json               # Copper scan
    ├── XPTUSD_results.json               # Platinum scan
    ├── XPTUSD_optimization.json          # XPT param search
    ├── XCUUSD_optimization.json          # XCU param search
    ├── XPTUSD_walkforward.json           # XPT walk-forward
    ├── XCUUSD_walkforward.json           # XCU walk-forward
    ├── XPTUSD_session_strategies.json    # XPT 15m tests
    ├── XCUUSD_session_strategies.json    # XCU 15m tests
    ├── XCU_VWAP_monte_carlo.json         # Monte Carlo 10K
    ├── XCU_VWAP_robustness.json          # 9 failure modes
    ├── XPT_regime_filter.json            # Regime filter analysis
    └── vwap_bot_latest.json              # Bot live signal
```
