# Project9 Trading System — Full Validation Suite

## OBJECTIVE

Run a complete quantitative validation on the **project9 trading system** to determine if it's ready for paper trading. This includes:
1. Parameter optimization on all strategies
2. Walk-forward validation (proper out-of-sample testing)
3. Monte Carlo simulation (10,000+ paths)
4. Comprehensive quant tests (stress tests, regime analysis, correlation, etc.)

**The system passes ONLY if ALL tests meet acceptance criteria.**

---

## SYSTEM OVERVIEW

The project9 system contains:

### Strategies (16 total)
1. **momentum_orb.py** — Momentum ORB (NVDA, AMD, PLTR, MRVL)
2. **xcu_vwap_reversion.py** — Copper VWAP Mean Reversion
3. **multi_asset_momentum.py** — Multi-Asset Time-Series Momentum
4. **regime_factor_strategy.py** — Regime-Conditional Factor
5. **pltr_gap_fill_strategy.py** — PLTR Gap Fill
6. **gap_fill_universe.py** — Gap Fill Universe Scanner
7. **momentum_50vol.py** — Momentum with Vol Filter
8. **combined_mom_vrp.py** — Combined Momentum + VRP
9. **vrp_options_strategy.py** — VRP Options Strategy
10. **vol_premium_strategy.py** — Volatility Premium
11. **high_freq_momentum.py** — High Frequency Momentum
12. **kalman_trend.py** — Kalman Filter Trend
13. **mean_reversion.py** — Adaptive Bollinger Mean Reversion
14. **vwap_mean_reversion.py** — VWAP Mean Reversion
15. **orb_strategy.py** — Opening Range Breakout (original)
16. **trend_following.py** — EMA Crossover Trend Following

### Risk Modules
- **regime_filter.py** — Regime-Conditional Filter (UpFraction > 0.55)
- **nr7_filter.py** — NR7 Volatility Contraction Filter
- **red_folder_filter.py** — Red Folder Day Filter (NFP, FOMC, CPI)
- **llm_sentiment.py** — LLM Sentiment Analysis (OpenRouter)
- **commodity_regime_filter.py** — Commodity Regime Filter

### Data
- **US Equities:** NVDA, AMD, PLTR, MRVL (5-min, 2022-2024)
- **Commodities:** XAU/USD, XAG/USD, XCU/USD, XPT/USD (1d, 1h, 4h)
- **Source:** London Strategic Edge API

---

## PHASE 1: PARAMETER OPTIMIZATION

### 1.1 Strategy-by-Strategy Optimization

For EACH strategy, run a grid search over all key parameters:

**Momentum ORB (equities):**
```
or_minutes: [5, 10, 15, 30, 60]
atr_mult_stop: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
trail_mult: [0.5, 1.0, 1.5, 2.0]
use_trend: [True, False]
```

**XCU VWAP Reversion:**
```
lookback: [10, 15, 20, 25, 30, 50]
entry_mult: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
exit_at_vwap: [True, False]
```

**Multi-Asset Momentum:**
```
lookback_days: [20, 40, 60, 80, 100, 120]
signal_type: [simple, risk_adjusted, multi_tf]
rebalance_freq: [weekly, monthly]
use_vol_target: [True, False]
target_vol: [0.10, 0.15, 0.20, 0.25]
```

**Regime-Conditional Factor:**
```
lookback_days: [20, 40, 63, 100, 126]
entry_threshold: [0.50, 0.55, 0.60, 0.65]
exit_threshold: [0.40, 0.45, 0.50]
```

**PLTR Gap Fill:**
```
min_gap_pct: [0.3, 0.5, 0.7, 1.0]
max_hold_bars: [5, 10, 15, 20]
```

**VWAP Mean Reversion (equities):**
```
z_entry: [1.5, 2.0, 2.5, 3.0]
z_exit: [0.0, 0.3, 0.5]
atr_mult: [1.0, 1.5, 2.0, 2.5]
max_hold: [4, 8, 12, 18, 24]
```

**Kalman Trend:**
```
Q: [0.001, 0.01, 0.02, 0.05, 0.1]
R: [0.1, 0.5, 1.0, 2.0, 5.0]
trail_atr_mult: [1.5, 2.0, 2.5, 3.0]
```

**EMA Trend Following:**
```
fast: [5, 9, 13, 21]
slow: [21, 34, 50, 100, 200]
atr_stop: [1.0, 1.5, 2.0, 2.5]
trail_mult: [1.5, 2.0, 2.5, 3.0]
```

### 1.2 Optimization Metrics

For each parameter combination, calculate:
- Win Rate (%)
- Profit Factor
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown (%)
- Calmar Ratio
- Total Trades
- Average Trade Duration
- Win/Loss Ratio
- Expectancy (avg win × win_rate - avg loss × loss_rate)

### 1.3 Selection Criteria

Keep parameter combos that meet ALL:
- Win Rate >= 50%
- Profit Factor >= 1.2
- Sharpe >= 0.3
- Max Drawdown <= 15%
- Total Trades >= 20

---

## PHASE 2: WALK-FORWARD VALIDATION

### 2.1 Walk-Forward Protocol

For each strategy with passing optimization results:

**Split:**
- Train Period 1: 2019-01-01 to 2021-12-31
- Test Period 1: 2022-01-01 to 2022-12-31
- Train Period 2: 2019-01-01 to 2022-12-31
- Test Period 2: 2023-01-01 to 2023-12-31
- Train Period 3: 2019-01-01 to 2023-12-31
- Test Period 3: 2024-01-01 to 2024-12-31

**Process:**
1. Optimize parameters on each train period
2. Run best parameters on corresponding test period
3. Calculate test metrics
4. Check robustness (test/train Sharpe ratio)

### 2.2 Walk-Forward Acceptance Criteria

Strategy passes walk-forward if:
- Test Sharpe > 0.2 in ALL 3 test periods
- Test Profit Factor > 1.0 in ALL 3 test periods
- Test Win Rate > 45% in ALL 3 test periods
- Test Total Return > 0% in ALL 3 test periods
- Robustness (test_sharpe / train_sharpe) > 0.4 in at least 2/3 periods

### 2.3 Anchored vs Rolling Walk-Forward

Run BOTH:
- **Anchored:** Train start fixed, test window slides forward
- **Rolling:** Fixed train window (3 years), slides forward

Compare results — if anchored and rolling give different conclusions, the strategy is fragile.

---

## PHASE 3: MONTE CARLO SIMULATION

### 3.1 Bootstrap Monte Carlo (10,000 iterations)

For each strategy that passes walk-forward:

1. Collect all trade returns from walk-forward test periods
2. Bootstrap resample (with replacement) to create 10,000 synthetic equity curves
3. Each curve = same number of trades as original, random order
4. Calculate distribution of:
   - Total Return
   - Max Drawdown
   - Win Rate
   - Profit Factor
   - Sharpe Ratio
   - Time to recovery from max DD

### 3.2 Monte Carlo Acceptance Criteria

Strategy passes Monte Carlo if:
- P(Win Rate >= 60%) > 50%
- P(Profit Factor >= 1.3) > 50%
- P(Max Drawdown <= 20%) > 70%
- P(Max Drawdown <= 30%) > 90%
- P(Total Return > 0%) > 70%
- 5th percentile Sharpe > 0

### 3.3 Parameter Perturbation Test

For each optimal parameter set:
1. Perturb each parameter by ±10%, ±20%, ±30%
2. Run backtest with perturbed parameters
3. Calculate how many perturbations still pass acceptance criteria
4. **Pass if > 50% of perturbations still profitable**

### 3.4 Trade Sequence Sensitivity

1. Remove the best 5% of trades → does strategy still work?
2. Remove the worst 5% of trades → how much does it improve?
3. Reverse the order of trades → does equity curve change shape?
4. **If removing top 5% kills profitability → strategy depends on outliers (fragile)**

---

## PHASE 4: COMPREHENSIVE QUANT TESTS

### 4.1 Deflated Sharpe Ratio

**Purpose:** Account for multiple testing bias (we tested many strategies, so some will look good by chance)

**Method:**
1. Count total strategies tested (N)
2. Calculate expected maximum Sharpe under null hypothesis (no edge)
3. Calculate deflated Sharpe = (Sharpe - expected_max) / std(Sharpe)
4. **Pass if deflated Sharpe > 1.96 (95% confidence)**

### 4.2 Minimum Backtest Length (MinBTL)

**Purpose:** Determine if we have enough data to trust the results

**Method:**
1. Calculate minimum number of trades needed for statistical significance
2. Use formula: MinBTL = (1.96 / (Sharpe / sqrt(252)))^2
3. **Pass if actual trades >= MinBTL**

### 4.3 Probability of Backtest Overfitting (PBO)

**Purpose:** Estimate probability that the strategy is overfitted

**Method:**
1. Split data into N partitions (e.g., 5)
2. Optimize on N-1 partitions, test on 1
3. Count how many times test performance < median performance
4. PBO = fraction of underperforming test partitions
5. **Pass if PBO < 0.5 (less than 50% chance of overfitting)**

### 4.4 Regime Analysis

**Purpose:** Understand when the strategy works and when it fails

**For each strategy:**
1. Classify market regime for each trade:
   - Bull market (200-day SMA rising)
   - Bear market (200-day SMA falling)
   - High volatility (VIX > 25 or equivalent)
   - Low volatility (VIX < 15 or equivalent)
   - Trending (ADX > 25)
   - Range-bound (ADX < 20)
2. Calculate metrics separately for each regime
3. **Pass if strategy is profitable in at least 2/3 regimes**

### 4.5 Correlation Analysis

**Purpose:** Ensure strategies are diversified when combined

**Method:**
1. Calculate pairwise correlation of trade returns between all strategies
2. **Pass if average correlation < 0.5**
3. **Pass if no pair has correlation > 0.7**

### 4.6 Slippage Sensitivity Analysis

**Purpose:** Understand how much execution costs can eat into profits

**Method:**
1. Run backtest with 0%, 0.05%, 0.1%, 0.2%, 0.5% slippage
2. Calculate breakeven slippage (where PF = 1.0)
3. **Pass if breakeven slippage > 0.1% (10 bps)**
4. Compare to realistic slippage for each instrument

### 4.7 Maximum Adverse Excursion (MAE) Analysis

**Purpose:** Understand worst-case intra-trade drawdowns

**Method:**
1. For each trade, calculate maximum adverse price movement
2. Plot MAE vs final trade P&L
3. Calculate optimal stop loss level (where MAE > stop = trade fails)
4. **Pass if optimal stop improves risk-adjusted returns**

### 4.8 Trade Duration Analysis

**Purpose:** Understand holding period characteristics

**Method:**
1. Calculate distribution of trade durations
2. Check if profitable trades are systematically longer/shorter
3. Check if there's an optimal holding period
4. **Pass if trade duration is consistent (low variance)**

### 4.9 Day-of-Week / Time-of-Day Analysis

**Purpose:** Find timing patterns

**Method:**
1. Calculate returns by day of week
2. Calculate returns by hour of day (for intraday strategies)
3. Check if certain times are consistently better/worse
4. **Pass if no single day/time accounts for > 30% of profits**

### 4.10 Instrument Concentration Risk

**Purpose:** Ensure portfolio isn't dependent on one symbol

**Method:**
1. Calculate contribution of each symbol to total portfolio return
2. Remove each symbol one at a time → does portfolio still work?
3. **Pass if no single symbol contributes > 50% of returns**

### 4.11 News Event Impact Analysis

**Purpose:** Understand how red folder days affect performance

**Method:**
1. Tag trades that occurred within 24h of NFP, FOMC, CPI, GDP
2. Compare performance on event days vs non-event days
3. **Pass if event day performance is not significantly worse**

### 4.12 Equity Curve Quality Metrics

**Purpose:** Assess smoothness and consistency of returns

**Calculate:**
- Ulcer Index (lower is better)
- Serenity Ratio (return / Ulcer Index)
- Common Sense Ratio (tail ratio × gain-to-pain ratio)
- **Pass if Ulcer Index < 10 and Serenity Ratio > 1.0**

---

## PHASE 5: PORTFOLIO CONSTRUCTION

### 5.1 Strategy Selection

From all strategies that passed Phases 1-4, select the top strategies:
- Must be uncorrelated (correlation < 0.5)
- Must complement each other (momentum + mean-reversion + breakout)
- Must cover multiple asset classes (equities + commodities)

### 5.2 Portfolio Optimization

**Method:**
1. Equal weight (1/N allocation)
2. Risk parity (equal risk contribution)
3. Minimum variance
4. Maximum Sharpe (Markowitz)

Compare all 4 methods and choose the most robust.

### 5.3 Position Sizing

For each strategy in the portfolio:
- Risk per trade: 1% of account
- Max position size: 5% of account
- Max portfolio exposure: 20% of account
- Use Kelly criterion (half-Kelly for safety)

### 5.4 Risk Management Rules

**Daily:**
- Max daily loss: 3% of account
- Max trades per day: 6
- Stop trading after 2 consecutive losses

**Weekly:**
- Max weekly loss: 5% of account
- Reduce size by 50% after losing week

**Monthly:**
- Max monthly loss: 10% of account
- Stop trading for the month if hit

**Prop Firm Specific:**
- FTMO: Daily DD 5%, Max DD 10%, Target 10%
- The5ers: Daily DD 3%, Max DD 6%, Target 8%
- FundingPips: Daily DD 5%, Max DD 10%, Target 8%

---

## PHASE 6: FINAL REPORT

Generate a comprehensive report with:

### 6.1 Strategy Scorecard

For each strategy tested:
```
| Strategy | Symbol | WR | PF | Sharpe | MaxDD | WF Pass | MC Pass | Ready |
|----------|--------|-----|-----|--------|-------|---------|---------|-------|
```

### 6.2 Portfolio Composition

```
| Strategy | Symbol | Weight | Expected Return | Expected DD | Correlation |
|----------|--------|--------|-----------------|-------------|-------------|
```

### 6.3 Risk Metrics

```
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Expected Annual Return | X% | > 10% | |
| Expected Max Drawdown | X% | < 20% | |
| Expected Sharpe | X | > 0.5 | |
| Expected Win Rate | X% | > 55% | |
| Deflated Sharpe | X | > 1.96 | |
| PBO | X% | < 50% | |
| MinBTL | N trades | < actual | |
```

### 6.4 Paper Trading Readiness Score

```
READINESS SCORECARD
├── Phase 1 (Optimization): PASS/FAIL (N/M strategies passed)
├── Phase 2 (Walk-Forward): PASS/FAIL (N/M strategies passed)
├── Phase 3 (Monte Carlo): PASS/FAIL (N/M strategies passed)
├── Phase 4 (Quant Tests): PASS/FAIL (N/12 tests passed)
├── Phase 5 (Portfolio): PASS/FAIL
└── OVERALL: READY / NOT READY / CONDITIONAL

If CONDITIONAL: List specific conditions that must be met
```

### 6.5 Paper Trading Plan

If system passes, generate a detailed paper trading plan:
1. Starting capital: $50,000 (simulated)
2. Instruments: [list]
3. Strategies: [list with parameters]
4. Risk rules: [specific limits]
5. Monitoring: [what to track daily/weekly/monthly]
6. Stop conditions: [when to stop paper trading and reassess]

---

## ACCEPTANCE CRITERIA SUMMARY

The system is **READY FOR PAPER TRADING** if:

### Must Pass (all required):
- [ ] At least 3 strategies pass walk-forward validation
- [ ] At least 3 strategies pass Monte Carlo (P(DD<20%) > 70%)
- [ ] Deflated Sharpe > 1.96 for portfolio
- [ ] PBO < 50% for at least 3 strategies
- [ ] Portfolio correlation < 0.5 average
- [ ] Slippage breakeven > 10 bps for at least 3 strategies
- [ ] No single symbol contributes > 50% of portfolio return
- [ ] Portfolio passes all 3 prop firm rule simulations

### Should Pass (at least 8/12):
- [ ] MinBTL < actual trades for at least 3 strategies
- [ ] Strategy profitable in at least 2/3 market regimes
- [ ] No single day/time accounts for > 30% of profits
- [ ] Ulcer Index < 10 for portfolio
- [ ] Serenity Ratio > 1.0 for portfolio
- [ ] Event day performance not significantly worse
- [ ] MAE analysis confirms stop loss levels
- [ ] Trade duration consistent (low variance)
- [ ] Parameter perturbation > 50% still profitable
- [ ] Removing top 5% trades doesn't kill profitability
- [ ] Anchored and walk-forward agree
- [ ] No strategy depends on outliers

---

## EXECUTION INSTRUCTIONS

1. **Load the project9 codebase** from the uploaded files
2. **Install dependencies:** `pip install pandas numpy scipy`
3. **Run Phase 1** — Parameter optimization (estimate: 1-2 hours)
4. **Run Phase 2** — Walk-forward validation (estimate: 30-60 min)
5. **Run Phase 3** — Monte Carlo simulation (estimate: 30-60 min)
6. **Run Phase 4** — Quant tests (estimate: 30-60 min)
7. **Run Phase 5** — Portfolio construction (estimate: 15-30 min)
8. **Generate Phase 6** — Final report

**Total estimated time: 3-5 hours** (fits within MiMo Claw session)

---

## DATA SOURCE

Use the **London Strategic Edge API** for any additional data fetching:

```python
import requests

def fetch_candles(symbol, timeframe, start, end):
    url = "https://api.londonstrategicedge.com/vault/candles"
    params = {"symbol": symbol, "timeframe": timeframe, "start": start, "end": end}
    headers = {"x-api-key": "lse_live_f4c9a7419371ecdd9365e146247b0289"}
    response = requests.get(url, params=params, headers=headers, timeout=60)
    return response.json()
```

**Rate limit:** 10 downloads/hour, 1M rows max per download

---

## CRITICAL NOTES

1. **Be honest** — If the system fails, report it. Don't fake results.
2. **No look-ahead bias** — All signals must use data available at time of trade
3. **Realistic costs** — Include 0.05% commission + 0.05% slippage minimum
4. **Walk-forward is mandatory** — No in-sample-only results accepted
5. **Monte Carlo is mandatory** — Single backtest path is not enough
6. **Deflated Sharpe** — Account for multiple testing (we tested 16+ strategies)
7. **Save all results** — JSON files for later analysis
8. **Log everything** — Timestamp and progress for each phase
