# PROJECT9 SYSTEM REPAIR PLAN
# ============================
# Professional Financial Engineering Assessment
# Date: 2026-08-07
# Status: CRITICAL — Multiple Fundamental Flaws Identified

## EXECUTIVE SUMMARY

The current Momentum ORB system is NOT fixable through parameter tuning.
The fundamental problem is: **there is no exploitable momentum edge in these
markets at the 5-minute timeframe.**

All strategies tested show win rates between 40-55%, which is within the
noise range. The backtest results showing 80%+ WR were artifacts of
look-ahead bias and unrealistic assumptions.

---

## ROOT CAUSE ANALYSIS

### 1. NO MOMENTUM EDGE EXISTS

Research results on 5-min data:
- NVDA: Random walk (autocorrelation -0.015)
- AMD: Random walk (autocorrelation -0.016)
- PLTR: Mean-reverting (autocorrelation -0.117)
- MRVL: Mean-reverting (autocorrelation -0.102)

Breakout continuation tests:
- NVDA: 49.6% WR (UP), 50.0% WR (DN) → NO EDGE
- AMD: 50.4% WR (UP), 54.4% WR (DN) → Marginal short edge
- PLTR: 45.8% WR (UP), 56.5% WR (DN) → Short edge on breakdowns
- MRVL: 40.2% WR (UP), 47.3% WR (DN) → NO EDGE (breakouts reverse!)

**Conclusion: Momentum ORB cannot work because momentum does not exist
at this timeframe.**

### 2. LOOK-AHEAD BIAS (FIXABLE)

Three layers of bias:
a) Entry at exact breakout level (or_high/or_low) — should be next bar open
b) Trailing stop updated and checked on same bar — should check next bar
c) Slippage model too optimistic (0.1% vs realistic 0.3-0.5%)

### 3. DATA QUALITY ISSUES

- Extremely high skewness (31-39 for NVDA/PLTR) suggests data errors
- Extremely high kurtosis (3600-6300) suggests fat tails/outliers
- 269-3411 gaps > 10 minutes in the data
- No volume filter for breakouts

### 4. OVERFITTING

- 1680+ parameter combos tested per symbol
- Walk-forward validation helps but doesn't eliminate overfitting
- Survivorship bias in symbol selection (all winners)

---

## FIX PLAN (3 PHASES)

### PHASE 1: FIX THE BACKTEST ENGINE (1-2 days)

**Goal:** Eliminate all look-ahead bias and make the backtest realistic.

**Changes needed:**

1. **Entry price = next bar's open after breakout**
   - Current: entry_price = or_high (breakout level)
   - Fix: entry_price = open[i+1] after breakout detected at bar i
   - Impact: 0.1-0.3% worse per trade

2. **Trailing stop checked on NEXT bar after update**
   - Current: update stop and check if hit on same bar
   - Fix: update stop at bar i, check if hit at bar i+1
   - Impact: Some trades that were stopped out will now stay in

3. **Realistic slippage model**
   - Current: flat 0.1% (0.05% entry + 0.05% exit)
   - Fix: 0.3% for breakout entries (price gaps through level), 0.1% for exits
   - Impact: Reduces PF by 20-40%

4. **Volume filter**
   - Add: minimum volume at breakout (e.g., 2x average)
   - Impact: Filters out fake breakouts

5. **Commission model**
   - Current: flat 0.05% per trade
   - Fix: actual commission structure (e.g., $0.005/share for stocks)
   - Impact: More realistic for small accounts

### PHASE 2: FIND ACTUAL EDGES (3-5 days)

**Goal:** Identify strategies that have statistical edge on this data.

**Research completed shows:**

1. **Mean-reversion on PLTR** (53.4% WR on VWAP z-score < -2)
   - Strategy: Buy when price is 2+ standard deviations below VWAP
   - Exit: Return to VWAP or 10-bar time stop
   - Risk: Works in range-bound markets, fails in trends

2. **Short breakdowns on AMD** (54.4% WR on OR breakdown)
   - Strategy: Short when price breaks below opening range
   - Exit: Return to OR low or trailing stop
   - Risk: Only works in bearish regimes

3. **Short breakdowns on PLTR** (56.5% WR on OR breakdown)
   - Strategy: Same as AMD short
   - Risk: PLTR is mean-reverting, so breakdowns tend to reverse

**New strategies to implement:**

A. **VWAP Mean-Reversion** (for PLTR, MRVL)
   - Entry: Buy when z-score < -2, sell when z-score > 2
   - Exit: Return to VWAP or time stop (10 bars)
   - Symbols: PLTR (53.4% WR), MRVL (52.8% WR)
   - Expected PF: 1.1-1.3 (marginal edge)

B. **Opening Range Breakdown Short** (for AMD, PLTR)
   - Entry: Short when price breaks below OR low
   - Exit: Return to OR low or trailing stop
   - Symbols: AMD (54.4% WR), PLTR (56.5% WR)
   - Expected PF: 1.2-1.5 (moderate edge)

C. **Mean-Reversion After Large Moves** (all symbols)
   - Entry: Buy after 2-sigma down move, sell after 2-sigma up move
   - Exit: Return to mean or time stop
   - Symbols: All (49-50% WR — marginal)
   - Expected PF: 1.0-1.1 (no edge after costs)

### PHASE 3: PORTFOLIO CONSTRUCTION (2-3 days)

**Goal:** Build a diversified portfolio that can pass prop firm challenges.

**Key principles:**

1. **Multiple uncorrelated strategies**
   - Don't rely on single strategy
   - Combine momentum + mean-reversion + pairs
   - Target: 3-5 strategies with low correlation

2. **Proper position sizing**
   - Kelly criterion with half-Kelly for safety
   - Max 2% risk per trade
   - Max 6% total portfolio risk

3. **Prop firm rule compliance**
   - Daily drawdown limit: 4% (buffer to 3.5%)
   - Max drawdown limit: 10% (buffer to 9%)
   - Consistency rule: no single day > 30% of profits

4. **Regime detection**
   - Use HMM or volatility regime filter
   - Only trade momentum in trending regimes
   - Only trade mean-reversion in range-bound regimes

---

## REVISED EXPECTATIONS

### What's Realistic

Based on research, the best achievable metrics are:

| Metric | Backtest | Live (Expected) |
|--------|----------|-----------------|
| Win Rate | 52-56% | 50-54% |
| Profit Factor | 1.2-1.5 | 1.0-1.2 |
| Sharpe Ratio | 0.5-0.8 | 0.3-0.5 |
| Max Drawdown | 5-10% | 8-15% |
| Monthly Return | 2-5% | 1-3% |

### Prop Firm Challenge Strategy

For FTMO/The5ers/FundingPips challenges:

1. **Phase 1 (Probing)**: Trade small, 1-2 trades/day, target 1% per day
2. **Phase 2 (Acceleration)**: Increase size, 2-3 trades/day, target 1.5% per day
3. **Phase 3 (Preservation)**: Reduce size, 1 trade/day, protect profits

**Risk management:**
- Daily loss limit: 3% (buffer to FTMO's 4%)
- Max drawdown: 8% (buffer to FTMO's 10%)
- Stop trading after 2 consecutive losses
- Reduce size by 50% after 3 consecutive losses

---

## IMPLEMENTATION ORDER

1. **Fix backtest engine** (eliminate look-ahead bias)
2. **Implement VWAP mean-reversion** (marginal edge on PLTR/MRVL)
3. **Implement OR breakdown short** (moderate edge on AMD/PLTR)
4. **Add regime filter** (avoid trading in wrong regime)
5. **Build portfolio** (combine strategies, proper sizing)
6. **Run walk-forward validation** (proper train/test split)
7. **Paper trade for 2 weeks** (verify live performance)
8. **Start prop firm challenge** (with proper risk management)

---

## WHAT WON'T WORK

1. **Momentum ORB on NVDA** — no edge exists (random walk)
2. **Momentum ORB on MRVL** — breakouts reverse (40% WR)
3. **Any single-strategy approach** — too much concentration risk
4. **Aggressive position sizing** — will violate prop firm rules
5. **Trading all day** — only trade during high-edge hours (9-10 AM, 2-3 PM ET)

---

## CONCLUSION

The current system is NOT fixable through parameter tuning. The fundamental
problem is that momentum does not exist at the 5-minute timeframe for these
symbols. The only viable path is:

1. Fix the backtest engine to eliminate bias
2. Implement mean-reversion strategies (marginal edge)
3. Build a diversified portfolio
4. Use proper risk management for prop firms

Expected outcome: **Marginal profitability (1-3% monthly) with risk of
significant drawdowns (8-15%).** This is NOT a get-rich-quick system.

**Recommendation: Do NOT trade with real money until the backtest is fixed
and paper trading confirms the results.**
