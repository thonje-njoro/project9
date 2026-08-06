# System Diagnostic & Fix Plan
## Professional Assessment by Financial + Software Engineering Analysis

**Date:** 2026-08-07
**Status:** PLAN ONLY — Do not execute until reviewed

---

## Part 1: Root Cause Analysis — What Went Wrong

### 1.1 Software Engineering Failures

| # | Bug | Impact | Root Cause |
|---|-----|--------|------------|
| 1 | **Look-ahead bias in trailing stop** | Fake 66-73% WR, PF 10+ | Used `close[j]` to update trail, then checked `low[j]` against it. Same-bar information leak. |
| 2 | **Stale day_groups after slicing** | Walk-forward produced -999 | Precomputed indices didn't update when dataframe was sliced by train/test mask. |
| 3 | **API 5,000 bar limit not detected** | Only 36 days of data used | No validation of unique dates after fetch. Silent truncation. |
| 4 | **Entry at exact breakout level** | Impossible fill assumption | Assumed entry at `or_high` exactly. Real breakout fills gap through the level. |
| 5 | **Zero slippage model** | 10-20x overstatement of returns | No slippage, no commission in v1 backtest. |

### 1.2 Financial Engineering Failures

| # | Issue | Impact | Why It Matters |
|---|-------|--------|----------------|
| 1 | **No edge source identified** | Strategy is random | Never asked "WHO is on the other side of this trade and WHY are they paying me?" (Ryan Wright, professional trader) |
| 2 | **Survivorship bias** | NVDA/AMD/PLTR/MRVL are all massive winners | Strategy is effectively "buy tech stocks that went up 300%+" — that's not ORB edge, that's long bias on winners. |
| 3 | **No regime filter** | Trades in choppy AND trending markets | ORB works in trending regimes, fails in choppy. No filter = noise. |
| 4 | **No volatility contraction filter** | Enters on random breakouts | Toby Crabel's documented edge uses NR7 (narrow range) filter. Breakout after contraction has edge; random breakout doesn't. |
| 5 | **Structural stop wrong** | 89.7% stopped out | Stop at opposite OR end is too wide for some symbols, too narrow for others. Needs volatility-adjusted sizing. |
| 6 | **No prop firm rule enforcement** | Would blow up in live trading | Daily DD limit (5%) not checked. Combined portfolio correlated drawdowns not modeled. |

### 1.3 The Core Question: Does ORB Have Edge?

**Academic evidence:**
- SSRN paper "Beat the Market" (2024): Intraday momentum on SPY has edge with proper implementation
- Toby Crabel (Crabel Capital Management, $10B+ AUM): ORB with NR7 filter has documented edge for 30+ years
- Reddit "17 strategies dead on MNQ/NQ": Basic ORB without filters has no edge after costs

**Conclusion:** Basic ORB without filters has **no edge**. ORB with volatility contraction filter (NR7/narrow range) has **documented edge** but it's small (PF 1.2-1.8, WR 48-55%).

---

## Part 2: What Actually Has Edge (Research Summary)

### 2.1 Documented Intraday Edges (2024-2025)

| Edge Source | Mechanism | Who Pays | Status |
|-------------|-----------|----------|--------|
| **Volatility contraction → expansion** | NR7/narrow range → breakout | Trapped traders on wrong side | Still works (Crabel Capital, 30+ years) |
| **News-driven liquidity withdrawal** | Liquidity pulls before announcements, rebuilds after | Late retail, systematic algos | Works but shrinking window |
| **Options dealer hedging (EOD)** | Gamma exposure forces dealer buying/selling | Options market makers | Works 2022+ but well-known |
| **Session structure** | London open creates direction, NY confirms | Trapped Asian session traders | Still works on forex/commodities |
| **Momentum continuation** | Strong opening range → continuation through day | Mean-reversion traders who fade too early | Conditional — needs trend regime |

### 2.2 What Doesn't Work

- Basic breakout without volatility filter (too many false breakouts)
- Mean reversion on 1h forex (too efficient)
- Indicator-based strategies (RSI, MACD, BB) without edge source
- Any strategy with WR > 65% and PF > 5 (almost certainly overfit or biased)

---

## Part 3: Fix Plan — 6 Phases

### Phase 1: Fix the Backtest Engine (Software Engineering)

**Objective:** Eliminate ALL look-ahead bias and implement realistic cost model.

**Changes required:**

```
1. SIGNAL GENERATION
   - Entry signal: Generated when bar CLOSES beyond OR level
   - Entry execution: NEXT bar's open (not current bar's close)
   - This adds 1 bar of latency — realistic for retail execution

2. TRAILING STOP
   - Trail update: Use bar[j-1] close (completed bar)
   - Trail check: bar[j] low/high (current bar)
   - This is the CORRECT order: know where stop is BEFORE checking if hit

3. SLIPPAGE MODEL
   - Breakout entry: 0.3-0.5% slippage (price gaps through level)
   - Stop exit: 0.2% slippage (gap through stop)
   - Limit exit: 0.1% slippage
   - Commission: 0.05% per side

4. POSITION SIZING
   - Risk per trade: 1% of account
   - Size = (Account × 1%) / (Entry - Stop)
   - Max 1 position per symbol
   - Max 4 positions total (portfolio limit)

5. PROP FIRM RULES
   - Check daily P&L after each trade
   - If daily loss > 3%: stop trading for the day
   - If total drawdown > 8%: stop trading entirely
   - Track consecutive losing days
```

**Files to modify:** `portfolio_backtest_v2.py` → `portfolio_backtest_v3.py`

### Phase 2: Add Volatility Contraction Filter (Financial Engineering)

**Objective:** Only trade breakouts after volatility contraction (the documented edge).

**Implementation:**

```
NR7 FILTER (Toby Crabel):
- Calculate 7-day range: max(high) - min(low) over last 7 bars
- NR7 day: today's range is the SMALLEST of the last 7 days
- Only take ORB trades on NR7 days
- Logic: Volatility contracts → expands. Breakout after contraction has edge.

ADDITIONAL FILTERS:
- ATR contraction: ATR(14) < ATR(14) from 5 days ago (volatility shrinking)
- Range contraction: Today's OR range < average OR range of last 10 days
- Volume expansion on breakout: Breakout bar volume > 1.5x average volume
```

**Why this works:**
- When volatility contracts, stop losses are tight (close to entry)
- When expansion comes, winners are large (volatility expands)
- Win rate may drop to 45-50%, but W/L ratio improves to 2-3x
- This is the documented Crabel edge

### Phase 3: Add Regime Filter

**Objective:** Only trade in trending regimes, avoid choppy markets.

**Implementation:**

```
REGIME DETECTION:
- 20-period SMA slope: positive = uptrend, negative = downtrend
- ADX > 20: trending market (trade ORB)
- ADX < 20: choppy market (skip ORB, or use mean reversion)
- VIX > 25: high volatility regime (wider stops, smaller size)

SESSION FILTER:
- Only trade during first 2 hours of session (9:30-11:30 ET)
- Avoid last hour (15:00-16:00 ET) — low edge, high noise
- Avoid first 5 minutes (9:30-9:35) — too much noise
```

### Phase 4: Fix Symbol Selection (Survivorship Bias)

**Objective:** Test on a broader universe to confirm edge isn't just long bias.

**Approach:**

```
1. CURRENT BIAS: All 4 symbols (NVDA, AMD, PLTR, MRVL) are massive winners
   - NVDA: +349% over test period
   - This means ANY long-biased strategy looks good

2. FIX: Add losers and range-bound stocks to test universe
   - Test on: AAPL, MSFT, GOOGL (large cap, less volatile)
   - Test on: META, AMZN (different character)
   - Test on: IWM (Russell 2000 ETF — small caps)
   - If strategy only works on NVDA/AMD/PLTR/MRVL = no edge, just long bias

3. ALSO: Test on short side
   - If ORB only works long (buying breakouts up), it's momentum beta, not edge
   - Real ORB should work on both long and short breakouts
```

### Phase 5: Walk-Forward Validation (Proper)

**Objective:** Confirm edge persists out-of-sample.

**Protocol:**

```
1. SPLIT: 2022-01 to 2023-06 (train), 2023-07 to 2024-07 (test)
2. OPTIMIZE on train: Grid search over NR7 filter params, ATR stops, trail configs
3. VALIDATE on test: Run best params on held-out data
4. ACCEPTANCE CRITERIA:
   - Train Sharpe > 0.3 AND Test Sharpe > 0.2
   - Train PF > 1.2 AND Test PF > 1.0
   - Test WR > 42%
   - Test return > 0%
   - Robustness = Test_Sharpe / Train_Sharpe > 0.5

5. MONTE CARLO: Bootstrap resample test trades 1000 times
   - 95th percentile drawdown < 15%
   - Median profit factor > 1.1
```

### Phase 6: Portfolio Construction

**Objective:** Build a properly diversified portfolio with risk management.

**Approach:**

```
1. POSITION SIZING: Risk parity
   - Each symbol gets equal risk allocation (1% per trade)
   - Correlation matrix: if symbols are >0.7 correlated, reduce allocation
   - Max 3 concurrent positions

2. DAILY RISK MANAGEMENT:
   - Max 3 trades per symbol per day
   - Max 6 trades total per day
   - If 2 consecutive losses: stop for the day
   - Daily loss limit: 3% of account

3. PROP FIRM COMPLIANCE:
   - Track daily drawdown in real-time
   - Emergency stop at 8% total drawdown
   - Minimum 10 trading days before profit target
```

---

## Part 4: Expected Realistic Outcomes

### If Edge Exists (optimistic)
- Win rate: 48-55%
- Profit factor: 1.3-2.0
- Win/loss ratio: 1.5-2.5x
- Sharpe: 0.3-0.7 (annualized)
- Max drawdown: 5-12%
- Trades per day: 1-3 across portfolio
- Hold time: 30min-4hrs

### If Edge Doesn't Exist (realistic)
- Win rate: 42-48%
- Profit factor: 0.8-1.1
- Sharpe: -0.2 to 0.2
- Strategy breaks even or loses slowly after costs

### The Honest Truth
**Most intraday strategies don't have edge.** The ones that do:
1. Require specific market conditions (volatility contraction)
2. Have small edge (PF 1.2-1.5, not 10+)
3. Go through losing periods
4. Require discipline to execute
5. Decay over time as others find the same pattern

---

## Part 5: Execution Order

| Step | Phase | Time Estimate | Depends On |
|------|-------|---------------|------------|
| 1 | Fix backtest engine (v3) | 2-3 hours | Nothing |
| 2 | Add NR7 filter | 1 hour | Step 1 |
| 3 | Add regime filter | 1 hour | Step 1 |
| 4 | Expand symbol universe | 30 min (data fetch) | Nothing |
| 5 | Run walk-forward validation | 1-2 hours | Steps 1-4 |
| 6 | Monte Carlo simulation | 30 min | Step 5 |
| 7 | Portfolio construction | 1 hour | Step 5 |
| 8 | Final report | 30 min | Steps 1-7 |

**Total estimated time: 7-10 hours**

---

## Part 6: Decision Points

After Phase 5 (walk-forward), we'll know:

**If test Sharpe > 0.3 and test PF > 1.2:**
→ Edge exists. Proceed to portfolio construction and prop firm evaluation.

**If test Sharpe is 0.0-0.3 and test PF is 0.9-1.2:**
→ Marginal edge. Consider if it's worth the effort. May work with better execution or different symbols.

**If test Sharpe < 0.0 or test PF < 0.9:**
→ No edge. Stop. Do not trade this strategy. The ORB approach on these symbols with OHLCV data does not have positive expected value.

---

## Appendix: Key References

1. **Toby Crabel** — "Day Trading with Short Term Price Patterns" (documented NR7 edge)
2. **Ryan Wright** — "Edge isn't yours" (understanding edge sources, who pays you)
3. **SSRN "Beat the Market"** (2024) — Intraday momentum on SPY with proper implementation
4. **SSRN "Structural Limits of OHLCV-Based Intraday Signals"** (2026) — Why most OHLCV strategies fail
5. **Reddit "17 strategies dead on MNQ/NQ"** — Real-world confirmation that basic ORB doesn't work

---

**DO NOT EXECUTE until this plan is reviewed and approved.**

---

## Part 7: Deep Research Findings — Quant Strategies with Documented Edge

**Date added:** 2026-08-07 (deep research follow-up)
**Sources:** Academic papers (arxiv, SSRN), professional trader insights, quant fund research

### 7.1 Regime-Conditional Factor Activation (HIGHEST PRIORITY)

**Source:** "Discovery of a 13-Sharpe OOS Factor" (arxiv, Nov 2025, NASA researcher)

**Key Insight:** Signals that appear weak on average become **extraordinarily powerful** when applied selectively during specific market conditions. The paper documents a cross-sectional equity factor achieving **out-of-sample Sharpe ratios exceeding 13** through regime-conditional signal activation.

**How it works:**
- Combine value + reversal signals (70% value, 30% 10-day reversal)
- Only activate when stock is in "drift regime": >60% positive days in trailing 63-day window
- Signal = BASE × REGIME (binary gate: 0 or 1)
- ~35% of stock-days qualify on average

**Results:**
- Annualized return: 158.6%
- Volatility: 12.0%
- Max drawdown: -11.9%
- Walk-forward validated over 20 years (2004-2024)
- 1,000 randomization trials, p-value < 0.001
- Sharpe > 7 across 30% parameter variations
- Near-zero factor exposure (R² < 3%)

**Why this matters for our system:**
- The ORB strategy's failure is partly because it trades in ALL regimes
- Adding a regime gate (only trade when symbol shows >60% positive days) could transform a mediocre strategy into a profitable one
- This is NOT overfitting — the regime filter is a structural insight about when signals work

**Implementation for our system:**
```
REGIME FILTER:
- Calculate UpFraction = % of positive days in trailing 63-day window
- Only take ORB trades when UpFraction > 0.55 (bullish drift regime)
- Skip trades when UpFraction < 0.45 (bearish regime)
- This should improve WR and PF by avoiding false breakouts in choppy/bearish markets
```

### 7.2 Volatility Risk Premium (Options Selling)

**Source:** Multiple academic papers, CME Group research, "Kelly, VIX, and Hybrid Approaches in Put-Writing" (2025)

**Key Insight:** Implied volatility **consistently exceeds** realized volatility. This is the "volatility risk premium" (VRP) — options sellers are compensated for bearing crash risk.

**Documented edge:**
- Selling SPY puts (30-delta, 30-45 DTE) has Sharpe ~0.5-0.8 over decades
- Edge is largest when IV percentile > 50% (sell when vol is elevated)
- Kelly-criterion sizing improves risk-adjusted returns

**Why this matters:**
- This is a STRUCTURAL edge — exists because of risk transfer (buyers pay premium for protection)
- Not easily arbitraged away because of crash risk
- Works alongside directional strategies for diversification

**Implementation for our system:**
```
OPTION SELLING OVERLAY:
- When VIX > 20: Sell 30-delta SPY puts (30-45 DTE)
- Size: 1-2% of account per trade
- Stop: Close if loss exceeds 2x premium received
- This adds a non-correlated return stream to the ORB strategy
```

### 7.3 Cross-Asset Momentum & Carry

**Source:** "Value and Momentum Everywhere" (Asness, Moskowitz, Pedersen, 2013, NYU Stern)

**Key Insight:** Value and momentum factors work across **ALL asset classes** — equities, currencies, bonds, commodities. The edges are cross-correlated, providing diversification.

**Documented edges:**
- Time-series momentum (trend following): Sharpe 0.5-1.0 across asset classes
- Carry (high-yield minus low-yield): Sharpe 0.4-0.7
- Value (cheap vs expensive): Sharpe 0.3-0.6
- Combining all three: Sharpe 1.0-1.5

**Why this matters:**
- Our system only trades US equities (single asset class)
- Adding forex/commodity momentum could provide non-correlated returns
- The XAG/USD strategy failed partly because we used mean reversion (wrong approach for trending commodity)

**Implementation for our system:**
```
CROSS-ASSET MOMENTUM:
- Trade XAG/USD, EUR/USD, CL with TREND FOLLOWING (not mean reversion)
- Use 20-day and 50-day SMA crossover
- Only trade in direction of 200-day SMA (regime filter)
- This is the OPPOSITE of what we tried (mean reversion on silver)
```

### 7.4 NR7 Volatility Contraction → Expansion (Already in Plan)

**Source:** Toby Crabel, "Day Trading with Short Term Price Patterns" (documented edge for 30+ years)

**Key Insight:** The ONLY documented intraday breakout edge. Breakouts after volatility contraction have higher continuation probability.

**Mechanism:**
- NR7 = today's range is smallest of last 7 days
- After NR7, breakout has ~55% continuation probability (vs 50% random)
- Edge is small but consistent over decades

**Already in Phase 2 of repair plan.**

### 7.5 LLM Sentiment Analysis (Emerging Edge)

**Source:** "From Deep Learning to LLMs: A survey of AI in Quantitative Investment" (2025), Citi Research

**Key Insight:** LLMs can extract predictive signals from news headlines, earnings calls, and social media faster than traditional NLP.

**Documented edge:**
- ChatGPT sentiment on earnings calls: Sharpe 0.3-0.5
- News headline sentiment: Sharpe 0.2-0.4
- Edge is time-decaying (works for minutes to hours after news)

**Why this matters for our system:**
- We could add a news sentiment filter to ORB trades
- Only take breakout trades when sentiment is aligned with breakout direction
- This requires API access to news feeds (not available with current data)

**Implementation (if news API available):**
```
SENTIMENT FILTER:
- Score news headlines for each symbol (bullish/bearish/neutral)
- Only take long ORB trades when sentiment > 0 (bullish)
- Only take short ORB trades when sentiment < 0 (bearish)
- Skip trades when sentiment is neutral or conflicting
```

### 7.6 Market Microstructure (Order Flow Imbalance)

**Source:** Multiple arxiv papers on LOB dynamics

**Key Insight:** Order flow imbalance (more market buys hitting ask vs sells hitting bid) predicts short-term price movement.

**Documented edge:**
- Order flow imbalance predicts next 1-5 minute returns with R² of 5-15%
- Edge decays rapidly (seconds to minutes)
- Requires Level 2 data (not available from our API)

**Why this matters:**
- This is the TRUE source of intraday edge — not OHLCV patterns
- Retail traders generally can't access this edge (requires co-location, Level 2 data)
- Our OHLCV-based ORB strategy is trying to capture a shadow of this edge

**Conclusion:** OHLCV-based intraday strategies have LIMITED edge. Real intraday edge comes from:
1. Order flow (need Level 2 data)
2. News flow (need real-time news API)
3. Volatility regime (achievable with OHLCV)
4. Structural patterns like NR7 (achievable with OHLCV)

### 7.7 Summary: Strategy Hierarchy by Edge Quality

| Strategy | Edge Source | Sharpe | Data Needed | Retail Feasible? |
|----------|-----------|--------|-------------|------------------|
| Regime-conditional factor | Behavioral bias amplification | 5-13 | Daily OHLCV | ✅ YES |
| Volatility risk premium | Risk transfer (structural) | 0.5-0.8 | Options data | ✅ YES |
| Cross-asset momentum | Risk premia persistence | 0.5-1.0 | Daily OHLCV | ✅ YES |
| NR7 breakout | Volatility contraction | 0.3-0.5 | Daily OHLCV | ✅ YES |
| LLM sentiment | Information asymmetry | 0.3-0.5 | News API | ⚠️ PARTIAL |
| Order flow imbalance | Microstructure | 0.5-2.0 | Level 2 data | ❌ NO |
| Basic ORB (no filter) | None documented | ~0 | OHLCV | ❌ NO EDGE |

### 7.8 Revised Strategy Recommendations

Based on deep research, the REPAIR PLAN should be updated as follows:

**PRIORITY 1: Add regime-conditional filter to ORB**
- This is the single highest-impact change
- Only trade when symbol is in drift regime (>55% positive days in 63-day window)
- Expected improvement: transform negative-PF strategy into positive-PF

**PRIORITY 2: Switch XAG/USD from mean reversion to trend following**
- Silver is a trending asset (documented in cross-asset momentum research)
- Our mean reversion approach was WRONG for this instrument
- Use SMA crossover with regime filter instead

**PRIORITY 3: Add NR7 volatility contraction filter**
- Already in Phase 2 of repair plan
- Combines with regime filter for double confirmation

**PRIORITY 4: Consider options selling overlay**
- If options data available, sell puts when IV > RV
- Non-correlated return stream
- Requires different data source

**PRIORITY 5: Consider cross-asset expansion**
- Trade momentum across equities, forex, commodities
- Reduces single-asset-class concentration risk
- Requires multi-asset data (partially available from our API)

---

## Part 8: Updated Execution Order

| Step | Phase | Time Estimate | Priority |
|------|-------|---------------|----------|
| 1 | Fix backtest engine (v3) | 2-3 hours | P0 |
| 2 | **Add regime-conditional filter** | 2 hours | **P1 (NEW)** |
| 3 | Add NR7 volatility contraction filter | 1 hour | P2 |
| 4 | Add regime filter (ADX) | 1 hour | P2 |
| 5 | **Fix XAG/USD: mean reversion → trend following** | 1 hour | **P1 (NEW)** |
| 6 | Expand symbol universe (survivorship bias) | 30 min | P3 |
| 7 | Run walk-forward validation | 1-2 hours | P1 |
| 8 | Monte Carlo simulation | 30 min | P2 |
| 9 | Portfolio construction | 1 hour | P2 |
| 10 | Final report | 30 min | - |

**Total estimated time: 9-12 hours**

---

## Part 9: Key Academic References

1. Singha, M. (2025) "Discovery of a 13-Sharpe OOS Factor: Drift Regimes Unlock Hidden Cross-Sectional Predictability" — arxiv.org/abs/2511.12490
2. Asness, Moskowitz, Pedersen (2013) "Value and Momentum Everywhere" — NYU Stern
3. Moskowitz, Ooi, Pedersen (2012) "Time Series Momentum" — Journal of Financial Economics
4. Crabel, T. (1990) "Day Trading with Short Term Price Patterns" — documented NR7 edge
5. Ryan Wright (2025) "Edge isn't yours: what actually works in modern markets" — professional trader insight
6. "Kelly, VIX, and Hybrid Approaches in Put-Writing on Index Options" (2025) — options VRP
7. "Beat the Market: An Effective Intraday Momentum Strategy for SPY" (2024) — SSRN
8. "Structural Limits of OHLCV-Based Intraday Signals" (2026) — SSRN
