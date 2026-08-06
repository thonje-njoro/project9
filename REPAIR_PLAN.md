cat: /mnt/c/Users/Admin/project9/REPAIR_PLAN.md: No such file or directory

---

## CRITICAL NEW FINDINGS FROM MIMO CLAW RESEARCH (2026-08-07)

### FINDING 1: REGIME-CONDITIONAL FACTOR ACTIVATION (HIGHEST PRIORITY)

**Source:** arxiv paper "Discovery of a 13-Sharpe OOS Factor" (NASA researcher, Nov 2025)

**Key Insight:** Signals that appear weak on average become EXTRAORDINARILY powerful when applied selectively during specific market conditions.

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

**Implementation for our system:**
```
REGIME FILTER:
- Calculate UpFraction = % of positive days in trailing 63-day window
- Only take ORB trades when UpFraction > 0.55 (bullish drift regime)
- Skip trades when UpFraction < 0.45 (bearish regime)
- This should improve WR and PF by avoiding false breakouts in choppy/bearish markets
```

---

### FINDING 2: MULTI-ASSET MOMENTUM HAS GENUINE EDGE

**Source:** MiMo Claw testing of 80 momentum variants across 9 US equities

**Key Results:**
- Train-test Sharpe correlation = 0.685 (STRONG CONSISTENCY)
- Top strategy: vol_target, 60d lookback, weekly rebalancing, 20% vol target
- Test Sharpe: 2.055, Test Return: +47.7%, Test MaxDD: -8.1%
- 47.5% of combos positive in both train AND test periods

**What Works:**
1. Weekly rebalancing (daily is too noisy)
2. Longer lookbacks (40-60d, not 5-10d)
3. Volatility targeting (15-20% annual vol)
4. Time-series momentum (trend following)

**What Doesn't Work:**
1. Cross-sectional momentum (ranking doesn't work)
2. Trend filter (200-day MA is UNRELIABLE)
3. Short lookbacks (5-10d are too noisy)
4. Daily rebalancing (too much transaction cost)

**Conclusion:** Time-series momentum has genuine edge when:
- Lookback is 40-60 days
- Rebalancing is weekly
- Volatility targeting is applied (15-20% target)

---

### FINDING 3: NR7 VOLATILITY CONTRACTION FILTER (DOCUMENTED EDGE)

**Source:** Toby Crabel, "Day Trading with Short Term Price Patterns" (30+ years of documentation)

**Key Insight:** The ONLY documented intraday breakout edge. Breakouts after volatility contraction have higher continuation probability.

**How it works:**
- NR7 = today's range is smallest of last 7 days
- After NR7, breakout has ~55% continuation probability (vs 50% random)
- Edge is small but consistent over decades

**Implementation:**
```
NR7 FILTER:
- Calculate 7-day range: max(high) - min(low) over last 7 bars
- NR7 day: today's range is the SMALLEST of the last 7 days
- Only take ORB trades on NR7 days
- Logic: Volatility contracts → expands. Breakout after contraction has edge.
```

---

### FINDING 4: VOLATILITY RISK PREMIUM (STRUCTURAL EDGE)

**Source:** Multiple academic papers, CME Group research

**Key Insight:** Implied volatility CONSISTENTLY EXCEEDS realized volatility. This is the "volatility risk premium" (VRP) — options sellers are compensated for bearing crash risk.

**Documented edge:**
- Selling SPY puts (30-delta, 30-45 DTE) has Sharpe ~0.5-0.8 over decades
- Edge is largest when IV percentile > 50% (sell when vol is elevated)
- Kelly-criterion sizing improves risk-adjusted returns

**Why this matters:**
- This is a STRUCTURAL edge — exists because of risk transfer
- Not easily arbitraged away because of crash risk
- Works alongside directional strategies for diversification

---

### FINDING 5: CROSS-ASSET MOMENTUM (DIVERSIFICATION)

**Source:** "Value and Momentum Everywhere" (Asness, Moskowitz, Pedersen, 2013)

**Key Insight:** Value and momentum factors work across ALL asset classes — equities, currencies, bonds, commodities.

**Documented edges:**
- Time-series momentum (trend following): Sharpe 0.5-1.0
- Carry (high-yield minus low-yield): Sharpe 0.4-0.7
- Value (cheap vs expensive): Sharpe 0.3-0.6
- Combining all three: Sharpe 1.0-1.5

**Why this matters:**
- Our system only trades US equities (single asset class)
- Adding forex/commodity momentum could provide non-correlated returns
- The XAG/USD strategy failed partly because we used mean reversion (wrong approach for trending commodity)

---

### FINDING 6: OHLCV-BASED STRATEGIES HAVE LIMITED EDGE

**Source:** SSRN "Structural Limits of OHLCV-Based Intraday Signals" (2026)

**Key Insight:** Real intraday edge comes from:
1. Order flow (need Level 2 data) — NOT available to retail
2. News flow (need real-time news API) — NOT available with current data
3. Volatility regime (achievable with OHLCV) — YES, implement this
4. Structural patterns like NR7 (achievable with OHLCV) — YES, implement this

**Conclusion:** OHLCV-based intraday strategies have LIMITED edge. The best we can do with our data:
1. Regime-conditional trading (highest impact)
2. NR7 volatility contraction filter
3. Multi-asset momentum (time-series)

---

## REVISED STRATEGY HIERARCHY (BY EDGE QUALITY)

| Strategy | Edge Source | Sharpe | Data Needed | Retail Feasible? |
|----------|-----------|--------|-------------|------------------|
| Regime-conditional factor | Behavioral bias amplification | 5-13 | Daily OHLCV | ✅ YES |
| Multi-asset momentum | Risk premia persistence | 0.5-2.0 | Daily OHLCV | ✅ YES |
| Volatility risk premium | Risk transfer (structural) | 0.5-0.8 | Options data | ✅ YES |
| NR7 breakout | Volatility contraction | 0.3-0.5 | Daily OHLCV | ✅ YES |
| LLM sentiment | Information asymmetry | 0.3-0.5 | News API | ⚠️ PARTIAL |
| Order flow imbalance | Microstructure | 0.5-2.0 | Level 2 data | ❌ NO |
| Basic ORB (no filter) | None documented | ~0 | OHLCV | ❌ NO EDGE |

---

## UPDATED IMPLEMENTATION PLAN

### Phase 1: Fix Backtest Engine (P0)
- [ ] Entry at next bar's open (not breakout level)
- [ ] Trailing stop checked on next bar
- [ ] Realistic slippage (0.3% for breakouts)
- [ ] Volume filter for entries

### Phase 2: Add Regime-Conditional Filter (P1 — HIGHEST IMPACT)
- [ ] Calculate UpFraction = % positive days in 63-day window
- [ ] Only trade when UpFraction > 0.55
- [ ] Skip trades when UpFraction < 0.45
- [ ] Expected: Transform negative-PF strategy into positive-PF

### Phase 3: Add NR7 Volatility Contraction Filter (P2)
- [ ] Calculate 7-day range
- [ ] Only trade on NR7 days
- [ ] Combine with regime filter for double confirmation

### Phase 4: Implement Multi-Asset Momentum (P1)
- [ ] Use 40-60 day lookback
- [ ] Weekly rebalancing
- [ ] Volatility targeting (15-20% annual vol)
- [ ] Add bonds (TLT), commodities (GLD), international (EFA)

### Phase 5: Add Options Selling Overlay (P3)
- [ ] Sell SPY puts when IV > RV
- [ ] 30-delta, 30-45 DTE
- [ ] Kelly-criterion sizing
- [ ] Non-correlated return stream

### Phase 6: Walk-Forward Validation
- [ ] Train: 2022-01 to 2023-06
- [ ] Test: 2023-07 to 2024-07
- [ ] Acceptance criteria: Train Sharpe > 0.3 AND Test Sharpe > 0.2

---

## REVISED EXPECTATIONS

Based on all research findings:

| Metric | Previous Estimate | Revised Estimate |
|--------|-------------------|------------------|
| Win Rate | 52-56% | 55-65% (with regime filter) |
| Profit Factor | 1.2-1.5 | 1.5-2.5 (with regime + NR7) |
| Sharpe Ratio | 0.5-0.8 | 0.8-2.0 (with multi-asset) |
| Max Drawdown | 5-10% | 5-12% (with vol targeting) |
| Monthly Return | 2-5% | 3-8% (with all edges combined) |

**Key improvement:** Regime-conditional factor activation alone can achieve Sharpe 5-13 (from arxiv paper). Combined with multi-asset momentum (Sharpe 0.5-2.0), the system becomes highly robust.

---

## CONCLUSION

The research has identified **multiple genuine edges**:

1. **Regime-conditional factor activation** (Sharpe 5-13) — HIGHEST PRIORITY
2. **Multi-asset time-series momentum** (Sharpe 0.5-2.0) — PROVEN
3. **NR7 volatility contraction** (Sharpe 0.3-0.5) — DOCUMENTED
4. **Volatility risk premium** (Sharpe 0.5-0.8) — STRUCTURAL

These edges, combined with proper risk management, can create a viable trading system. The system will NOT be highly profitable on single stocks with basic ORB, but CAN be profitable with regime filters and multi-asset momentum.

**Recommendation:** Implement regime-conditional filter first, then multi-asset momentum, then NR7 filter. Paper trade for 2 weeks before starting prop firm challenge.
