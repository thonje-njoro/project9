# Momentum ORB Strategy — Replacement Symbol Candidates for TSLA & AMD

**Date:** 2026-08-06  
**Context:** Replacing TSLA (no edge found) and AMD (already works, keeping it) with additional symbols for intraday Momentum ORB backtesting.

---

## Why TSLA Failed

TSLA's intraday behavior is **regime-dependent and chaotic** — it whipsaws between trending and mean-reverting states unpredictably. The ATR-based entry + trailing stop that works on NVDA/AMD gets chopped up by TSLA's erratic opening ranges. The stock has high vol but lacks *consistent directional character*.

## What Makes ORB Work

From the successful NVDA/AMD results, the key characteristics are:
- **Strong intraday trending behavior** — once price breaks the opening range, it tends to keep going
- **Consistent volatility regime** — not swinging between dead calm and chaos
- **High institutional participation** — creates sustained directional moves, not retail-driven chop
- **Clean price action** — minimal gaps within the trading day, orderly candles

---

## Candidate Analysis

### 1. PLTR — Palantir Technologies ⭐ TOP PICK

| Attribute | Value |
|-----------|-------|
| Sector | AI / Defense Software / Data Analytics |
| Market Cap | ~$250B+ (large-cap) |
| Avg Daily Volume | ~50-60M shares (~$3-4B dollar volume) |
| Annual Volatility | ~45-55% |
| Beta | ~1.5 |
| Options | Very liquid, tight spreads |

**Why it would work:**
- Extremely high volume with strong retail AND institutional participation
- Clear trending character — PLTR has been in a sustained uptrend since mid-2023 with strong momentum legs
- AI/defense narrative creates catalyst-driven directional moves
- Price action is orderly; breakouts tend to follow through
- Similar profile to NVDA (AI narrative, high vol, trending)

**Risks:**
- Can gap significantly on news (earnings, government contracts)
- Elevated valuation creates potential for sharp reversals
- 2024 saw some parabolic moves that may be regime-specific

**Verdict: STRONG CANDIDATE** — Closest behavioral analog to NVDA. Must-test.

---

### 2. ARM — ARM Holdings ⭐ TOP PICK

| Attribute | Value |
|-----------|-------|
| Sector | Semiconductor IP / AI Chips |
| Market Cap | ~$150B+ (large-cap) |
| Avg Daily Volume | ~10-15M shares (~$1.5-2.5B dollar volume) |
| Annual Volatility | ~45-55% |
| Beta | ~1.5-2.0 |
| Options | Liquid |

**Why it would work:**
- Pure semiconductor IP play — benefits from AI capex cycle like NVDA
- Strong trending character since IPO (2023); persistent directional moves
- Institutional ownership growing steadily
- Cleaner price action than TSLA — more orderly intraday structure
- Volatility in the sweet spot (not too low, not chaotic)

**Risks:**
- Relatively recent IPO (Sept 2023) — less historical data for backtest
- Lower volume than PLTR/NVDA
- Can gap on earnings significantly

**Verdict: STRONG CANDIDATE** — Best semiconductor complement to NVDA. High conviction.

---

### 3. MSTR — MicroStrategy (Strategy Inc) ⭐ STRONG PICK

| Attribute | Value |
|-----------|-------|
| Sector | Bitcoin Treasury / Software |
| Market Cap | ~$50-80B (large-cap) |
| Avg Daily Volume | ~15-25M shares (~$1.5-3B dollar volume) |
| Annual Volatility | ~70-90% |
| Beta | ~2.5-3.0 |
| Options | Extremely liquid, very active options market |

**Why it would work:**
- **Massive intraday moves** — daily ranges of 3-8% are common
- Strong trending character — moves directionally with Bitcoin, which itself trends intraday
- Enormous options volume creates gamma-driven momentum
- Retail favorite with sustained volume
- The BTC correlation means it has a clear macro driver (unlike TSLA's chaos)

**Risks:**
- Volatility may be TOO high for the ATR-based strategy — need to test if the trailing stop gets hit too often
- Bitcoin-driven regime changes can be abrupt
- Not a "pure" equity — behaves more like a leveraged BTC play
- Higher vol than the 30-60% target range

**Verdict: STRONG CANDIDATE** — Volume and trending character are excellent. Test with wider ATR multipliers.

---

### 4. MRVL — Marvell Technology ✅ SOLID PICK

| Attribute | Value |
|-----------|-------|
| Sector | Semiconductors / AI Infrastructure |
| Market Cap | ~$60-70B (large-cap) |
| Avg Daily Volume | ~12-18M shares (~$800M-1.5B dollar volume) |
| Annual Volatility | ~40-50% |
| Beta | ~1.4-1.7 |
| Options | Liquid |

**Why it would work:**
- Pure-play AI infrastructure semiconductors (custom ASICs, networking)
- Strong trending character — driven by AI capex narrative similar to NVDA
- Volatility in the ideal 30-60% range
- Institutional-heavy ownership creates sustained directional moves
- Clean intraday price action

**Risks:**
- Lower volume than PLTR/NVDA
- Can be choppy during sector rotation periods
- Earnings reactions can be violent

**Verdict: SOLID CANDIDATE** — Good semiconductor complement. Worth testing.

---

### 5. COIN — Coinbase Global ⚠️ CAUTIOUS PICK

| Attribute | Value |
|-----------|-------|
| Sector | Cryptocurrency Exchange / Fintech |
| Market Cap | ~$50-70B (large-cap) |
| Avg Daily Volume | ~10-15M shares (~$1.5-2.5B dollar volume) |
| Annual Volatility | ~65-80% |
| Beta | ~2.5 |
| Options | Very liquid |

**Why it might work:**
- High volume, strong retail participation
- Can trend strongly during crypto bull runs
- Options market creates gamma-driven momentum

**Why it might NOT work:**
- Volatility is on the high end (65-80%) — similar concern to MSTR
- Highly regime-dependent — works during crypto uptrends, fails during consolidation
- Can be very choppy when BTC is range-bound
- More mean-reverting than trending during non-trending crypto periods

**Verdict: CAUTIOUS CANDIDATE** — Test but expect regime-dependent results. May behave like TSLA.

---

### 6. SMCI — Super Micro Computer ⚠️ CAUTIOUS PICK

| Attribute | Value |
|-----------|-------|
| Sector | AI Servers / Infrastructure |
| Market Cap | ~$19B (mid-cap) |
| Avg Daily Volume | ~50-55M shares (~$1.5-2B dollar volume) |
| Annual Volatility | ~60-80% |
| Beta | ~1.97 |
| Options | Liquid |

**Why it might work:**
- Extremely high volume for its cap size
- AI server narrative creates strong directional moves
- High beta = big intraday swings

**Why it might NOT work:**
- Accounting/audit concerns in 2024 created erratic, news-driven price action
- Very high volatility (60-80%) — may cause ATR stop-outs
- Price action can be chaotic — similar failure mode to TSLA
- Smaller cap ($19B) means more susceptible to manipulation

**Verdict: CAUTIOUS CANDIDATE** — High volume is appealing, but chaotic character is a red flag. Test with caution.

---

## Symbols Considered but Rejected

| Symbol | Reason for Rejection |
|--------|---------------------|
| **RIVN** (Rivian) | EV maker, ~$15B cap. Too range-bound, more mean-reverting than trending. Similar failure mode to TSLA. |
| **SOFI** (SoFi) | Fintech, ~$10B cap. Volatility ~40-50% but tends to be choppy/range-bound. Not strong trending character. |
| **CVNA** (Carvana) | Used cars, ~$15B cap. Volatility 80-100%+ — too chaotic. Short-squeeze dynamics create unpredictable regimes. |
| **SNOW** (Snowflake) | Data cloud, ~$50B cap. Volatility declining, becoming more range-bound. Not enough directional character. |
| **CRWD** (CrowdStrike) | Cybersecurity, ~$70B cap. Volatility ~30-35% — on the low end. Trends well but moves may be too small for ORB. |
| **MDB** (MongoDB) | Database, ~$25B cap. Volatility ~40-50%. Decent but lower conviction than top picks. |
| **AVGO** (Broadcom) | Semis, ~$700B+ cap. Becoming too large/stable. Volatility dropping below 30%. |
| **META** (Meta) | Big tech, ~$1.5T cap. Too large, vol too low for ORB edge. |
| **PANW** (Palo Alto) | Cybersecurity, ~$110B cap. Volatility ~30%. Below threshold. |

---

## Final Recommendation — 5 Symbols to Test

**Priority 1 (High Confidence):**
1. **PLTR** — Closest behavioral match to NVDA. AI narrative, massive volume, strong trending. Must-test.
2. **ARM** — Best semiconductor complement. Clean price action, ideal volatility range.

**Priority 2 (Strong but needs ATR tuning):**
3. **MSTR** — Incredible trending character and volume, but volatility is higher than target. Test with 2-3x ATR multiplier.
4. **MRVL** — Solid AI semi play with good volatility. Reliable but lower edge potential than PLTR/ARM.

**Priority 3 (Test with caution):**
5. **COIN** — Good volume and trending during crypto uptrends, but regime-dependent. May need additional filters.

**Note on AMD:** AMD already works (63.6% WR, +4.5%, PF 5.6). Keep it in the portfolio. The 5 symbols above are replacements for TSLA only (and potentially additions alongside NVDA+AMD).

---

## Suggested Testing Approach

1. **Run the same Momentum ORB strategy** (10-30 min OR, ATR entry, ATR trailing stop, 20-SMA trend filter) on all 5 candidates using 1-min data from 2022-2024
2. **For MSTR and COIN:** Also test with 1.5x and 2x ATR stop multipliers (wider stops for higher-vol names)
3. **Compare:** Win rate, profit factor, Sharpe, max drawdown, number of trades
4. **Target:** At least 3 symbols with PF > 2.0 and WR > 55% to build a diversified ORB portfolio
5. **Portfolio sizing:** If multiple work, equal-weight across the ORB basket for diversification
