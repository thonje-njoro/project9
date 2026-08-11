# MiMo Claw Prompt — Find 6 Diversification Instruments + Strategies

## OBJECTIVE

Find **6 additional instruments** to trade alongside Copper (XCU/USD) for portfolio diversification. For each instrument, find and validate a profitable strategy that passes ALL acceptance criteria.

**The loop must continue until 6 passing instruments are found or data sources are exhausted.**

---

## CONTEXT

We already have a working strategy:
- **XCU VWAP Reversion** (Copper): WR=76%, PF=2.31, Sharpe=5.21, DD=14.8%
- Passed Monte Carlo: P(DD<=20%)=71%, P(Profitable)=94.6%
- Deflated Sharpe: 2.631 (>1.96 threshold)

**Problem:** Single instrument = concentration risk. We need 6 more uncorrelated instruments.

---

## STEP 1: INSTRUMENT CANDIDATE SELECTION

### Asset Classes to Search

**1. Commodities (3 candidates)**
- Gold (XAU/USD)
- Silver (XAG/USD)
- Platinum (XPT/USD)
- Palladium (XPD/USD)
- Crude Oil (CL or WTI/USD)
- Natural Gas (NG or NATGAS/USD)
- Wheat (ZW)
- Corn (ZC)
- Soybeans (ZS)
- Coffee (KC)
- Sugar (SB)
- Cotton (CT)

**2. Forex (2 candidates)**
- EUR/USD
- GBP/USD
- USD/JPY
- AUD/USD
- NZD/USD
- USD/CHF
- EUR/GBP
- EUR/JPY

**3. Indices (2 candidates)**
- S&P 500 (SPX or US500)
- NASDAQ (NAS100 or US100)
- Dow Jones (US30)
- Russell 2000 (US2000)
- DAX (GER40)
- Nikkei (JPN225)
- FTSE (UK100)

**4. Bonds (1 candidate)**
- US 10-Year (ZN)
- US 30-Year (ZB)
- German Bund (BUND)

### Selection Criteria

For each instrument, check:
1. **Data availability** — At least 3 years of daily data from LSE API
2. **Liquidity** — Sufficient volume for prop firm trading
3. **Volatility** — Enough range for profitable trading (not too low)
4. **Correlation to XCU** — Must be < 0.5 correlation to copper
5. **Session overlap** — Trading hours should overlap with XCU for portfolio management

---

## STEP 2: STRATEGY RESEARCH

For each candidate instrument, test ALL these strategies:

### Tier 1: Mean Reversion (highest probability)
1. **VWAP Reversion** — Buy below VWAP-1σ, sell at VWAP
2. **Bollinger Band Reversion** — Buy at lower band, sell at mid
3. **RSI Oversold/Overbought** — Buy RSI<30, sell RSI>70
4. **Z-Score Mean Reversion** — Buy z<-2, sell z>0
5. **Keltner Channel Reversion** — Buy at lower band

### Tier 2: Trend Following
6. **SMA Crossover** — Fast/Slow SMA with trend filter
7. **EMA Crossover** — Fast/Slow EMA
8. **Donchian Channel Breakout** — N-day high/low breakout
9. **Turtle Trading** — 20-day breakout, 10-day exit
10. **Momentum** — Rate of change (ROC)

### Tier 3: Breakout
11. **NR7 Breakout** — Breakout after narrowest range in 7 days
12. **Volatility Breakout** — Breakout after ATR contraction
13. **Inside Bar Breakout** — Breakout of inside bar pattern

### Tier 4: Session-Based (for forex/commodities)
14. **London Session Breakout** — Breakout of London open range
15. **NY Session Momentum** — Momentum continuation in NY
16. **Asian Range Breakout** — Breakout of Asian session range

### Tier 5: Multi-Timeframe
17. **Daily Trend + 4H Entry** — Use daily for direction, 4H for entry
18. **Weekly Trend + Daily Entry** — Use weekly for direction, daily for entry

---

## STEP 3: PARAMETER OPTIMIZATION

For each strategy, run grid search:

**Mean Reversion:**
```
lookback: [10, 15, 20, 25, 30, 50]
entry_mult: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
exit_mult: [0.0, 0.25, 0.5]
```

**Trend Following:**
```
fast: [5, 10, 20, 50]
slow: [20, 50, 100, 200]
atr_stop: [1.0, 1.5, 2.0, 2.5]
```

**Breakout:**
```
lookback: [5, 10, 20, 50]
atr_mult: [0.5, 1.0, 1.5, 2.0]
```

---

## STEP 4: WALK-FORWARD VALIDATION

For each strategy with passing optimization:

**Split (3 windows):**
- Train: 2019-01-01 to 2021-12-31 → Test: 2022-01-01 to 2022-12-31
- Train: 2019-01-01 to 2022-12-31 → Test: 2023-01-01 to 2023-12-31
- Train: 2019-01-01 to 2023-12-31 → Test: 2024-01-01 to 2024-12-31

**Pass if:**
- Test Sharpe > 0.2 in ALL 3 windows
- Test PF > 1.0 in ALL 3 windows
- Test WR > 45% in ALL 3 windows
- Test Return > 0% in ALL 3 windows

---

## STEP 5: MONTE CARLO SIMULATION

For each strategy that passes walk-forward:

**10,000 bootstrap iterations:**
- P(WR >= 60%) > 50%
- P(PF >= 1.3) > 50%
- P(DD <= 20%) > 70%
- P(DD <= 30%) > 90%
- P(Profitable) > 70%

---

## STEP 6: COMPREHENSIVE QUANT TESTS

For each passing strategy:

1. **Deflated Sharpe** — Account for multiple testing
2. **MinBTL** — Minimum backtest length
3. **Slippage Sensitivity** — Breakeven at 10+ bps
4. **Regime Analysis** — Profitable in 2/3 regimes
5. **Correlation to XCU** — Must be < 0.5
6. **Equity Curve Quality** — Ulcer Index < 10
7. **Trade Sequence Sensitivity** — Not dependent on outliers

---

## ACCEPTANCE CRITERIA

An instrument/strategy PASSES if ALL of these are met:

### Must Pass:
- [ ] Win Rate >= 60%
- [ ] Profit Factor >= 1.3
- [ ] Sharpe >= 0.5
- [ ] Max Drawdown <= 15%
- [ ] Total Trades >= 30
- [ ] Walk-Forward: 3/3 windows pass
- [ ] Monte Carlo: P(DD<=20%) > 70%
- [ ] Monte Carlo: P(Profitable) > 70%
- [ ] Correlation to XCU < 0.5
- [ ] Slippage breakeven > 10 bps

### Should Pass (at least 5/7):
- [ ] Deflated Sharpe > 1.96
- [ ] MinBTL < actual trades
- [ ] Profitable in 2/3 regimes
- [ ] Ulcer Index < 10
- [ ] Not dependent on top 5% trades
- [ ] Day-of-week consistency
- [ ] Session time consistency

---

## ITERATION PROTOCOL

```
INSTRUMENT_COUNT = 0
PASSING_INSTRUMENTS = []

FOR each asset_class in [commodities, forex, indices, bonds]:
    FOR each candidate in asset_class:
        IF INSTRUMENT_COUNT >= 6:
            BREAK
        
        1. Fetch data from LSE API
        2. Check correlation to XCU
        3. IF correlation >= 0.5: SKIP
        
        FOR each strategy in [Tier 1, Tier 2, Tier 3, Tier 4, Tier 5]:
            4. Run parameter optimization
            5. Run walk-forward validation
            6. Run Monte Carlo simulation
            7. Run quant tests
            
            IF ALL acceptance criteria met:
                ADD to PASSING_INSTRUMENTS
                INSTRUMENT_COUNT += 1
                PRINT "FOUND: {instrument} with {strategy}"
                BREAK (move to next instrument)
            
            PRINT "  {strategy} failed: {reason}"
        
        IF no strategy passed:
            PRINT "  No passing strategy for {instrument}"

PRINT "\nFINAL PORTFOLIO:"
FOR each instrument in PASSING_INSTRUMENTS:
    PRINT "  {instrument}: {strategy} (WR={wr}%, PF={pf}, Sharpe={sharpe})"
```

---

## DATA SOURCE

Use **London Strategic Edge API** for all data:

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

**Available symbols on LSE:**
- XAU/USD, XAG/USD, XPT/USD, XPD/USD
- EUR/USD, GBP/USD, USD/JPY, AUD/USD
- SPX, NAS100, US30, GER40
- CL (crude oil), NG (natural gas)

---

## OUTPUT FORMAT

### For each passing instrument:
```
INSTRUMENT: {symbol}
├── Asset Class: {commodity/forex/index/bond}
├── Strategy: {strategy_name}
├── Parameters: {key: value, ...}
├── Optimization:
│   ├── WR: X%
│   ├── PF: X.XX
│   ├── Sharpe: X.XX
│   ├── MaxDD: X%
│   └── Trades: N
├── Walk-Forward:
│   ├── Window 1: PASS/FAIL
│   ├── Window 2: PASS/FAIL
│   └── Window 3: PASS/FAIL
├── Monte Carlo:
│   ├── P(DD<=20%): X%
│   ├── P(Profitable): X%
│   └── Median Sharpe: X.XX
├── Quant Tests:
│   ├── Deflated Sharpe: X.XX
│   ├── Correlation to XCU: X.XX
│   ├── Slippage breakeven: Xbps
│   └── Regime pass: X/3
└── Prop Firm Assessment:
    ├── FTMO (10% target, 10% DD): PASS/FAIL
    ├── The5ers (8% target, 6% DD): PASS/FAIL
    └── FundingPips (8% target, 10% DD): PASS/FAIL
```

### Final Portfolio Summary:
```
DIVERSIFIED PORTFOLIO (7 instruments)
├── 1. XCU/USD — VWAP Reversion (WR=76%, PF=2.31)
├── 2. {symbol} — {strategy} (WR=X%, PF=X.XX)
├── 3. {symbol} — {strategy} (WR=X%, PF=X.XX)
├── 4. {symbol} — {strategy} (WR=X%, PF=X.XX)
├── 5. {symbol} — {strategy} (WR=X%, PF=X.XX)
├── 6. {symbol} — {strategy} (WR=X%, PF=X.XX)
├── 7. {symbol} — {strategy} (WR=X%, PF=X.XX)
├── Portfolio Correlation Matrix: [7x7]
├── Expected Portfolio Sharpe: X.XX
├── Expected Portfolio DD: X%
└── Prop Firm Pass Probability: X%
```

---

## STOP CONDITIONS

Stop the loop when:
1. **6 passing instruments found** — Portfolio complete
2. **All candidates exhausted** — Report how many found
3. **4 hours elapsed** — Report progress
4. **100 strategies tested per instrument** — Move to next instrument

---

## CRITICAL NOTES

1. **Correlation is key** — Reject any instrument with XCU correlation >= 0.5
2. **Walk-forward is mandatory** — No in-sample-only results accepted
3. **Be honest** — If no passing strategy found, report it
4. **Save everything** — JSON files for later analysis
5. **Log progress** — Print every instrument/strategy tested
6. **Realistic costs** — Include 0.05% commission + 0.05% slippage
7. **Session times matter** — London/NY overlap preferred
8. **News events** — Test impact of NFP, FOMC, CPI on each instrument

---

## EXECUTION TIME ESTIMATE

| Phase | Time |
|-------|------|
| Data fetching | 30-60 min |
| Strategy testing (per instrument) | 15-30 min |
| Walk-forward (per instrument) | 5-10 min |
| Monte Carlo (per instrument) | 5-10 min |
| Quant tests (per instrument) | 5-10 min |
| **Total for 6 instruments** | **3-4 hours** |

Fits within MiMo Claw session limit.
