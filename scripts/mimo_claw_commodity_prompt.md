# MiMo Claw Prompt — Commodity Strategy Optimization Loop

## OBJECTIVE

Find a profitable trading strategy for **4 commodities** (Gold, Silver, Copper, + 1 commodity of your choice) that:
- Win rate >= 65%
- Profit factor >= 1.5
- Sharpe ratio >= 0.5
- Max drawdown <= 15%
- Passes prop firm challenge rules (FTMO, The5ers, FundingPips)

## DATA SOURCE

Use the **London Strategic Edge API** for all data fetching.

**API Details:**
- Base URL: `https://api.londonstrategicedge.com/vault`
- API Key: `lse_live_f4c9a7419371ecdd9365e146247b0289`
- Rate limit: 10 downloads/hour, 1M rows max per download
- Python client: `pip install lse-data`
- Timeframes available: `1m`, `5m`, `15m`, `1h`, `4h`, `1d`

**Data Fetching Example:**
```python
import requests

def fetch_candles(symbol, timeframe, start, end):
    url = "https://api.londonstrategicedge.com/vault/candles"
    params = {"symbol": symbol, "timeframe": timeframe, "start": start, "end": end}
    headers = {"x-api-key": "lse_live_f4c9a7419371ecdd9365e146247b0289"}
    response = requests.get(url, params=params, headers=headers, timeout=60)
    return response.json()
```

## SYMBOLS TO TRADE

1. **XAU/USD** (Gold) — Primary target
2. **XAG/USD** (Silver) — Secondary target
3. **XCU/USD** (Copper) — Tertiary target
4. **[YOU CHOOSE]** — Pick one from: Platinum (XPT/USD), Palladium (XPD/USD), Natural Gas (NG), Crude Oil (CL), or another commodity with good liquidity

## STRATEGIES TO TEST

Test ALL of the following strategies, in order of priority:

### Tier 1: Trend Following (Highest probability for commodities)
1. **SMA Crossover** — Fast/Slow SMA crossover with trend filter
2. **EMA Crossover** — Fast/Slow EMA crossover with trend filter
3. **Donchian Channel Breakout** — Breakout of N-day high/low
4. **Turtle Trading** — 20-day breakout with 10-day exit
5. **Momentum** — Rate of change (ROC) with trend filter

### Tier 2: Mean Reversion (Works in range-bound markets)
6. **Bollinger Band Reversion** — Buy at lower band, sell at upper band
7. **RSI Oversold/Overbought** — Buy when RSI < 30, sell when RSI > 70
8. **Z-Score Mean Reversion** — Buy when z-score < -2, sell when z-score > 2
9. **VWAP Reversion** — Buy when price < VWAP, sell when price > VWAP

### Tier 3: Breakout Strategies
10. **NR7 Breakout** — Breakout after narrowest range in 7 days
11. **Volatility Breakout** — Breakout after volatility contraction
12. **Opening Range Breakout (ORB)** — Breakout of first hour's range
13. **Session Breakout** — Breakout of Asian/London/NY session range

### Tier 4: Pattern-Based
14. **Inside Bar Breakout** — Breakout of inside bar pattern
15. **Engulfing Pattern** — Bullish/bearish engulfing candles
16. **Pin Bar Reversal** — Rejection at key levels

### Tier 5: Multi-Timeframe
17. **Multi-Timeframe Momentum** — Daily trend + 4H entry + 1H exit
18. **Multi-Timeframe Mean Reversion** — Daily trend + 4H oversold entry

### Tier 6: Session-Based (For forex/commodities)
19. **London Session Breakout** — Breakout of London open range
20. **NY Session Momentum** — Momentum continuation in NY session
21. **Asian Range Breakout** — Breakout of Asian session range

## PARAMETER GRIDS

For each strategy, test these parameter combinations:

### Trend Following
- Fast SMA: [5, 10, 20, 50]
- Slow SMA: [20, 50, 100, 200]
- Trend filter: [None, SMA 200, SMA 100]
- ATR stop: [1.0, 1.5, 2.0, 2.5, 3.0]

### Mean Reversion
- BB period: [10, 20, 50]
- BB std: [1.5, 2.0, 2.5]
- RSI period: [7, 14, 21]
- RSI oversold: [20, 25, 30]
- RSI overbought: [70, 75, 80]
- Z-score threshold: [1.5, 2.0, 2.5]

### Breakout
- Lookback period: [5, 10, 20, 50]
- ATR multiplier: [0.5, 1.0, 1.5, 2.0]
- Volume filter: [None, 1.5x, 2.0x average]

### Session-Based
- Session start hour: [0, 7, 13, 21] (UTC)
- Session end hour: [7, 13, 21, 24] (UTC)
- Breakout threshold: [0, 0.1%, 0.2%, 0.5% of range]

## WALK-FORWARD VALIDATION

Use proper walk-forward validation with:

**Split:**
- Train: 2019-01-01 to 2022-12-31 (~4 years)
- Test: 2023-01-01 to 2025-12-31 (~3 years)

**Process:**
1. Optimize parameters on train period
2. Run best parameters on test period
3. Check if test performance is within 50% of train performance (robustness)
4. If robust, keep. If not, discard.

**Acceptance Criteria:**
- Train Sharpe > 0.3
- Test Sharpe > 0.2
- Train PF > 1.2
- Test PF > 1.0
- Test WR > 42%
- Test return > 0%
- Robustness = Test_Sharpe / Train_Sharpe > 0.5

## PROP FIRM RULES

The strategy must pass these prop firm rules:

### FTMO 2-Step
- Profit target: 10% (Phase 1), 5% (Phase 2)
- Max drawdown: 10%
- Daily drawdown: 5%
- Minimum trading days: 10
- Maximum trading days: 30 (Phase 1), 60 (Phase 2)

### The5ers High Stakes
- Profit target: 8%
- Max drawdown: 6%
- Daily drawdown: 3%
- Minimum trading days: 10

### FundingPips
- Profit target: 8%
- Max drawdown: 10%
- Daily drawdown: 5%
- Minimum trading days: 5

## ITERATIVE LOOP STRUCTURE

Run the following loop until acceptance criteria are met:

```
LOOP 1: SYMBOL SELECTION
  For each symbol in [XAU/USD, XAG/USD, XCU/USD, YOUR_CHOICE]:
    LOOP 2: STRATEGY TYPE
      For each strategy in [Tier 1, Tier 2, Tier 3, Tier 4, Tier 5, Tier 6]:
        LOOP 3: PARAMETER GRID
          For each parameter combination:
            1. Fetch data from LSE API
            2. Calculate indicators
            3. Generate signals
            4. Run walk-forward validation
            5. Calculate metrics (WR, PF, Sharpe, MaxDD)
            6. Check acceptance criteria
            7. If PASS: save to results
            8. If FAIL: continue to next combination
          
          END LOOP 3
        
        END LOOP 2
      
      END LOOP 1
    
    If any strategy passes:
      1. Run Monte Carlo simulation (1000 paths)
      2. Calculate deflated Sharpe ratio
      3. Run prop firm simulation
      4. Generate final report
    
    If no strategy passes:
      1. Report findings
      2. Suggest new approaches
      3. Consider different symbols or timeframes
```

## STOP CONDITIONS

Stop the loop if:
1. A strategy passes ALL acceptance criteria
2. 10,000 parameter combinations tested without success
3. 4 hours of runtime elapsed
4. All symbols and strategies exhausted

## OUTPUT FORMAT

Generate a final report with:

### For each symbol:
```
SYMBOL: XAU/USD
├── Best Strategy: [Strategy Name]
├── Parameters: [Key: Value, ...]
├── Train Metrics:
│   ├── Trades: N
│   ├── Win Rate: X%
│   ├── Profit Factor: X.XX
│   ├── Sharpe: X.XX
│   ├── Max Drawdown: -X%
│   └── Total Return: +X%
├── Test Metrics:
│   ├── Trades: N
│   ├── Win Rate: X%
│   ├── Profit Factor: X.XX
│   ├── Sharpe: X.XX
│   ├── Max Drawdown: -X%
│   └── Total Return: +X%
├── Robustness: X.XX
├── Prop Firm Pass:
│   ├── FTMO: PASS/FAIL
│   ├── The5ers: PASS/FAIL
│   └── FundingPips: PASS/FAIL
└── Monte Carlo:
    ├── 95th percentile DD: -X%
    ├── Median PF: X.XX
    └── Probability of ruin: X%
```

### Summary:
```
PORTFOLIO SUMMARY
├── Symbols with passing strategies: N/4
├── Best symbol: [Symbol]
├── Best strategy: [Strategy]
├── Expected monthly return: X%
├── Expected max drawdown: -X%
└── Prop firm ready: YES/NO
```

## ADDITIONAL REQUIREMENTS

1. **Save all data** to parquet files for later analysis
2. **Save all results** to JSON for comparison
3. **Log every iteration** with timestamp and progress
4. **Handle API errors gracefully** (retry on failure, skip if unavailable)
5. **Use realistic costs**: 0.05% commission per side, 0.1% slippage
6. **Account for spread**: Use average spread for each symbol
7. **Time zone handling**: All times in UTC, convert to local session times

## CRITICAL NOTES

1. **Commodities trend strongly** — Prioritize trend-following strategies
2. **Gold and Silver are correlated** — Don't double-count signals
3. **Copper is a leading indicator** — May have different characteristics
4. **Session times matter** — London and NY sessions have different behavior
5. **News events cause spikes** — Filter out NFP, FOMC, CPI days
6. **Use 1H or 4H timeframes** — Lower timeframes are noisy for commodities
7. **Walk-forward is mandatory** — No in-sample-only results accepted
8. **Realistic costs** — Include commission, slippage, and spread

## EXPECTED RUNTIME

- Data fetching: ~30 minutes (4 symbols × multiple timeframes)
- Strategy testing: ~2-3 hours (6 tiers × 20+ strategies × 100+ parameter combos)
- Walk-forward validation: ~30 minutes
- Monte Carlo: ~15 minutes
- **Total: ~4 hours** (fits within MiMo Claw session limit)

## START COMMAND

```python
# Start the optimization loop
print("=" * 80)
print("COMMODITY STRATEGY OPTIMIZATION LOOP")
print("=" * 80)
print(f"Start time: {datetime.now()}")
print(f"Symbols: XAU/USD, XAG/USD, XCU/USD, [YOUR_CHOICE]")
print(f"Target: WR >= 65%, PF >= 1.5, Sharpe >= 0.5")
print("=" * 80)

# Begin loop...
```
