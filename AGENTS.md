# AGENTS.md — Multi-Instrument Backtesting System

## Quick Start

```bash
cd backtest
pip install --break-system-packages -r requirements.txt
python main.py              # Full backtest with regime filter
python main.py --sweep      # Parameter sweep heatmaps
```

## Architecture

```
backtest/
├── main.py              # Entry point — orchestrates full pipeline
├── config.py            # ALL parameters centralized here
├── engine.py            # vectorbt BacktestEngine with HMM regime filter
├── data/
│   ├── fetcher.py       # Alpaca IEX/crypto fetching + parquet cache
│   ├── resampler.py     # 1Min → 15min/1h/4h OHLCV resampling
│   ├── synthetic.py     # Fallback when no API keys
│   └── cache/           # Parquet files (auto-created)
├── strategies/
│   ├── mean_reversion.py    # Adaptive Bollinger + trailing stops
│   ├── momentum_breakout.py # Breakout strength + volume ratio filter
│   └── trend_following.py   # EMA crossover
├── risk/
│   ├── position_sizer.py    # ATR-based sizing with 25% max exposure cap
│   ├── correlation_filter.py
│   └── regime_filter.py     # HMM 2-state regime detection
├── prop_firm/
│   └── rule_simulator.py    # Daily/max DD + consistency checks
├── reporting/
│   ├── metrics.py           # Daily-return Sharpe (not vectorbt's broken annualized)
│   └── plotter.py           # Equity curves, trades, param sweep heatmaps
└── results/                 # PNG + CSV outputs
```

## Key Gotchas

### Position Sizing Bug Prevention
- `atr_position_sizes()` requires `price` parameter now — old 3-arg signature will crash
- Max exposure capped at 25% of equity per trade (configurable in `RISK_CONFIG["max_exposure_pct"]`)
- Without this cap, SPY/QQQ ATR sizing produced >100% losses

### Sharpe Ratio
- **Do NOT use** `vbt.Portfolio.sharpe_ratio()` — it annualizes by sqrt(bars/year), producing values like -26,000 for 15min data
- Use `_compute_sharpe()` from `reporting/metrics.py` which computes daily-return Sharpe

### Signal Returns
- `mean_reversion.generate_signals()` returns **5-tuple**: `(le, lx, se, sx, trailing_stops)`
- Other strategies return 4-tuple. Engine handles both via `len(result)` check.
- All signals must be shifted by 1 bar to avoid lookahead bias. Engine asserts no signal fires on bar 0.

### Regime Filter
- HMM fits on returns + realized volatility, classifies as "mean_reverting" or "trending"
- Falls back to rolling-window method when HMM fitting fails
- If <10% of bars are "trending", filter is bypassed (too restrictive)
- Configure via `ENGINE_CONFIG["use_regime_filter"]` in config.py

### Alpaca Data
- **Free tier**: IEX feed for stocks, free crypto. No SIP/pre-market data.
- Rate limit: 200 req/min. Cache avoids re-fetching.
- BTC/USD volume is in BTC (tiny numbers like 0.003) — use ratio-based volume filter, not absolute

### vectorbt 1.0 API Changes
- `freq` parameter must use lowercase: `"15min"`, `"1h"`, `"4h"` (not `"15T"`, `"1H"`)
- `trail_atr_mult` from config must be filtered before passing to `generate_signals()` — it's engine-only
- Combined portfolio requires `cash_sharing=True` and `freq="1min"` base

## Modifying Strategies

1. Edit `strategies/<name>.py` — signal generation only, no position sizing
2. Update `STRATEGY_PARAMS` in `config.py` if adding parameters
3. If parameter is engine-only (like `trail_atr_mult`), filter it out in `engine.py:_generate_portfolio_signals()`
4. Run `python main.py --sweep` to compare parameter impact

## Testing Changes

```bash
# Quick sanity check (uses cache, ~30 seconds)
python main.py 2>&1 | grep -E "(return=|PASS|FAIL)"

# Full parameter comparison
python main.py --sweep
```

No formal test suite exists. Validate by comparing Sharpe ratios and max drawdown before/after changes.

## Environment

`.env` file required for Alpaca API:
```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```
Without valid keys, system auto-falls back to synthetic data generation.
