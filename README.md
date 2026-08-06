# Project9 — Automated Trading System

## Overview

Multi-strategy automated trading system with 6 symbols, prop firm evaluation, and AI-driven optimization.

## Directory Structure

```
project9/
├── backtest/                    # Main backtesting framework
│   ├── config.py                # All parameters centralized
│   ├── engine.py                # vectorbt BacktestEngine with HMM regime filter
│   ├── main.py                  # Entry point
│   ├── strategies/              # 15 trading strategies
│   │   ├── orb_strategy.py     # Opening Range Breakout (SPY/QQQ/TSLA/NVDA/AMD)
│   │   ├── xauusd_session_mr.py # XAUUSD session mean reversion
│   │   ├── kalman_trend.py     # Kalman filter trend following
│   │   ├── mean_reversion.py   # Adaptive Bollinger + trailing stops
│   │   ├── vwap_mean_reversion.py # VWAP Z-score framework
│   │   └── ...                 # 10 more strategies
│   ├── risk/                    # Risk management modules
│   ├── reporting/               # Metrics, plotting, deflated Sharpe
│   ├── optimization/            # Walk-forward, grid search, regime detection
│   ├── paper_trading/           # Live signal engine, order manager
│   ├── prop_firm/               # Prop firm rule simulator
│   └── data/                    # Data fetching (Alpaca, yfinance, LSE)
│
├── trading-system/              # AI-driven strategy optimization loop
│   ├── loop/                    # LLM-powered iteration loop
│   ├── evaluator/               # Prop firm rule evaluation
│   ├── strategies/              # Candidate strategies
│   └── data/                    # XAUUSD data (15m, 1h)
│
├── strategies/                  # Standalone strategy implementations
├── research/                    # Deep research documents
│   ├── orb_strategy_pitfalls.md
│   └── xauusd_session_mean_reversion_pitfalls.md
├── scripts/                     # Utility scripts
│   ├── mimo_claw_pipeline.py    # MiMo Claw data fetch + optimization
│   ├── mimo_claw_pipeline_expanded.py # Full 6-symbol pipeline
│   └── ...                      # Analysis and testing scripts
└── config/                      # Config additions for new symbols
```

## Symbols

| Symbol | Strategy | Timeframe | Status |
|--------|----------|-----------|--------|
| SPY | ORB (5-min) | 15-min bars | ✓ Implemented |
| QQQ | ORB (5-min) | 15-min bars | ✓ Implemented |
| TSLA | ORB (5-min) | 1-min bars | ✓ Config ready |
| NVDA | ORB (5-min) | 1-min bars | ✓ Config ready |
| AMD | ORB (5-min) | 1-min bars | ✓ Config ready |
| XAU/USD | Session MR | 1-hour bars | ✓ Implemented |

## Quick Start

```bash
cd backtest
pip install -r requirements.txt
python main.py              # Full backtest
python main.py --sweep      # Parameter sweep
```

## MiMo Claw Pipeline

Run the expanded pipeline in MiMo Claw to:
1. Fetch 1-min data for all 6 symbols from London Strategic Edge API
2. Deep research each symbol's characteristics
3. Optimize parameters with walk-forward validation
4. Assess prop firm readiness (FTMO, The5ers, FundingPips)

See `scripts/mimo_claw_pipeline_expanded.py` for the full script.

## Prop Firm Targets

| Firm | Profit Target | Max DD | Daily Loss |
|------|--------------|--------|------------|
| FTMO 2-Step | 10% | 10% | 5% |
| The5ers High Stakes | 8% | 6% | 3% |
| FundingPips | 8% | 10% | 5% |

## Research Documents

- `research/orb_strategy_pitfalls.md` — 10 professional pitfalls with code mitigations
- `research/xauusd_session_mean_reversion_pitfalls.md` — 10 pitfalls for XAUUSD session MR
- `prop_firm_intraday_strategy.md` — Prop firm challenge design

## API Keys

- **London Strategic Edge**: `lse_live_f4c9a7419371ecdd9365e146247b0289` (free tier)
- **Alpaca**: Set in `backtest/.env`
- **yfinance**: No key needed

## Key Features

- **Regime detection**: 3-method consensus (Hurst + Variance Ratio + Half-life)
- **News event filter**: NFP, FOMC, CPI calendar blackout
- **Prop firm simulation**: Daily DD, max DD, consistency checks
- **Walk-forward validation**: Train/test splits with robustness scoring
- **Deflated Sharpe Ratio**: Accounts for multiple testing
- **Paper trading**: Live signal engine with order management
