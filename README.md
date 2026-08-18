# Project9 — Autonomous Quantitative Trading System

A fully autonomous, self-improving quantitative trading system designed to run on a single GCP e2-medium VM (2 vCPU / 4 GB RAM).

## Architecture

```
backtest/           → Core backtesting engine + strategies
  config.py         → Single source of truth for ALL parameters
  engine.py         → BacktestEngine orchestrator
  main.py           → CLI entry point
  data/             → Data fetching (Alpaca, LSE, yfinance) + caching
  strategies/       → 6 strategy implementations
  risk/             → Regime filter, position sizing, correlation, Monte Carlo
  optimization/     → Purged walk-forward validation
  reporting/        → Metrics, deflated Sharpe, plotting
  prop_firm/        → FTMO/The5ers/FundingPips rule simulation
  paper_trading/    → Live paper trading via Alpaca
ai_loop/            → Nightly AI self-improvement (MiMo v2.5 Pro)
scripts/            → VM setup + health check
systemd/            → Service files for 24/7 operation
```

## Strategies

| Instrument | Strategy | Timeframe | Data Source |
|------------|----------|-----------|-------------|
| GLD, TLT, IWM, CPER | Kalman Trend | 4H | Alpaca IEX |
| CPER/GLD Ratio | Mean Reversion | 4H | Alpaca IEX |
| SPY, QQQ | Opening Range Breakout | 15Min | Alpaca IEX |
| NVDA, AMD, PLTR, MRVL | Momentum ORB | 5Min | Alpaca IEX |
| SPY, IWM | VWAP Mean Reversion | 15Min | Alpaca IEX |
| XAU/USD | Session Mean Reversion | 1H | LSE API |

## Quick Start

```bash
# 1. Clone and setup
cd project9
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure API keys
cp backtest/.env.example backtest/.env
# Edit backtest/.env with real keys

# 3. Run backtest
python backtest/main.py

# 4. Run validation
python backtest/main.py --validate

# 5. Single instrument
python backtest/main.py --symbol GLD

# 6. JSON output
python backtest/main.py --json
```

## GCP Deployment

```bash
# On a fresh e2-medium VM:
chmod +x scripts/setup_vm.sh
./scripts/setup_vm.sh

# Edit .env, then:
sudo systemctl start paper-trader
sudo systemctl start ai-loop.timer
```

## Risk Management

- **Regime Gate**: Only trades when >55% of recent days are positive
- **ATR Position Sizing**: 1% risk per trade, 25% max exposure cap
- **Correlation Filter**: Blocks trades correlated >0.7 with existing positions
- **Monte Carlo**: 2,000 bootstrap simulations, kills if 99th-pct DD > 12%

## Validation Pipeline

1. Deflated Sharpe Ratio > 0.95
2. Purged Walk-Forward (6 folds, 20-bar embargo) — consistency > 60%
3. Monte Carlo survival rate > 85%
4. Prop firm rule simulation (FTMO/The5ers/FundingPips)

## AI Self-Improvement

Runs nightly at 18:00 ET via systemd timer:
1. Backtest with current config
2. MiMo v2.5 Pro proposes ONE parameter change
3. Validated via walk-forward before applying
4. All changes logged

## License

Private — not for distribution.
