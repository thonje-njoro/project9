"""Main paper trading runner."""

import os
import sys
import signal
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import INSTRUMENTS, RISK_CONFIG, PROP_FIRM_RULES
from data.resampler import resample_ohlcv
from paper_trading.config import PAPER_TRADING_CONFIG
from paper_trading.signal_engine import LiveSignalEngine
from paper_trading.order_manager import OrderManager
from paper_trading.risk_guard import RiskGuard
from paper_trading.state import StateManager
from paper_trading.market_calendar import MarketCalendar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("paper_trading/state/paper_trading.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("runner")


class PaperTradingRunner:
    """Orchestrates live paper trading on Alpaca."""

    def __init__(self) -> None:
        load_dotenv(Path(__file__).parent.parent / ".env")
        api_key = os.getenv("ALPACA_API_KEY", "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        dry_run = PAPER_TRADING_CONFIG.get("dry_run", True)

        self.signal_engine = LiveSignalEngine()
        self.order_manager = OrderManager(api_key, secret_key, dry_run=dry_run)
        self.risk_guard = RiskGuard(PROP_FIRM_RULES, PAPER_TRADING_CONFIG["state_dir"])
        self.state = StateManager(PAPER_TRADING_CONFIG["state_dir"])

        self._restore_state()
        self._shutdown = False

    def _restore_state(self) -> None:
        self.order_manager._sync_positions()
        logger.info(f"Restored {len(self.order_manager.open_positions)} open positions")

        equity_df = self.state.load_equity_curve()
        equity_history = []
        if not equity_df.empty:
            equity_history = list(zip(
                equity_df["timestamp"].tolist(),
                equity_df["equity"].tolist(),
            ))

        current_equity = self.order_manager.get_account_equity()
        self.risk_guard.initialize(current_equity, equity_history)

        historical_data = self._fetch_historical_bars()
        if historical_data:
            self.signal_engine.initialize_regime_filters(historical_data)
            self.state.save_regime_state(self.signal_engine.regime_filters)
            logger.info("Regime filters fitted from historical data")

    def _fetch_historical_bars(self) -> dict:
        from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        import pandas as pd

        api_key = os.getenv("ALPACA_API_KEY", "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "")

        stock_client = StockHistoricalDataClient(api_key, secret_key)
        crypto_client = CryptoHistoricalDataClient()

        data = {}
        now = datetime.utcnow()

        for symbol, info in INSTRUMENTS.items():
            days_back = 30
            start = now - timedelta(days=days_back)
            try:
                if info["asset_class"] == "crypto":
                    request = CryptoBarsRequest(
                        symbol_or_symbols=symbol,
                        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                        start=start,
                    )
                    bars = crypto_client.get_crypto_bars(request)
                else:
                    request = StockBarsRequest(
                        symbol_or_symbols=symbol,
                        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                        start=start,
                        feed="iex",
                    )
                    bars = stock_client.get_stock_bars(request)

                df = bars.df
                if isinstance(df.index, pd.MultiIndex):
                    df = df.droplevel("symbol")
                df = df[["open", "high", "low", "close", "volume"]]
                df.index = pd.to_datetime(df.index).tz_convert("UTC")

                target_tf = INSTRUMENTS[symbol]["target_tf"]
                data[symbol] = resample_ohlcv(df, target_tf, info["asset_class"])
                logger.info(f"Fetched {len(data[symbol])} bars for {symbol}")
            except Exception as e:
                logger.error(f"Failed to fetch historical bars for {symbol}: {e}")
        return data

    def _fetch_latest_bar(self, symbol: str, info: dict):
        from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        import pandas as pd

        api_key = os.getenv("ALPACA_API_KEY", "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "")

        stock_client = StockHistoricalDataClient(api_key, secret_key)
        crypto_client = CryptoHistoricalDataClient()

        try:
            now = datetime.utcnow()
            start = now - timedelta(hours=2)

            if info["asset_class"] == "crypto":
                request = CryptoBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                    start=start,
                )
                bars = crypto_client.get_crypto_bars(request)
            else:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                    start=start,
                    feed="iex",
                )
                bars = stock_client.get_stock_bars(request)

            df = bars.df
            if isinstance(df.index, pd.MultiIndex):
                df = df.droplevel("symbol")
            df = df[["open", "high", "low", "close", "volume"]]
            df.index = pd.to_datetime(df.index).tz_convert("UTC")
            target_tf = INSTRUMENTS[symbol]["target_tf"]
            return resample_ohlcv(df, target_tf, info["asset_class"])
        except Exception as e:
            logger.error(f"Failed to fetch latest bar for {symbol}: {e}")
            return None

    def _evaluate_and_trade(self, symbol: str, info: dict) -> None:
        if not MarketCalendar.can_trade(symbol, info["asset_class"]):
            return

        df = self._fetch_latest_bar(symbol, info)
        if df is None or len(df) < 30:
            logger.warning(f"Insufficient data for {symbol}, skipping")
            return

        signal = self.signal_engine.evaluate(symbol, df)
        logger.info(f"{symbol}: action={signal['action']} price={signal['price']:.2f} "
                     f"regime={signal['regime']}")

        if signal["action"] == "hold":
            return

        equity = self.order_manager.get_account_equity()
        self.risk_guard.update_equity(datetime.utcnow().isoformat(), equity)

        if signal["action"] in ("long_exit", "short_exit"):
            pos = self.order_manager.get_position(symbol)
            if pos:
                self.order_manager.close_position(symbol)
                self._log_trade(symbol, "exit", 0, signal["price"], "exit_signal")
            return

        if signal["action"] in ("long_entry", "short_entry"):
            if not self.risk_guard.can_trade(equity):
                logger.warning(f"Trade BLOCKED by risk guard for {symbol}")
                return

            from risk.position_sizer import compute_atr, atr_position_sizes
            atr = compute_atr(df, RISK_CONFIG["atr_period"])
            sizes = atr_position_sizes(
                equity, atr, df["close"],
                RISK_CONFIG["max_risk_per_trade_pct"],
                RISK_CONFIG.get("max_exposure_pct", 0.25),
            )
            trade_notional = float(sizes.iloc[-1] * signal["price"])

            max_notional = equity * RISK_CONFIG.get("max_exposure_pct", 0.25)
            trade_notional = min(trade_notional, max_notional)

            if trade_notional < 10:
                logger.info(f"Trade size too small for {symbol}: ${trade_notional:.2f}")
                return

            side = "buy" if signal["action"] == "long_entry" else "sell"
            result = self.order_manager.submit_market_order(
                symbol, trade_notional, side, signal["trailing_stop_pct"]
            )
            self._log_trade(symbol, side, trade_notional, signal["price"], signal["action"])

    def _log_trade(self, symbol, side, notional, price, reason) -> None:
        self.state.append_trade({
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "side": side,
            "notional": notional,
            "price": price,
            "reason": reason,
            "equity": self.order_manager.get_account_equity(),
        })

    def _snapshot_equity(self) -> None:
        equity = self.order_manager.get_account_equity()
        self.state.save_equity_snapshot(datetime.utcnow().isoformat(), equity, 0.0)
        self.risk_guard.update_equity(datetime.utcnow().isoformat(), equity)

    def tick(self) -> None:
        logger.info("--- Tick start ---")
        for symbol, info in INSTRUMENTS.items():
            try:
                self._evaluate_and_trade(symbol, info)
            except Exception as e:
                logger.error(f"Error evaluating {symbol}: {e}", exc_info=True)
        self._snapshot_equity()
        self.state.save_positions(self.order_manager.open_positions)
        self.state.save_risk_state({
            "equity_history": self.risk_guard.equity_history[-100:],
            "peak_equity": self.risk_guard.peak_equity,
        })
        logger.info("--- Tick end ---")

    def run(self) -> None:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger

        scheduler = BlockingScheduler(timezone="UTC")

        scheduler.add_job(
            self.tick, CronTrigger(
                minute="*/15", day_of_week="mon-fri",
                hour="9-15", timezone="America/New_York",
            ),
            id="stock_tick",
        )

        scheduler.add_job(
            self.tick, CronTrigger(minute="0", hour="*/1"),
            id="crypto_tick",
        )

        scheduler.add_job(
            self.tick, CronTrigger(minute="0", hour="*/4"),
            id="gld_uso_tick",
        )

        def shutdown_handler(signum, frame):
            logger.info("Shutdown signal received, stopping...")
            self._shutdown = True
            scheduler.shutdown(wait=False)

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        logger.info("Paper trading system started")
        self.tick()

        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped")
        finally:
            self._snapshot_equity()
            self.state.save_positions(self.order_manager.open_positions)
            logger.info("State saved, exiting")


if __name__ == "__main__":
    runner = PaperTradingRunner()
    runner.run()
