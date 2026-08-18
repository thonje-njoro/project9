"""
paper_trading/runner.py — Main loop: fetch → signal → order.
Runs continuously via systemd.
"""

import gc
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.config import ALPACA_API_KEY, ALPACA_SECRET_KEY, INSTRUMENTS
from backtest.paper_trading.signal_engine import LiveSignalEngine
from backtest.paper_trading.order_manager import OrderManager
from backtest.paper_trading.risk_guard import RiskGuard
from backtest.paper_trading.state import StateManager
from backtest.paper_trading.market_calendar import is_trading_day, market_is_open

logger = logging.getLogger(__name__)


class PaperTradingRunner:
    """
    Main paper trading loop.
    Architecture:
        on_bar() — called every minute
            → fetch latest bar
            → generate signals
            → check risk
            → submit orders
        on_day_open() — 09:30 ET
            → load regime gate
            → check daily loss limit
        on_day_close() — 16:00 ET
            → force-close intraday positions
            → log daily P&L
    """

    def __init__(self):
        self.signal_engine = LiveSignalEngine()
        self.order_manager = OrderManager(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
        )
        self.risk_guard = RiskGuard(firm_rules="ftmo_2step")
        self.state = StateManager()

        self.running = True
        self.last_day_open = None
        self.last_day_close = None

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        logger.info("Shutdown signal received")
        self.running = False

    def run(self):
        """Main loop — runs until stopped."""
        logger.info("Paper trading runner started")

        # Sync positions on startup
        self.order_manager.sync_positions()

        while self.running:
            try:
                now = datetime.utcnow()

                # Check if it's a trading day
                if not is_trading_day(now.date()):
                    logger.debug("Not a trading day — sleeping 60s")
                    time.sleep(60)
                    continue

                # Day open logic
                if self._is_day_open(now):
                    self._on_day_open()

                # Main bar processing
                if market_is_open(now):
                    self._on_bar()

                # Day close logic
                if self._is_day_close(now):
                    self._on_day_close()

                # Heartbeat
                self.state.save_heartbeat("ok")

                # Sleep until next minute
                time.sleep(60)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Runner error: {e}", exc_info=True)
                time.sleep(30)

        logger.info("Paper trading runner stopped")

    def _is_day_open(self, now: datetime) -> bool:
        """Check if we should run day-open logic (9:30 ET)."""
        import pytz
        et = pytz.timezone("US/Eastern")
        now_et = now.replace(tzinfo=pytz.UTC).astimezone(et)

        if now_et.hour == 9 and now_et.minute >= 30:
            if self.last_day_open != now_et.date():
                return True
        return False

    def _is_day_close(self, now: datetime) -> bool:
        """Check if we should run day-close logic (16:00 ET)."""
        import pytz
        et = pytz.timezone("US/Eastern")
        now_et = now.replace(tzinfo=pytz.UTC).astimezone(et)

        if now_et.hour == 16 and now_et.minute < 5:
            if self.last_day_close != now_et.date():
                return True
        return False

    def _on_day_open(self):
        """Day open: load regime, reset daily risk."""
        logger.info("=== Market Open ===")
        self.last_day_open = datetime.utcnow().date()

        # Get current equity
        account = self.order_manager.get_account()
        equity = account.get("equity", 0)
        if equity > 0:
            self.risk_guard.new_day(equity)

        # Sync positions
        self.order_manager.sync_positions()

    def _on_day_close(self):
        """Day close: force-close intraday positions, log P&L."""
        logger.info("=== Market Close ===")
        self.last_day_close = datetime.utcnow().date()

        # Force close all intraday positions
        intraday_symbols = [
            sym for sym, inst in INSTRUMENTS.items()
            if inst["timeframe"] in ("5min", "15min")
        ]

        for symbol in intraday_symbols:
            if symbol in self.order_manager.open_positions:
                logger.info(f"Force-closing intraday position: {symbol}")
                self.order_manager.close_position(symbol)

        # Log equity
        account = self.order_manager.get_account()
        equity = account.get("equity", 0)
        if equity > 0:
            self.state.append_equity(equity)

        # Sync
        self.order_manager.sync_positions()

    def _on_bar(self):
        """Main bar processing: fetch → signal → risk check → order."""
        # Generate signals
        signals = self.signal_engine.generate(lookback_bars=200)

        if not signals:
            return

        # Get current equity for risk check
        account = self.order_manager.get_account()
        equity = account.get("equity", 0)

        if equity <= 0:
            logger.warning("Could not get account equity")
            return

        # Risk check
        risk_check = self.risk_guard.check(equity)
        if not risk_check["allow_trading"]:
            logger.info(f"Trading blocked: {risk_check['reason']}")
            return

        # Process each signal
        for symbol, signal_data in signals.items():
            self._process_signal(symbol, signal_data, equity)

    def _process_signal(self, symbol: str, signal_data: dict, equity: float):
        """Process a single trading signal."""
        side = signal_data.get("side")
        price = signal_data.get("price", 0)

        if side is None or price <= 0:
            return

        # Check if already in position
        positions = self.order_manager.get_positions()
        if symbol in positions:
            existing_side = positions[symbol].get("side")
            if existing_side == side:
                return  # Already in same direction
            # Close opposite position first
            self.order_manager.close_position(symbol)

        # Compute position size using ATR from signal engine
        from backtest.risk.position_sizer import atr_position_size
        atr_estimate = signal_data.get("atr") or price * 0.02  # fall back only if missing
        qty = atr_position_size(equity, atr_estimate, price)

        if qty < 1:
            return

        order_side = "buy" if side == "long" else "sell"
        stop_price = price - atr_estimate * 2 if side == "long" else price + atr_estimate * 2

        order_id = self.order_manager.submit(
            symbol=symbol,
            qty=int(qty),
            side=order_side,
            stop_price=round(stop_price, 2),
        )

        if order_id:
            logger.info(f"Signal processed: {side} {int(qty)} {symbol} @ {price}")
            self.state.log_trade(
                symbol=symbol, side=side, qty=int(qty),
                entry_price=price, exit_price=0, pnl=0,
            )


def main():
    """Entry point for paper trading runner."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    runner = PaperTradingRunner()
    runner.run()


if __name__ == "__main__":
    main()
