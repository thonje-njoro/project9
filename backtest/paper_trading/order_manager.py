"""
paper_trading/order_manager.py — Alpaca paper order placement + position sync.
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Manages paper trading orders via Alpaca API.
    Submits bracket orders, tracks fills, syncs positions.
    """

    def __init__(self, api_key: str = "", secret_key: str = "",
                 base_url: str = "https://paper-api.alpaca.markets"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.client = None
        self.open_positions = {}  # {symbol: position_info}

        self._init_client()

    def _init_client(self):
        """Initialize Alpaca trading client."""
        if not self.api_key or self.api_key == "your_alpaca_api_key_here":
            logger.warning("Alpaca API keys not configured — paper trading disabled")
            return

        try:
            from alpaca.trading.client import TradingClient
            self.client = TradingClient(
                self.api_key, self.secret_key,
                paper=True, base_url=self.base_url,
            )
            logger.info("Alpaca trading client initialized")
        except Exception as e:
            logger.error(f"Failed to init Alpaca client: {e}")

    def submit(self, symbol: str, qty: float, side: str,
               stop_price: float | None = None) -> str | None:
        """
        Submit a market order with optional stop-loss.

        Args:
            symbol: Instrument symbol.
            qty: Number of shares.
            side: 'buy' or 'sell'.
            stop_price: Stop-loss price. If None, no stop attached.

        Returns:
            Order ID string, or None if failed.
        """
        if self.client is None:
            logger.warning("No Alpaca client — order not submitted")
            return None

        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )

            order = self.client.submit_order(request)
            order_id = str(order.id)

            logger.info(f"Order submitted: {side} {qty} {symbol} (id={order_id})")

            # Submit stop-loss if specified
            if stop_price is not None:
                self._submit_stop(symbol, qty, side, stop_price)

            return order_id

        except Exception as e:
            logger.error(f"Order failed for {symbol}: {e}")
            return None

    def _submit_stop(self, symbol: str, qty: float, side: str,
                     stop_price: float):
        """Submit a stop-loss order."""
        try:
            from alpaca.trading.requests import StopOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            # Stop is opposite side: if we bought, stop is sell
            stop_side = OrderSide.SELL if side == "buy" else OrderSide.BUY

            request = StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=stop_side,
                stop_price=stop_price,
                time_in_force=TimeInForce.GTC,
            )

            self.client.submit_order(request)
            logger.info(f"Stop order: {stop_side} {qty} {symbol} @ {stop_price}")

        except Exception as e:
            logger.error(f"Stop order failed for {symbol}: {e}")

    def get_positions(self) -> dict:
        """Get current Alpaca positions."""
        if self.client is None:
            return {}

        try:
            positions = self.client.get_all_positions()
            result = {}
            for pos in positions:
                result[pos.symbol] = {
                    "qty": float(pos.qty),
                    "side": "long" if float(pos.qty) > 0 else "short",
                    "avg_entry": float(pos.avg_entry_price),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc),
                }
            return result
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}

    def get_account(self) -> dict:
        """Get account info."""
        if self.client is None:
            return {}

        try:
            account = self.client.get_account()
            return {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
            }
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            return {}

    def sync_positions(self):
        """Reconcile local state with Alpaca positions."""
        remote = self.get_positions()
        self.open_positions = remote
        logger.info(f"Synced {len(remote)} positions from Alpaca")
        return remote

    def close_position(self, symbol: str) -> bool:
        """Close a position."""
        if self.client is None:
            return False

        try:
            self.client.close_position(symbol)
            logger.info(f"Closed position: {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to close {symbol}: {e}")
            return False

    def close_all(self) -> bool:
        """Close all positions."""
        if self.client is None:
            return False

        try:
            self.client.close_all_positions()
            logger.info("All positions closed")
            return True
        except Exception as e:
            logger.error(f"Failed to close all: {e}")
            return False
