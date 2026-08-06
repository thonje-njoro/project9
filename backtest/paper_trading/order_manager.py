"""Alpaca order management for paper trading."""

import os
import logging
from datetime import datetime

logger = logging.getLogger("order_manager")


class OrderManager:
    """Manages orders and positions via Alpaca paper trading API."""

    def __init__(self, api_key: str, secret_key: str, dry_run: bool = True) -> None:
        self.dry_run = dry_run
        self.open_positions: dict = {}

        if not dry_run:
            from alpaca.trading.client import TradingClient
            self.client = TradingClient(api_key, secret_key, paper=True)
        else:
            self.client = None
            logger.info("DRY RUN mode — orders will be logged but not submitted")

    def get_account_equity(self) -> float:
        if self.dry_run:
            return 10000.0
        try:
            account = self.client.get_account()
            return float(account.equity)
        except Exception as e:
            logger.error(f"Failed to get account equity: {e}")
            return 0.0

    def get_position(self, symbol: str) -> dict | None:
        if self.dry_run:
            return self.open_positions.get(symbol)
        try:
            position = self.client.get_open_position(symbol)
            if position:
                return {
                    "symbol": symbol,
                    "qty": float(position.qty),
                    "side": position.side.value,
                    "avg_entry": float(position.avg_entry_price),
                    "market_value": float(position.market_value),
                    "unrealized_pl": float(position.unrealized_pl),
                }
            return None
        except Exception:
            return self.open_positions.get(symbol)

    def submit_market_order(
        self, symbol: str, notional: float, side: str, trailing_stop_pct: float = 0.0
    ) -> dict:
        now = datetime.utcnow().isoformat()

        if self.dry_run:
            price = 100.0
            qty = notional / price if price > 0 else 0
            order_info = {
                "id": f"dry_run_{now}",
                "symbol": symbol,
                "side": side,
                "notional": notional,
                "qty": qty,
                "status": "accepted",
                "timestamp": now,
            }
            self.open_positions[symbol] = {
                "symbol": symbol,
                "qty": qty if side == "buy" else -qty,
                "side": side,
                "avg_entry": price,
            }
            logger.info(f"DRY RUN ORDER: {side} ${notional:.2f} of {symbol}")
            return order_info

        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            tif = TimeInForce.DAY
            if symbol == "BTC/USD":
                tif = TimeInForce.GTC

            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
            request = MarketOrderRequest(
                symbol=symbol,
                notional=notional,
                side=order_side,
                time_in_force=tif,
            )
            order = self.client.submit_order(request)

            order_info = {
                "id": str(order.id),
                "symbol": symbol,
                "side": side,
                "notional": notional,
                "status": order.status.value,
                "timestamp": now,
            }
            logger.info(f"ORDER SUBMITTED: {side} ${notional:.2f} of {symbol}, id={order.id}")

            if trailing_stop_pct > 0:
                self._submit_trailing_stop(symbol, trailing_stop_pct)

            return order_info
        except Exception as e:
            logger.error(f"Order failed for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}

    def _submit_trailing_stop(self, symbol: str, trailing_stop_pct: float) -> None:
        if self.dry_run:
            logger.info(f"DRY RUN TRAILING STOP: {symbol} at {trailing_stop_pct*100:.2f}%")
            return

        try:
            from alpaca.trading.requests import TrailingStopOrderRequest
            from alpaca.trading.enums import TimeInForce

            position = self.get_position(symbol)
            if not position:
                return

            qty = abs(position["qty"])
            side = "sell" if position["side"] == "long" else "buy"

            tif = TimeInForce.DAY
            if symbol == "BTC/USD":
                tif = TimeInForce.GTC

            request = TrailingStopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                trail_percent=trailing_stop_pct * 100,
                time_in_force=tif,
            )
            order = self.client.submit_order(request)
            logger.info(f"TRAILING STOP: {symbol} at {trailing_stop_pct*100:.2f}%, id={order.id}")
        except Exception as e:
            logger.error(f"Trailing stop failed for {symbol}: {e}")

    def close_position(self, symbol: str) -> dict:
        if self.dry_run:
            if symbol in self.open_positions:
                del self.open_positions[symbol]
            logger.info(f"DRY RUN CLOSE: {symbol}")
            return {"status": "closed", "symbol": symbol}

        try:
            self.client.close_position(symbol)
            logger.info(f"POSITION CLOSED: {symbol}")
            return {"status": "closed", "symbol": symbol}
        except Exception as e:
            logger.error(f"Close position failed for {symbol}: {e}")
            return {"error": str(e)}

    def _sync_positions(self) -> None:
        if self.dry_run:
            self.open_positions = {}
            return

        try:
            positions = self.client.get_all_positions()
            self.open_positions = {}
            for pos in positions:
                self.open_positions[pos.symbol] = {
                    "symbol": pos.symbol,
                    "qty": float(pos.qty),
                    "side": pos.side.value,
                    "avg_entry": float(pos.avg_entry_price),
                    "market_value": float(pos.market_value),
                }
        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")
