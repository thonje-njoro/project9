"""Market hours filter for stocks vs crypto."""

from datetime import datetime, time


class MarketCalendar:
    """Determine if an instrument is currently tradable."""

    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)

    @staticmethod
    def can_trade(symbol: str, asset_class: str) -> bool:
        if asset_class == "crypto":
            return True

        now = datetime.now()
        if now.weekday() >= 5:
            return False

        current_time = now.time()
        return MarketCalendar.MARKET_OPEN <= current_time <= MarketCalendar.MARKET_CLOSE

    @staticmethod
    def next_bar_close(target_tf: str) -> datetime:
        now = datetime.now()
        minutes = {"15Min": 15, "1H": 60, "4H": 240}
        interval = minutes.get(target_tf, 15)

        current_minute = (now.minute // interval + 1) * interval
        if current_minute >= 60:
            return now.replace(minute=0, second=0, microsecond=0).replace(
                hour=now.hour + 1
            )
        return now.replace(minute=current_minute, second=0, microsecond=0)
