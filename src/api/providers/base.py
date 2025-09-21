from typing import List
from src.api.models import Quote, OHLCV, SearchResult

class MarketDataProvider:
    name = "base"

    async def quote(self, symbol: str) -> Quote:
        raise NotImplementedError

    async def ohlcv(self, symbol: str, interval: str = "1m", range_: str = "1d") -> OHLCV:
        raise NotImplementedError

    async def search(self, query: str) -> List[SearchResult]:
        raise NotImplementedError