import yfinance as yf
from datetime import datetime, timezone
from typing import List
from src.api.models import Quote, OHLCV, Candle, SearchResult
from src.api.providers.base import MarketDataProvider
import json, pathlib
class YahooProvider(MarketDataProvider):
    name = "yahoo"

    async def quote(self, symbol: str) -> Quote:
        tk = yf.Ticker(symbol)
        info = getattr(tk, "fast_info", None)
        price = currency = prev_close = open_ = high = low = vol = None

        try:
            if info is not None:
                price = float(getattr(info, "last_price", None)) if getattr(info, "last_price", None) is not None else None
                currency = getattr(info, "currency", None)
                prev_close = float(getattr(info, "previous_close", None)) if getattr(info, "previous_close", None) is not None else None
                open_ = float(getattr(info, "open", None)) if getattr(info, "open", None) is not None else None
                high = float(getattr(info, "day_high", None)) if getattr(info, "day_high", None) is not None else None
                low  = float(getattr(info, "day_low", None))  if getattr(info, "day_low", None)  is not None else None
                vol  = int(getattr(info, "last_volume", None)) if getattr(info, "last_volume", None) is not None else None
        except Exception:
            pass

        name = exchange = None
        try:
            meta = tk.get_info()
            name = meta.get("shortName") or meta.get("longName")
            exchange = meta.get("exchange")
            currency = currency or meta.get("currency")
        except Exception:
            pass

        change = change_pct = None
        if price is not None and prev_close not in (None, 0):
            change = price - prev_close
            change_pct = (change / prev_close) * 100.0

        return Quote(
            symbol=symbol, name=name, currency=currency, exchange=exchange,
            price=price, previous_close=prev_close, open=open_, high=high, low=low, volume=vol,
            change=change, change_percent=change_pct, time=datetime.now(timezone.utc),
            is_delayed=True
        )

    async def ohlcv(self, symbol: str, interval: str = "1m", range_: str = "1d") -> OHLCV:
        df = yf.download(tickers=symbol, interval=interval, period=range_, progress=False, auto_adjust=False)
        df = df.dropna()
        data = []
        if not df.empty:
            for ts, row in df.iterrows():
                t = ts.to_pydatetime().astimezone(timezone.utc) if getattr(ts, "tzinfo", None) else ts.to_pydatetime().replace(tzinfo=timezone.utc)
                data.append(Candle(
                    time=t,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row.get("Volume", 0.0))
                ))
        return OHLCV(symbol=symbol, interval=interval, data=data)

    async def search(self, query: str):
        # Recherche locale dans le fichier JSON
        
        data_path = pathlib.Path(__file__).resolve().parents[1] / "data" / "cac40_symbols.json"
        try:
            items = json.loads(data_path.read_text(encoding="utf-8"))
        except Exception:
            # fallback minimal si le fichier est absent
            items = [
                {"symbol": "^FCHI", "name": "CAC 40 Index", "exchange": "INDEX"},
                {"symbol": "BNP.PA", "name": "BNP Paribas", "exchange": "PARIS"},
                {"symbol": "MC.PA", "name": "LVMH", "exchange": "PARIS"},
                {"symbol": "OR.PA", "name": "L'Oréal", "exchange": "PARIS"},
            ]

        q = query.lower()
        out = []
        for row in items:
            if q in row["symbol"].lower() or q in row["name"].lower():
                out.append(SearchResult(
                    symbol=row["symbol"],
                    name=row["name"],
                    exchange=row.get("exchange"),
                    type="EQUITY" if row["symbol"].endswith(".PA") else "INDEX"
                ))
        return out[:25]
