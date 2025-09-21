from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class Quote(BaseModel):
    symbol: str
    name: Optional[str] = None
    currency: Optional[str] = None
    exchange: Optional[str] = None
    price: Optional[float] = Field(None, description="Last trade price (may be delayed)")
    change: Optional[float] = None
    change_percent: Optional[float] = None
    previous_close: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[int] = None
    time: Optional[datetime] = None
    is_delayed: bool = True

class Candle(BaseModel):
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = 0

class OHLCV(BaseModel):
    symbol: str
    interval: str
    data: List[Candle]

class SearchResult(BaseModel):
    symbol: str
    name: str
    exchange: Optional[str] = None
    type: Optional[str] = None