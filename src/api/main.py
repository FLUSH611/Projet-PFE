import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from dotenv import load_dotenv

from src.api.models import Quote, OHLCV, SearchResult
from src.api.providers.yahoo import YahooProvider
from src.api.utils import is_euronext_open, paris_now

load_dotenv()

app = FastAPI(title="IT-Storm Market API", version="1.0.0",
              description="Lightweight market data API (FR). Default provider: Yahoo (delayed).")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

PROVIDER = YahooProvider()

@app.get("/v1/health")
async def health():
    return {"status": "ok", "time": paris_now().isoformat()}

@app.get("/v1/marketstatus")
async def market_status():
    return {"market": "Euronext Paris", "is_open": is_euronext_open(), "time": paris_now().isoformat()}

@app.get("/v1/search", response_model=List[SearchResult])
async def search(q: str = Query(..., min_length=1)):
    return await PROVIDER.search(q)

@app.get("/v1/quote/{symbol}", response_model=Quote)
async def get_quote(symbol: str):
    try:
        return await PROVIDER.quote(symbol)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/v1/ohlcv/{symbol}", response_model=OHLCV)
async def get_ohlcv(symbol: str, interval: str = "1m", range: str = "1d"):
    try:
        return await PROVIDER.ohlcv(symbol, interval=interval, range_=range)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))