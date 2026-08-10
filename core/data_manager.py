"""
Market-data layer.

Responsibilities:
  * build & refresh the tradable universe (>= MIN_VOLUME_USDT 24h quote volume)
  * cache symbol filters (tick size, step size, min notional)
  * cache OHLCV per (symbol, timeframe) with a TTL tuned per timeframe so the
    REST weight budget survives 200+ symbols
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.modes import TF_TTL
from config.settings import Settings
from core.exchange import BinanceFutures
from core.indicators import enrich
from utils.helpers import gather_limited, safe_float, step_size_to_precision
from utils.logger import get_logger

log = get_logger("data")

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume",
              "close_time", "quote_volume", "trades",
              "taker_buy_base", "taker_buy_quote", "ignore"]


@dataclass
class SymbolInfo:
    symbol: str
    base: str
    tick_size: float = 0.0001
    step_size: float = 0.001
    min_qty: float = 0.001
    min_notional: float = 5.0
    price_precision: int = 4
    qty_precision: int = 3
    quote_volume: float = 0.0
    last_price: float = 0.0
    price_change_pct: float = 0.0
    funding_rate: float = 0.0

    @property
    def tick_precision(self) -> int:
        return step_size_to_precision(self.tick_size)


@dataclass
class CacheEntry:
    df: pd.DataFrame
    fetched_at: float
    interval: str

    def is_fresh(self, now: float) -> bool:
        return (now - self.fetched_at) < TF_TTL.get(self.interval, 60)


class DataManager:
    def __init__(self, exchange: BinanceFutures, settings: Settings):
        self.ex = exchange
        self.s = settings
        self.symbols: Dict[str, SymbolInfo] = {}
        self.universe: List[str] = []
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self.last_universe_refresh: float = 0.0

    # ------------------------------------------------------------------ #
    # universe
    # ------------------------------------------------------------------ #
    async def refresh_universe(self) -> List[str]:
        """Pull exchangeInfo + 24h tickers, filter by volume, cache filters."""
        async with self._lock:
            try:
                info, tickers = await asyncio.gather(
                    self.ex.exchange_info(), self.ex.ticker_24h())
            except Exception as exc:                       # noqa: BLE001
                log.error("Universe refresh failed: %s", exc)
                return self.universe

            vol_map, price_map, chg_map = {}, {}, {}
            for t in tickers:
                sym = t.get("symbol", "")
                vol_map[sym] = safe_float(t.get("quoteVolume"))
                price_map[sym] = safe_float(t.get("lastPrice"))
                chg_map[sym] = safe_float(t.get("priceChangePercent"))

            blacklist = self.s.blacklist
            new_symbols: Dict[str, SymbolInfo] = {}

            for sym_data in info.get("symbols", []):
                sym = sym_data.get("symbol", "")
                if sym_data.get("status") != "TRADING":
                    continue
                if sym_data.get("contractType") != "PERPETUAL":
                    continue
                if sym_data.get("quoteAsset") != self.s.quote_asset:
                    continue
                if sym in blacklist or any(b in sym for b in blacklist if b):
                    continue

                qv = vol_map.get(sym, 0.0)
                if qv < self.s.min_volume_usdt:
                    continue

                si = SymbolInfo(
                    symbol=sym,
                    base=sym_data.get("baseAsset", ""),
                    price_precision=int(sym_data.get("pricePrecision", 4)),
                    qty_precision=int(sym_data.get("quantityPrecision", 3)),
                    quote_volume=qv,
                    last_price=price_map.get(sym, 0.0),
                    price_change_pct=chg_map.get(sym, 0.0),
                )

                for f in sym_data.get("filters", []):
                    ftype = f.get("filterType")
                    if ftype == "PRICE_FILTER":
                        si.tick_size = safe_float(f.get("tickSize"), si.tick_size)
                    elif ftype == "LOT_SIZE":
                        si.step_size = safe_float(f.get("stepSize"), si.step_size)
                        si.min_qty = safe_float(f.get("minQty"), si.min_qty)
                    elif ftype == "MARKET_LOT_SIZE":
                        si.step_size = safe_float(f.get("stepSize"), si.step_size)
                    elif ftype == "MIN_NOTIONAL":
                        si.min_notional = safe_float(f.get("notional"), 5.0)

                # carry forward previously known metadata
                prev = self.symbols.get(sym)
                if prev:
                    si.funding_rate = prev.funding_rate

                new_symbols[sym] = si

            ranked = sorted(new_symbols.values(),
                            key=lambda x: x.quote_volume, reverse=True)
            ranked = ranked[:self.s.max_symbols]

            self.symbols = {si.symbol: si for si in ranked}
            self.universe = [si.symbol for si in ranked]
            self.last_universe_refresh = time.time()

            log.info("Universe refreshed: %d symbols (vol >= $%.0fM)",
                     len(self.universe), self.s.min_volume_usdt / 1e6)

        await self.refresh_funding()
        return self.universe

    async def refresh_funding(self) -> None:
        try:
            data = await self.ex.mark_price()
            if isinstance(data, dict):
                data = [data]
            for item in data:
                sym = item.get("symbol")
                if sym in self.symbols:
                    self.symbols[sym].funding_rate = safe_float(item.get("lastFundingRate"))
        except Exception as exc:                           # noqa: BLE001
            log.debug("Funding refresh failed: %s", exc)

    def get_symbol(self, symbol: str) -> Optional[SymbolInfo]:
        return self.symbols.get(symbol.upper())

    # ------------------------------------------------------------------ #
    # OHLCV
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_df(raw: List[List]) -> pd.DataFrame:
        if not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw, columns=KLINE_COLS[:len(raw[0])])
        numeric = ["open", "high", "low", "close", "volume",
                   "quote_volume", "taker_buy_base", "taker_buy_quote"]
        for col in numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
        df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce").astype("int64")
        df["trades"] = pd.to_numeric(df.get("trades", 0), errors="coerce").fillna(0)
        df = df.dropna(subset=["open", "high", "low", "close"])
        # NOTE: naming the index "dt" matters — leaving it as "open_time" makes
        # any later reset_index() collide with the column of the same name.
        idx = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        idx.name = "dt"
        df = df.set_index(idx)
        return df

    def _key(self, symbol: str, interval: str) -> str:
        return f"{symbol}|{interval}"

    def cached(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        entry = self._cache.get(self._key(symbol, interval))
        return entry.df if entry else None

    async def get_klines(self, symbol: str, interval: str, limit: int = 300,
                         force: bool = False) -> Optional[pd.DataFrame]:
        key = self._key(symbol, interval)
        now = time.time()
        entry = self._cache.get(key)
        if entry and not force and entry.is_fresh(now) and len(entry.df) >= limit * 0.8:
            return entry.df

        try:
            raw = await self.ex.klines(symbol, interval, limit)
        except Exception as exc:                           # noqa: BLE001
            log.debug("klines %s %s failed: %s", symbol, interval, exc)
            return entry.df if entry else None

        df = self._to_df(raw)
        if df.empty or len(df) < 30:
            return entry.df if entry else None

        df = enrich(df)
        self._cache[key] = CacheEntry(df=df, fetched_at=now, interval=interval)
        return df

    async def get_multi(self, symbol: str, timeframes: List[str],
                        candles: Dict[str, int]) -> Dict[str, pd.DataFrame]:
        """Fetch several timeframes for one symbol concurrently."""
        coros = [self.get_klines(symbol, tf, candles.get(tf, 300)) for tf in timeframes]
        results = await gather_limited(coros, limit=len(timeframes))
        out: Dict[str, pd.DataFrame] = {}
        for tf, res in zip(timeframes, results):
            if isinstance(res, Exception) or res is None or len(res) < 30:
                continue
            out[tf] = res
        return out

    async def batch_get(self, symbols: List[str], interval: str,
                        limit: int = 300) -> Dict[str, pd.DataFrame]:
        coros = [self.get_klines(s, interval, limit) for s in symbols]
        results = await gather_limited(coros, limit=self.s.http_concurrency)
        return {s: r for s, r in zip(symbols, results)
                if isinstance(r, pd.DataFrame) and not r.empty}

    # ------------------------------------------------------------------ #
    # housekeeping
    # ------------------------------------------------------------------ #
    def prune_cache(self, keep_symbols: Optional[List[str]] = None) -> int:
        keep = set(keep_symbols or self.universe)
        removed = 0
        for key in list(self._cache.keys()):
            sym = key.split("|")[0]
            if sym not in keep:
                del self._cache[key]
                removed += 1
        return removed

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    async def spread_pct(self, symbol: str) -> float:
        try:
            bt = await self.ex.book_ticker(symbol)
            bid = safe_float(bt.get("bidPrice"))
            ask = safe_float(bt.get("askPrice"))
            if bid > 0 and ask > 0:
                return (ask - bid) / ((ask + bid) / 2) * 100
        except Exception:                                  # noqa: BLE001
            pass
        return 0.0
