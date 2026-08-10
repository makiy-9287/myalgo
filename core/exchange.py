"""
Binance USDⓈ-M Futures — public market data client.

Deliberately keyless. Every endpoint here is public, so there is no HMAC
signing, no timestamp/recvWindow dance, and no way for this process to place,
modify or cancel an order even if something went badly wrong. That is a
security property, not just a simplification: the code that could lose money
does not exist.

Production details that matter:
  * request weight is read back from the response headers and used to throttle
    before Binance bans the IP, rather than reacting to a 418 afterwards
  * 429 responses honour Retry-After
  * a bounded semaphore caps concurrency so a 200-symbol scan cannot open 200
    sockets at once
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import aiohttp

from config.settings import Settings
from utils.logger import get_logger

log = get_logger("exchange")

# Binance allows 2400 weight/minute per IP. We self-throttle well below it.
WEIGHT_LIMIT = 2400
WEIGHT_SOFT_CEILING = 1900


class BinanceError(Exception):
    def __init__(self, code: int, msg: str, endpoint: str = ""):
        self.code = code
        self.msg = msg
        self.endpoint = endpoint
        super().__init__(f"[{code}] {msg} ({endpoint})")


class BinanceFutures:
    """Async public-data client. Safe to share across tasks."""

    def __init__(self, settings: Settings):
        self.s = settings
        self.base = settings.rest_base
        self._session: Optional[aiohttp.ClientSession] = None
        self._sem = asyncio.Semaphore(settings.http_concurrency)
        self._weight = 0
        self._weight_reset = time.time() + 60
        self._throttle_until = 0.0

    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        if self._session and not self._session.closed:
            return
        timeout = aiohttp.ClientTimeout(total=self.s.http_timeout)
        connector = aiohttp.TCPConnector(limit=self.s.http_concurrency * 2,
                                         ttl_dns_cache=300)
        self._session = aiohttp.ClientSession(
            timeout=timeout, connector=connector,
            headers={"User-Agent": "bfs-signal-engine/2.0"})

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_):
        await self.close()

    # ------------------------------------------------------------------ #
    @property
    def weight_used(self) -> int:
        if time.time() > self._weight_reset:
            return 0
        return self._weight

    async def _wait_for_budget(self) -> None:
        now = time.time()
        if now < self._throttle_until:
            await asyncio.sleep(self._throttle_until - now)
        if now > self._weight_reset:
            self._weight = 0
            self._weight_reset = now + 60
        elif self._weight > WEIGHT_SOFT_CEILING:
            wait = max(0.0, self._weight_reset - now)
            log.warning("Weight budget %d/%d — pausing %.1fs",
                        self._weight, WEIGHT_LIMIT, wait)
            await asyncio.sleep(wait + 0.25)
            self._weight = 0
            self._weight_reset = time.time() + 60

    # ------------------------------------------------------------------ #
    async def _request(self, path: str, params: Dict[str, Any] | None = None,
                       retries: int = 3) -> Any:
        if self._session is None or self._session.closed:
            await self.start()

        url = f"{self.base}{path}"
        params = {k: v for k, v in (params or {}).items() if v is not None}
        last_exc: Exception | None = None

        for attempt in range(retries):
            await self._wait_for_budget()
            try:
                async with self._sem:
                    async with self._session.get(url, params=params) as resp:
                        used = resp.headers.get("X-MBX-USED-WEIGHT-1M")
                        if used and used.isdigit():
                            self._weight = int(used)
                            self._weight_reset = max(self._weight_reset,
                                                     time.time() + 60)

                        if resp.status == 429 or resp.status == 418:
                            retry_after = float(resp.headers.get("Retry-After", 5))
                            self._throttle_until = time.time() + retry_after
                            log.warning("Rate limited on %s — backing off %.0fs",
                                        path, retry_after)
                            await asyncio.sleep(retry_after)
                            continue

                        data = await resp.json(content_type=None)

                        if resp.status >= 400:
                            code = (data or {}).get("code", resp.status)
                            msg = (data or {}).get("msg", str(data)[:200])
                            raise BinanceError(int(code), str(msg), path)
                        return data

            except BinanceError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_exc = exc
                wait = 0.6 * (2 ** attempt)
                log.debug("%s attempt %d failed (%s), retry in %.1fs",
                          path, attempt + 1, exc, wait)
                await asyncio.sleep(wait)

        raise BinanceError(-1, f"network failure after {retries} attempts: {last_exc}",
                           path)

    # ------------------------------------------------------------------ #
    # public market data
    # ------------------------------------------------------------------ #
    async def ping(self) -> Dict:
        return await self._request("/fapi/v1/ping")

    async def server_time(self) -> int:
        data = await self._request("/fapi/v1/time")
        return int(data.get("serverTime", 0))

    async def exchange_info(self) -> Dict:
        return await self._request("/fapi/v1/exchangeInfo")

    async def ticker_24h(self) -> List[Dict]:
        return await self._request("/fapi/v1/ticker/24hr")

    async def ticker_price(self, symbol: str | None = None):
        return await self._request("/fapi/v1/ticker/price",
                                   {"symbol": symbol} if symbol else None)

    async def klines(self, symbol: str, interval: str, limit: int = 300) -> List[List]:
        return await self._request("/fapi/v1/klines",
                                   {"symbol": symbol, "interval": interval,
                                    "limit": min(limit, 1500)})

    async def book_ticker(self, symbol: str | None = None):
        return await self._request("/fapi/v1/ticker/bookTicker",
                                   {"symbol": symbol} if symbol else None)

    async def mark_price(self, symbol: str | None = None):
        return await self._request("/fapi/v1/premiumIndex",
                                   {"symbol": symbol} if symbol else None)

    async def open_interest_hist(self, symbol: str, period: str = "5m",
                                 limit: int = 30):
        return await self._request("/futures/data/openInterestHist",
                                   {"symbol": symbol, "period": period,
                                    "limit": limit})

    async def long_short_ratio(self, symbol: str, period: str = "15m",
                               limit: int = 30):
        return await self._request("/futures/data/globalLongShortAccountRatio",
                                   {"symbol": symbol, "period": period,
                                    "limit": limit})
