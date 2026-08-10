from __future__ import annotations

import asyncio
import functools
import math
import random
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP, InvalidOperation
from typing import Any, Callable, Iterable, List, Sequence

from utils.logger import get_logger

log = get_logger("helpers")


# --------------------------------------------------------------------------- #
# time
# --------------------------------------------------------------------------- #
def now_ms() -> int:
    return int(time.time() * 1000)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def fmt_ts(ts: float | int | None = None, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    if ts is None:
        dt = utcnow()
    else:
        if ts > 1e11:            # milliseconds
            ts = ts / 1000.0
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime(fmt) + " UTC"


def is_weekend(dt: datetime | None = None) -> bool:
    """Retained for compatibility; the engine uses in_danger_window()."""
    d = dt or utcnow()
    return d.weekday() >= 5


def in_danger_window(dt: datetime | None = None) -> bool:
    """
    Friday 23:00 UTC through Monday 20:00 UTC.

    This window brackets the traditional market close and reopen. Crypto never
    stops, but the capital that moves it does: desks flatten into the weekend,
    books thin out, and the same liquidity pools that would take a week to
    build get raided in hours by size that no longer has anything to trade
    against. Breaks fail, structure lies, and a level that held all week gives
    way on a Sunday for no reason that survives Monday.

    Signals are not blocked here — they are simply made to work much harder.
    """
    d = dt or utcnow()
    wd, hour = d.weekday(), d.hour        # Mon=0 ... Sun=6
    if wd == 4 and hour >= 23:            # Friday night
        return True
    if wd in (5, 6):                      # Saturday, Sunday
        return True
    if wd == 0 and hour < 20:             # Monday until the reopen settles
        return True
    return False


def danger_window_label(dt: datetime | None = None) -> str:
    d = dt or utcnow()
    wd = d.weekday()
    return {4: "Friday close", 5: "Saturday", 6: "Sunday",
            0: "Monday reopen"}.get(wd, "")


def human_delta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}d {h}h {m}m"
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# --------------------------------------------------------------------------- #
# numeric / precision
# --------------------------------------------------------------------------- #
def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def step_size_to_precision(step: float | str) -> int:
    """0.001 -> 3 ; 1 -> 0 ; 10 -> 0"""
    try:
        d = Decimal(str(step)).normalize()
    except InvalidOperation:
        return 8
    exp = d.as_tuple().exponent
    return max(0, -int(exp))


def round_step(value: float, step: float, mode: str = "down") -> float:
    """Round `value` to a multiple of `step` (Binance LOT_SIZE / PRICE_FILTER)."""
    if step is None or step <= 0:
        return float(value)
    dv, ds = Decimal(str(value)), Decimal(str(step))
    rounding = ROUND_UP if mode == "up" else ROUND_DOWN
    quantised = (dv / ds).quantize(Decimal("1"), rounding=rounding) * ds
    return float(quantised)


def fmt_price(value: float, tick: float | None = None) -> str:
    """Human formatting that keeps enough decimals for micro-cap coins."""
    v = safe_float(value)
    if tick:
        p = step_size_to_precision(tick)
        return f"{v:.{p}f}"
    av = abs(v)
    if av >= 1000:
        return f"{v:,.2f}"
    if av >= 10:
        return f"{v:.3f}"
    if av >= 0.1:
        return f"{v:.4f}"
    if av >= 0.001:
        return f"{v:.6f}"
    return f"{v:.8f}"


def pct(a: float, b: float) -> float:
    """Percentage distance of a from b."""
    if not b:
        return 0.0
    return (a - b) / abs(b) * 100.0


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def chunked(seq: Sequence, size: int) -> Iterable[List]:
    for i in range(0, len(seq), size):
        yield list(seq[i:i + size])


# --------------------------------------------------------------------------- #
# async retry
# --------------------------------------------------------------------------- #
class RetryableError(Exception):
    """Raised for transient failures that should be retried."""


class FatalError(Exception):
    """Raised for failures where retrying is pointless (bad params, no margin)."""


def async_retry(attempts: int = 5, base_delay: float = 0.8,
                max_delay: float = 12.0, exceptions: tuple = (Exception,)):
    """Exponential backoff with jitter. FatalError always aborts immediately."""
    def decorator(fn: Callable):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            last = None
            for i in range(1, attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except FatalError:
                    raise
                except exceptions as exc:            # noqa: PERF203
                    last = exc
                    if i >= attempts:
                        break
                    delay = min(max_delay, base_delay * (2 ** (i - 1)))
                    delay += random.uniform(0, delay * 0.35)
                    log.warning("%s failed (attempt %d/%d): %s | retry in %.1fs",
                                fn.__name__, i, attempts, exc, delay)
                    await asyncio.sleep(delay)
            raise last if last else RuntimeError("retry exhausted")
        return wrapper
    return decorator


async def gather_limited(coros, limit: int = 12):
    """asyncio.gather with a concurrency ceiling; exceptions returned, not raised."""
    sem = asyncio.Semaphore(max(1, limit))

    async def _run(c):
        async with sem:
            try:
                return await c
            except Exception as exc:                 # noqa: BLE001
                return exc

    return await asyncio.gather(*[_run(c) for c in coros])
