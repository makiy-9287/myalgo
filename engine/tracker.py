"""
Signal tracker.

Watches every published signal against live market data and drives the
lifecycle transitions that generate notifications:

    PENDING  --price enters the zone-->  FILLED     "entry filled" alert
    FILLED   --price reaches TP1/2/3-->  partial    "target hit" alert
    FILLED   --price reaches SL-------->  CLOSED    "stopped out" alert
    PENDING  --validity window ends--->  dropped    "expired" alert + card deleted

Two deliberate design choices worth knowing about:

1. **Fills are detected from bar highs/lows, not closes.** Price frequently
   wicks into an entry zone and closes outside it. A close-only check would
   miss most real entries.

2. **When one bar contains both a target and the stop, the stop wins.** From
   1-minute OHLC there is no way to know which came first, and assuming the
   good outcome would quietly inflate the win rate. Reporting a slightly
   pessimistic result is the only honest option.
"""
from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from config.settings import RuntimeConfig
from core.exchange import BinanceError, BinanceFutures
from core.models import Signal
from core.state import StateStore
from utils.logger import get_logger

log = get_logger("tracker")

NotifyFn = Callable[..., Awaitable[None]]
DeleteFn = Callable[[int], Awaitable[bool]]


class SignalTracker:
    def __init__(self, exchange: BinanceFutures, state: StateStore,
                 runtime: RuntimeConfig,
                 notify: Optional[NotifyFn] = None,
                 delete_message: Optional[DeleteFn] = None):
        self.ex = exchange
        self.state = state
        self.rt = runtime
        self.notify = notify
        self.delete_message = delete_message
        self.last_run = 0.0
        self._last_seen: Dict[str, int] = {}     # symbol -> last kline open_time

    # ------------------------------------------------------------------ #
    async def _notify(self, text: str, silent: bool = False) -> None:
        if self.notify:
            try:
                await self.notify(text, silent=silent)
            except Exception as exc:                        # noqa: BLE001
                log.error("notify failed: %s", exc)

    # ------------------------------------------------------------------ #
    async def _bar_ranges(self, symbols: List[str]) -> Dict[str, Tuple[float, float, float]]:
        """
        (low, high, close) covering everything that happened since the last
        poll, per symbol. Built from 1m klines so nothing is missed between
        polling intervals.
        """
        out: Dict[str, Tuple[float, float, float]] = {}
        if not symbols:
            return out

        async def one(sym: str) -> None:
            try:
                rows = await self.ex.klines(sym, "1m", limit=5)
            except (BinanceError, asyncio.TimeoutError) as exc:
                log.debug("kline fetch failed for %s: %s", sym, exc)
                return
            if not rows:
                return

            since = self._last_seen.get(sym, 0)
            fresh = [r for r in rows if int(r[0]) >= since] or rows[-1:]
            try:
                low = min(float(r[3]) for r in fresh)
                high = max(float(r[2]) for r in fresh)
                close = float(fresh[-1][4])
            except (TypeError, ValueError, IndexError):
                return
            self._last_seen[sym] = int(rows[-1][0])
            out[sym] = (low, high, close)

        await asyncio.gather(*(one(s) for s in symbols), return_exceptions=True)
        return out

    # ------------------------------------------------------------------ #
    async def run_once(self) -> None:
        tracked = self.state.all_tracked()
        if not tracked:
            self.last_run = time.time()
            return

        symbols = sorted({s.symbol for s in tracked})
        ranges = await self._bar_ranges(symbols)

        for sig in tracked:
            rng = ranges.get(sig.symbol)
            if not rng:
                continue
            low, high, close = rng
            try:
                if sig.status == "PENDING":
                    await self._check_fill(sig, low, high, close)
                elif sig.status == "FILLED":
                    await self._check_progress(sig, low, high, close)
            except Exception as exc:                        # noqa: BLE001
                log.error("tracking %s failed: %s", sig.symbol, exc, exc_info=True)

        self.last_run = time.time()

    # ------------------------------------------------------------------ #
    async def _check_fill(self, sig: Signal, low: float, high: float,
                          close: float) -> None:
        if not sig.in_entry_zone(low, high):
            return

        # Price reached the reference level, so that is the fill — clamped
        # into the range the market actually traded so we never claim a price
        # that never printed.
        fill = min(max(sig.entry, low), high)
        self.state.mark_filled(sig, fill)
        log.info("FILLED %s %s %s @ %.6f", sig.symbol, sig.side, sig.mode, fill)

        if self.rt.alert_on_fill:
            from notify.formatter import format_fill
            await self._notify(format_fill(sig, fill, close))

        # A single bar can enter the zone and run straight to a target or the
        # stop, so evaluate progress immediately rather than waiting a cycle.
        await self._check_progress(sig, low, high, close)

    # ------------------------------------------------------------------ #
    async def _check_progress(self, sig: Signal, low: float, high: float,
                              close: float) -> None:
        if sig.is_long:
            sig.peak_price = max(sig.peak_price or high, high)
            sig.trough_price = min(sig.trough_price or low, low)
            stop_hit = low <= sig.stop_loss
        else:
            sig.peak_price = min(sig.peak_price or low, low)
            sig.trough_price = max(sig.trough_price or high, high)
            stop_hit = high >= sig.stop_loss

        newly_hit = []
        for tp in sig.take_profits:
            if tp.hit:
                continue
            reached = (high >= tp.price) if sig.is_long else (low <= tp.price)
            if reached:
                tp.hit = True
                tp.hit_at = time.time()
                newly_hit.append(tp)
            else:
                break                     # targets are ordered; stop at the first miss

        # Stop and target in the same bar: assume the stop came first unless a
        # target was already banked on an earlier bar.
        if stop_hit and newly_hit and sig.tp_hits == len(newly_hit):
            for tp in newly_hit:
                tp.hit = False
                tp.hit_at = 0.0
            newly_hit = []

        for tp in newly_hit:
            log.info("TP%d hit %s %s @ %.6f", tp.level, sig.symbol, sig.side, tp.price)
            if self.rt.alert_on_tp:
                from notify.formatter import format_tp_hit
                await self._notify(format_tp_hit(sig, tp, close))

        if newly_hit and newly_hit[-1].level >= len(sig.take_profits):
            outcome = self.state.close_signal(sig, newly_hit[-1].price, "TP3")
            if self.rt.alert_on_tp:
                from notify.formatter import format_closed
                await self._notify(format_closed(sig, outcome))
            return

        if stop_hit:
            reason = "SL" if sig.tp_hits == 0 else f"SL after TP{sig.tp_hits}"
            outcome = self.state.close_signal(sig, sig.stop_loss, reason)
            log.info("STOPPED %s %s (%s)", sig.symbol, sig.side, reason)
            if self.rt.alert_on_sl:
                from notify.formatter import format_closed
                await self._notify(format_closed(sig, outcome))

    # ------------------------------------------------------------------ #
    async def handle_expiries(self) -> int:
        """
        Drop pending signals whose window closed, delete their Telegram card,
        and (optionally) post a one-line notice.
        """
        expired = self.state.drop_expired()
        for sig in expired:
            log.info("EXPIRED %s %s %s (never filled)", sig.symbol, sig.side, sig.mode)

            if self.rt.delete_expired_message and sig.message_id and self.delete_message:
                try:
                    await self.delete_message(sig.message_id)
                except Exception as exc:                    # noqa: BLE001
                    log.debug("could not delete message %s: %s", sig.message_id, exc)

            if self.rt.alert_on_expiry:
                from notify.formatter import format_expired
                await self._notify(format_expired(sig), silent=True)

        if expired:
            self.state.save()
        return len(expired)

    # ------------------------------------------------------------------ #
    async def snapshot(self) -> List[Dict]:
        """Current PnL of every live signal — backs /pnl."""
        live = self.state.live_signals()
        if not live:
            return []

        ranges = await self._bar_ranges(sorted({s.symbol for s in live}))
        rows = []
        for sig in live:
            rng = ranges.get(sig.symbol)
            price = rng[2] if rng else (sig.fill_price or sig.entry)
            u = sig.unrealised(price)
            nxt = next((t for t in sig.take_profits if not t.hit), None)
            rows.append({
                "signal": sig, "price": price,
                "pct": u["pct"], "r": u["r"], "to_target": u["to_target"],
                "next_tp": nxt, "tp_hits": sig.tp_hits,
                "age_min": (time.time() - sig.filled_at) / 60.0,
            })
        rows.sort(key=lambda r: r["r"], reverse=True)
        return rows
