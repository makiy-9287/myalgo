#!/usr/bin/env python3
"""
Binance Futures Multi-Timeframe Signal Engine
=============================================

Entry point. Run with:  python main.py

Signal-only: this process reads public market data and sends Telegram
messages. It holds no API keys and cannot place an order.

Preflight runs before anything touches the network, so a misconfigured .env
fails loudly in one second instead of silently producing no signals.
"""
from __future__ import annotations

import asyncio
import signal
import sys

from config.settings import LOG_DIR, RUNTIME, SETTINGS
from utils.logger import get_logger, setup_logging

setup_logging(SETTINGS.log_level, LOG_DIR)
log = get_logger("main")

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║   BINANCE FUTURES · MULTI-TIMEFRAME SIGNAL ENGINE            ║
║   Liquidity Sweep + MSS   ·   Order Block + FVG              ║
║   Modes: DAY · SWING              Signal-only, no API keys   ║
╚══════════════════════════════════════════════════════════════╝
"""


def preflight() -> bool:
    ok = True

    if not SETTINGS.telegram_token:
        log.error("TELEGRAM_BOT_TOKEN is not set — no signals can be delivered.")
        ok = False
    if not SETTINGS.telegram_chat_id:
        log.error("TELEGRAM_CHAT_ID is not set — no signals can be delivered.")
        ok = False

    if SETTINGS.analysis_interval < 30:
        log.warning("ANALYSIS_INTERVAL below 30s risks hitting Binance rate limits.")
    if SETTINGS.tracker_interval < 5:
        log.warning("TRACKER_INTERVAL below 5s spends weight budget for little gain.")

    low = [m for m, v in RUNTIME.min_score.items() if v < 80]
    if low:
        log.warning("Minimum score below 80 for %s — expect more signals of "
                    "lower quality.", ", ".join(low))

    if not any(RUNTIME.modes_enabled.values()):
        log.error("Every mode is disabled — nothing will ever be scanned.")
        ok = False
    if not any(RUNTIME.strategies_enabled.values()):
        log.error("Both strategies are disabled — nothing will ever fire.")
        ok = False

    return ok


async def _run() -> None:
    from engine.orchestrator import Application

    app = Application(SETTINGS, RUNTIME)
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _request_stop() -> None:
        if not stop_event.is_set():
            log.info("Shutdown signal received…")
            stop_event.set()

    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(getattr(signal, sig_name), _request_stop)
        except (NotImplementedError, AttributeError):
            pass                                   # Windows

    try:
        await app.start()
        await stop_event.wait()
    finally:
        await app.stop()


def main() -> int:
    print(BANNER)
    if not preflight():
        log.error("Preflight failed. Fix the .env and try again.")
        return 1
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
