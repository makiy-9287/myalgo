from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False
_FMT = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_dir / "bot.log", maxBytes=8 * 1024 * 1024,
            backupCount=5, encoding="utf-8",
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)

        eh = RotatingFileHandler(
            log_dir / "errors.log", maxBytes=4 * 1024 * 1024,
            backupCount=3, encoding="utf-8",
        )
        eh.setFormatter(fmt)
        eh.setLevel(logging.WARNING)
        root.addHandler(eh)

    # Silence noisy third-party loggers
    for noisy in ("aiohttp.access", "asyncio", "aiohttp.client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
