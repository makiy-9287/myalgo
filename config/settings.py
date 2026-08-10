"""
Configuration layer.

Two tiers:
  1. Static config  -> .env  (Telegram credentials, universe, timing)
  2. Runtime config -> data/runtime_config.json (changed live from Telegram)

This build is SIGNAL-ONLY. It never places an order, so it never needs a
Binance API key — every endpoint it touches is public market data.

Anything a user may want to flip mid-session lives in RuntimeConfig so that a
Telegram button or command can mutate it atomically and it survives a restart.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

RUNTIME_FILE = DATA_DIR / "runtime_config.json"


# --------------------------------------------------------------------------- #
# .env loading (no external dependency, tolerant parser)
# --------------------------------------------------------------------------- #
def load_dotenv(path: Path | None = None) -> None:
    path = path or (BASE_DIR / ".env")
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if " #" in val:                       # trailing inline comment
            val = val.split(" #")[0].strip()
        if key and key not in os.environ:
            os.environ[key] = val


def _env(key: str, default: Any = None, cast=str):
    val = os.environ.get(key)
    if val is None or val == "":
        return default
    try:
        if cast is bool:
            return str(val).strip().lower() in ("1", "true", "yes", "on", "y")
        return cast(val)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Static settings
# --------------------------------------------------------------------------- #
@dataclass
class Settings:
    # --- Telegram ---
    telegram_token: str = ""
    telegram_chat_id: str = ""
    telegram_admin_ids: str = ""          # comma separated; empty = chat_id only

    # --- Universe ---
    min_volume_usdt: float = 10_000_000.0
    max_symbols: int = 220
    quote_asset: str = "USDT"
    symbol_blacklist: str = ""

    # --- Loop timing (seconds) ---
    analysis_interval: int = 300          # deep scan every 5 minutes
    universe_refresh_interval: int = 10800  # rebuild the coin list every 3h
    tracker_interval: int = 15            # entry-fill / TP / SL polling
    telegram_poll_timeout: int = 25

    # --- Networking ---
    http_concurrency: int = 12
    http_timeout: int = 20

    # --- Logging ---
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            telegram_token=_env("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=_env("TELEGRAM_CHAT_ID", ""),
            telegram_admin_ids=_env("TELEGRAM_ADMIN_IDS", ""),
            min_volume_usdt=_env("MIN_VOLUME_USDT", 10_000_000.0, float),
            max_symbols=_env("MAX_SYMBOLS", 220, int),
            quote_asset=_env("QUOTE_ASSET", "USDT"),
            symbol_blacklist=_env("SYMBOL_BLACKLIST", ""),
            analysis_interval=_env("ANALYSIS_INTERVAL", 300, int),
            universe_refresh_interval=_env("UNIVERSE_REFRESH_INTERVAL", 10800, int),
            tracker_interval=_env("TRACKER_INTERVAL", 15, int),
            telegram_poll_timeout=_env("TELEGRAM_POLL_TIMEOUT", 25, int),
            http_concurrency=_env("HTTP_CONCURRENCY", 12, int),
            http_timeout=_env("HTTP_TIMEOUT", 20, int),
            log_level=_env("LOG_LEVEL", "INFO").upper(),
        )

    @property
    def admin_ids(self) -> set[str]:
        ids = {i.strip() for i in self.telegram_admin_ids.split(",") if i.strip()}
        if self.telegram_chat_id:
            ids.add(self.telegram_chat_id.strip())
        return ids

    @property
    def blacklist(self) -> set[str]:
        return {s.strip().upper() for s in self.symbol_blacklist.split(",") if s.strip()}

    @property
    def rest_base(self) -> str:
        return "https://fapi.binance.com"


# --------------------------------------------------------------------------- #
# Runtime (Telegram-mutable) settings
# --------------------------------------------------------------------------- #
@dataclass
class RuntimeConfig:
    """Everything toggleable from Telegram. Thread/async safe via lock."""
    signals_enabled: bool = True
    paused: bool = False

    modes_enabled: Dict[str, bool] = field(
        default_factory=lambda: {"DAY": True, "SWING": True}
    )
    strategies_enabled: Dict[str, bool] = field(
        default_factory=lambda: {"SWEEP_MSS": True, "OB_FVG": True}
    )

    # Minimum confluence score (0-100) required per mode.
    # 90 is deliberately brutal: it demands nearly every confirmation in the
    # 100-point budget, so only textbook setups survive.
    min_score: Dict[str, float] = field(
        default_factory=lambda: {"DAY": 90.0, "SWING": 90.0}
    )
    # Friday 23:00 UTC -> Monday 20:00 UTC. Not a weekend flag: it brackets
    # the traditional close and reopen, when fakeouts are most frequent.
    danger_score_bonus: float = 5.0
    danger_extra_confirmations: int = 3
    danger_window_enabled: bool = True

    min_rr: float = 6.0
    max_signals_per_cycle: int = 4
    signal_cooldown_min: int = 120       # per symbol+mode+strategy
    max_spread_pct: float = 0.08
    min_atr_pct: float = 0.12           # dead-market filter
    max_atr_pct: float = 6.0            # chaos filter
    funding_extreme: float = 0.0009     # abs funding rate blocker

    # --- alerting ---
    alert_on_fill: bool = True
    alert_on_tp: bool = True
    alert_on_sl: bool = True
    alert_on_expiry: bool = True
    delete_expired_message: bool = True   # remove the original card on expiry

    _lock: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_lock", threading.RLock())

    # ---------------- persistence ---------------- #
    def to_dict(self) -> Dict[str, Any]:
        # NOTE: dataclasses.asdict() deep-copies every field and chokes on the
        # RLock, so the dict is assembled field-by-field instead.
        out: Dict[str, Any] = {}
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            val = getattr(self, f.name)
            out[f.name] = dict(val) if isinstance(val, dict) else val
        return out

    def save(self) -> None:
        with self._lock:
            tmp = RUNTIME_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
            tmp.replace(RUNTIME_FILE)

    @classmethod
    def load(cls, settings: Settings | None = None) -> "RuntimeConfig":
        rc = cls()
        if RUNTIME_FILE.exists():
            try:
                saved = json.loads(RUNTIME_FILE.read_text(encoding="utf-8"))
                for k, v in saved.items():
                    if k.startswith("_") or not hasattr(rc, k):
                        continue
                    cur = getattr(rc, k)
                    if isinstance(cur, dict) and isinstance(v, dict):
                        cur.update(v)
                    else:
                        setattr(rc, k, v)
            except (json.JSONDecodeError, OSError):
                pass
        return rc

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            setattr(self, key, value)
        self.save()

    def toggle(self, key: str) -> bool:
        with self._lock:
            new = not bool(getattr(self, key))
            setattr(self, key, new)
        self.save()
        return new


SETTINGS = Settings.from_env()
RUNTIME = RuntimeConfig.load(SETTINGS)
