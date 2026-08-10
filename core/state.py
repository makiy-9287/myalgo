"""
Durable state.

Storage policy, as requested: **only signals that actually filled are ever
written to disk.** A pending signal that expires without being touched is
deleted outright — no record, no history line, nothing. The store therefore
only ever grows with things that really happened.

    pending{}   in-memory only, lost on restart (by design)
    live{}      filled, still running — persisted
    outcomes[]  completed — persisted, feeds /report

Writes are atomic (tmp + replace) so a crash mid-write cannot corrupt the file.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional

from config.settings import DATA_DIR
from core.models import CLOSED, FILLED, PENDING, Outcome, Signal
from utils.logger import get_logger

log = get_logger("state")

STATE_FILE = DATA_DIR / "state.json"
MAX_OUTCOMES = 2000


class StateStore:
    def __init__(self):
        self._lock = threading.RLock()
        self.pending: Dict[str, Signal] = {}      # key -> Signal (memory only)
        self.live: Dict[str, Signal] = {}         # key -> Signal (persisted)
        self.outcomes: List[Outcome] = []         # completed (persisted)
        self.cooldowns: Dict[str, float] = {}
        self.stats: Dict[str, Any] = {
            "signals_published": 0,
            "signals_filled": 0,
            "signals_expired": 0,
            "started_at": time.time(),
        }
        self.load()

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        if not STATE_FILE.exists():
            return
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.error("State load failed: %s", exc)
            return

        with self._lock:
            for key, sd in (data.get("live") or {}).items():
                try:
                    self.live[key] = Signal.from_dict(sd)
                except (TypeError, KeyError) as exc:
                    log.warning("Skipping bad live signal %s: %s", key, exc)
            for od in (data.get("outcomes") or []):
                try:
                    self.outcomes.append(Outcome.from_dict(od))
                except (TypeError, KeyError):
                    continue
            self.cooldowns = {k: float(v)
                              for k, v in (data.get("cooldowns") or {}).items()}
            saved = data.get("stats") or {}
            self.stats.update({k: v for k, v in saved.items() if k in self.stats})

        log.info("State restored: %d live signals, %d outcomes",
                 len(self.live), len(self.outcomes))

    def save(self) -> None:
        with self._lock:
            payload = {
                "live": {k: v.to_dict(include_confirmations=False)
                         for k, v in self.live.items()},
                "outcomes": [o.to_dict() for o in self.outcomes[-MAX_OUTCOMES:]],
                "cooldowns": self.cooldowns,
                "stats": self.stats,
                "saved_at": time.time(),
            }
        try:
            tmp = STATE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2, default=str),
                           encoding="utf-8")
            tmp.replace(STATE_FILE)
        except OSError as exc:
            log.error("State save failed: %s", exc)

    # ------------------------------------------------------------------ #
    # cooldowns
    # ------------------------------------------------------------------ #
    def on_cooldown(self, key: str) -> bool:
        with self._lock:
            until = self.cooldowns.get(key, 0)
            if until > time.time():
                return True
            self.cooldowns.pop(key, None)
            return False

    def set_cooldown(self, key: str, minutes: int) -> None:
        with self._lock:
            self.cooldowns[key] = time.time() + minutes * 60

    def purge_cooldowns(self) -> None:
        now = time.time()
        with self._lock:
            for k in [k for k, v in self.cooldowns.items() if v < now]:
                del self.cooldowns[k]

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def publish(self, sig: Signal) -> None:
        """Signal has been sent to Telegram; now waiting for an entry fill."""
        with self._lock:
            sig.status = PENDING
            self.pending[sig.key] = sig
            self.stats["signals_published"] += 1

    def mark_filled(self, sig: Signal, price: float) -> None:
        with self._lock:
            sig.status = FILLED
            sig.filled_at = time.time()
            sig.fill_price = price
            sig.peak_price = price
            sig.trough_price = price
            sig.outcome = "OPEN"
            self.pending.pop(sig.key, None)
            self.live[sig.key] = sig
            self.stats["signals_filled"] += 1
        self.save()

    def drop_expired(self) -> List[Signal]:
        """
        Remove pending signals past their validity window.

        They are dropped, not archived: an entry that never triggered is not a
        trade and would only pollute the statistics.
        """
        now = time.time()
        gone: List[Signal] = []
        with self._lock:
            for key, sig in list(self.pending.items()):
                if sig.expires_at and now > sig.expires_at:
                    sig.status = "EXPIRED"
                    gone.append(sig)
                    del self.pending[key]
                    self.stats["signals_expired"] += 1
        return gone

    def close_signal(self, sig: Signal, price: float, reason: str) -> Outcome:
        """Finalise a live signal into an immutable Outcome."""
        entry = sig.fill_price or sig.entry
        move = (price - entry) if sig.is_long else (entry - price)
        risk = abs(entry - sig.stop_loss)
        r_mult = round(move / risk, 3) if risk > 0 else 0.0
        pct = round(move / entry * 100.0, 3) if entry else 0.0

        outcome = Outcome(
            signal_id=sig.signal_id, symbol=sig.symbol, side=sig.side,
            mode=sig.mode, strategy=sig.strategy, entry=sig.entry,
            stop_loss=sig.stop_loss, fill_price=entry, exit_price=price,
            result=reason, tp_hits=sig.tp_hits, r_multiple=r_mult, pct=pct,
            filled_at=sig.filled_at, closed_at=time.time(),
        )
        with self._lock:
            sig.status = CLOSED
            sig.closed_at = outcome.closed_at
            sig.close_reason = reason
            sig.outcome = "WIN" if outcome.is_win else "LOSS"
            self.live.pop(sig.key, None)
            self.outcomes.append(outcome)
            if len(self.outcomes) > MAX_OUTCOMES:
                del self.outcomes[:-MAX_OUTCOMES]
        self.save()
        return outcome

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #
    def all_tracked(self) -> List[Signal]:
        with self._lock:
            return list(self.pending.values()) + list(self.live.values())

    def live_signals(self) -> List[Signal]:
        with self._lock:
            return sorted(self.live.values(), key=lambda s: s.filled_at)

    def pending_signals(self) -> List[Signal]:
        with self._lock:
            return sorted(self.pending.values(), key=lambda s: s.created_at)

    def find(self, symbol: str) -> List[Signal]:
        symbol = symbol.upper()
        return [s for s in self.all_tracked() if s.symbol == symbol]

    @property
    def live_count(self) -> int:
        with self._lock:
            return len(self.live)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self.pending)

    # ------------------------------------------------------------------ #
    # reporting
    # ------------------------------------------------------------------ #
    def report(self, mode: Optional[str] = None,
               strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        Aggregate completed outcomes.

        Win definition, per spec: reaching TP1 before the stop is a win, even
        if the trade later reverses and stops out at breakeven. That measures
        whether the entry was right, which is what a signal engine is judged on.
        """
        with self._lock:
            rows = [o for o in self.outcomes
                    if (mode is None or o.mode == mode)
                    and (strategy is None or o.strategy == strategy)]

        total = len(rows)
        tp1 = sum(1 for o in rows if o.tp_hits >= 1)
        tp2 = sum(1 for o in rows if o.tp_hits >= 2)
        tp3 = sum(1 for o in rows if o.tp_hits >= 3)
        # A stop reached after banking TP1 is still a stop-out — it just is not
        # a losing one. `clean_sl` is the subset that never reached a target.
        sl = sum(1 for o in rows if o.result.startswith("SL"))
        clean_sl = sum(1 for o in rows if o.result.startswith("SL") and o.tp_hits == 0)
        wins = tp1
        losses = total - wins

        by_mode: Dict[str, Dict[str, int]] = {}
        for o in rows:
            m = by_mode.setdefault(o.mode, {"n": 0, "w": 0})
            m["n"] += 1
            if o.is_win:
                m["w"] += 1

        by_strategy: Dict[str, Dict[str, int]] = {}
        for o in rows:
            m = by_strategy.setdefault(o.strategy, {"n": 0, "w": 0})
            m["n"] += 1
            if o.is_win:
                m["w"] += 1

        reversed_tp1 = sum(1 for o in rows if o.reversed_after_tp1)
        stalled_tp2 = sum(1 for o in rows if o.stalled_at_tp2)
        # of everything that reached TP1, how much went on to TP2?
        followed_through = (tp2 / tp1 * 100.0) if tp1 else 0.0

        r_values = [o.r_multiple for o in rows]
        return {
            "total": total,
            "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "sl": sl, "clean_sl": clean_sl,
            "wins": wins, "losses": losses,
            "reversed_after_tp1": reversed_tp1,
            "stalled_at_tp2": stalled_tp2,
            "followed_through": followed_through,
            "win_rate": (wins / total * 100.0) if total else 0.0,
            "avg_r": (sum(r_values) / len(r_values)) if r_values else 0.0,
            "total_r": sum(r_values),
            "best_r": max(r_values) if r_values else 0.0,
            "worst_r": min(r_values) if r_values else 0.0,
            "by_mode": by_mode,
            "by_strategy": by_strategy,
            "published": self.stats.get("signals_published", 0),
            "filled": self.stats.get("signals_filled", 0),
            "expired": self.stats.get("signals_expired", 0),
        }
