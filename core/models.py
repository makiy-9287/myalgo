"""
Domain models.

There is no `Trade` any more — nothing is executed. A Signal now carries its
own lifecycle instead:

    PENDING  -> published, waiting for price to trade into the entry zone
    FILLED   -> price entered the zone; the signal is now live and tracked
    CLOSED   -> full TP ladder completed, or stop hit
    EXPIRED  -> never filled inside the validity window (discarded entirely)

Only FILLED and CLOSED signals are ever written to disk. Pending signals that
expire leave no trace, which is what keeps the store tidy.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# lifecycle constants
PENDING = "PENDING"
FILLED = "FILLED"
CLOSED = "CLOSED"
EXPIRED = "EXPIRED"


# --------------------------------------------------------------------------- #
@dataclass
class Confirmation:
    """One discrete reason the setup is valid. Weight feeds the score."""
    name: str
    detail: str
    weight: float
    timeframe: str = ""
    category: str = "general"      # bias | structure | liquidity | momentum | volume | risk | context

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TakeProfit:
    level: int
    price: float
    rr: float
    reason: str                     # which liquidity pool / zone it targets
    allocation: float               # portion of the position closed here
    hit: bool = False
    hit_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MTFView:
    """Per-timeframe read shown in the Telegram message."""
    timeframe: str
    role: str                       # BIAS | STRUCTURE | SETUP | TRIGGER
    trend: str
    note: str
    aligned: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
@dataclass
class Signal:
    symbol: str
    side: str                       # LONG | SHORT
    mode: str                       # DAY | SWING
    strategy: str                   # SWEEP_MSS | OB_FVG

    entry: float
    entry_low: float
    entry_high: float
    stop_loss: float
    take_profits: List[TakeProfit] = field(default_factory=list)

    score: float = 0.0
    required_score: float = 0.0
    confirmations: List[Confirmation] = field(default_factory=list)
    mtf: List[MTFView] = field(default_factory=list)

    risk_pct: float = 0.0           # SL distance as % of entry
    rr_total: float = 0.0
    atr_pct: float = 0.0
    quote_volume: float = 0.0
    funding_rate: float = 0.0

    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    weekend: bool = False
    signal_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10].upper())

    # ---- lifecycle ----
    status: str = PENDING
    message_id: Optional[int] = None      # Telegram card, so it can be deleted
    filled_at: float = 0.0
    fill_price: float = 0.0
    closed_at: float = 0.0
    close_reason: str = ""                # TP1 | TP2 | TP3 | SL | ...
    outcome: str = ""                     # WIN | LOSS | OPEN
    peak_price: float = 0.0               # best excursion since fill
    trough_price: float = 0.0             # worst excursion since fill
    notes: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    @property
    def key(self) -> str:
        return f"{self.symbol}:{self.mode}:{self.strategy}"

    @property
    def is_long(self) -> bool:
        return self.side == "LONG"

    @property
    def expired(self) -> bool:
        return self.expires_at > 0 and time.time() > self.expires_at

    @property
    def risk(self) -> float:
        """Absolute price distance from entry to stop."""
        base = self.fill_price or self.entry
        return abs(base - self.stop_loss)

    @property
    def tp_hits(self) -> int:
        return sum(1 for t in self.take_profits if t.hit)

    def in_entry_zone(self, low: float, high: float) -> bool:
        """
        Has price actually taken the entry?

        NOT mere overlap with the zone. Price has to trade through `entry` —
        the reference level at the middle of the zone — in the direction that
        gives the trade its edge: down to it for a long, up to it for a short.

        Grazing the near edge is not a fill. On a short published at
        201.89-200.28 with TP1 at 199.50, recording a fill the moment price
        touches 200.28 books an entry nobody could realistically have got and
        hands back most of the reward before the trade starts. Requiring the
        midpoint means the level has to be respected, not brushed.

        The check uses the bar's high/low rather than its close, because a
        wick through the level is a genuine fill.
        """
        return (low <= self.entry) if self.is_long else (high >= self.entry)

    def unrealised(self, price: float) -> Dict[str, float]:
        """Progress of a filled signal, from entry toward TP or SL."""
        entry = self.fill_price or self.entry
        if entry <= 0:
            return {"pct": 0.0, "r": 0.0, "to_target": 0.0}

        move = (price - entry) if self.is_long else (entry - price)
        pct = move / entry * 100.0
        risk = self.risk
        r = (move / risk) if risk > 0 else 0.0

        # distance to the next unhit target, as a percentage of the way there
        nxt = next((t for t in self.take_profits if not t.hit), None)
        if nxt:
            span = abs(nxt.price - entry)
            to_target = (move / span * 100.0) if span > 0 else 0.0
        else:
            to_target = 100.0
        return {"pct": pct, "r": r, "to_target": max(-999.0, min(999.0, to_target))}

    # ------------------------------------------------------------------ #
    def to_dict(self, include_confirmations: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        d["take_profits"] = [t.to_dict() for t in self.take_profits]
        d["mtf"] = [m.to_dict() for m in self.mtf]
        if include_confirmations:
            d["confirmations"] = [c.to_dict() for c in self.confirmations]
        else:
            d.pop("confirmations", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Signal":
        d = dict(d)
        tps = [TakeProfit(**t) for t in d.pop("take_profits", [])]
        mtf = [MTFView(**m) for m in d.pop("mtf", [])]
        confs = [Confirmation(**c) for c in d.pop("confirmations", [])]
        allowed = set(cls.__dataclass_fields__)                    # noqa: SLF001
        clean = {k: v for k, v in d.items() if k in allowed}
        return cls(take_profits=tps, mtf=mtf, confirmations=confs, **clean)


# --------------------------------------------------------------------------- #
@dataclass
class Outcome:
    """A completed signal, reduced to what /report needs."""
    signal_id: str
    symbol: str
    side: str
    mode: str
    strategy: str
    entry: float
    stop_loss: float
    fill_price: float
    exit_price: float
    result: str                     # TP1 | TP2 | TP3 | SL | BE
    tp_hits: int
    r_multiple: float
    pct: float
    filled_at: float
    closed_at: float

    @property
    def is_win(self) -> bool:
        """Reaching TP1 before the stop counts as a win."""
        return self.tp_hits >= 1

    @property
    def reversed_after_tp1(self) -> bool:
        """
        Banked TP1, then gave it all back.

        Tracked separately because it is the specific failure this build is
        tuned against. It still counts as a win by the TP1 definition, but a
        strategy that keeps doing it is placing TP1 where price reverses,
        which is a targeting problem, not a bad-luck problem.
        """
        return self.tp_hits == 1 and self.result.startswith("SL")

    @property
    def stalled_at_tp2(self) -> bool:
        return self.tp_hits == 2 and self.result.startswith("SL")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Outcome":
        allowed = set(cls.__dataclass_fields__)                    # noqa: SLF001
        return cls(**{k: v for k, v in d.items() if k in allowed})
