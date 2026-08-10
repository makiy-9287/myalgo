"""
Trading-mode definitions.

Two modes only. Each owns its timeframe stack, cache TTLs, expiry and entry
geometry.

    DAY   : 4H  bias -> 1H  structure -> 15m setup -> 5m  sniper trigger
    SWING : 1D  bias -> 4H  structure -> 1H  setup -> 15m sniper trigger
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class ModeSpec:
    name: str
    label: str
    bias_tf: str          # highest timeframe - directional context
    structure_tf: str     # market structure / POI mapping
    setup_tf: str         # refined zone (OB / FVG)
    trigger_tf: str       # sniper entry confirmation
    candles: Dict[str, int]        # how many candles per tf
    expiry_minutes: int            # how long the entry stays valid
    sl_atr_mult: float             # ATR buffer beyond invalidation
    entry_zone_atr: float          # MAX half-width of the entry zone, in ATR
    min_tp3_rr: float              # total reward:risk floor
    min_tp1_rr: float              # TP1 must be worth taking on its own
    min_tp2_rr: float              # and TP2 must be genuinely reachable
    max_zone_frac_of_tp1: float    # zone width as a fraction of entry->TP1
    tp_split: List[float] = field(default_factory=lambda: [0.4, 0.35, 0.25])

    @property
    def timeframes(self) -> List[str]:
        seen, out = set(), []
        for tf in (self.bias_tf, self.structure_tf, self.setup_tf, self.trigger_tf):
            if tf not in seen:
                seen.add(tf)
                out.append(tf)
        return out


DAY = ModeSpec(
    name="DAY",
    label="DAY",
    bias_tf="4h", structure_tf="1h", setup_tf="15m", trigger_tf="5m",
    candles={"4h": 320, "1h": 420, "15m": 450, "5m": 350},
    # A narrow zone needs time to be revisited. Eight hours spans a full
    # session rotation, which is roughly how long a 15m POI stays relevant.
    expiry_minutes=480,
    sl_atr_mult=0.65,
    entry_zone_atr=0.22,
    min_tp3_rr=6.0,
    min_tp1_rr=1.5,
    min_tp2_rr=3.0,
    max_zone_frac_of_tp1=0.18,
)

SWING = ModeSpec(
    name="SWING",
    label="SWING",
    bias_tf="1d", structure_tf="4h", setup_tf="1h", trigger_tf="15m",
    candles={"1d": 300, "4h": 400, "1h": 450, "15m": 350},
    # A 1H point of interest can take a day or more to be retested; expiring
    # sooner would throw away setups that were simply early.
    expiry_minutes=1440,
    sl_atr_mult=0.85,
    entry_zone_atr=0.28,
    min_tp3_rr=6.0,
    min_tp1_rr=1.5,
    min_tp2_rr=3.0,
    max_zone_frac_of_tp1=0.20,
)

MODES: Dict[str, ModeSpec] = {"DAY": DAY, "SWING": SWING}

# Cache TTL per timeframe in seconds. Shorter than the bar length on the
# trigger TFs so we always see the live forming candle; HTF data is refreshed
# lazily to protect the REST weight budget.
TF_TTL: Dict[str, int] = {
    "5m": 60,
    "15m": 160,
    "1h": 450,
    "4h": 1600,
    "1d": 5400,
}

TF_MINUTES: Dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440,
}

# Timeframes used only for the cheap pre-filter stage of the scan.
PREFILTER_TFS = ["4h", "1h"]
