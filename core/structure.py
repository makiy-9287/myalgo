"""
Market-structure / Smart-Money engine.

This is the part that actually models what desks do to retail:
  * swing mapping (fractals)
  * BOS  - break of structure  (continuation)
  * CHoCH- change of character (reversal)
  * liquidity pools (equal highs/lows, swing clusters, round numbers)
  * liquidity sweeps / stop hunts
  * order blocks (last opposing candle before displacement)
  * fair value gaps (3-candle imbalance)
  * premium / discount of the dealing range
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

Side = Literal["LONG", "SHORT"]


# --------------------------------------------------------------------------- #
# data classes
# --------------------------------------------------------------------------- #
@dataclass
class Swing:
    index: int
    price: float
    kind: str              # "high" | "low"
    ts: int = 0


@dataclass
class Zone:
    low: float
    high: float
    index: int
    kind: str              # "OB" | "FVG" | "BREAKER"
    side: str              # "bull" | "bear"
    strength: float = 1.0
    mitigated: bool = False

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2

    def contains(self, price: float, tol: float = 0.0) -> bool:
        return (self.low - tol) <= price <= (self.high + tol)


@dataclass
class LiquidityPool:
    price: float
    kind: str              # "buyside" | "sellside"
    strength: float        # 0-1, based on touch count / equality tightness
    touches: int = 1
    index: int = 0
    label: str = ""


@dataclass
class StructureReport:
    trend: str = "RANGE"                 # BULL | BEAR | RANGE
    last_bos: Optional[str] = None       # "bull" | "bear"
    last_bos_index: int = -1
    last_choch: Optional[str] = None
    last_choch_index: int = -1
    swing_highs: List[Swing] = field(default_factory=list)
    swing_lows: List[Swing] = field(default_factory=list)
    range_high: float = 0.0
    range_low: float = 0.0
    equilibrium: float = 0.0
    premium_discount: str = "EQUILIBRIUM"   # PREMIUM | DISCOUNT | EQUILIBRIUM
    position_in_range: float = 0.5
    strength: float = 0.0                   # 0-1 conviction in the trend read


# --------------------------------------------------------------------------- #
# swing detection
# --------------------------------------------------------------------------- #
def find_swings(df: pd.DataFrame, left: int = 2, right: int = 2) -> Tuple[List[Swing], List[Swing]]:
    """Fractal swing points. `right` bars must confirm, so the last `right`
    candles can never produce a confirmed swing (no repainting)."""
    highs, lows = [], []
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    ts = (df["open_time"].to_numpy() if "open_time" in df.columns
          else np.zeros(len(df), dtype=np.int64))
    n = len(df)

    for i in range(left, n - right):
        wh = h[i - left:i + right + 1]
        wl = l[i - left:i + right + 1]
        if h[i] == wh.max() and (wh.argmax() == left):
            highs.append(Swing(i, float(h[i]), "high", int(ts[i])))
        if l[i] == wl.min() and (wl.argmin() == left):
            lows.append(Swing(i, float(l[i]), "low", int(ts[i])))
    return highs, lows


# --------------------------------------------------------------------------- #
# structure read
# --------------------------------------------------------------------------- #
def analyse_structure(df: pd.DataFrame, left: int = 2, right: int = 2,
                      lookback: int = 120) -> StructureReport:
    rep = StructureReport()
    if df is None or len(df) < max(30, left + right + 10):
        return rep

    window = df.iloc[-lookback:] if len(df) > lookback else df
    offset = len(df) - len(window)
    highs, lows = find_swings(window, left, right)
    # shift indices back to the full-frame coordinate system
    highs = [Swing(s.index + offset, s.price, s.kind, s.ts) for s in highs]
    lows = [Swing(s.index + offset, s.price, s.kind, s.ts) for s in lows]

    rep.swing_highs, rep.swing_lows = highs, lows
    if len(highs) < 2 or len(lows) < 2:
        return rep

    # --- trend from the last two confirmed HH/HL or LH/LL pairs
    h1, h2 = highs[-2].price, highs[-1].price
    l1, l2 = lows[-2].price, lows[-1].price
    hh, hl = h2 > h1, l2 > l1
    lh, ll = h2 < h1, l2 < l1

    if hh and hl:
        rep.trend, rep.strength = "BULL", 0.9
    elif ll and lh:
        rep.trend, rep.strength = "BEAR", 0.9
    elif hh or hl:
        rep.trend, rep.strength = "BULL", 0.55
    elif ll or lh:
        rep.trend, rep.strength = "BEAR", 0.55
    else:
        rep.trend, rep.strength = "RANGE", 0.3

    # --- BOS / CHoCH
    # Walk forward through the window. At each bar only swings already CONFIRMED
    # by that point are usable (index + right <= i), which is what a live chart
    # would have shown. A close through the active swing is a break of
    # structure; a break opposite to the previous one is a change of character.
    closes = df["close"].to_numpy()
    start = max(highs[0].index, lows[0].index)
    hi_ptr = lo_ptr = 0
    active_high: Optional[Swing] = None
    active_low: Optional[Swing] = None

    for i in range(start, len(df)):
        while hi_ptr < len(highs) and highs[hi_ptr].index + right <= i:
            active_high = highs[hi_ptr]
            hi_ptr += 1
        while lo_ptr < len(lows) and lows[lo_ptr].index + right <= i:
            active_low = lows[lo_ptr]
            lo_ptr += 1

        if active_high and closes[i] > active_high.price:
            if rep.last_bos == "bear":
                rep.last_choch, rep.last_choch_index = "bull", i
            rep.last_bos, rep.last_bos_index = "bull", i
            active_high = None                 # consumed until the next swing forms
        elif active_low and closes[i] < active_low.price:
            if rep.last_bos == "bull":
                rep.last_choch, rep.last_choch_index = "bear", i
            rep.last_bos, rep.last_bos_index = "bear", i
            active_low = None

    # --- dealing range / premium-discount
    rng = df.iloc[-lookback:] if len(df) > lookback else df
    rep.range_high = float(rng["high"].max())
    rep.range_low = float(rng["low"].min())
    span = rep.range_high - rep.range_low
    rep.equilibrium = (rep.range_high + rep.range_low) / 2
    price = float(df["close"].iloc[-1])
    rep.position_in_range = float((price - rep.range_low) / span) if span > 0 else 0.5

    if rep.position_in_range > 0.62:
        rep.premium_discount = "PREMIUM"
    elif rep.position_in_range < 0.38:
        rep.premium_discount = "DISCOUNT"
    else:
        rep.premium_discount = "EQUILIBRIUM"

    return rep


# --------------------------------------------------------------------------- #
# liquidity pools
# --------------------------------------------------------------------------- #
def find_liquidity_pools(df: pd.DataFrame, structure: StructureReport,
                         tolerance_atr: float = 0.18,
                         max_pools: int = 14,
                         lookback: int = 260) -> List[LiquidityPool]:
    """
    Liquidity sits where stops sit: above equal/clustered highs and below
    equal/clustered lows. Clusters are merged inside an ATR-scaled tolerance
    and scored by touch count + recency.

    Swings are re-derived over `lookback` bars rather than reusing the trend
    read's shorter window — stop clusters built weeks ago still hold orders,
    and they are exactly the levels price travels to.
    """
    pools: List[LiquidityPool] = []
    if df is None or len(df) < 30:
        return pools

    window = df.iloc[-lookback:] if len(df) > lookback else df
    offset = len(df) - len(window)
    w_highs, w_lows = find_swings(window, 2, 2)
    swing_highs = [Swing(s.index + offset, s.price, s.kind, s.ts) for s in w_highs]
    swing_lows = [Swing(s.index + offset, s.price, s.kind, s.ts) for s in w_lows]

    atr_val = float(df["atr"].iloc[-1]) if "atr" in df.columns else \
        float((df["high"] - df["low"]).tail(14).mean())
    if atr_val <= 0:
        return pools
    tol = atr_val * tolerance_atr
    n = len(df)

    def cluster(swings: List[Swing], kind: str) -> List[LiquidityPool]:
        if not swings:
            return []
        groups: List[List[Swing]] = []
        for s in sorted(swings, key=lambda x: x.price):
            if groups and abs(s.price - groups[-1][-1].price) <= tol:
                groups[-1].append(s)
            else:
                groups.append([s])

        out = []
        for g in groups:
            prices = [x.price for x in g]
            idx = max(x.index for x in g)
            # equal highs/lows are stronger; recency adds weight
            equality = 1.0 - (max(prices) - min(prices)) / tol if tol else 0.5
            equality = max(0.0, min(1.0, equality))
            recency = idx / max(1, n - 1)
            touch_score = min(1.0, len(g) / 3.0)
            strength = 0.45 * touch_score + 0.35 * equality + 0.20 * recency
            label = "EQH" if (kind == "buyside" and len(g) > 1) else \
                    "EQL" if (kind == "sellside" and len(g) > 1) else "Swing"
            out.append(LiquidityPool(
                price=float(np.mean(prices)), kind=kind,
                strength=round(float(strength), 3), touches=len(g),
                index=idx, label=label))
        return out

    pools.extend(cluster(swing_highs, "buyside"))
    pools.extend(cluster(swing_lows, "sellside"))

    # psychological round numbers act as magnets too
    price = float(df["close"].iloc[-1])
    if price > 0:
        magnitude = 10 ** np.floor(np.log10(price))
        for mult in (0.25, 0.5, 1.0):
            step = magnitude * mult
            if step <= 0:
                continue
            for direction in (1, -1):
                lvl = (np.floor(price / step) + (1 if direction > 0 else 0)) * step
                if lvl <= 0 or abs(lvl - price) > atr_val * 8:
                    continue
                pools.append(LiquidityPool(
                    price=float(lvl),
                    kind="buyside" if lvl > price else "sellside",
                    strength=0.35, touches=1, index=n - 1, label="Round"))

    # merge duplicates then keep the strongest
    pools.sort(key=lambda p: p.price)
    merged: List[LiquidityPool] = []
    for p in pools:
        if merged and merged[-1].kind == p.kind and abs(merged[-1].price - p.price) <= tol * 0.6:
            prev = merged[-1]
            prev.price = (prev.price + p.price) / 2
            prev.strength = max(prev.strength, p.strength)
            prev.touches += p.touches
            prev.label = prev.label if prev.label != "Round" else p.label
        else:
            merged.append(p)

    merged.sort(key=lambda p: p.strength, reverse=True)
    return merged[:max_pools]


def detect_liquidity_sweeps(df: pd.DataFrame, pools: List[LiquidityPool],
                            lookback: int = 8, atr_frac: float = 0.05,
                            limit: int = 8) -> List[Dict]:
    """
    All sweeps in the window, strongest first.

    A sweep = price wicks through a pool and CLOSES back on the origin side.
    Returning the full list rather than only the best one matters: the most
    recent raid is often too fresh to have produced a structure shift yet, and
    the setup that is actually tradeable may be the slightly older sweep two
    bars behind it. The caller decides which one earned its place.
    """
    if df is None or len(df) < lookback + 2 or not pools:
        return []

    atr_val = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0
    if atr_val <= 0:
        return []
    min_pen = atr_val * atr_frac

    recent = df.iloc[-lookback:]
    found: List[Dict] = []

    for pool in pools:
        for pos in range(len(recent)):
            row = recent.iloc[pos]
            bars_ago = len(recent) - 1 - pos
            if pool.kind == "buyside":
                swept = (row["high"] > pool.price + min_pen and
                         row["close"] < pool.price)
                direction = "SHORT"
            else:
                swept = (row["low"] < pool.price - min_pen and
                         row["close"] > pool.price)
                direction = "LONG"
            if not swept:
                continue
            penetration = (abs(row["high"] - pool.price) if pool.kind == "buyside"
                           else abs(pool.price - row["low"]))
            score = pool.strength * (1.0 - bars_ago / max(1, lookback)) \
                + min(1.0, penetration / atr_val) * 0.3
            found.append({
                "pool": pool, "direction": direction, "bars_ago": bars_ago,
                "penetration_atr": round(penetration / atr_val, 2),
                "score": round(float(score), 3),
                "sweep_extreme": float(row["high"] if pool.kind == "buyside"
                                       else row["low"]),
            })

    found.sort(key=lambda c: c["score"], reverse=True)
    return found[:limit]


def detect_liquidity_sweep(df: pd.DataFrame, pools: List[LiquidityPool],
                           lookback: int = 6, atr_frac: float = 0.05) -> Optional[Dict]:
    """The single strongest sweep, or None."""
    found = detect_liquidity_sweeps(df, pools, lookback, atr_frac, limit=1)
    return found[0] if found else None


# --------------------------------------------------------------------------- #
# order blocks / FVG
# --------------------------------------------------------------------------- #
def find_order_blocks(df: pd.DataFrame, lookback: int = 60,
                      displacement_atr: float = 1.1) -> List[Zone]:
    """
    Order block = last opposing candle immediately before a displacement leg.
    Displacement is measured against ATR so it adapts per symbol/timeframe.
    """
    zones: List[Zone] = []
    if df is None or len(df) < 30:
        return zones

    start = max(1, len(df) - lookback)
    atr_arr = (df["atr"].to_numpy() if "atr" in df.columns
               else np.full(len(df), (df["high"] - df["low"]).mean()))
    o, h, l, c = (df["open"].to_numpy(), df["high"].to_numpy(),
                  df["low"].to_numpy(), df["close"].to_numpy())
    n = len(df)
    leg = 3            # displacement is an impulse LEG, not one candle

    for i in range(start, n - 1):
        a = atr_arr[i] if atr_arr[i] > 0 else 1e-12
        j = min(n, i + 1 + leg)
        up_move = float(h[i + 1:j].max() - o[i + 1])
        dn_move = float(o[i + 1] - l[i + 1:j].min())

        # bullish OB: down candle followed by a strong up impulse
        if c[i] < o[i] and up_move > a * displacement_atr:
            strength = min(1.0, up_move / (a * 2))
            zones.append(Zone(float(l[i]), float(max(o[i], c[i])), i, "OB", "bull", strength))

        # bearish OB: up candle followed by a strong down impulse
        if c[i] > o[i] and dn_move > a * displacement_atr:
            strength = min(1.0, dn_move / (a * 2))
            zones.append(Zone(float(min(o[i], c[i])), float(h[i]), i, "OB", "bear", strength))

    # mark mitigation (price traded back through it afterwards)
    for z in zones:
        after = df.iloc[z.index + 2:]
        if len(after) == 0:
            continue
        if z.side == "bull":
            z.mitigated = bool((after["low"] <= z.low).any())
        else:
            z.mitigated = bool((after["high"] >= z.high).any())

    return zones


def find_fvgs(df: pd.DataFrame, lookback: int = 60,
              min_gap_atr: float = 0.12) -> List[Zone]:
    """Fair Value Gap: candle1.high < candle3.low (bullish) — a 3-bar imbalance."""
    zones: List[Zone] = []
    if df is None or len(df) < 10:
        return zones

    start = max(2, len(df) - lookback)
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    atr_arr = (df["atr"].to_numpy() if "atr" in df.columns
               else np.full(len(df), (df["high"] - df["low"]).mean()))

    for i in range(start, len(df)):
        a = atr_arr[i] if atr_arr[i] > 0 else 1e-12
        # bullish gap between bar i-2 high and bar i low
        if l[i] > h[i - 2]:
            gap = l[i] - h[i - 2]
            if gap >= a * min_gap_atr:
                zones.append(Zone(float(h[i - 2]), float(l[i]), i - 1, "FVG", "bull",
                                  min(1.0, gap / a)))
        # bearish gap
        if h[i] < l[i - 2]:
            gap = l[i - 2] - h[i]
            if gap >= a * min_gap_atr:
                zones.append(Zone(float(h[i]), float(l[i - 2]), i - 1, "FVG", "bear",
                                  min(1.0, gap / a)))

    for z in zones:
        after = df.iloc[z.index + 2:]
        if len(after) == 0:
            continue
        # a gap is "filled" once price closes through its midpoint
        if z.side == "bull":
            z.mitigated = bool((after["low"] <= z.mid).any())
        else:
            z.mitigated = bool((after["high"] >= z.mid).any())

    return zones


def nearest_zone(zones: List[Zone], price: float, side: str,
                 max_distance: float, unmitigated_only: bool = True) -> Optional[Zone]:
    """Closest valid POI on the correct side of price."""
    best, best_dist = None, float("inf")
    for z in zones:
        if z.side != side:
            continue
        if unmitigated_only and z.mitigated:
            continue
        if side == "bull" and z.high > price * 1.002:
            continue                                  # bullish POI must be at/below price
        if side == "bear" and z.low < price * 0.998:
            continue
        dist = abs(price - z.mid)
        if dist <= max_distance and dist < best_dist:
            best, best_dist = z, dist
    return best


# --------------------------------------------------------------------------- #
# divergence
# --------------------------------------------------------------------------- #
def detect_divergence(df: pd.DataFrame, col: str = "rsi",
                      lookback: int = 40) -> Optional[str]:
    """Regular divergence between price extremes and an oscillator/CVD."""
    if df is None or col not in df.columns or len(df) < lookback + 5:
        return None

    seg = df.iloc[-lookback:]
    highs, lows = find_swings(seg, 2, 2)
    if len(highs) >= 2:
        a, b = highs[-2], highs[-1]
        pa, pb = seg[col].iloc[a.index], seg[col].iloc[b.index]
        if b.price > a.price and pb < pa:
            return "bearish"
    if len(lows) >= 2:
        a, b = lows[-2], lows[-1]
        pa, pb = seg[col].iloc[a.index], seg[col].iloc[b.index]
        if b.price < a.price and pb > pa:
            return "bullish"
    return None


# --------------------------------------------------------------------------- #
# market structure shift, inducement and major zones
# --------------------------------------------------------------------------- #
def detect_mss(df: pd.DataFrame, side: str, sweep_index: int,
               displacement_atr: float = 0.9) -> Optional[Dict]:
    """
    Market Structure Shift — the confirmation that a sweep was a reversal and
    not just a deeper pullback.

    A plain break of structure is not enough. After liquidity is taken, the
    move away from it has to be *decisive*: a candle whose body carries real
    range, closing beyond the last opposing swing point. That displacement is
    what leaves the imbalance an institution has to come back and fill, and it
    is the difference between a genuine reversal and price drifting back to
    where it came from.

    Returns the break level, the displacement leg, and the index range of the
    leg so the caller can hunt for the FVG/OB it created.
    """
    if df is None or len(df) < 30 or sweep_index < 0:
        return None

    atr_val = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0
    if atr_val <= 0:
        return None

    highs, lows = find_swings(df, 2, 2)
    closes = df["close"].to_numpy()
    opens = df["open"].to_numpy()
    high_arr = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    n = len(df)

    want_bull = side == "LONG"

    # The level to break is the last opposing swing formed BEFORE the sweep.
    if want_bull:
        candidates = [s for s in highs if s.index <= sweep_index]
    else:
        candidates = [s for s in lows if s.index <= sweep_index]
    if not candidates:
        return None
    level = candidates[-1]

    for i in range(max(sweep_index, 1), n):
        broke = closes[i] > level.price if want_bull else closes[i] < level.price
        if not broke:
            continue

        # Walk back to where the impulse began. Displacement is a LEG, not
        # necessarily one candle — three consecutive strong closes displace
        # exactly as much as one large one, and measuring only the breaking
        # candle throws away most real structure shifts.
        start = i
        while start > max(0, i - 8):
            prev = start - 1
            same_way = (closes[prev] > opens[prev]) if want_bull else (closes[prev] < opens[prev])
            if not same_way:
                break
            start = prev

        body = abs(closes[i] - opens[start])
        if body < atr_val * displacement_atr:
            continue                      # a break without conviction is noise

        leg_low = float(low_arr[start:i + 1].min())
        leg_high = float(high_arr[start:i + 1].max())
        return {
            "index": i,
            "level": float(level.price),
            "leg_start": int(start),
            "leg_end": int(i),
            "leg_low": leg_low,
            "leg_high": leg_high,
            "displacement_atr": round(body / atr_val, 2),
            "bars_ago": int(n - 1 - i),
        }
    return None


def find_inducement(df: pd.DataFrame, side: str, before_index: int,
                    lookback: int = 40) -> Optional[Dict]:
    """
    Inducement — the minor liquidity taken *before* the real move.

    Institutions rarely reverse from an untouched level. They first let price
    take a small, obvious pool (the last minor high/low retail uses for stops
    and breakout entries), which supplies the fills needed to position. A sweep
    that had inducement beneath it is far more likely to hold than one that did
    not, because the fuel for the move has already been collected.
    """
    if df is None or before_index <= 5:
        return None

    window = df.iloc[max(0, before_index - lookback):before_index + 1]
    if len(window) < 10:
        return None
    offset = len(df) - len(window) if len(df) > len(window) else 0
    highs, lows = find_swings(window, 1, 1)

    # For a long, price sweeps sell-side; the inducement is a minor low taken
    # on the way down before the deeper raid.
    pool_swings = lows if side == "LONG" else highs
    if len(pool_swings) < 2:
        return None

    minor = pool_swings[-2]
    later = window.iloc[minor.index + 1:]
    if later.empty:
        return None

    if side == "LONG":
        taken = bool((later["low"] < minor.price).any())
    else:
        taken = bool((later["high"] > minor.price).any())
    if not taken:
        return None

    return {"price": float(minor.price), "index": int(minor.index + offset)}


def find_major_zones(df: pd.DataFrame, lookback: int = 300,
                     top_n: int = 6) -> List[Zone]:
    """
    Major supply and demand zones — the origins of the largest moves on the
    chart, not every small imbalance.

    Take-profits belong at levels that can actually absorb size. A zone earns
    its place here by the magnitude of the move that left it, so the strongest
    few survive and the noise does not.
    """
    if df is None or len(df) < 40:
        return []

    window = df.iloc[-lookback:] if len(df) > lookback else df
    offset = len(df) - len(window)
    atr_val = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0
    if atr_val <= 0:
        return []

    o = window["open"].to_numpy()
    h = window["high"].to_numpy()
    l = window["low"].to_numpy()
    c = window["close"].to_numpy()
    n = len(window)
    zones: List[Zone] = []
    leg = 5

    for i in range(2, n - leg - 1):
        j = min(n, i + 1 + leg)
        up_move = float(h[i + 1:j].max() - o[i + 1]) if j > i + 1 else 0.0
        dn_move = float(o[i + 1] - l[i + 1:j].min()) if j > i + 1 else 0.0

        # demand: a down candle that produced a large up-leg
        if c[i] < o[i] and up_move > atr_val * 2.5:
            zones.append(Zone(low=float(l[i]), high=float(max(o[i], c[i])),
                              index=int(i + offset), kind="DEMAND", side="bull",
                              strength=min(1.0, up_move / (atr_val * 5))))
        # supply: an up candle that produced a large down-leg
        elif c[i] > o[i] and dn_move > atr_val * 2.5:
            zones.append(Zone(low=float(min(o[i], c[i])), high=float(h[i]),
                              index=int(i + offset), kind="SUPPLY", side="bear",
                              strength=min(1.0, dn_move / (atr_val * 5))))

    # Drop zones price has already eaten through. Mitigation is measured on
    # CLOSES, not wicks: price spiking through a zone and immediately
    # reversing has not consumed the orders resting in it — that rejection is
    # the zone doing its job. Requiring the extreme to hold would discard
    # almost every level that ever worked.
    fresh = []
    for z in zones:
        after = window.iloc[z.index - offset + 1:]
        if after.empty:
            continue
        closes_after = after["close"]
        if z.side == "bull" and float(closes_after.min()) < z.low:
            continue
        if z.side == "bear" and float(closes_after.max()) > z.high:
            continue
        fresh.append(z)

    fresh.sort(key=lambda z: z.strength, reverse=True)
    return fresh[:top_n]


def zone_overlap(a: Zone, b: Zone) -> Optional[Tuple[float, float]]:
    """Intersection of two zones, or None. Used for OB∩FVG confluence."""
    low = max(a.low, b.low)
    high = min(a.high, b.high)
    return (low, high) if low < high else None
