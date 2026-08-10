"""
Strategy contract.

A strategy receives a fully-built `AnalysisContext` (multi-timeframe dataframes
plus pre-computed structure) and returns either None or a `StrategyResult`
containing a direction, an invalidation level, a POI and a weighted list of
confirmations. The engine turns that into a Signal (TP placement, RR, expiry).

Separating "why" (strategy) from "where" (signal_builder) keeps TP logic
consistent across both strategies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from config.modes import ModeSpec
from core.models import Confirmation, MTFView
from core.structure import (LiquidityPool, StructureReport, Zone,
                            analyse_structure, find_fvgs, find_liquidity_pools,
                            find_order_blocks)


@dataclass
class TFContext:
    timeframe: str
    role: str
    df: pd.DataFrame
    structure: StructureReport
    pools: List[LiquidityPool] = field(default_factory=list)
    obs: List[Zone] = field(default_factory=list)
    fvgs: List[Zone] = field(default_factory=list)

    @property
    def last(self) -> pd.Series:
        return self.df.iloc[-1]

    @property
    def prev(self) -> pd.Series:
        return self.df.iloc[-2]

    @property
    def price(self) -> float:
        return float(self.df["close"].iloc[-1])

    @property
    def atr(self) -> float:
        return float(self.df["atr"].iloc[-1]) if "atr" in self.df.columns else 0.0


@dataclass
class AnalysisContext:
    symbol: str
    mode: ModeSpec
    price: float
    bias: TFContext
    structure: TFContext
    setup: TFContext
    trigger: TFContext
    quote_volume: float = 0.0
    funding_rate: float = 0.0
    weekend: bool = False
    spread_pct: float = 0.0
    # market regime, filled in by the scanner once per cycle (cheap: one fetch)
    btc_trend: str = ""            # BULL | BEAR | RANGE
    btc_change_pct: float = 0.0

    @property
    def tfs(self) -> Dict[str, TFContext]:
        return {"BIAS": self.bias, "STRUCTURE": self.structure,
                "SETUP": self.setup, "TRIGGER": self.trigger}

    def mtf_views(self, side: str) -> List[MTFView]:
        want = "BULL" if side == "LONG" else "BEAR"
        views = []
        for role, ctx in self.tfs.items():
            st = ctx.structure
            note_bits = [f"{st.premium_discount.title()}"]
            if st.last_bos:
                note_bits.append(f"BOS {st.last_bos}")
            if st.last_choch:
                note_bits.append(f"CHoCH {st.last_choch}")
            note_bits.append(f"ADX {ctx.last.get('adx', 0):.0f}")
            views.append(MTFView(
                timeframe=ctx.timeframe.upper(), role=role, trend=st.trend,
                note=" · ".join(note_bits),
                aligned=(st.trend == want or st.trend == "RANGE"),
            ))
        return views


def build_tf_context(timeframe: str, role: str, df: pd.DataFrame,
                     deep: bool = True) -> Optional[TFContext]:
    if df is None or len(df) < 40:
        return None
    st = analyse_structure(df)
    ctx = TFContext(timeframe=timeframe, role=role, df=df, structure=st)
    if deep:
        ctx.pools = find_liquidity_pools(df, st)
        ctx.obs = find_order_blocks(df)
        ctx.fvgs = find_fvgs(df)
    return ctx


@dataclass
class StrategyResult:
    side: str                          # LONG | SHORT
    entry_ref: float                   # reference entry price
    entry_low: float
    entry_high: float
    invalidation: float                # raw structural level SL sits beyond
    confirmations: List[Confirmation]
    score: float
    target_pools: List[LiquidityPool] = field(default_factory=list)
    poi: Optional[Zone] = None
    major_zones: List[Zone] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


class BaseStrategy:
    name: str = "BASE"
    label: str = "Base"

    #: minimum number of confirmations regardless of score
    min_confirmations: int = 9

    #: the strategy's 100-point weight table (see strategies/scoring.py)
    weights: dict = {}

    def evaluate(self, ctx: AnalysisContext) -> Optional[StrategyResult]:
        raise NotImplementedError

    # -- helpers shared by concrete strategies -------------------------- #
    @staticmethod
    def score_of(confs: List[Confirmation]) -> float:
        """Weighted score against a 100-point budget."""
        total = sum(c.weight for c in confs)
        return round(min(100.0, total), 1)

    def regime_confirmations(self, ctx: AnalysisContext, side: str,
                             confs: List[Confirmation]) -> None:
        """
        Confirmations that depend on the wider market rather than the chart:
        agreement with BTC, session timing and volatility regime. Shared by
        both strategies because a setup fighting BTC in the Asian afternoon is
        a bad setup regardless of which engine found it.
        """
        import time as _t

        from core.models import Confirmation as _C
        from strategies.scoring import scaled, session_now
        from utils.helpers import safe_float

        w = self.weights
        is_long = side == "LONG"

        # --- BTC as the market's tide -------------------------------------- #
        if ctx.btc_trend:
            agrees = ((is_long and ctx.btc_trend == "BULL")
                      or (not is_long and ctx.btc_trend == "BEAR"))
            neutral = ctx.btc_trend == "RANGE"
            if ctx.symbol.startswith("BTC") or agrees:
                confs.append(_C(
                    name="Market Regime",
                    detail=f"BTC is {ctx.btc_trend} ({ctx.btc_change_pct:+.2f}% 24h) "
                           f"— the tide is with this trade",
                    weight=w.get("btc_regime", 2.0), category="context"))
            elif neutral:
                confs.append(_C(
                    name="Market Regime",
                    detail=f"BTC is ranging ({ctx.btc_change_pct:+.2f}% 24h) "
                           f"— no headwind",
                    weight=scaled(w.get("btc_regime", 2.0), 0.0),
                    category="context"))
            # actively fighting BTC scores nothing

        # --- session ------------------------------------------------------- #
        label, quality = session_now(_t.time())
        if quality > 0:
            confs.append(_C(
                name="Session Timing",
                detail=f"{label} session — where follow-through actually happens",
                weight=scaled(w.get("session", 3.0), quality),
                category="context"))

        # --- volatility regime --------------------------------------------- #
        atr_pct = safe_float(ctx.setup.last.get("atr_pct"))
        if atr_pct <= 0 and ctx.setup.price:
            atr_pct = ctx.setup.atr / ctx.setup.price * 100
        try:
            series = ctx.setup.df["atr_pct"].tail(120).dropna()
            pctile = float((series < atr_pct).mean()) if len(series) > 30 else 0.5
        except (KeyError, TypeError, ValueError):
            pctile = 0.5

        # 35th-85th percentile is the sweet spot: enough movement to reach a
        # target, not so much that stops become lottery tickets.
        if 0.35 <= pctile <= 0.85:
            confs.append(_C(
                name="Volatility Regime",
                detail=f"ATR {atr_pct:.2f}% sits at the {pctile:.0%} percentile "
                       f"of its own range — expansion without chaos",
                weight=scaled(w.get("volatility_regime", 3.0),
                              1.0 - abs(pctile - 0.6) / 0.25),
                timeframe=ctx.setup.timeframe.upper(), category="context"))

    @staticmethod
    def has(confs: List[Confirmation], category: str) -> bool:
        return any(c.category == category for c in confs)

    # ------------------------------------------------------------------ #
    # shared mechanics
    # ------------------------------------------------------------------ #
    def _trigger(self, ctx: "AnalysisContext", is_long: bool,
                 confs: List[Confirmation], atr: float) -> bool:
        """
        Sniper trigger on the lowest timeframe.

        The higher timeframes say *where*; this says *now*. At least one of
        displacement, engulfing or wick rejection must be present in the
        signal direction, otherwise we are guessing at a level rather than
        responding to what price is doing at it.
        """
        from core.models import Confirmation as _C
        from strategies.scoring import scaled
        from utils.helpers import safe_float

        trig = ctx.trigger
        last, prev = trig.last, trig.prev
        rng = safe_float(last["high"]) - safe_float(last["low"])
        body_ratio = safe_float(last.get("body_ratio"))

        # Measure the trigger candle against the TRIGGER timeframe's own ATR.
        # Comparing a 15m candle to the 1H ATR demands a bar four times normal
        # size and silently suppresses almost every entry.
        t_atr = trig.atr or atr
        displacement = rng >= t_atr * 0.9 and body_ratio >= 0.55
        bullish = last["close"] > last["open"]
        if is_long:
            engulf = bullish and last["close"] > prev["open"] and prev["close"] < prev["open"]
            rejection = safe_float(last.get("lower_wick")) > safe_float(last.get("upper_wick")) * 1.5
        else:
            engulf = (not bullish) and last["close"] < prev["open"] and prev["close"] > prev["open"]
            rejection = safe_float(last.get("upper_wick")) > safe_float(last.get("lower_wick")) * 1.5

        bits = []
        if displacement:
            bits.append(f"displacement ({rng / t_atr:.2f} ATR)")
        if engulf:
            bits.append("engulfing close")
        if rejection:
            bits.append("wick rejection")
        if not bits:
            return False

        confs.append(_C(
            name="Sniper Trigger",
            detail=f"{trig.timeframe.upper()} {', '.join(bits)} in the signal direction",
            weight=scaled(self.weights.get("sniper_trigger", 9.0),
                          min(1.0, (len(bits) - 1) / 2.0)),
            timeframe=trig.timeframe.upper(), category="momentum"))
        return True

    @staticmethod
    def opposing_pools(ctx: "AnalysisContext", side: str,
                       price: float) -> List[LiquidityPool]:
        """Untapped liquidity in the direction of the trade, nearest first."""
        want = "buyside" if side == "LONG" else "sellside"
        pools: List[LiquidityPool] = []
        for tf_ctx in (ctx.structure, ctx.setup, ctx.bias):
            for p in tf_ctx.pools:
                if p.kind != want:
                    continue
                if side == "LONG" and p.price <= price * 1.0008:
                    continue
                if side == "SHORT" and p.price >= price * 0.9992:
                    continue
                pools.append(p)

        pools.sort(key=lambda p: p.price, reverse=(side == "SHORT"))
        out: List[LiquidityPool] = []
        for p in pools:
            if out and abs(p.price - out[-1].price) / max(price, 1e-9) < 0.0012:
                if p.strength > out[-1].strength:
                    out[-1] = p
                continue
            out.append(p)
        return out

    @staticmethod
    def zone_from_poi(poi: Zone, is_long: bool, price: float, atr: float,
                      mode) -> tuple:
        """
        Build a NARROW entry zone anchored to the point of interest, and pick
        the exact price that must trade for the entry to count as taken.

        This exists because a wide zone quietly destroys the trade. If a short
        is published as 201.89–200.28 with TP1 at 199.50, a fill recorded at
        the moment price grazes 200.28 leaves 0.78 of upside against a stop
        measured from 201.89 — the reward has been given away before the trade
        starts, and every statistic downstream is flattered by an entry nobody
        could have got.

        So two things are enforced. The zone is clamped to a fraction of an
        ATR around the POI, and the reference price sits at its MIDPOINT
        rather than its near edge. Price has to commit to the level, not brush
        past it.
        """
        half = max(atr * mode.entry_zone_atr, price * 0.0004)

        if is_long:
            high = float(poi.high)
            low = float(max(poi.low, high - 2 * half))
            if low >= high:
                low = high - 2 * half
        else:
            low = float(poi.low)
            high = float(min(poi.high, low + 2 * half))
            if high <= low:
                high = low + 2 * half

        ref = (low + high) / 2.0
        return round(low, 10), round(high, 10), round(ref, 10)
