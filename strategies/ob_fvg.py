"""
STRATEGY 2 — ORDER BLOCK + FAIR VALUE GAP  (continuation)
=========================================================

Where strategy 1 catches the turn, this one joins a move that is already
running — but only at the price institutions themselves left behind.

When size moves a market it moves it inefficiently. Orders that could not be
filled during the impulse remain resting at the origin of the move (the order
block), and the impulse itself skips price levels entirely, leaving a gap that
was never traded fairly (the fair value gap). Price returns to those levels not
because a pattern says so, but because unfilled business is still there.

The sequence, in strict order:

  1. HTF trend        - established direction on the bias timeframe
  2. BOS              - structure broken IN that direction: continuation
                        confirmed, not hoped for
  3. Order block      - the last opposing candle before the impulse that
                        caused the break, still unmitigated
  4. FVG              - the imbalance inside that same impulse leg
  5. Retracement      - price is returning into the OB/FVG confluence
  6. Sniper trigger   - entry confirmation on the lowest timeframe

Links 1-4 and 6 are mandatory. The A+ version is when the order block and the
fair value gap overlap: one narrow pocket containing both the unfilled orders
and the untraded prices.
"""
from __future__ import annotations

from typing import List, Optional

from core.models import Confirmation
from core.structure import (find_major_zones,
                            Zone, find_fvgs, find_inducement,
                            find_order_blocks, zone_overlap)
from strategies.base import AnalysisContext, BaseStrategy, StrategyResult
from strategies.scoring import OB_FVG_WEIGHTS as W
from strategies.scoring import scaled
from utils.helpers import safe_float
from utils.logger import get_logger

log = get_logger("ob_fvg")


class OrderBlockFVGStrategy(BaseStrategy):
    name = "OB_FVG"
    label = "Order Block + FVG"
    short = "OB+FVG"
    min_confirmations = 9
    weights = W

    def evaluate(self, ctx: AnalysisContext) -> Optional[StrategyResult]:
        bias, struct, setup, trig = ctx.bias, ctx.structure, ctx.setup, ctx.trigger
        price = ctx.price
        atr = setup.atr or trig.atr
        if atr <= 0:
            return None

        confs: List[Confirmation] = []

        # ------------------------------------------------------------------ #
        # 1. Higher-timeframe trend  (MANDATORY)
        # ------------------------------------------------------------------ #
        htf = bias.structure.trend
        if htf == "RANGE":
            return None                   # continuation needs something to continue
        side = "LONG" if htf == "BULL" else "SHORT"
        is_long = side == "LONG"

        confs.append(Confirmation(
            name="HTF Trend",
            detail=f"{bias.timeframe.upper()} structure is {htf} "
                   f"(conviction {bias.structure.strength:.0%})",
            weight=scaled(W["htf_trend"], bias.structure.strength),
            timeframe=bias.timeframe.upper(), category="bias"))

        # ------------------------------------------------------------------ #
        # 2. Break of structure in the trend direction  (MANDATORY)
        # ------------------------------------------------------------------ #
        want = "bull" if is_long else "bear"
        st = struct.structure
        setup_st = setup.structure
        if st.last_bos != want and setup_st.last_bos != want:
            return None

        on_struct = st.last_bos == want
        confs.append(Confirmation(
            name="BOS Continuation",
            detail=f"{(struct if on_struct else setup).timeframe.upper()} broke "
                   f"structure to the {want} side — the trend is extending, "
                   f"not stalling",
            weight=scaled(W["bos_continuation"], 1.0 if on_struct else 0.35),
            timeframe=(struct if on_struct else setup).timeframe.upper(),
            category="structure"))

        # ------------------------------------------------------------------ #
        # 3. + 4. Order block and FVG from the impulse  (BOTH MANDATORY)
        # ------------------------------------------------------------------ #
        zone_side = "bull" if is_long else "bear"
        obs = [z for z in find_order_blocks(setup.df)
               if z.side == zone_side and not z.mitigated]
        fvgs = [z for z in find_fvgs(setup.df)
                if z.side == zone_side and not z.mitigated]
        if not obs or not fvgs:
            return None

        # only zones price has not yet come back to
        if is_long:
            obs = [z for z in obs if z.high < price]
            fvgs = [z for z in fvgs if z.high < price]
        else:
            obs = [z for z in obs if z.low > price]
            fvgs = [z for z in fvgs if z.low > price]
        if not obs or not fvgs:
            return None

        ob = min(obs, key=lambda z: abs(price - z.mid))
        fvg = min(fvgs, key=lambda z: abs(price - z.mid))

        confs.append(Confirmation(
            name="Order Block",
            detail=f"{setup.timeframe.upper()} unmitigated {zone_side} OB at "
                   f"{ob.low:.6g}–{ob.high:.6g} — the origin of the impulse",
            weight=scaled(W["order_block"], ob.strength),
            timeframe=setup.timeframe.upper(), category="structure"))
        confs.append(Confirmation(
            name="Fair Value Gap",
            detail=f"{setup.timeframe.upper()} unfilled imbalance at "
                   f"{fvg.low:.6g}–{fvg.high:.6g}",
            weight=scaled(W["fvg"], fvg.strength),
            timeframe=setup.timeframe.upper(), category="structure"))

        # ------------------------------------------------------------------ #
        # zone confluence — the pocket where both sit
        # ------------------------------------------------------------------ #
        overlap = zone_overlap(ob, fvg)
        if overlap:
            poi = Zone(low=overlap[0], high=overlap[1],
                       index=max(ob.index, fvg.index), kind="OB+FVG",
                       side=zone_side,
                       strength=min(1.0, (ob.strength + fvg.strength) / 2 + 0.15))
            confs.append(Confirmation(
                name="Zone Confluence",
                detail=f"Order block and FVG overlap at "
                       f"{overlap[0]:.6g}–{overlap[1]:.6g} — unfilled orders "
                       f"and untraded price in one pocket",
                weight=W["zone_confluence"],
                timeframe=setup.timeframe.upper(), category="structure"))
        else:
            # no overlap: trade the one closer to price, at reduced conviction
            poi = ob if abs(price - ob.mid) <= abs(price - fvg.mid) else fvg

        # ------------------------------------------------------------------ #
        # 5. Sniper trigger  (MANDATORY)
        # ------------------------------------------------------------------ #
        if not self._trigger(ctx, is_long, confs, atr):
            return None

        # ------------------------------------------------------------------ #
        # EXTRA CONFIRMATIONS
        # ------------------------------------------------------------------ #
        # (a) fibonacci confluence with the impulse
        fib = self._fib_position(setup.df, is_long, poi.mid)
        if fib is not None and 0.5 <= fib <= 0.9:
            golden = 0.618 <= fib <= 0.79
            confs.append(Confirmation(
                name="Fib Confluence",
                detail=f"Zone sits at the {fib:.0%} retracement of the impulse"
                       + (" — golden pocket" if golden else ""),
                weight=scaled(W["fib_confluence"], 1.0 if golden else 0.3),
                timeframe=setup.timeframe.upper(), category="structure"))

        # (b) pullback quality — retracement, not distribution
        quality = self._pullback_quality(setup.df, is_long)
        if quality > 0:
            confs.append(Confirmation(
                name="Pullback Quality",
                detail="Retracement on contracting volume and range — "
                       "profit-taking, not a reversal",
                weight=scaled(W["pullback_quality"], quality),
                timeframe=setup.timeframe.upper(), category="volume"))

        # (c) inducement swept into the zone
        ind = find_inducement(setup.df, side, len(setup.df) - 1)
        if ind:
            confs.append(Confirmation(
                name="Inducement Taken",
                detail=f"Minor pool at {ind['price']:.6g} swept on the way into "
                       f"the zone — late entries flushed first",
                weight=W["inducement"],
                timeframe=setup.timeframe.upper(), category="liquidity"))

        # (d) not overextended from the mean
        ema200 = safe_float(setup.last.get("ema200"))
        if ema200 > 0:
            ext = abs(price - ema200) / atr
            if ext > 7.0:
                return None               # chasing a move that already happened
            confs.append(Confirmation(
                name="Not Overextended",
                detail=f"{ext:.1f} ATR from the {setup.timeframe.upper()} EMA200 "
                       f"— room left in the move",
                weight=scaled(W["not_overextended"], 1.0 - ext / 7.0),
                timeframe=setup.timeframe.upper(), category="risk"))

        # (e) target liquidity
        targets = self.opposing_pools(ctx, side, price)
        if not targets:
            return None
        confs.append(Confirmation(
            name="Target Liquidity",
            detail=f"{len(targets)} untapped "
                   f"{'buy-side' if is_long else 'sell-side'} pool(s), "
                   f"first at {targets[0].price:.6g}",
            weight=scaled(W["target_liquidity"], targets[0].strength),
            timeframe=struct.timeframe.upper(), category="liquidity"))

        # (f) session, BTC regime, volatility
        self.regime_confirmations(ctx, side, confs)

        # ------------------------------------------------------------------ #
        # invalidation & entry zone
        # ------------------------------------------------------------------ #
        # If price closes through the far side of the order block, the orders
        # that created it are gone and the continuation thesis is void.
        invalidation = float(ob.low if is_long else ob.high)
        entry_low, entry_high, entry_ref = self.zone_from_poi(
            poi, is_long, price, atr, ctx.mode)

        return StrategyResult(
            side=side, entry_ref=entry_ref,
            entry_low=entry_low, entry_high=entry_high,
            invalidation=invalidation,
            confirmations=confs, score=self.score_of(confs),
            target_pools=targets, poi=poi,
            major_zones=find_major_zones(ctx.structure.df),
            tags=["continuation", "ob", "fvg"],
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _fib_position(df, is_long: bool, level: float) -> Optional[float]:
        """Where `level` sits as a retracement of the last 60-bar impulse."""
        window = df.tail(60)
        if len(window) < 20:
            return None
        hi = float(window["high"].max())
        lo = float(window["low"].min())
        if hi <= lo:
            return None
        return (hi - level) / (hi - lo) if is_long else (level - lo) / (hi - lo)

    @staticmethod
    def _pullback_quality(df, is_long: bool, bars: int = 6) -> float:
        """
        A healthy pullback contracts. Volume and range should fall as price
        retraces; if they expand instead, that is supply arriving, not a dip.
        """
        window = df.tail(bars * 3)
        if len(window) < bars * 2 or "volume" not in window.columns:
            return 0.0
        recent = window.tail(bars)
        earlier = window.head(bars)

        v_recent = float(recent["volume"].mean())
        v_earlier = float(earlier["volume"].mean())
        r_recent = float((recent["high"] - recent["low"]).mean())
        r_earlier = float((earlier["high"] - earlier["low"]).mean())
        if v_earlier <= 0 or r_earlier <= 0:
            return 0.0

        v_ratio = v_recent / v_earlier
        r_ratio = r_recent / r_earlier
        if v_ratio >= 1.0 or r_ratio >= 1.1:
            return 0.0
        return max(0.0, min(1.0, (1.0 - v_ratio) * 1.5 + (1.0 - r_ratio) * 0.5))
