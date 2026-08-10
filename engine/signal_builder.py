"""
Converts a StrategyResult into a fully-specified Signal.

Rules enforced here (identical for both strategies so results are comparable):
  * SL sits BEYOND the structural invalidation plus an ATR buffer, then is
    widened past the nearest opposing pool so the obvious stop cluster is not
    where our stop rests.
  * TP1/TP2/TP3 are anchored to real liquidity pools. Only if fewer than three
    valid pools exist do we fall back to R-multiples, and the fallback is
    labelled honestly in the message.
  * Minimum RR is enforced on TP3; a setup that cannot pay is discarded.
  * Expiry is per mode, so a scalp signal cannot be taken two hours later.
"""
from __future__ import annotations

import time
from typing import List, Optional

from config.modes import ModeSpec
from config.settings import RuntimeConfig
from core.data_manager import SymbolInfo
from core.models import Confirmation, Signal, TakeProfit
from core.structure import LiquidityPool
from strategies.base import AnalysisContext, StrategyResult
from utils.helpers import round_step, safe_float
from utils.logger import get_logger

log = get_logger("signal")

MIN_SL_PCT = 0.15          # never place a stop tighter than this (noise floor)
MAX_SL_PCT = 9.0           # absurdly wide stop = broken structure read


class SignalBuilder:
    def __init__(self, runtime: RuntimeConfig):
        self.rt = runtime

    # ------------------------------------------------------------------ #
    def build(self, ctx: AnalysisContext, result: StrategyResult,
              strategy_name: str, symbol_info: Optional[SymbolInfo],
              required_score: float) -> Optional[Signal]:
        mode: ModeSpec = ctx.mode
        is_long = result.side == "LONG"
        atr = ctx.setup.atr or ctx.trigger.atr
        if atr <= 0:
            return None

        entry = float(result.entry_ref)
        tick = symbol_info.tick_size if symbol_info else 0.0

        # ---------------- stop loss ---------------- #
        buffer = atr * mode.sl_atr_mult
        if ctx.weekend:
            buffer *= 1.30                       # close/reopen wicks run wider

        sl = (result.invalidation - buffer) if is_long else (result.invalidation + buffer)

        # push the stop past the closest same-side pool (stops cluster there)
        shelter = self._shelter_level(ctx, result.side, entry, sl, atr)
        if shelter is not None:
            sl = min(sl, shelter) if is_long else max(sl, shelter)

        risk = abs(entry - sl)
        risk_pct = risk / entry * 100 if entry else 0.0
        if risk <= 0 or risk_pct < MIN_SL_PCT or risk_pct > MAX_SL_PCT:
            return None

        # ---------------- take profits ---------------- #
        tps = self._build_tps(result, entry, sl, risk, is_long, mode, atr,
                              major_zones=getattr(result, 'major_zones', None))
        if len(tps) < 3:
            return None

        rr_total = tps[-1].rr
        if rr_total < max(self.rt.min_rr, mode.min_tp3_rr):
            return None

        # TP1 has to be worth taking on its own, and TP2 has to be genuinely
        # reachable — a ladder that front-loads everything into TP1 is how a
        # setup "wins" and still loses money when it reverses straight after.
        if tps[0].rr < mode.min_tp1_rr or tps[1].rr < mode.min_tp2_rr:
            return None

        # The entry zone must be narrow relative to the distance to TP1.
        # A zone that spans a fifth of the way to the first target means the
        # difference between a good fill and a bad one is most of the trade.
        zone_width = abs(float(result.entry_high) - float(result.entry_low))
        tp1_distance = abs(tps[0].price - entry)
        if tp1_distance <= 0:
            return None
        if zone_width > tp1_distance * mode.max_zone_frac_of_tp1:
            shrink = tp1_distance * mode.max_zone_frac_of_tp1 / 2.0
            result.entry_low = entry - shrink
            result.entry_high = entry + shrink

        # ---------------- rounding to tick ---------------- #
        if tick and tick > 0:
            entry = round_step(entry, tick, "down")
            sl = round_step(sl, tick, "down" if is_long else "up")
            for t in tps:
                t.price = round_step(t.price, tick, "down" if is_long else "up")
            result.entry_low = round_step(result.entry_low, tick, "down")
            result.entry_high = round_step(result.entry_high, tick, "up")

        # sanity: monotonic TP ladder on the correct side of entry
        seq = [t.price for t in tps]
        ordered = seq == sorted(seq) if is_long else seq == sorted(seq, reverse=True)
        if not ordered:
            return None
        if is_long and not (sl < entry < tps[0].price):
            return None
        if not is_long and not (sl > entry > tps[0].price):
            return None

        # ---------------- risk annotations ---------------- #
        confs = list(result.confirmations)
        confs.append(Confirmation(
            name="Risk Geometry",
            detail=f"Stop {risk_pct:.2f}% away, TP3 at {rr_total:.2f}R "
                   f"(mode minimum {max(self.rt.min_rr, mode.min_tp3_rr):.1f}R)",
            weight=0.0, category="risk"))

        if ctx.weekend:
            confs.append(Confirmation(
                name="Danger Window Filter Passed",
                detail=f"Friday-close / Monday-reopen window — required score "
                       f"raised to {required_score:.0f}, extra confirmations "
                       f"demanded and stop buffer widened 30%",
                weight=0.0, category="risk"))

        sig = Signal(
            symbol=ctx.symbol, side=result.side, mode=mode.name, strategy=strategy_name,
            entry=entry, entry_low=float(result.entry_low), entry_high=float(result.entry_high),
            stop_loss=float(sl), take_profits=tps,
            score=result.score, required_score=required_score,
            confirmations=confs, mtf=ctx.mtf_views(result.side),
            risk_pct=round(risk_pct, 3), rr_total=round(rr_total, 2),
            atr_pct=round(safe_float(ctx.trigger.last.get("atr_pct")), 3),
            quote_volume=ctx.quote_volume, funding_rate=ctx.funding_rate,
            expires_at=time.time() + mode.expiry_minutes * 60,
            weekend=ctx.weekend,
        )

        return sig

    # ------------------------------------------------------------------ #
    def _build_tps(self, result: StrategyResult, entry: float, sl: float,
                   risk: float, is_long: bool, mode: ModeSpec,
                   atr: float,
                   major_zones: Optional[List] = None) -> List[TakeProfit]:
        """
        Anchor TPs to places that can actually absorb size: untapped liquidity
        pools, and the major supply/demand zones where the last large moves
        originated. Pools are already sorted by distance.

        Major zones are merged in as targets in their own right, because a
        pool sitting just in front of a large supply block is not where price
        stops — the block is.
        """
        floors = [mode.min_tp1_rr, mode.min_tp2_rr, mode.min_tp3_rr]
        min_gap = risk * floors[0]

        if is_long:
            pools = [p for p in result.target_pools if p.price > entry + min_gap]
        else:
            pools = [p for p in result.target_pools if p.price < entry - min_gap]

        # fold major supply/demand zones into the target list
        for z in (major_zones or []):
            edge = z.low if is_long else z.high
            if is_long and edge > entry + min_gap and z.side == "bear":
                pools.append(LiquidityPool(price=float(edge), kind="buyside",
                                           strength=z.strength, touches=1,
                                           index=z.index, label="Supply"))
            elif not is_long and edge < entry - min_gap and z.side == "bull":
                pools.append(LiquidityPool(price=float(edge), kind="sellside",
                                           strength=z.strength, touches=1,
                                           index=z.index, label="Demand"))
        pools.sort(key=lambda p: p.price, reverse=not is_long)

        label_for = {"EQH": "equal highs (buy-side stops)",
                     "EQL": "equal lows (sell-side stops)",
                     "Round": "round-number magnet",
                     "Swing": "prior swing liquidity",
                     "Supply": "major supply zone",
                     "Demand": "major demand zone"}
        alloc = mode.tp_split
        tps: List[TakeProfit] = []
        used: set = set()
        last_price = entry

        # Build the ladder FROM the reward floors rather than filtering after
        # the fact. Each level must clear its own minimum: a target picked
        # simply because it was the next pool along is how TP1 ends up close
        # enough that the trade "wins" and still gives everything back.
        for i, floor in enumerate(floors):
            need = risk * floor
            spacing = max(risk * 0.5, atr * 0.5)
            pick = None
            for p in pools:
                if id(p) in used:
                    continue
                dist = (p.price - entry) if is_long else (entry - p.price)
                if dist < need:
                    continue
                if tps and abs(p.price - last_price) < spacing:
                    continue
                pick = p
                break

            if pick is not None:
                used.add(id(pick))
                rr = abs(pick.price - entry) / risk
                reason = (f"{label_for.get(pick.label, 'liquidity pool')} · "
                          f"strength {pick.strength:.0%} · {pick.touches} touch(es)")
                price = float(pick.price)
            else:
                rr = floor
                price = entry + risk * rr if is_long else entry - risk * rr
                reason = f"measured {rr:.1f}R extension (no clean pool at this range)"

            if tps and ((is_long and price <= tps[-1].price)
                        or (not is_long and price >= tps[-1].price)):
                continue                    # would break the ladder ordering

            tps.append(TakeProfit(
                level=i + 1, price=price, rr=round(rr, 2), reason=reason,
                allocation=alloc[i] if i < len(alloc) else 0.25))
            last_price = price

        return tps[:3]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _shelter_level(ctx: AnalysisContext, side: str, entry: float,
                       sl: float, atr: float) -> Optional[float]:
        """
        Find the nearest same-side pool between entry and our stop; if one
        exists, move the stop just beyond it so we are not parked inside the
        obvious stop cluster.
        """
        want = "sellside" if side == "LONG" else "buyside"
        candidates = []
        for tf_ctx in (ctx.setup, ctx.trigger, ctx.structure):
            for p in tf_ctx.pools:
                if p.kind != want:
                    continue
                if side == "LONG" and sl - atr * 1.2 <= p.price <= entry:
                    candidates.append(p.price)
                elif side == "SHORT" and entry <= p.price <= sl + atr * 1.2:
                    candidates.append(p.price)
        if not candidates:
            return None
        pad = atr * 0.22
        return (min(candidates) - pad) if side == "LONG" else (max(candidates) + pad)

