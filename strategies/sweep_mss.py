"""
STRATEGY 1 — LIQUIDITY SWEEP + MARKET STRUCTURE SHIFT  (reversal)
=================================================================

This is the manipulation pattern, traded from the other side.

Large participants cannot fill size where retail can. To buy, they need
someone selling; to sell, they need someone buying. The counterparty they need
is sitting in the most obvious places on the chart — the stops resting under
equal lows and above equal highs, and the breakout orders that trigger there.
So price is driven *into* those pools, the stops are absorbed, and the real
move begins in the opposite direction.

The sequence, in strict order:

  1. Pool mapped        - equal highs/lows, session or swing extremes
  2. Inducement         - a minor pool taken first, supplying initial fills
  3. THE SWEEP          - the pool is raided and price CLOSES back inside
  4. MSS                - a displacement candle closes beyond the last
                          opposing swing: intent has flipped, with conviction
  5. POI                - the FVG or order block left by that displacement
  6. Sniper trigger     - entry confirmation on the lowest timeframe

Links 1, 3, 4, 5 and 6 are mandatory — no amount of score substitutes for a
missing one. What separates this from "price bounced off support" is step 4:
without displacement through structure, a sweep is just a deeper pullback.
"""
from __future__ import annotations

from typing import List, Optional

from core.models import Confirmation
from core.structure import (find_major_zones,
                            LiquidityPool, Zone, detect_divergence,
                            detect_liquidity_sweeps, detect_mss, find_fvgs,
                            find_inducement, find_order_blocks, nearest_zone)
from strategies.base import AnalysisContext, BaseStrategy, StrategyResult
from strategies.scoring import SWEEP_MSS_WEIGHTS as W
from strategies.scoring import scaled
from utils.helpers import safe_float
from utils.logger import get_logger

log = get_logger("sweep_mss")


class SweepMSSStrategy(BaseStrategy):
    name = "SWEEP_MSS"
    label = "Liquidity Sweep + MSS"
    short = "Sweep+MSS"
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
        # 1. THE SWEEP  (MANDATORY) — this defines the direction
        # ------------------------------------------------------------------ #
        # Take the strongest sweep that ALSO produced a structure shift. The
        # freshest raid is frequently too new to have displaced yet, and
        # trading it on that basis is exactly the mistake this strategy exists
        # to avoid.
        # Take the strongest sweep that produced a structure shift AND whose
        # direction survives the higher-timeframe test. The freshest raid is
        # frequently too new to have displaced yet, and the loudest one often
        # points into the wrong half of the dealing range — evaluating every
        # candidate rather than only the top-scoring one is the difference
        # between finding the setup and missing it.
        htf_trend = bias.structure.trend
        struct_pd = struct.structure.premium_discount
        pos = struct.structure.position_in_range

        sweep = mss = None
        for cand in detect_liquidity_sweeps(setup.df, setup.pools, lookback=16):
            idx = len(setup.df) - 1 - int(cand["bars_ago"])
            found = detect_mss(setup.df, cand["direction"], idx)
            if not found:
                continue
            long_cand = cand["direction"] == "LONG"
            aligned = ((long_cand and htf_trend == "BULL")
                       or (not long_cand and htf_trend == "BEAR"))
            counter_ok = (pos < 0.42) if long_cand else (pos > 0.58)
            if not aligned and not counter_ok:
                continue      # fading the HTF from mid-range is how accounts die
            sweep, mss = cand, found
            break
        if not sweep or not mss:
            return None

        side = sweep["direction"]
        is_long = side == "LONG"
        pool: LiquidityPool = sweep["pool"]
        sweep_extreme = sweep["sweep_extreme"]

        confs.append(Confirmation(
            name="Liquidity Sweep",
            detail=f"{pool.label} {'buy-side' if pool.kind == 'buyside' else 'sell-side'} "
                   f"pool at {pool.price:.6g} raided by {sweep['penetration_atr']} ATR "
                   f"then reclaimed, {sweep['bars_ago']} bar(s) ago",
            weight=scaled(W["liquidity_sweep"], pool.strength),
            timeframe=setup.timeframe.upper(), category="liquidity"))

        # ------------------------------------------------------------------ #
        # 2. MARKET STRUCTURE SHIFT  (MANDATORY)
        # ------------------------------------------------------------------ #
        sweep_idx = len(setup.df) - 1 - int(sweep["bars_ago"])
        confs.append(Confirmation(
            name="Market Structure Shift",
            detail=f"{setup.timeframe.upper()} displacement close through "
                   f"{mss['level']:.6g} ({mss['displacement_atr']} ATR body) — "
                   f"intent flipped {mss['bars_ago']} bar(s) ago",
            weight=scaled(W["mss"], min(1.0, mss["displacement_atr"] / 1.8)),
            timeframe=setup.timeframe.upper(), category="structure"))

        # ------------------------------------------------------------------ #
        # 3. POI left by the displacement leg  (MANDATORY)
        # ------------------------------------------------------------------ #
        zone_side = "bull" if is_long else "bear"
        leg_zones: List[Zone] = []
        for z in find_fvgs(setup.df) + find_order_blocks(setup.df):
            if z.side != zone_side or z.mitigated:
                continue
            # only zones created by the MSS leg itself are valid here
            if mss["leg_start"] - 2 <= z.index <= mss["leg_end"] + 1:
                leg_zones.append(z)

        poi: Optional[Zone] = None
        if leg_zones:
            # the FVG is the cleanest fill; prefer it, then the order block
            fvgs = [z for z in leg_zones if z.kind == "FVG"]
            poi = max(fvgs or leg_zones, key=lambda z: z.strength)
        else:
            poi = nearest_zone(setup.fvgs, price, zone_side, atr * 2.0) \
                or nearest_zone(setup.obs, price, zone_side, atr * 2.0)
        if poi is None:
            return None

        from_leg = bool(leg_zones)
        confs.append(Confirmation(
            name=f"Displacement {poi.kind}",
            detail=f"{setup.timeframe.upper()} {zone_side} {poi.kind} at "
                   f"{poi.low:.6g}–{poi.high:.6g}"
                   + (" left by the MSS leg" if from_leg else " (nearest unmitigated)"),
            weight=scaled(W["poi_entry"], poi.strength if from_leg else 0.3),
            timeframe=setup.timeframe.upper(), category="structure"))

        # ------------------------------------------------------------------ #
        # 4. Higher-timeframe context  (MANDATORY)
        # ------------------------------------------------------------------ #
        # A reversal is only worth taking if it runs toward, not against, the
        # higher-timeframe draw on liquidity. Already screened above; recorded
        # here so it carries its weight in the score.
        aligned = (is_long and htf_trend == "BULL") or (not is_long and htf_trend == "BEAR")

        confs.append(Confirmation(
            name="HTF Context",
            detail=(f"{bias.timeframe.upper()} {htf_trend} and the sweep runs with it"
                    if aligned else
                    f"Counter-trend reversal from the {struct_pd.lower()} extreme "
                    f"({pos:.0%} of the {struct.timeframe.upper()} range)"),
            weight=scaled(W["htf_context"],
                          bias.structure.strength if aligned else 0.45),
            timeframe=bias.timeframe.upper(), category="bias"))

        # ------------------------------------------------------------------ #
        # 5. Sniper trigger  (MANDATORY)
        # ------------------------------------------------------------------ #
        if not self._trigger(ctx, is_long, confs, atr):
            return None

        # ------------------------------------------------------------------ #
        # EXTRA CONFIRMATIONS
        # ------------------------------------------------------------------ #
        # (a) premium / discount location
        if (is_long and struct_pd == "DISCOUNT") or (not is_long and struct_pd == "PREMIUM"):
            confs.append(Confirmation(
                name="Premium/Discount",
                detail=f"Entry from the {struct_pd.lower()} of the "
                       f"{struct.timeframe.upper()} dealing range ({pos:.0%})",
                weight=W["premium_discount"],
                timeframe=struct.timeframe.upper(), category="bias"))
        elif struct_pd == "EQUILIBRIUM":
            confs.append(Confirmation(
                name="Premium/Discount",
                detail=f"Entry at equilibrium ({pos:.0%} of range)",
                weight=scaled(W["premium_discount"], 0.0),
                timeframe=struct.timeframe.upper(), category="bias"))

        # (b) volume on the raid itself
        sweep_row = setup.df.iloc[max(0, sweep_idx)]
        vol_z = safe_float(sweep_row.get("vol_z"))
        if vol_z >= 1.0:
            confs.append(Confirmation(
                name="Sweep Volume",
                detail=f"Raid candle traded {vol_z:.1f}σ above its 50-bar mean — "
                       f"real size absorbed the stops",
                weight=scaled(W["sweep_volume"], min(1.0, (vol_z - 1.0) / 1.5)),
                timeframe=setup.timeframe.upper(), category="volume"))

        # (c) inducement — was there fuel before the raid?
        ind = find_inducement(setup.df, side, sweep_idx)
        if ind:
            confs.append(Confirmation(
                name="Inducement Taken",
                detail=f"Minor pool at {ind['price']:.6g} was swept first — "
                       f"the fills were collected before the reversal",
                weight=W["inducement"],
                timeframe=setup.timeframe.upper(), category="liquidity"))

        # (d) divergence at the extreme
        want = "bullish" if is_long else "bearish"
        divs = []
        if detect_divergence(setup.df, "rsi") == want:
            divs.append("RSI")
        if detect_divergence(setup.df, "cvd") == want:
            divs.append("delta")
        if divs:
            confs.append(Confirmation(
                name="Divergence",
                detail=f"{' and '.join(divs)} divergence into the sweep — "
                       f"price made the extreme, momentum did not",
                weight=scaled(W["divergence"], 1.0 if len(divs) > 1 else 0.4),
                timeframe=setup.timeframe.upper(), category="momentum"))

        # (e) did the sweep happen at a higher-timeframe zone?
        htf_zone = (nearest_zone(struct.obs, sweep_extreme, zone_side, atr * 2.5)
                    or nearest_zone(struct.fvgs, sweep_extreme, zone_side, atr * 2.5))
        if htf_zone:
            confs.append(Confirmation(
                name="HTF Zone Confluence",
                detail=f"The raid landed in a {struct.timeframe.upper()} "
                       f"{htf_zone.kind} at {htf_zone.low:.6g}–{htf_zone.high:.6g}",
                weight=scaled(W["htf_poi_confluence"], htf_zone.strength),
                timeframe=struct.timeframe.upper(), category="structure"))

        # (f) opposing liquidity to actually target
        targets = self.opposing_pools(ctx, side, price)
        if not targets:
            return None                   # nothing to pay for the risk
        confs.append(Confirmation(
            name="Target Liquidity",
            detail=f"{len(targets)} untapped "
                   f"{'buy-side' if is_long else 'sell-side'} pool(s), "
                   f"first at {targets[0].price:.6g}",
            weight=scaled(W["target_liquidity"], targets[0].strength),
            timeframe=struct.timeframe.upper(), category="liquidity"))

        # (g) session, BTC regime, volatility
        self.regime_confirmations(ctx, side, confs)

        # ------------------------------------------------------------------ #
        # invalidation & entry zone
        # ------------------------------------------------------------------ #
        # The stop belongs beyond the sweep extreme: if price trades back
        # through the level that was raided, the manipulation reading was wrong.
        if is_long:
            invalidation = min(sweep_extreme, poi.low)
        else:
            invalidation = max(sweep_extreme, poi.high)

        entry_low, entry_high, entry_ref = self.zone_from_poi(
            poi, is_long, price, atr, ctx.mode)

        return StrategyResult(
            side=side, entry_ref=entry_ref,
            entry_low=entry_low, entry_high=entry_high,
            invalidation=float(invalidation),
            confirmations=confs, score=self.score_of(confs),
            target_pools=targets, poi=poi,
            major_zones=find_major_zones(ctx.structure.df),
            tags=["sweep", "mss", pool.label.lower()],
        )
