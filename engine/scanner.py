"""
The scanning engine.

Two stages, because fetching 4 timeframes for 220 symbols every minute would
blow the REST weight budget within seconds:

  STAGE 1 - CHEAP PRE-FILTER
      Uses only the slow, heavily-cached HTF data (bias + structure TF). Kills
      anything with no HTF trend, dead volatility, extreme funding or an active
      cooldown. Typically removes 80-90% of the universe.

  STAGE 2 - DEEP ANALYSIS
      Only survivors get their setup/trigger timeframes pulled and the full
      structure engine run (swings, pools, order blocks, FVGs, sweeps).

Both strategies are evaluated per mode. Weekend sessions raise the score
threshold AND demand extra confirmations, because thin weekend books produce
fake breaks that reverse the moment Monday liquidity arrives.
"""
from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional, Tuple

from config.modes import MODES, ModeSpec
from config.settings import RuntimeConfig, Settings
from core.data_manager import DataManager
from core.models import Signal
from core.state import StateStore
from core.structure import analyse_structure
from engine.signal_builder import SignalBuilder
from strategies.base import AnalysisContext, BaseStrategy, build_tf_context
from strategies.sweep_mss import SweepMSSStrategy
from strategies.ob_fvg import OrderBlockFVGStrategy
from utils.helpers import gather_limited, in_danger_window, safe_float
from utils.logger import get_logger

log = get_logger("scanner")


class Scanner:
    def __init__(self, data: DataManager, state: StateStore,
                 settings: Settings, runtime: RuntimeConfig):
        self.data = data
        self.state = state
        self.s = settings
        self.rt = runtime
        self.builder = SignalBuilder(runtime)
        self.strategies: Dict[str, BaseStrategy] = {
            SweepMSSStrategy.name: SweepMSSStrategy(),
            OrderBlockFVGStrategy.name: OrderBlockFVGStrategy(),
        }
        self.last_scan_at: float = 0.0
        self.last_scan_duration: float = 0.0
        self.last_scanned_count: int = 0
        self.last_candidates: Dict[str, int] = {}
        # market regime, refreshed once per cycle rather than per symbol
        self.btc_trend: str = ""
        self.btc_change_pct: float = 0.0

    # ------------------------------------------------------------------ #
    def required_score(self, mode_name: str) -> float:
        base = float(self.rt.min_score.get(mode_name, 90.0))
        if in_danger_window():
            base += float(self.rt.danger_score_bonus)
        return min(98.0, base)

    def min_confirmations(self, strategy: BaseStrategy) -> int:
        base = strategy.min_confirmations
        if in_danger_window():
            base += int(self.rt.danger_extra_confirmations)
        return base

    # ------------------------------------------------------------------ #
    # stage 1
    # ------------------------------------------------------------------ #
    async def prefilter(self, mode: ModeSpec) -> List[str]:
        """Cheap HTF screen. Returns symbols worth a deep scan."""
        symbols = list(self.data.universe)
        if not symbols:
            return []

        bias_tf, struct_tf = mode.bias_tf, mode.structure_tf
        await self.data.batch_get(symbols, bias_tf, mode.candles.get(bias_tf, 300))

        survivors: List[str] = []
        weekend = in_danger_window()
        # one setup per coin at a time: a second signal on a symbol that is
        # already pending or running is almost always the same idea restated,
        # and it doubles the notification noise for no extra information.
        busy = {s.symbol for s in self.state.all_tracked()}

        for sym in symbols:
            if sym in busy:
                continue

            si = self.data.get_symbol(sym)
            if not si:
                continue

            if abs(si.funding_rate) > self.rt.funding_extreme:
                continue                                   # funding squeeze risk

            df = self.data.cached(sym, bias_tf)
            if df is None or len(df) < 60:
                continue

            last = df.iloc[-1]
            atr_pct = safe_float(last.get("atr_pct"))
            if atr_pct < self.rt.min_atr_pct or atr_pct > self.rt.max_atr_pct:
                continue

            ema50 = safe_float(last.get("ema50"))
            ema200 = safe_float(last.get("ema200"))
            close = safe_float(last.get("close"))
            adx = safe_float(last.get("adx"))

            trending = (close > ema50 > ema200) or (close < ema50 < ema200)
            # Strategy 1 can trade counter to the EMA stack after a sweep, so the
            # gate is either a clean stack OR a meaningfully strong ADX.
            if not trending and adx < 20:
                continue

            if weekend and adx < 22:
                continue                                   # weekends need real trend

            survivors.append(sym)

        # cap deep scans so one cycle can never exceed the weight budget
        cap = max(20, min(90, self.s.max_symbols // 2))
        survivors.sort(key=lambda s: self.data.symbols[s].quote_volume, reverse=True)
        return survivors[:cap]

    # ------------------------------------------------------------------ #
    # stage 2
    # ------------------------------------------------------------------ #
    async def build_context(self, symbol: str, mode: ModeSpec) -> Optional[AnalysisContext]:
        frames = await self.data.get_multi(symbol, mode.timeframes, mode.candles)
        needed = [mode.bias_tf, mode.structure_tf, mode.setup_tf, mode.trigger_tf]
        if any(tf not in frames for tf in needed):
            return None

        bias = build_tf_context(mode.bias_tf, "BIAS", frames[mode.bias_tf])
        struct = build_tf_context(mode.structure_tf, "STRUCTURE", frames[mode.structure_tf])
        setup = build_tf_context(mode.setup_tf, "SETUP", frames[mode.setup_tf])
        trigger = build_tf_context(mode.trigger_tf, "TRIGGER", frames[mode.trigger_tf])
        if not all((bias, struct, setup, trigger)):
            return None

        si = self.data.get_symbol(symbol)
        return AnalysisContext(
            symbol=symbol, mode=mode, price=float(trigger.df["close"].iloc[-1]),
            bias=bias, structure=struct, setup=setup, trigger=trigger,
            quote_volume=si.quote_volume if si else 0.0,
            funding_rate=si.funding_rate if si else 0.0,
            weekend=in_danger_window(),
            btc_trend=self.btc_trend, btc_change_pct=self.btc_change_pct,
        )

    async def analyse_symbol(self, symbol: str, mode: ModeSpec) -> List[Signal]:
        out: List[Signal] = []
        try:
            ctx = await self.build_context(symbol, mode)
        except Exception as exc:                           # noqa: BLE001
            log.debug("context build failed %s %s: %s", symbol, mode.name, exc)
            return out
        if ctx is None:
            return out

        required = self.required_score(mode.name)

        for strat_name, strat in self.strategies.items():
            if not self.rt.strategies_enabled.get(strat_name, True):
                continue

            cooldown_key = f"{symbol}:{mode.name}:{strat_name}"
            if self.state.on_cooldown(cooldown_key):
                continue

            try:
                result = strat.evaluate(ctx)
            except Exception as exc:                       # noqa: BLE001
                log.debug("strategy %s failed on %s: %s", strat_name, symbol, exc)
                continue
            if result is None:
                continue

            # --- score gate
            if result.score < required:
                continue

            # --- confirmation-count gate (weekend adds to this)
            scoring = [c for c in result.confirmations if c.weight > 0]
            if len(scoring) < self.min_confirmations(strat):
                continue

            # --- weekend hard requirements
            if ctx.weekend and not self._weekend_ok(result):
                continue

            si = self.data.get_symbol(symbol)
            sig = self.builder.build(ctx, result, strat_name, si, required)
            if sig is None:
                continue

            out.append(sig)
            self.state.set_cooldown(cooldown_key, self.rt.signal_cooldown_min)

        return out

    def _weekend_ok(self, result) -> bool:
        """
        Close/reopen rule set. Thin books mean structure breaks lie, so we demand:
          * volume expansion must be present (no ghost moves)
          * at least one bias-category and one liquidity-category confirmation
          * a momentum confirmation on top of the trigger itself
        """
        cats = {c.category for c in result.confirmations if c.weight > 0}
        if "volume" not in cats:
            return False
        if "bias" not in cats:
            return False
        if "liquidity" not in cats:
            return False
        if "momentum" not in cats:
            return False
        return True

    # ------------------------------------------------------------------ #
    # full cycle
    # ------------------------------------------------------------------ #
    async def refresh_regime(self) -> None:
        """
        Read BTC once per cycle and reuse it for every symbol.

        Alts that fight the majors tend to fail regardless of how clean their
        own chart looks, so both strategies score agreement with BTC. Doing it
        here costs one cached fetch instead of one per symbol.
        """
        try:
            df = await self.data.get_klines("BTCUSDT", "4h", 220)
            if df is None or len(df) < 60:
                return
            st = analyse_structure(df)
            self.btc_trend = st.trend
            closes = df["close"]
            ref = float(closes.iloc[-7]) if len(closes) > 7 else float(closes.iloc[0])
            last = float(closes.iloc[-1])
            self.btc_change_pct = ((last - ref) / ref * 100.0) if ref else 0.0
        except Exception as exc:                            # noqa: BLE001
            log.debug("regime refresh failed: %s", exc)

    async def scan_mode(self, mode_name: str) -> List[Signal]:
        mode = MODES[mode_name]
        candidates = await self.prefilter(mode)
        self.last_candidates[mode_name] = len(candidates)
        if not candidates:
            return []

        coros = [self.analyse_symbol(sym, mode) for sym in candidates]
        results = await gather_limited(coros, limit=max(4, self.s.http_concurrency // 2))

        signals: List[Signal] = []
        for res in results:
            if isinstance(res, Exception):
                log.debug("analyse error: %s", res)
                continue
            signals.extend(res)

        signals.sort(key=lambda s: (s.score, s.rr_total), reverse=True)
        return signals[:self.rt.max_signals_per_cycle]

    async def scan_all(self) -> Dict[str, List[Signal]]:
        started = time.time()
        out: Dict[str, List[Signal]] = {}
        total_candidates = 0

        await self.refresh_regime()

        for mode_name in ("DAY", "SWING"):
            if not self.rt.modes_enabled.get(mode_name, True):
                continue
            if in_danger_window() and not self.rt.danger_window_enabled:
                continue
            try:
                sigs = await self.scan_mode(mode_name)
            except Exception as exc:                       # noqa: BLE001
                log.error("scan_mode %s failed: %s", mode_name, exc, exc_info=True)
                sigs = []
            if sigs:
                out[mode_name] = sigs
            total_candidates += self.last_candidates.get(mode_name, 0)

        self.last_scan_at = time.time()
        self.last_scan_duration = self.last_scan_at - started
        self.last_scanned_count = total_candidates

        found = sum(len(v) for v in out.values())
        log.info("Scan complete in %.1fs — %d deep scans, %d signal(s), weight %d",
                 self.last_scan_duration, total_candidates, found,
                 self.data.ex.weight_used)
        return out

    # ------------------------------------------------------------------ #
    async def scan_single(self, symbol: str) -> Tuple[List[Signal], List[str]]:
        """On-demand /scan for one symbol across every enabled mode."""
        symbol = symbol.upper()
        if not symbol.endswith(self.s.quote_asset):
            symbol += self.s.quote_asset

        notes: List[str] = []
        if symbol not in self.data.symbols:
            return [], [f"{symbol} is not in the tradable universe "
                        f"(needs >${self.s.min_volume_usdt / 1e6:.0f}M 24h volume)."]

        signals: List[Signal] = []
        for mode_name, mode in MODES.items():
            if not self.rt.modes_enabled.get(mode_name, True):
                continue
            ctx = await self.build_context(symbol, mode)
            if ctx is None:
                notes.append(f"{mode_name}: not enough candle history")
                continue

            required = self.required_score(mode_name)
            hit = False
            for strat_name, strat in self.strategies.items():
                if not self.rt.strategies_enabled.get(strat_name, True):
                    continue
                try:
                    result = strat.evaluate(ctx)
                except Exception as exc:                   # noqa: BLE001
                    notes.append(f"{mode_name}/{strat_name}: error {exc}")
                    continue
                if result is None:
                    notes.append(f"{mode_name}/{strat_name}: no valid setup "
                                 f"(mandatory conditions unmet)")
                    continue
                if result.score < required:
                    notes.append(f"{mode_name}/{strat_name}: score "
                                 f"{result.score:.0f} < required {required:.0f}")
                    continue
                si = self.data.get_symbol(symbol)
                sig = self.builder.build(ctx, result, strat_name, si, required)
                if sig is None:
                    notes.append(f"{mode_name}/{strat_name}: rejected by risk "
                                 f"geometry (RR or stop distance)")
                    continue
                signals.append(sig)
                hit = True
            if not hit:
                continue

        return signals, notes
