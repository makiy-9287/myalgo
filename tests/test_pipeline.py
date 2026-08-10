#!/usr/bin/env python3
"""
Offline pipeline test — no network, no API keys, no Telegram.

Generates synthetic OHLCV that contains the patterns the strategies look for
(trend legs, pullbacks, equal highs/lows, stop-hunt wicks) and pushes it all
the way through:

    indicators -> structure -> strategy -> signal builder -> Telegram formatter

Run:  python tests/test_pipeline.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config.modes import MODES, TF_MINUTES
from config.settings import RuntimeConfig, Settings
from core.data_manager import SymbolInfo
from core.indicators import enrich
from core.structure import (analyse_structure, detect_liquidity_sweep, find_fvgs,
                            find_liquidity_pools, find_order_blocks)
from engine.signal_builder import SignalBuilder
from notify.formatter import (format_closed, format_expired, format_fill,
                              format_pending, format_pnl, format_report,
                              format_signal, format_status, format_tp_hit,
                              split_message)
from strategies.base import AnalysisContext, build_tf_context
from strategies.sweep_mss import SweepMSSStrategy
from strategies.ob_fvg import OrderBlockFVGStrategy

PASS, FAIL = "\033[92m✓\033[0m", "\033[91m✗\033[0m"
results = {"pass": 0, "fail": 0}


def check(name: str, cond: bool, extra: str = "") -> bool:
    if cond:
        results["pass"] += 1
        print(f"  {PASS} {name} {extra}")
    else:
        results["fail"] += 1
        print(f"  {FAIL} {name} {extra}")
    return cond


# --------------------------------------------------------------------------- #
def _base_series(n: int, start: float, drift: float, vol: float,
                 seed: int, bars_ms: int) -> pd.DataFrame:
    """
    Random-walk OHLCV with periodic pullbacks.

    `drift` is the TOTAL fractional return across the whole series (e.g. 0.45
    for +45%). Log-returns are de-meaned and re-centred on that target so the
    injected pullbacks cannot quietly cancel out the trend — without this the
    generator produced "uptrends" that actually finished 37% down.
    """
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, vol, n)
    direction = 1.0 if drift >= 0 else -1.0
    for i in range(0, n, 53):                 # scheduled counter-trend pullbacks
        shocks[i] -= direction * abs(rng.normal(0, vol * 2.5))

    target = np.log1p(drift) / max(1, n - 1)
    shocks = shocks - shocks.mean() + target

    closes = start * np.exp(np.cumsum(shocks))
    closes[0] = start
    open_ = np.concatenate([[start], closes[:-1]])
    spread = np.abs(rng.normal(0, vol * 0.7, n)) * closes
    high = np.maximum(open_, closes) + spread
    low = np.minimum(open_, closes) - spread
    volume = np.abs(rng.normal(1_000_000, 220_000, n)) + 1_000

    ot = np.arange(n, dtype=np.int64) * bars_ms + 1_700_000_000_000
    return pd.DataFrame({"open_time": ot, "open": open_, "high": high,
                         "low": low, "close": closes, "volume": volume})


def _inject_setup(df: pd.DataFrame, bullish: bool, span: int = 400) -> pd.DataFrame:
    """
    Plant a complete textbook sequence in the tail of the series so the
    resampled higher timeframes all see it:

        [-32 .. -28]  raid of an equal-high/low pool (the stop hunt)
        [-27 .. -16]  displacement leg away from the pool
        [-15 ..  -4]  retracement back into the imbalance
        [ -3 ..  -1]  sniper trigger back in the signal direction

    The earlier version wrote the sweep and the retracement to the same bars,
    so the retracement silently erased the sweep and nothing could ever fire.
    """
    d = df.copy()
    n = len(d)
    span = min(span, n - 40)
    if n < 120 or span < 60:
        return d

    hi = d["high"].to_numpy(copy=True); lo = d["low"].to_numpy(copy=True)
    op = d["open"].to_numpy(copy=True); cl = d["close"].to_numpy(copy=True)
    vo = d["volume"].to_numpy(copy=True)

    sgn = 1.0 if bullish else -1.0

    # ---- 1. build the pool of equal lows (bull) / equal highs (bear)
    if bullish:
        pool = float(lo[-span:-40].min())
        for idx in (-span + 40, -span + 110, -span + 180):
            lo[idx] = pool * 1.0003
            cl[idx] = max(cl[idx], pool * 1.002)
            vo[idx] *= 1.6
    else:
        pool = float(hi[-span:-40].max())
        for idx in (-span + 40, -span + 110, -span + 180):
            hi[idx] = pool * 0.9997
            cl[idx] = min(cl[idx], pool * 0.998)
            vo[idx] *= 1.6

    def _set(idx, o, c, pad=0.0015, volx=1.0):
        op[idx] = o
        cl[idx] = c
        hi[idx] = max(o, c) * (1 + pad)
        lo[idx] = min(o, c) * (1 - pad)
        vo[idx] *= volx

    # ---- 2. the raid: wick through the pool, close back inside
    raid = -32
    if bullish:
        op[raid] = pool * 1.003
        cl[raid] = pool * 1.005
        lo[raid] = pool * 0.990          # stops taken here
        hi[raid] = pool * 1.008
    else:
        op[raid] = pool * 0.997
        cl[raid] = pool * 0.995
        hi[raid] = pool * 1.010
        lo[raid] = pool * 0.992
    vo[raid] *= 4.0

    # ---- 3. displacement leg (creates the FVG / order block)
    price = cl[raid]
    for k in range(-31, -15):
        step = 1 + sgn * 0.0035
        _set(k, price, price * step, pad=0.0012, volx=2.2)
        price = cl[k]

    leg_end = price
    leg_start = cl[raid]

    # ---- 4. retracement halfway back into the imbalance
    target = leg_end - (leg_end - leg_start) * 0.5
    path = np.linspace(leg_end, target, 12)
    for j, k in enumerate(range(-15, -3)):
        _set(k, cl[k - 1], path[j], pad=0.0010, volx=0.7)

    # ---- 5. sniper trigger: engulfing close back in the signal direction
    for k, mult in ((-3, 1.0), (-2, 1.0), (-1, 1.0)):
        pass
    _set(-3, cl[-4], cl[-4] * (1 - sgn * 0.0015), pad=0.0008, volx=0.6)
    _set(-2, cl[-3], cl[-3] * (1 + sgn * 0.0055), pad=0.0010, volx=2.6)
    _set(-1, cl[-2], cl[-2] * (1 + sgn * 0.0035), pad=0.0010, volx=2.4)

    d["high"], d["low"] = hi, lo
    d["open"], d["close"], d["volume"] = op, cl, vo
    return d


def _resample(base: pd.DataFrame, factor: int) -> pd.DataFrame:
    """Aggregate the base series into a higher timeframe."""
    if factor <= 1:
        out = base.copy()
    else:
        usable = (len(base) // factor) * factor
        b = base.iloc[len(base) - usable:].reset_index(drop=True)
        g = b.index // factor
        out = pd.DataFrame({
            "open_time": b.groupby(g)["open_time"].first(),
            "open": b.groupby(g)["open"].first(),
            "high": b.groupby(g)["high"].max(),
            "low": b.groupby(g)["low"].min(),
            "close": b.groupby(g)["close"].last(),
            "volume": b.groupby(g)["volume"].sum(),
        }).reset_index(drop=True)

    out["close_time"] = out["open_time"] + 1
    out["quote_volume"] = out["volume"] * out["close"]
    out["trades"] = (out["volume"] / 100).astype(int)
    out["taker_buy_base"] = out["volume"] * 0.5
    out["taker_buy_quote"] = out["quote_volume"] * 0.5
    out.index = pd.to_datetime(out["open_time"], unit="ms", utc=True)
    return enrich(out)


def _retrace_into_poi(df: pd.DataFrame, bullish: bool, bars: int = 4) -> pd.DataFrame:
    """
    After displacement, price normally returns to the imbalance before
    continuing. Without this the series ends at the extreme of the leg, which
    both strategies correctly refuse to chase.
    """
    d = df.copy()
    n = len(d)
    if n < bars + 12:
        return d
    hi = d["high"].to_numpy(copy=True); lo = d["low"].to_numpy(copy=True)
    op = d["open"].to_numpy(copy=True); cl = d["close"].to_numpy(copy=True)
    vo = d["volume"].to_numpy(copy=True)

    leg_start = cl[-(bars + 4)]
    leg_end = cl[-1]
    target = leg_end - (leg_end - leg_start) * 0.5      # 50% back into the gap

    path = np.linspace(leg_end, target, bars + 1)[1:]
    for k in range(bars, 0, -1):
        idx = -k
        prev = cl[idx - 1]
        cl[idx] = path[bars - k]
        op[idx] = prev
        hi[idx] = max(op[idx], cl[idx]) * 1.0012
        lo[idx] = min(op[idx], cl[idx]) * 0.9988
        vo[idx] *= 0.8

    # final bar turns back in the signal direction: the sniper trigger
    if bullish:
        op[-1] = cl[-2]
        cl[-1] = op[-1] * 1.004
        hi[-1] = cl[-1] * 1.001
        lo[-1] = op[-1] * 0.9985
    else:
        op[-1] = cl[-2]
        cl[-1] = op[-1] * 0.996
        lo[-1] = cl[-1] * 0.999
        hi[-1] = op[-1] * 1.0015
    vo[-1] *= 2.2

    d["high"], d["low"] = hi, lo
    d["open"], d["close"], d["volume"] = op, cl, vo
    return d


def synth(n: int = 400, start: float = 100.0, drift: float = 0.35,
          vol: float = 0.006, seed: int = 7, sweep: bool = True,
          interval_ms: int = 300_000) -> pd.DataFrame:
    """Single-timeframe convenience wrapper used by the structure tests."""
    base = _base_series(n, start, drift, vol, seed, interval_ms)
    if sweep:
        base = _inject_setup(base, bullish=drift > 0, span=min(300, n - 40))
    return _resample(base, 1)


def make_ctx(mode_name: str, seed: int = 7, drift: float = 0.5,
             weekend: bool = False) -> AnalysisContext:
    """
    Build a genuinely multi-timeframe context: one base series resampled into
    every timeframe, so the HTF and LTF agree the way real data does.
    The base interval is the GCD of the mode's timeframes (3m and 5m are not
    multiples of each other, so SCALP has to be built from 1m).
    """
    from math import gcd
    from functools import reduce

    mode = MODES[mode_name]
    mins = [TF_MINUTES[tf] for tf in mode.timeframes]
    base_min = reduce(gcd, mins)
    need = max(mode.candles.get(tf, 300) * TF_MINUTES[tf] for tf in mode.timeframes)
    n_base = int(need / base_min) + 60

    # `drift` here is the total return across the series (+0.5 = +50%)
    vol = max(0.0009, 0.0040 * (base_min / 60.0) ** 0.5)

    base = _base_series(n_base, 100.0, drift, vol, seed, base_min * 60_000)
    span = min(600, max(150, n_base // 5))
    base = _inject_setup(base, bullish=drift > 0, span=span)

    frames = {tf: _resample(base, TF_MINUTES[tf] // base_min)
              for tf in mode.timeframes}

    return AnalysisContext(
        symbol="TESTUSDT", mode=mode,
        price=float(frames[mode.trigger_tf]["close"].iloc[-1]),
        bias=build_tf_context(mode.bias_tf, "BIAS", frames[mode.bias_tf]),
        structure=build_tf_context(mode.structure_tf, "STRUCTURE", frames[mode.structure_tf]),
        setup=build_tf_context(mode.setup_tf, "SETUP", frames[mode.setup_tf]),
        trigger=build_tf_context(mode.trigger_tf, "TRIGGER", frames[mode.trigger_tf]),
        quote_volume=85_000_000, funding_rate=0.00012, weekend=weekend,
    )


# --------------------------------------------------------------------------- #
def test_indicators():
    print("\n▸ Indicators")
    df = synth(300)
    for col in ("ema21", "ema200", "atr", "rsi", "adx", "macd_hist",
                "vwap", "vol_z", "cvd", "st_dir", "bb_width", "mfi"):
        check(f"{col} present & finite", col in df.columns and
              np.isfinite(df[col].iloc[-1]))
    check("RSI within 0-100", 0 <= df["rsi"].iloc[-1] <= 100,
          f"({df['rsi'].iloc[-1]:.1f})")
    check("ATR positive", df["atr"].iloc[-1] > 0, f"({df['atr'].iloc[-1]:.4f})")
    check("no NaN in core columns",
          not df[["ema21", "atr", "rsi", "adx"]].tail(50).isna().any().any())

    tiny = synth(30)
    check("short frame handled without crash", tiny is not None)


def test_structure():
    print("\n▸ Market structure")
    df = synth(400, sweep=True)
    st = analyse_structure(df)
    check("trend classified", st.trend in ("BULL", "BEAR", "RANGE"), f"({st.trend})")
    check("swings found", len(st.swing_highs) > 2 and len(st.swing_lows) > 2,
          f"({len(st.swing_highs)}H/{len(st.swing_lows)}L)")
    check("dealing range valid", st.range_high > st.range_low)
    check("premium/discount set",
          st.premium_discount in ("PREMIUM", "DISCOUNT", "EQUILIBRIUM"),
          f"({st.premium_discount} @ {st.position_in_range:.0%})")

    pools = find_liquidity_pools(df, st)
    check("liquidity pools built", len(pools) >= 4, f"({len(pools)} pools)")
    check("pools have both sides",
          {p.kind for p in pools} == {"buyside", "sellside"})
    check("pool strengths in range", all(0 <= p.strength <= 1.05 for p in pools))

    # the raid is planted ~32 bars back, so the window has to reach it
    sweep = detect_liquidity_sweep(df, pools, lookback=36)
    check("stop-hunt detected on seeded sweep", sweep is not None,
          f"({sweep['direction'] if sweep else 'none'})")
    check("sweep outside the lookback window is ignored",
          detect_liquidity_sweep(df, pools, lookback=4) is None)

    obs = find_order_blocks(df)
    fvgs = find_fvgs(df)
    check("order blocks found", len(obs) > 0, f"({len(obs)})")
    check("FVGs found", len(fvgs) > 0, f"({len(fvgs)})")
    check("zones well-formed", all(z.low <= z.high for z in obs + fvgs))

    flat = synth(200, drift=0.0, vol=0.0001, seed=3)
    check("flat market does not crash structure",
          analyse_structure(flat) is not None)


def test_strategies():
    print("\n▸ Strategies")
    strats = [SweepMSSStrategy(), OrderBlockFVGStrategy()]
    fired = 0
    for mode_name in MODES:
        for seed in (7, 21, 42, 99, 123, 256):
            for drift in (0.55, -0.45):
                try:
                    ctx = make_ctx(mode_name, seed=seed, drift=drift)
                except Exception as exc:                 # noqa: BLE001
                    check(f"context {mode_name}/{seed}", False, str(exc))
                    continue
                for s in strats:
                    try:
                        res = s.evaluate(ctx)
                    except Exception as exc:             # noqa: BLE001
                        check(f"{s.name} crashed ({mode_name}/{seed})", False, repr(exc))
                        continue
                    if res is None:
                        continue
                    fired += 1
                    ok = (res.side in ("LONG", "SHORT")
                          and res.entry_low <= res.entry_high
                          and 0 <= res.score <= 100
                          and len(res.confirmations) >= 3)
                    check(f"{s.name} {mode_name} s{seed} valid result", ok,
                          f"({res.side} score={res.score:.0f} "
                          f"confs={len(res.confirmations)})")
    check("strategies produced setups across the sweep of seeds", fired > 0,
          f"({fired} raw setups)")


def test_signal_builder():
    print("\n▸ Signal builder")
    rt = RuntimeConfig()
    rt.min_rr = 1.2
    builder = SignalBuilder(rt)
    si = SymbolInfo(symbol="TESTUSDT", base="TEST", tick_size=0.001,
                    step_size=0.1, min_qty=0.1, min_notional=5.0,
                    quote_volume=85e6)
    strats = [SweepMSSStrategy(), OrderBlockFVGStrategy()]
    built = 0
    for mode_name in MODES:
        for seed in (7, 21, 42, 99, 123, 256, 512):
            for drift in (0.55, -0.45):
                ctx = make_ctx(mode_name, seed=seed, drift=drift)
                for s in strats:
                    res = s.evaluate(ctx)
                    if res is None:
                        continue
                    sig = builder.build(ctx, res, s.name, si, 60.0)
                    if sig is None:
                        continue
                    built += 1
                    long = sig.side == "LONG"
                    tps = [t.price for t in sig.take_profits]
                    check(f"{s.name}/{mode_name} 3 TPs", len(tps) == 3)
                    check(f"{s.name}/{mode_name} SL correct side",
                          (sig.stop_loss < sig.entry) if long else (sig.stop_loss > sig.entry),
                          f"(SL {sig.stop_loss:.4f} vs entry {sig.entry:.4f})")
                    check(f"{s.name}/{mode_name} TP ladder ordered",
                          tps == sorted(tps) if long else tps == sorted(tps, reverse=True))
                    check(f"{s.name}/{mode_name} RR increases",
                          all(sig.take_profits[i].rr < sig.take_profits[i + 1].rr
                              for i in range(2)))
                    check(f"{s.name}/{mode_name} risk sane",
                          0.1 < sig.risk_pct < 10, f"({sig.risk_pct:.2f}%)")
                    check(f"{s.name}/{mode_name} expiry set",
                          sig.expires_at > sig.created_at)
                    check(f"{s.name}/{mode_name} TP reasons non-empty",
                          all(t.reason for t in sig.take_profits))
    check("signal builder produced signals", built > 0, f"({built} signals)")
    return built


def test_formatting():
    print("\n▸ Telegram formatting")
    rt = RuntimeConfig()
    rt.min_rr = 1.2
    builder = SignalBuilder(rt)
    si = SymbolInfo(symbol="TESTUSDT", base="TEST", tick_size=0.0001,
                    step_size=0.1, min_qty=0.1, quote_volume=85e6)
    strats = [SweepMSSStrategy(), OrderBlockFVGStrategy()]

    sample = None
    for mode_name in MODES:
        for seed in (7, 42, 99, 256, 512, 1024):
            for drift in (0.55, -0.45):
                ctx = make_ctx(mode_name, seed=seed, drift=drift)
                for s in strats:
                    res = s.evaluate(ctx)
                    if res is None:
                        continue
                    sig = builder.build(ctx, res, s.name, si, 60.0)
                    if sig is None:
                        continue
                    text = format_signal(sig, si.tick_size)
                    check(f"{s.name}/{mode_name} message non-empty", 120 < len(text) < 500,
                          f"({len(text)} chars)")
                    check(f"{s.name}/{mode_name} balanced <b> tags",
                          text.count("<b>") == text.count("</b>"))
                    check(f"{s.name}/{mode_name} balanced <i> tags",
                          text.count("<i>") == text.count("</i>"))
                    check(f"{s.name}/{mode_name} balanced <code> tags",
                          text.count("<code>") == text.count("</code>"))
                    check(f"{s.name}/{mode_name} no raw ampersand",
                          " & " not in text)
                    for part in split_message(text):
                        check(f"{s.name}/{mode_name} chunk <= 4096",
                              len(part) <= 4096, f"({len(part)} chars)")
                    if sample is None:
                        sample = text
    if sample:
        print("\n" + "─" * 66)
        print("SAMPLE TELEGRAM SIGNAL (HTML tags shown raw)")
        print("─" * 66)
        print(sample[:2600])
        print("─" * 66)

    # the card must NOT carry the confirmation list any more
    if sample:
        # the minimal card: coin, direction, entry, TPs, SL, mode, strategy
        check("card omits confirmations",
              "Sniper Trigger" not in sample and "CONFIRMATIONS" not in sample)
        check("card omits the MTF block", "MULTI-TIMEFRAME" not in sample)
        check("card omits volume/funding clutter",
              "Funding" not in sample and "24h volume" not in sample)
        check("card keeps the entry zone", "Entry" in sample)
        check("card keeps all three targets",
              all(f"TP{i}" in sample for i in (1, 2, 3)))
        check("card keeps the stop", "SL" in sample)
        check("card names the mode",
              "DAY" in sample or "SWING" in sample)
        check("card names the strategy",
              "Sweep" in sample or "Order Block" in sample)
        check("card mentions leverage nowhere", "leverage" not in sample.lower())
        check("card is genuinely minimal", len(sample) < 420,
              f"({len(sample)} chars)")

    status = format_status({
        "runtime": RuntimeConfig().to_dict(), "uptime_min": 192.0,
        "universe": 214, "min_volume": 10e6, "last_scan": "12:00:00 UTC",
        "scan_duration": 18.4, "pending": 3, "live": 2, "completed": 41,
        "win_rate": 63.4, "weight": 640, "queue": 0, "weekend": False,
    })
    check("status message renders", len(status) > 200)
    check("status tags balanced", status.count("<b>") == status.count("</b>"))


def test_lifecycle_messages():
    """Every notification the tracker can emit must render safely."""
    print("\n▸ Lifecycle notifications")
    from core.models import Outcome, Signal, TakeProfit

    entry = 100.0
    sig = Signal(symbol="SOLUSDT", side="LONG", mode="DAY",
                 strategy="SWEEP_MSS", entry=entry,
                 entry_low=99.5, entry_high=100.5, stop_loss=98.0,
                 take_profits=[TakeProfit(1, 102.0, 1.0, "EQH pool", 0.4),
                               TakeProfit(2, 104.0, 2.0, "swing high", 0.35),
                               TakeProfit(3, 106.0, 3.0, "range high", 0.25)],
                 score=93.0, required_score=90.0, risk_pct=2.0,
                 expires_at=time.time() + 3600)
    sig.filled_at = time.time() - 900
    sig.fill_price = entry

    msgs = {
        "fill": format_fill(sig, entry, 100.4),
        "tp1": format_tp_hit(sig, sig.take_profits[0], 102.1),
        "expired": format_expired(sig),
    }
    sig.take_profits[0].hit = True
    outcome = Outcome(signal_id=sig.signal_id, symbol=sig.symbol, side=sig.side,
                      mode=sig.mode, strategy=sig.strategy, entry=entry,
                      stop_loss=98.0, fill_price=entry, exit_price=106.0,
                      result="TP3", tp_hits=3, r_multiple=3.0, pct=6.0,
                      filled_at=sig.filled_at, closed_at=time.time())
    msgs["closed"] = format_closed(sig, outcome)

    for name, text in msgs.items():
        check(f"{name} message renders", len(text) > 60)
        for tag in ("b", "i", "code"):
            check(f"{name} balanced <{tag}> tags",
                  text.count(f"<{tag}>") == text.count(f"</{tag}>"))
        check(f"{name} fits one Telegram message", len(text) <= 4096)

    check("fill message names the fill price", "100.000" in msgs["fill"])
    check("TP message shows the R multiple", "1.00R" in msgs["tp1"])
    check("expiry message is short", len(msgs["expired"]) < 320,
          f"({len(msgs['expired'])} chars)")
    check("win is announced as a win", "WIN" in msgs["closed"])

    # --- /pnl and /report views ---
    rows = [{"signal": sig, "price": 103.0, "pct": 3.0, "r": 1.5,
             "to_target": 75.0, "next_tp": sig.take_profits[1],
             "tp_hits": 1, "age_min": 42.0}]
    pnl = format_pnl(rows)
    check("pnl renders", "SOLUSDT" in pnl and "1.50R" in pnl)
    check("pnl tags balanced", pnl.count("<b>") == pnl.count("</b>"))
    check("empty pnl handled", "NO ACTIVE" in format_pnl([]))
    check("empty pending handled", "NO PENDING" in format_pending([]))
    check("pending renders", "SOLUSDT" in format_pending([sig]))

    rep = {"total": 20, "tp1": 12, "tp2": 7, "tp3": 4, "sl": 12, "clean_sl": 8,
           "wins": 12, "losses": 8, "win_rate": 60.0, "avg_r": 0.85,
           "total_r": 17.0, "best_r": 3.0, "worst_r": -1.0,
           "by_mode": {"DAY": {"n": 12, "w": 8}, "SCALP": {"n": 8, "w": 4}},
           "by_strategy": {"LIQUIDITY_SMC": {"n": 20, "w": 12}},
           "published": 60, "filled": 20, "expired": 40}
    text = format_report(rep)
    check("report renders", "60.0%" in text)
    check("report shows full-TP count", "TP3 full" in text)
    check("report tags balanced", text.count("<b>") == text.count("</b>"))
    check("empty report handled",
          "No completed trades" in format_report(
              {"total": 0, "published": 3, "filled": 0, "expired": 3}))


def test_settings_panel():
    """Every inline keyboard must render and every callback must be routable."""
    print("\n▸ Settings panel")
    from notify import keyboards

    rt = RuntimeConfig()
    seen_callbacks = set()

    for name in keyboards.PANELS:
        text, kb = keyboards.render(name, rt)
        check(f"panel '{name}' renders", len(text) > 40)
        check(f"panel '{name}' tags balanced",
              text.count("<b>") == text.count("</b>")
              and text.count("<i>") == text.count("</i>"))
        rows = kb["inline_keyboard"]
        check(f"panel '{name}' has buttons", len(rows) > 0)
        for row in rows:
            for btn in row:
                data = btn["callback_data"]
                seen_callbacks.add(data)
                check(f"'{data}' within Telegram's 64-byte limit",
                      len(data.encode()) <= 64, f"({len(data.encode())}b)")
                check(f"'{data}' has a label", bool(btn["text"].strip()))

    check("unknown panel falls back to main",
          keyboards.render("nonsense", rt)[0].startswith("⚙️"))

    verbs = {c.split(":")[0] for c in seen_callbacks}
    check("only known verbs are emitted", verbs <= {"nav", "tog", "adj"},
          f"({sorted(verbs)})")

    numeric_keys = {c.split(":")[2] for c in seen_callbacks
                    if c.startswith("adj:num:")}
    check("every numeric callback has a range defined",
          numeric_keys <= set(keyboards.NUMERIC),
          f"({sorted(numeric_keys - set(keyboards.NUMERIC))})")

    flag_keys = {c.split(":")[2] for c in seen_callbacks if c.startswith("tog:flag:")}
    check("every flag callback maps to a real setting",
          all(hasattr(rt, k) for k in flag_keys),
          f"({sorted(k for k in flag_keys if not hasattr(rt, k))})")

    mode_keys = {c.split(":")[2] for c in seen_callbacks if c.startswith("tog:modes:")}
    check("mode callbacks map to real modes", mode_keys <= set(MODES))


def test_new_primitives():
    """MSS, inducement, major zones and the danger window."""
    print("\n▸ Structure primitives")
    from datetime import datetime, timezone

    from core.structure import (detect_liquidity_sweeps, detect_mss,
                                find_major_zones, zone_overlap)
    from core.structure import Zone as _Z
    from utils.helpers import in_danger_window

    ctx = make_ctx("SWING", seed=42, drift=0.55)
    df = ctx.setup.df

    sweeps = detect_liquidity_sweeps(df, ctx.setup.pools, lookback=16)
    check("multiple sweep candidates returned", len(sweeps) > 1,
          f"({len(sweeps)})")
    check("candidates sorted strongest first",
          all(sweeps[i]["score"] >= sweeps[i + 1]["score"]
              for i in range(len(sweeps) - 1)))

    found = None
    for c in sweeps:
        idx = len(df) - 1 - c["bars_ago"]
        m = detect_mss(df, c["direction"], idx)
        if m:
            found = (c, m)
            break
    check("MSS detected after a sweep", found is not None)
    if found:
        c, m = found
        check("MSS displacement measured in ATR", m["displacement_atr"] > 0.5,
              f"({m['displacement_atr']} ATR)")
        check("MSS records the leg it displaced through",
              m["leg_start"] <= m["leg_end"])
        check("MSS break comes at or after the sweep",
              m["index"] >= len(df) - 1 - c["bars_ago"])
    check("no MSS from an impossible index",
          detect_mss(df, "LONG", -5) is None)

    zones = find_major_zones(df)
    check("major supply/demand zones found", len(zones) > 0, f"({len(zones)})")
    check("major zones are ranked by strength",
          all(zones[i].strength >= zones[i + 1].strength
              for i in range(len(zones) - 1)))
    check("major zones are typed",
          all(z.kind in ("SUPPLY", "DEMAND") for z in zones))

    a = _Z(low=10.0, high=12.0, index=0, kind="OB", side="bull")
    b = _Z(low=11.0, high=14.0, index=1, kind="FVG", side="bull")
    far = _Z(low=20.0, high=22.0, index=2, kind="FVG", side="bull")
    check("overlapping zones intersect", zone_overlap(a, b) == (11.0, 12.0))
    check("separated zones do not", zone_overlap(a, far) is None)

    # --- the Friday-close / Monday-reopen window ---
    def at(day, hour):
        return in_danger_window(datetime(2026, 8, day, hour, 0,
                                         tzinfo=timezone.utc))
    check("Friday 22:00 is normal trading", not at(7, 22))
    check("Friday 23:00 enters the strict window", at(7, 23))
    check("Saturday is strict", at(8, 12))
    check("Sunday is strict", at(9, 12))
    check("Monday 19:00 is still strict", at(10, 19))
    check("Monday 20:00 returns to normal", not at(10, 20))
    check("midweek is normal", not at(12, 12))


def test_entry_zone_discipline():
    """The narrow-zone contract, which is what protects the reward."""
    print("\n▸ Entry zone discipline")
    from core.models import Signal, TakeProfit

    # the exact shape the user reported: a wide short zone with a near TP1
    wide = Signal(symbol="X", side="SHORT", mode="DAY", strategy="SWEEP_MSS",
                  entry=201.0, entry_low=200.28, entry_high=201.89,
                  stop_loss=203.0)
    check("grazing the near edge is not a fill",
          not wide.in_entry_zone(200.20, 200.40))
    check("trading through the reference level is a fill",
          wide.in_entry_zone(200.50, 201.20))

    long_sig = Signal(symbol="X", side="LONG", mode="DAY", strategy="OB_FVG",
                      entry=100.0, entry_low=99.5, entry_high=100.5,
                      stop_loss=98.0)
    check("long: grazing the top of the zone is not a fill",
          not long_sig.in_entry_zone(100.40, 100.80))
    check("long: reaching the reference level fills",
          long_sig.in_entry_zone(99.80, 100.60))

    # every produced signal must satisfy the geometry contract
    rt = RuntimeConfig()
    rt.min_rr = 6.0
    builder = SignalBuilder(rt)
    si = SymbolInfo(symbol="TESTUSDT", base="TEST", tick_size=0.0001,
                    step_size=0.1, min_qty=0.1, quote_volume=85e6)
    strats = [SweepMSSStrategy(), OrderBlockFVGStrategy()]
    made = 0
    for mode_name in MODES:
        mode = MODES[mode_name]
        for seed in (7, 42, 99, 256, 512, 1024):
            for drift in (0.55, -0.45):
                ctx = make_ctx(mode_name, seed=seed, drift=drift)
                for st in strats:
                    res = st.evaluate(ctx)
                    if res is None:
                        continue
                    sig = builder.build(ctx, res, st.name, si, 60.0)
                    if sig is None:
                        continue
                    made += 1
                    tag = f"{st.name}/{mode_name}"
                    width = abs(sig.entry_high - sig.entry_low)
                    tp1_dist = abs(sig.take_profits[0].price - sig.entry)
                    check(f"{tag} zone narrow vs TP1",
                          width <= tp1_dist * mode.max_zone_frac_of_tp1 + 1e-6,
                          f"({width / tp1_dist:.1%} of the way to TP1)")
                    check(f"{tag} reference sits inside the zone",
                          sig.entry_low <= sig.entry <= sig.entry_high)
                    check(f"{tag} TP1 clears its floor",
                          sig.take_profits[0].rr >= mode.min_tp1_rr - 1e-6,
                          f"({sig.take_profits[0].rr:.2f}R)")
                    check(f"{tag} TP2 clears its floor",
                          sig.take_profits[1].rr >= mode.min_tp2_rr - 1e-6,
                          f"({sig.take_profits[1].rr:.2f}R)")
                    check(f"{tag} TP3 meets the 6R contract",
                          sig.take_profits[2].rr >= mode.min_tp3_rr - 1e-6,
                          f"({sig.take_profits[2].rr:.2f}R)")
    check("entry-zone suite actually exercised signals", made > 0, f"({made})")


def test_scoring_budget():
    """The 100-point budget is a contract, not a suggestion."""
    print("\n▸ Scoring budget")
    from strategies.scoring import (OB_FVG_WEIGHTS, SWEEP_MSS_WEIGHTS,
                                    budget_total, scaled, session_now)

    check("Sweep+MSS budget totals exactly 100",
          budget_total(SWEEP_MSS_WEIGHTS) == 100.0,
          f"({budget_total(SWEEP_MSS_WEIGHTS)})")
    check("OB+FVG budget totals exactly 100",
          budget_total(OB_FVG_WEIGHTS) == 100.0,
          f"({budget_total(OB_FVG_WEIGHTS)})")
    check("no single confirmation dominates",
          max(SWEEP_MSS_WEIGHTS.values()) <= 20
          and max(OB_FVG_WEIGHTS.values()) <= 20)
    check("the sweep carries the most weight in strategy 1",
          SWEEP_MSS_WEIGHTS["liquidity_sweep"] == max(SWEEP_MSS_WEIGHTS.values()))
    check("the order block carries the most weight in strategy 2",
          OB_FVG_WEIGHTS["order_block"] == max(OB_FVG_WEIGHTS.values()))

    check("perfect quality earns the full weight", scaled(10.0, 1.0) == 10.0)
    check("weak quality still earns 75%", scaled(10.0, 0.0) == 7.5)
    check("scaling is monotonic", scaled(10.0, 0.3) < scaled(10.0, 0.7))
    check("out-of-range quality is clamped",
          scaled(10.0, 5.0) == 10.0 and scaled(10.0, -3.0) == 7.5)

    # a full sweep of every weight must be able to reach the 90 threshold
    for name, w in (("Sweep+MSS", SWEEP_MSS_WEIGHTS), ("OB+FVG", OB_FVG_WEIGHTS)):
        cheapest = sorted(w.values())[:3]
        check(f"{name}: 90 reachable while missing the 3 smallest extras",
              sum(w.values()) - sum(cheapest) >= 90,
              f"({sum(w.values()) - sum(cheapest):.0f})")

    london = session_now(_ts_at_hour(14))
    asia = session_now(_ts_at_hour(3))
    check("London/NY overlap scores highest", london[1] == 1.0, f"({london[0]})")
    check("dead Asian hours score zero", asia[1] == 0.0, f"({asia[0]})")


def _ts_at_hour(hour: int) -> float:
    import calendar
    return calendar.timegm((2026, 6, 10, hour, 30, 0, 0, 0, 0))


def test_state():
    print("\n▸ State store & lifecycle")
    import tempfile

    import core.state as state_mod
    from core.models import Signal, TakeProfit

    with tempfile.TemporaryDirectory() as tmp:
        state_mod.STATE_FILE = Path(tmp) / "state.json"
        st = state_mod.StateStore()

        def mk(symbol="BTCUSDT", side="LONG", entry=60000.0):
            return Signal(
                symbol=symbol, side=side, mode="DAY", strategy="SWEEP_MSS",
                entry=entry, entry_low=entry * 0.999, entry_high=entry * 1.001,
                stop_loss=entry * 0.99,
                take_profits=[TakeProfit(1, entry * 1.01, 1.0, "pool", 0.4),
                              TakeProfit(2, entry * 1.02, 2.0, "pool", 0.35),
                              TakeProfit(3, entry * 1.03, 3.0, "pool", 0.25)],
                expires_at=time.time() + 3600, score=92.0)

        sig = mk()
        st.publish(sig)
        check("published signal is pending", st.pending_count == 1)
        check("pending signals are NOT persisted",
              "live" not in state_mod.STATE_FILE.read_text()
              if state_mod.STATE_FILE.exists() else True)

        # --- expiry drops the record entirely ---
        doomed = mk("ETHUSDT")
        doomed.expires_at = time.time() - 1
        st.publish(doomed)
        gone = st.drop_expired()
        check("expired signal returned for notification", len(gone) == 1)
        check("expired signal removed from store", st.pending_count == 1)
        check("expired signal leaves no outcome", len(st.outcomes) == 0)
        check("expiry counted in stats", st.stats["signals_expired"] == 1)

        # --- fill ---
        st.mark_filled(sig, 60000.0)
        check("filled signal moves to live", st.live_count == 1 and st.pending_count == 0)
        check("fill price recorded", sig.fill_price == 60000.0)

        st2 = state_mod.StateStore()
        check("live signal survives reload", st2.live_count == 1)
        restored = st2.live_signals()[0]
        check("reloaded TPs intact", len(restored.take_profits) == 3)
        check("reloaded fill price intact", restored.fill_price == 60000.0)

        # --- unrealised pnl ---
        u = sig.unrealised(60600.0)
        check("unrealised percent correct", abs(u["pct"] - 1.0) < 0.01,
              f"({u['pct']:.2f}%)")
        check("R multiple correct", abs(u["r"] - 1.0) < 0.02, f"({u['r']:.2f}R)")
        check("progress toward TP1 correct", abs(u["to_target"] - 100.0) < 1.0,
              f"({u['to_target']:.0f}%)")
        short = mk("SOLUSDT", "SHORT", 100.0)
        short.fill_price = 100.0
        check("short PnL inverts correctly", short.unrealised(99.0)["pct"] > 0)

        # --- close as a win ---
        sig.take_profits[0].hit = True
        outcome = st.close_signal(sig, sig.take_profits[0].price, "SL after TP1")
        check("closed signal leaves live set", st.live_count == 0)
        check("outcome recorded", len(st.outcomes) == 1)
        check("TP1 before SL counts as a WIN", outcome.is_win)
        check("R multiple computed", abs(outcome.r_multiple - 1.0) < 0.05,
              f"({outcome.r_multiple:.2f}R)")

        # --- a clean loss ---
        loser = mk("XRPUSDT", "LONG", 0.6)
        st.publish(loser)
        st.mark_filled(loser, 0.6)
        st.close_signal(loser, loser.stop_loss, "SL")
        rep = st.report()
        check("report counts both trades", rep["total"] == 2)
        check("win rate is 50%", abs(rep["win_rate"] - 50.0) < 0.01,
              f"({rep['win_rate']:.0f}%)")
        check("TP1 tally correct", rep["tp1"] == 1)
        check("full-TP tally correct", rep["tp3"] == 0)
        check("stop tally correct", rep["sl"] == 2, f"({rep['sl']})")
        check("clean stops separated from post-TP1 stops",
              rep["clean_sl"] == 1, f"({rep['clean_sl']})")
        check("per-mode breakdown present", "DAY" in rep["by_mode"])
        check("mode filter works", st.report(mode="SWING")["total"] == 0)

        # --- cooldowns ---
        st.set_cooldown("BTCUSDT:DAY", 5)
        check("cooldown active", st.on_cooldown("BTCUSDT:DAY"))
        check("unknown key not on cooldown", not st.on_cooldown("NOPE"))


def test_helpers():
    print("\n▸ Helpers")
    from utils.helpers import (chunked, fmt_price, is_weekend, round_step,
                               step_size_to_precision)
    check("round down to step", abs(round_step(1.23456, 0.001) - 1.234) < 1e-9)
    check("round up to step", abs(round_step(1.23412, 0.001, "up") - 1.235) < 1e-9)
    check("integer step", abs(round_step(17.9, 1.0) - 17.0) < 1e-9)
    check("zero step is a no-op", round_step(5.5, 0) == 5.5)
    check("precision from step", step_size_to_precision(0.0001) == 4)
    check("precision of 1", step_size_to_precision(1) == 0)
    check("price formatting small", "0.000012" in fmt_price(0.0000123))
    check("price formatting large", "," in fmt_price(64231.5))
    check("chunking", len(list(chunked(list(range(10)), 3))) == 4)
    check("weekend flag boolean", isinstance(is_weekend(), bool))


def main() -> int:
    print("═" * 66)
    print("  OFFLINE PIPELINE TEST — synthetic data, no network")
    print("═" * 66)
    test_helpers()
    test_indicators()
    test_structure()
    test_strategies()
    built = test_signal_builder()
    test_formatting()
    test_lifecycle_messages()
    test_settings_panel()
    test_new_primitives()
    test_entry_zone_discipline()
    test_scoring_budget()
    test_state()

    print("\n" + "═" * 66)
    total = results["pass"] + results["fail"]
    print(f"  RESULT: {results['pass']}/{total} checks passed"
          f"  ({results['fail']} failed)")
    print("═" * 66)
    if built == 0:
        print("  NOTE: no signals were built from synthetic data — that is not")
        print("  necessarily a bug, the filters are strict by design.")
    return 0 if results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
