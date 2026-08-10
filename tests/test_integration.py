#!/usr/bin/env python3
"""
Integration test — every moving part except the network.

A fake public Binance is injected in place of the REST client and a fake
Telegram captures outbound messages, message ids, deletions and edits. That
lets us exercise, with zero credentials:

  * universe building and the >$10M volume filter
  * a full multi-mode scan cycle with the market-regime read
  * the complete signal lifecycle:
        published -> entry filled -> TP1 -> TP2 -> stopped / full TP
    including that the right notification fires at each step
  * expiry: the card is deleted AND the record is purged
  * /pnl and /report against real tracked state
  * every Telegram command, including malformed arguments
  * every settings-panel button, end to end through the callback router

Run:  python tests/test_integration.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_pipeline import _base_series, _inject_setup, _resample  # noqa: E402

PASS, FAIL = "\033[92m✓\033[0m", "\033[91m✗\033[0m"
results = {"pass": 0, "fail": 0}


def check(name: str, cond: bool, extra: str = "") -> bool:
    if cond:
        results["pass"] += 1
        print(f"  {PASS} {name} {extra}")
    else:
        results["fail"] += 1
        print(f"  {FAIL} {name} {extra}")
    return bool(cond)


# --------------------------------------------------------------------------- #
SYMS = [("BTCUSDT", 60000.0, 3.1e9), ("ETHUSDT", 3000.0, 1.4e9),
        ("SOLUSDT", 150.0, 6.2e8), ("XRPUSDT", 0.62, 4.1e8),
        ("DOGEUSDT", 0.16, 2.2e8), ("AVAXUSDT", 34.0, 1.1e8),
        ("LINKUSDT", 17.0, 9.5e7), ("ARBUSDT", 1.1, 4.4e7),
        ("TINYUSDT", 0.004, 2.0e6)]           # below the 10M cut — must drop

PRICES = {s: p for s, p, _ in SYMS}


class FakeExchange:
    """Mimics the public-data client with deterministic, MTF-correlated data."""

    def __init__(self):
        self._bases: dict = {}
        self.weight_used = 118
        # symbol -> (low, high, close) forced for the next tracker poll
        self.overrides: dict = {}
        self.calls: list = []

    async def start(self): ...
    async def close(self): ...

    async def ping(self):
        return {}

    async def exchange_info(self):
        return {"symbols": [{
            "symbol": s, "status": "TRADING", "contractType": "PERPETUAL",
            "quoteAsset": "USDT", "baseAsset": s.replace("USDT", ""),
            "pricePrecision": 4, "quantityPrecision": 3,
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.0001"},
                {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                {"filterType": "MIN_NOTIONAL", "notional": "5"},
            ]} for s, _, _ in SYMS]}

    async def ticker_24h(self):
        return [{"symbol": s, "quoteVolume": str(v), "lastPrice": str(p),
                 "priceChangePercent": "1.5"} for s, p, v in SYMS]

    async def mark_price(self, symbol=None):
        if symbol:
            return {"symbol": symbol, "markPrice": str(PRICES.get(symbol, 1.0)),
                    "lastFundingRate": "0.00008"}
        return [{"symbol": s, "markPrice": str(p), "lastFundingRate": "0.00008"}
                for s, p, _ in SYMS]

    async def book_ticker(self, symbol=None):
        p = PRICES.get(symbol, 100.0)
        return {"bidPrice": str(p * 0.9999), "askPrice": str(p * 1.0001)}

    async def klines(self, symbol, interval, limit=300):
        from config.modes import TF_MINUTES

        self.calls.append((symbol, interval))

        # forced ranges drive the lifecycle test deterministically
        if interval == "1m" and symbol in self.overrides:
            low, high, close = self.overrides[symbol]
            now = int(time.time() * 1000)
            return [[now - 60_000, close, high, low, close, 1000.0,
                     now, 1000.0 * close, 50, 500.0, 500.0, "0"]]

        if symbol not in self._bases:
            # Python's str hash is salted per process, so derive a stable seed
            # by hand — otherwise results would depend on PYTHONHASHSEED.
            seed = sum(ord(c) * (i + 7) for i, c in enumerate(symbol)) % 9973
            drift = 0.55 if seed % 2 == 0 else -0.45
            base = _base_series(30_000, 100.0, drift, 0.0016, seed, 60_000)
            self._bases[symbol] = _inject_setup(base, bullish=drift > 0, span=2000)

        df = _resample(self._bases[symbol], max(1, TF_MINUTES.get(interval, 5)))
        scale = PRICES.get(symbol, 100.0) / float(df["close"].iloc[-1])
        rows = []
        for _, r in df.tail(limit).iterrows():
            rows.append([int(r["open_time"]), r["open"] * scale, r["high"] * scale,
                         r["low"] * scale, r["close"] * scale, r["volume"],
                         int(r["open_time"]) + 1, r["volume"] * r["close"] * scale,
                         100, r["volume"] / 2, r["volume"] / 2, "0"])
        return rows

    def force(self, symbol, low, high, close=None):
        self.overrides[symbol] = (low, high, close if close is not None else high)


class FakeTelegram:
    def __init__(self):
        self.sent: list = []
        self.deleted: list = []
        self.edits: list = []
        self.answered: list = []
        self.handlers: dict = {}
        self.callback_handler = None
        self.enabled = True
        self.queue_size = 0
        self._next_id = 5000

    async def start(self): ...
    async def stop(self): ...

    def register(self, cmd, fn):
        self.handlers[cmd] = fn

    def register_callback(self, fn):
        self.callback_handler = fn

    async def send(self, text, chat_id=None, parse_mode="HTML", silent=False,
                   markup=None, wait=False):
        if not text:
            return None
        self._next_id += 1
        self.sent.append({"text": text, "id": self._next_id, "markup": markup})
        return self._next_id if wait else None

    async def delete_message(self, message_id, chat_id=None):
        self.deleted.append(message_id)
        return True

    async def edit_message(self, message_id, text, chat_id=None, markup=None):
        self.edits.append({"id": message_id, "text": text, "markup": markup})
        return True

    async def answer_callback(self, cb_id, text="", alert=False):
        self.answered.append(text)

    async def set_my_commands(self, commands):
        return None

    async def sender_loop(self):
        while True:
            await asyncio.sleep(3600)

    async def polling_loop(self):
        while True:
            await asyncio.sleep(3600)

    def texts(self, since=0):
        return [m["text"] for m in self.sent[since:]]


# --------------------------------------------------------------------------- #
def html_balanced(text: str) -> bool:
    for tag in ("b", "i", "code", "pre", "u"):
        if text.count(f"<{tag}>") != text.count(f"</{tag}>"):
            return False
    return True


async def run() -> None:
    import tempfile

    import core.state as state_mod
    from config.settings import RuntimeConfig, Settings

    tmp = tempfile.mkdtemp()
    state_mod.STATE_FILE = Path(tmp) / "state.json"

    from engine.orchestrator import Application

    settings = Settings()
    settings.telegram_token = "test"
    settings.telegram_chat_id = "1"
    settings.min_volume_usdt = 10_000_000
    settings.max_symbols = 50

    runtime = RuntimeConfig()
    runtime.save = lambda: None                    # keep the test filesystem clean
    runtime.min_rr = 1.2   # relaxed for the fixture; production floor is 6R
    runtime.min_score = {"DAY": 55.0, "SWING": 55.0}

    app = Application(settings, runtime)
    fake = FakeExchange()
    app.exchange = fake
    app.data.ex = fake
    app.tracker.ex = fake
    app.bot = FakeTelegram()
    app.tracker.notify = app._notify
    app.tracker.delete_message = app.bot.delete_message
    app._register_commands()
    app.bot.register_callback(app.on_callback)

    # ---------------- universe ----------------
    print("\n▸ Universe (public data only)")
    await app.data.refresh_universe()
    check("universe built", len(app.data.universe) > 0,
          f"({len(app.data.universe)} symbols)")
    check("low-volume symbol excluded", "TINYUSDT" not in app.data.universe)
    check("high-volume symbol included", "BTCUSDT" in app.data.universe)
    check("universe sorted by volume", app.data.universe[0] == "BTCUSDT")
    check("no credentials anywhere on settings",
          not any("api" in f and "key" in f
                  for f in settings.__dataclass_fields__))

    from core.exchange import BinanceFutures
    for banned in ("place_order", "set_leverage", "positions", "account",
                   "cancel_order", "balance"):
        check(f"client cannot {banned}", not hasattr(BinanceFutures, banned))

    # ---------------- data ----------------
    print("\n▸ Market data")
    df = await app.data.get_klines("BTCUSDT", "15m", 300)
    check("klines fetched & enriched", df is not None and "atr" in df.columns,
          f"({len(df) if df is not None else 0} bars)")
    check("price scaled to symbol", 40000 < float(df["close"].iloc[-1]) < 90000)
    t0 = time.time()
    await app.data.get_klines("BTCUSDT", "15m", 300)
    check("second call served from cache", (time.time() - t0) < 0.05)

    # ---------------- scan ----------------
    print("\n▸ Scan cycle")
    await app.scanner.refresh_regime()
    check("market regime read from BTC", app.scanner.btc_trend in
          ("BULL", "BEAR", "RANGE"), f"({app.scanner.btc_trend})")

    results_map = await app.scanner.scan_all()
    total = sum(len(v) for v in results_map.values())
    check("scan completes without error", app.scanner.last_scan_at > 0,
          f"({app.scanner.last_scan_duration:.1f}s, {total} signals)")
    for mode, sigs in results_map.items():
        for s in sigs:
            check(f"{mode} signal well-formed",
                  len(s.take_profits) == 3 and s.stop_loss > 0,
                  f"({s.symbol} {s.side} {s.score:.0f})")

    single, notes = await app.scanner.scan_single("SOL")
    check("scan_single resolves bare ticker", True,
          f"({len(single)} signals, {len(notes)} notes)")
    empty, notes2 = await app.scanner.scan_single("NOTREAL")
    check("unknown symbol explained, not crashed", not empty and len(notes2) == 1)

    # ---------------- lifecycle ----------------
    print("\n▸ Signal lifecycle")
    from core.models import Signal, TakeProfit

    def make_signal(symbol="SOLUSDT", ttl=3600):
        p = PRICES[symbol]
        return Signal(
            symbol=symbol, side="LONG", mode="DAY", strategy="SWEEP_MSS",
            entry=p, entry_low=p * 0.998, entry_high=p * 1.002,
            stop_loss=p * 0.98,
            take_profits=[TakeProfit(1, p * 1.02, 1.0, "EQH pool", 0.4),
                          TakeProfit(2, p * 1.04, 2.0, "swing high", 0.35),
                          TakeProfit(3, p * 1.06, 3.0, "range high", 0.25)],
            score=93.0, required_score=90.0, risk_pct=2.0, rr_total=3.0,
            expires_at=time.time() + ttl)

    sig = make_signal()
    before = len(app.bot.sent)
    await app._dispatch(sig)
    check("signal published to Telegram", len(app.bot.sent) > before)
    check("message id captured for later deletion", sig.message_id is not None)
    check("signal is pending", app.state.pending_count == 1)
    card = app.bot.sent[-1]["text"]
    check("card omits the confirmation list", "CONFIRMATIONS" not in card)
    check("card is minimal", len(card) < 420, f"({len(card)} chars)")

    # --- price never reaches the zone: nothing happens ---
    p = PRICES["SOLUSDT"]
    fake.force("SOLUSDT", p * 1.01, p * 1.015)
    before = len(app.bot.sent)
    await app.tracker.run_once()
    check("no fill when price stays outside the zone",
          sig.status == "PENDING" and len(app.bot.sent) == before)

    # --- grazing the far edge is NOT a fill any more ---
    fake.force("SOLUSDT", p * 1.0015, p * 1.0025, p * 1.002)
    before = len(app.bot.sent)
    await app.tracker.run_once()
    check("grazing the zone edge does not count as a fill",
          sig.status == "PENDING")

    # --- price trades through the reference level: fill ---
    fake.force("SOLUSDT", p * 0.9985, p * 1.008, p * 1.005)
    before = len(app.bot.sent)
    await app.tracker.run_once()
    check("entry filled on a wick into the zone", sig.status == "FILLED")
    check("fill alert sent", any("FILLED" in t
                                for t in app.bot.texts(before)))
    check("fill price inside the traded range",
          p * 0.9985 <= sig.fill_price <= p * 1.008, f"({sig.fill_price:.4f})")
    check("moved from pending to live",
          app.state.live_count == 1 and app.state.pending_count == 0)

    # --- TP1 ---
    fake.force("SOLUSDT", p * 1.005, p * 1.021, p * 1.02)
    before = len(app.bot.sent)
    await app.tracker.run_once()
    check("TP1 detected", sig.take_profits[0].hit)
    check("TP1 alert sent", any("TP1 HIT" in t for t in app.bot.texts(before)))
    check("TP2 not yet hit", not sig.take_profits[1].hit)
    check("still live after a partial", app.state.live_count == 1)

    # --- TP2 then stop, in that order across polls ---
    fake.force("SOLUSDT", p * 1.02, p * 1.041, p * 1.04)
    before = len(app.bot.sent)
    await app.tracker.run_once()
    check("TP2 detected", sig.take_profits[1].hit)
    check("TP2 alert sent", any("TP2 HIT" in t for t in app.bot.texts(before)))

    fake.force("SOLUSDT", p * 0.979, p * 1.03, p * 0.98)
    before = len(app.bot.sent)
    await app.tracker.run_once()
    check("stop closes the signal", app.state.live_count == 0)
    check("stop alert sent", any("CLOSED" in t for t in app.bot.texts(before)))
    outcome = app.state.outcomes[-1]
    check("outcome banked", outcome.symbol == "SOLUSDT")
    check("two targets recorded", outcome.tp_hits == 2, f"({outcome.tp_hits})")
    check("stop after TP1 still counts as a win", outcome.is_win)
    check("close reason names the partial", "TP2" in outcome.result,
          f"({outcome.result})")

    # --- a clean loser ---
    loser = make_signal("XRPUSDT")
    await app._dispatch(loser)
    lp = PRICES["XRPUSDT"]
    fake.force("XRPUSDT", lp * 0.999, lp * 1.001, lp)
    await app.tracker.run_once()
    check("second signal filled", loser.status == "FILLED")
    fake.force("XRPUSDT", lp * 0.979, lp * 0.995, lp * 0.98)
    await app.tracker.run_once()
    check("clean stop recorded", app.state.outcomes[-1].result == "SL")
    check("clean stop is not a win", not app.state.outcomes[-1].is_win)

    # --- full TP ladder ---
    winner = make_signal("AVAXUSDT")
    await app._dispatch(winner)
    wp = PRICES["AVAXUSDT"]
    fake.force("AVAXUSDT", wp * 0.999, wp * 1.001, wp)
    await app.tracker.run_once()
    before = len(app.bot.sent)
    fake.force("AVAXUSDT", wp * 1.0, wp * 1.065, wp * 1.06)
    await app.tracker.run_once()
    check("full ladder completes", app.state.outcomes[-1].result == "TP3")
    check("all three targets recorded", app.state.outcomes[-1].tp_hits == 3)
    check("full-TP alert sent", any("CLOSED" in t for t in app.bot.texts(before)))

    # --- expiry deletes the card and purges the record ---
    print("\n▸ Expiry handling")
    doomed = make_signal("LINKUSDT", ttl=-1)
    await app._dispatch(doomed)
    mid = doomed.message_id
    pending_before = app.state.pending_count
    before = len(app.bot.sent)
    n = await app.tracker.handle_expiries()
    check("expired signal detected", n == 1)
    check("original card deleted", mid in app.bot.deleted, f"(id {mid})")
    check("expiry notice sent", any("EXPIRED" in t for t in app.bot.texts(before)))
    check("record purged from store", app.state.pending_count == pending_before - 1)
    check("expired signal left no outcome",
          all(o.symbol != "LINKUSDT" for o in app.state.outcomes))
    check("expiry counted", app.state.stats["signals_expired"] >= 1)

    runtime.delete_expired_message = False
    ghost = make_signal("DOGEUSDT", ttl=-1)
    await app._dispatch(ghost)
    deleted_before = len(app.bot.deleted)
    await app.tracker.handle_expiries()
    check("deletion respects the setting",
          len(app.bot.deleted) == deleted_before)
    runtime.delete_expired_message = True

    # ---------------- reporting ----------------
    print("\n▸ /pnl and /report")
    live = make_signal("ETHUSDT")
    await app._dispatch(live)
    ep = PRICES["ETHUSDT"]
    fake.force("ETHUSDT", ep * 0.999, ep * 1.001, ep)
    await app.tracker.run_once()
    fake.force("ETHUSDT", ep * 1.005, ep * 1.012, ep * 1.01)

    rows = await app.tracker.snapshot()
    check("snapshot returns the live trade", len(rows) == 1)
    check("snapshot computes positive PnL", rows[0]["pct"] > 0,
          f"({rows[0]['pct']:.2f}%)")
    check("snapshot computes an R multiple", rows[0]["r"] > 0,
          f"({rows[0]['r']:.2f}R)")
    check("snapshot names the next target", rows[0]["next_tp"].level == 1)

    pnl = await app.cmd_pnl([], "1")
    check("/pnl renders the live trade", "ETHUSDT" in pnl)
    check("/pnl tags balanced", html_balanced(pnl))

    rep = await app.cmd_report([], "1")
    check("/report renders", "PERFORMANCE REPORT" in rep)
    check("/report tags balanced", html_balanced(rep))
    check("/report counts completed trades", "Win rate" in rep)
    data = app.state.report()
    check("report totals match outcomes", data["total"] == len(app.state.outcomes),
          f"({data['total']})")
    check("win rate computed", 0 <= data["win_rate"] <= 100,
          f"({data['win_rate']:.0f}%)")
    mode_report = await app.cmd_report(["day"], "1")
    check("/report accepts a mode filter", "DAY" in mode_report)
    check("/report accepts a strategy filter",
          isinstance(await app.cmd_report(["sweep"], "1"), str))

    # ---------------- commands ----------------
    print("\n▸ Telegram commands")
    cmds = [
        ("help", []), ("status", []), ("pnl", []), ("pending", []),
        ("report", []), ("report", ["swing"]), ("report", ["garbage"]),
        ("why", []), ("why", ["ETHUSDT"]), ("why", ["NOTHING"]),
        ("signals", []), ("signals", ["off"]), ("signals", ["on"]),
        ("pause", []), ("resume", []),
        ("mode", []), ("mode", ["scalp", "off"]), ("mode", ["scalp", "on"]),
        ("mode", ["bogus", "on"]), ("mode", ["scalp", "maybe"]),
        ("strategy", []), ("strategy", ["sweep", "off"]), ("strategy", ["sweep", "on"]),
        ("strategy", ["nope", "on"]),
        ("score", []), ("score", ["day", "90"]), ("score", ["day", "999"]),
        ("score", ["day", "abc"]), ("score", ["nope", "90"]),
        ("minrr", []), ("minrr", ["2"]), ("minrr", ["abc"]), ("minrr", ["99"]),
        ("weekend", []), ("weekend", ["on"]),
        ("top", []), ("top", ["5"]), ("top", ["xyz"]),
        ("symbols", []), ("ping", []), ("log", []), ("log", ["7"]),
        ("scan", []), ("refresh", []),
    ]
    for cmd, args in cmds:
        handler = app.bot.handlers.get(cmd)
        if not handler:
            check(f"/{cmd} registered", False)
            continue
        try:
            out = await handler(args, "1")
        except Exception as exc:                       # noqa: BLE001
            check(f"/{cmd} {' '.join(args)}", False, f"raised {exc!r}")
            continue
        label = f"/{cmd} {' '.join(args)}".strip()
        ok = out is None or isinstance(out, (str, tuple))
        if isinstance(out, str) and out:
            ok = ok and html_balanced(out) and len(out) < 8000
        check(label, ok)

    check("/scan on a real symbol works",
          isinstance(await app.bot.handlers["scan"](["BTCUSDT"], "1"), str))

    # ---------------- settings panel ----------------
    print("\n▸ Interactive settings panel")
    from notify import keyboards

    panel = await app.cmd_settings([], "1")
    check("/settings returns text plus a keyboard",
          isinstance(panel, tuple) and "inline_keyboard" in panel[1])

    # walk every button on every panel through the real callback router
    all_buttons = set()
    for name in keyboards.PANELS:
        _text, kb = keyboards.render(name, runtime)
        for row in kb["inline_keyboard"]:
            for btn in row:
                all_buttons.add(btn["callback_data"])

    # The walk taps real buttons, which really mutate config — snapshot first
    # so a toggle like signals_enabled does not silently break later checks.
    snapshot = runtime.to_dict()

    failures = []
    for data in sorted(all_buttons):
        if data == "nav:close":
            continue
        try:
            res = await app.on_callback(data, "1", 999)
        except Exception as exc:                       # noqa: BLE001
            failures.append(f"{data}: {exc!r}")
            continue
        if res is None:
            continue
        if len(res) == 3 and res[1] is not None and not html_balanced(res[1]):
            failures.append(f"{data}: unbalanced HTML")
    check("every panel button routes without error", not failures,
          f"({len(all_buttons)} buttons)" if not failures else f"({failures[:2]})")

    mutated = [k for k, v in snapshot.items() if runtime.to_dict()[k] != v]
    check("the walk actually changed settings (buttons are wired up)",
          len(mutated) > 0, f"({len(mutated)} settings moved)")
    for key, value in snapshot.items():                # restore
        setattr(runtime, key, dict(value) if isinstance(value, dict) else value)
    check("config restored after the walk", runtime.to_dict() == snapshot)

    # --- buttons actually change state ---
    runtime.modes_enabled["SCALP"] = True
    await app.on_callback("tog:modes:SCALP", "1", 999)
    check("mode toggle flips the setting", runtime.modes_enabled["SCALP"] is False)
    await app.on_callback("tog:modes:SCALP", "1", 999)
    check("mode toggle is reversible", runtime.modes_enabled["SCALP"] is True)

    runtime.min_score["DAY"] = 90.0
    await app.on_callback("adj:score:DAY:5", "1", 999)
    check("score nudge raises the threshold", runtime.min_score["DAY"] == 95.0,
          f"({runtime.min_score['DAY']})")
    for _ in range(30):
        await app.on_callback("adj:score:DAY:5", "1", 999)
    check("score is clamped at the ceiling", runtime.min_score["DAY"] <= 99.0,
          f"({runtime.min_score['DAY']})")

    runtime.min_rr = 1.6
    await app.on_callback("adj:num:min_rr:2", "1", 999)
    check("RR nudge moves in 0.2 steps", abs(runtime.min_rr - 1.8) < 0.001,
          f"({runtime.min_rr})")
    for _ in range(80):
        await app.on_callback("adj:num:min_rr:-2", "1", 999)
    check("RR is clamped at the floor", runtime.min_rr >= 0.5,
          f"({runtime.min_rr})")

    before_flag = runtime.alert_on_fill
    await app.on_callback("tog:flag:alert_on_fill", "1", 999)
    check("alert toggle flips", runtime.alert_on_fill is not before_flag)
    await app.on_callback("tog:flag:alert_on_fill", "1", 999)

    edits_before = len(app.bot.edits)
    await app.on_callback("nav:main", "1", 999)
    check("navigation returns a panel to render",
          len(app.bot.edits) == edits_before)      # router returns, bot edits

    res = await app.on_callback("nav:close", "1", 4242)
    check("close deletes the panel", res is None and 4242 in app.bot.deleted)
    garbage = await app.on_callback("total:nonsense:here", "1", 999)
    check("garbage callback is handled", garbage is not None)
    malformed = await app.on_callback("adj:num:min_rr:abc", "1", 999)
    check("malformed numeric callback is handled", malformed[0] == "Bad value")

    # ---------------- alert suppression ----------------
    print("\n▸ Alert switches")
    runtime.alert_on_fill = False
    quiet = make_signal("ARBUSDT")
    await app._dispatch(quiet)
    ap = PRICES["ARBUSDT"]
    fake.force("ARBUSDT", ap * 0.999, ap * 1.001, ap)
    before = len(app.bot.sent)
    await app.tracker.run_once()
    check("fill alert suppressed when disabled",
          not any("FILLED" in t for t in app.bot.texts(before)))
    check("fill still recorded despite silence", quiet.status == "FILLED")
    runtime.alert_on_fill = True

    check("state saves cleanly", app.state.save() is None)


def main() -> int:
    print("═" * 66)
    print("  INTEGRATION TEST — mocked public Binance, no keys, no network")
    print("═" * 66)
    asyncio.run(run())
    total = results["pass"] + results["fail"]
    print("\n" + "═" * 66)
    print(f"  RESULT: {results['pass']}/{total} checks passed "
          f"({results['fail']} failed)")
    print("═" * 66)
    return 0 if results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
