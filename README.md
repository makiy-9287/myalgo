# Binance Futures Signal Engine

Multi-timeframe signal generation for Binance USDⓈ-M Futures, controlled
entirely from Telegram.

Two independent strategies scan every USDT perpetual with more than $10M of
24h volume, once a minute. Signals fire only when the full timeframe stack
agrees and the setup scores **90 or more out of 100**. Every published signal
is then tracked live: you get an alert when the entry fills, on every target,
and on the stop.

**No API keys. No orders. Nothing to liquidate.** This build reads public
market data and sends messages. The code that could place a trade does not
exist in the project — `place_order`, `set_leverage`, `positions` and
`balance` are not methods on the client, and the test suite asserts they never
come back.

---

## Install

Requires Python 3.10+.

```bash
pip install -r requirements.txt
cp .env.example .env
```

You need exactly two values in `.env`:

| Setting | Where to get it |
|---|---|
| `TELEGRAM_BOT_TOKEN` | [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | [@userinfobot](https://t.me/userinfobot) |

Then:

```bash
python main.py
```

Preflight refuses to start on a broken config rather than half-running. You
should get a Telegram message within a few seconds — send `/settings`.

Everything else (modes, strategies, score thresholds, RR, alerts, weekend
rules) is adjusted from Telegram at runtime and persists in
`data/runtime_config.json`. You never edit it by hand.

---

## The signal lifecycle

This is the part that matters, so it is worth being precise about.

```
PUBLISHED ──price enters the zone──▶ FILLED ──▶ TP1 ──▶ TP2 ──▶ TP3
    │                                   │
    │                                   └──────────────▶ STOPPED
    │
    └──validity window ends──▶ EXPIRED (card deleted, record purged)
```

**Fills are detected from bar highs and lows, not closes.** Price routinely
wicks into an entry zone and closes outside it, and that wick is a real fill.
A close-only check would miss most entries and quietly under-report
performance.

**When one bar contains both a target and the stop, the stop wins.** From
1-minute OHLC there is no way to know which came first. Assuming the good
outcome would inflate the win rate, so the engine reports the pessimistic one.

**Expired signals are deleted, not archived.** The original Telegram card is
removed from the chat and the record is dropped entirely — no outcome, no
history line. An entry that never triggered is not a trade, and keeping it
would pollute the statistics. Only filled signals are ever written to disk.

---

## The two modes

Each mode scans independently with its own timeframe stack and expiry.

| | Bias | Structure | Setup | Sniper entry | Entry valid |
|---|---|---|---|---|---|
| **Day** | 4H | 1H | 15m | 5m | 8 hours |
| **Swing** | 1D | 4H | 1H | 15m | 24 hours |

Expiries are longer than you might expect, deliberately. The entry zone is
narrow, so price needs time to come back to it — expiring sooner would throw
away setups that were merely early.

Deep analysis runs **every 5 minutes**. The coin list (>$10M 24h volume)
rebuilds **every 3 hours**.

---

## The two strategies

Both hunt the same thing — the places institutions have to transact — from
opposite sides of the move.

### 1. Liquidity Sweep + MSS  (reversal)

Large participants cannot fill size where retail can. To buy they need
sellers; to sell they need buyers. That counterparty sits in the most obvious
places on the chart: stops resting under equal lows and above equal highs, and
the breakout orders that trigger there. Price is driven *into* those pools,
the stops are absorbed, and the real move starts the other way.

1. Pool mapped — equal highs/lows, session or swing extremes
2. **Inducement** — a minor pool taken first, supplying the initial fills
3. **The sweep** — the pool is raided and price closes back inside
4. **MSS** — a displacement *leg* closes beyond the last opposing swing
5. POI — the FVG or order block that displacement left behind
6. Sniper trigger on the entry timeframe

Steps 1, 3, 4, 5 and 6 are mandatory. Step 4 is what separates this from
"price bounced off support": without displacement through structure, a sweep
is just a deeper pullback.

The engine evaluates *every* sweep candidate in the window, not merely the
loudest one, and takes the strongest that both produced a structure shift and
survives the higher-timeframe test.

### 2. Order Block + FVG  (continuation)

When size moves a market it moves it inefficiently. Orders that could not be
filled during the impulse stay resting at its origin — the order block — and
the impulse skips price levels entirely, leaving a gap never traded fairly —
the fair value gap. Price returns there because unfilled business is still
there.

1. Established HTF trend
2. **BOS** in that direction — continuation confirmed, not hoped for
3. **Order block** at the origin of the impulse, unmitigated
4. **FVG** inside that same impulse leg
5. Retracement into the confluence
6. Sniper trigger

The A+ version is when the order block and the FVG overlap: one narrow pocket
holding both the unfilled orders and the untraded prices.

---

## Scoring: a strict 100-point budget

Each strategy allocates exactly 100 points, weighted so the links that decide
the trade carry the most. The default threshold of 90 means *almost everything
must be present*.

**Liquidity Sweep + MSS**

| Confirmation | Points | |
|---|---|---|
| Liquidity sweep | 16 | mandatory |
| Market structure shift | 15 | mandatory |
| POI from the displacement | 12 | mandatory |
| HTF context | 10 | mandatory |
| Sniper trigger | 9 | mandatory |
| Target liquidity | 7 | |
| Premium/discount location | 6 | |
| Sweep volume | 6 | |
| Inducement taken | 5 | |
| RSI / delta divergence | 5 | |
| HTF zone confluence | 4 | |
| Session · BTC regime | 2 each | |
| Volatility regime | 1 | |

**Order Block + FVG**

| Confirmation | Points | |
|---|---|---|
| Order block | 15 | mandatory |
| Fair value gap | 13 | mandatory |
| BOS continuation | 13 | mandatory |
| HTF trend | 11 | mandatory |
| Sniper trigger | 9 | mandatory |
| Zone confluence (OB∩FVG) | 7 | |
| Target liquidity | 7 | |
| Fib confluence | 6 | |
| Pullback quality | 6 | |
| Inducement taken | 5 | |
| Not overextended | 4 | |
| Session | 2 | |
| BTC regime · Volatility | 1 each | |

Both require **9 confirmations minimum** (12 inside the strict window). The
budgets live in one table the test suite asserts sums to exactly 100.

---

## Entry zones and targets

### The zone is narrow on purpose

A wide zone quietly destroys the trade. Published as a short at
`201.89 – 200.28` with TP1 at `199.50`, a fill recorded the moment price
grazes `200.28` leaves 0.78 of upside against a stop measured from `201.89`.
The reward is handed back before the trade starts, and every statistic
downstream is flattered by an entry nobody could have got.

Two things are enforced:

1. The zone is clamped to a fraction of an ATR around the POI, and further
   shrunk if it exceeds ~18–20% of the distance to TP1.
2. **A fill requires price to trade through the reference level at the middle
   of the zone**, not to touch its near edge. The level has to be respected,
   not brushed.

### Targets are built from the reward floors

The ladder is constructed *from* its minimums rather than filtered afterwards:
TP1 ≥ 1.5R, TP2 ≥ 3R, TP3 ≥ 6R. For each level the builder takes the nearest
untapped liquidity pool or major supply/demand zone at or beyond that floor,
and only falls back to a measured extension when nothing real sits there —
saying so on the card when it does.

Picking a target simply because it was the next pool along is exactly how TP1
ends up close enough that a setup "wins" and still gives everything back.
`/report` tracks that failure directly.

---

## Telegram

### Interactive panel

`/settings` opens an inline-keyboard control panel that edits itself in place
rather than spamming the chat:

```
⚙️ CONTROL PANEL
[⚡ Modes]        [🧠 Strategies]
[🎯 Score & RR]   [🔔 Alerts]
[🗓 Weekend]      [🔧 Engine]
[🔄 Refresh]      [✖️ Close]
```

Every runtime setting is reachable by tapping — nothing has to be typed on a
phone. Changes save instantly and survive a restart.

### Commands

**Monitoring**
```
/status    engine state and health
/pnl       live PnL of active trades
/pending   signals awaiting entry
/report    win rate and TP/SL breakdown
/why <SYM> the confirmations behind a signal
```

**Analysis**
```
/scan <SYMBOL>   force a full MTF scan and explain the verdict
/top [n]         best current candidates
/symbols         active universe
/refresh         rebuild the universe now
```

**Settings**
```
/settings                     interactive panel (recommended)
/mode scalp|day|swing on|off
/strategy sweep|obfvg on|off
/score <mode> <value>
/minrr <value>
/weekend on|off      strict Fri 23:00-Mon 20:00 window
/signals on|off
/pause  /resume
```

**System**
```
/ping  /log [n]  /help
```

`/why` exists because the confirmation list was removed from the signal card —
a fifteen-line rationale buries the numbers that matter. The reasoning is
still computed and scored, just printed on demand.

---

## What `/report` measures

```
📋 PERFORMANCE REPORT

Win rate  ██████░░░░  60.0%
12W / 8L across 20 completed

━━━ TARGET BREAKDOWN ━━━
🎯 TP1 reached    12  (60.0%)
🎯 TP2 reached     7  (35.0%)
🏁 Full TP (TP3)   4  (20.0%)
🛑 Stopped out    12  (60.0%)
     └ 8 without reaching any target
```

**A trade counts as a win if it reached TP1 before its stop** — your
definition. A signal that banks TP1 and then reverses to a stop is still a win,
because the entry was right; that is what a signal engine is judged on. The
report separates those from clean stops that never reached a target, so you can
see both numbers.

`/report day` or `/report sweep` filters by mode or strategy.

---

## The strict window: Friday 23:00 → Monday 20:00 UTC

This brackets the traditional market close and reopen. Crypto never stops, but
the capital that moves it does: desks flatten into the weekend, books thin
out, and the same liquidity pools that took a week to build get raided in
hours by size that no longer has anything to trade against. Breaks fail,
structure lies, and a level that held all week gives way on a Sunday for no
reason that survives Monday.

Signals are not blocked in that window — they are made to work much harder:

- Required score rises (+5 by default, capped at 98)
- Three additional confirmations required
- Confirmations must span **all four** categories: volume, bias, liquidity, momentum
- Prefilter demands ADX ≥ 22
- Stop buffer widened 30%

---

## Layout

```
main.py                  entry point, preflight, signal handlers
config/settings.py       .env settings + Telegram-mutable runtime config
config/modes.py          mode specs, timeframe TTLs
core/exchange.py         public-only async REST client (no keys, no signing)
core/indicators.py       indicator library (pandas/numpy only)
core/structure.py        swings, BOS/CHoCH, liquidity pools, sweeps, OB, FVG
core/data_manager.py     universe build, filters, OHLCV cache
core/models.py           Signal lifecycle + Outcome
core/state.py            pending/live/outcomes, atomic persistence
strategies/scoring.py    the two 100-point budgets
strategies/sweep_mss.py  liquidity sweep + market structure shift
strategies/ob_fvg.py     order block + fair value gap
strategies/base.py       MTF context, shared trigger, narrow-zone builder
engine/signal_builder.py SL placement, TPs at liquidity, RR validation
engine/scanner.py        two-stage scan, market regime, weekend rules
engine/tracker.py        fills, targets, stops, expiries
engine/orchestrator.py   four loops, commands, panel callbacks
notify/keyboards.py      inline keyboard panels
notify/                  HTML-safe formatting + raw Bot API client
tests/                   offline test suites
```

---

## Tests

Both run offline. No credentials, no network.

```bash
python tests/test_pipeline.py      # 678 checks
python tests/test_integration.py   # 129 checks
```

The integration suite injects a fake public Binance and a fake Telegram, then
drives a complete lifecycle — published → filled on a wick → TP1 → TP2 →
stopped — asserting the right notification fires at each step. It also walks
**every button on every settings panel** through the real callback router,
verifies each mutates config, then restores the snapshot.

Bugs found this way across both builds, all fixed:

1. BOS/CHoCH detection only examined the last two bars
2. Liquidity pools mapped over only 120 bars, hiding older stop clusters
3. Index/column name collision crashing every kline fetch
4. `RuntimeConfig.to_dict()` failing on a thread lock
5. A stale `get_trade` call left over from the removed execution layer
6. MSS measured displacement on a single candle instead of the leg, so real
   structure shifts were missed almost every time
7. The sniper trigger compared the trigger candle to the *setup* timeframe's
   ATR — demanding a 15m bar four times normal size, which silently
   suppressed nearly every entry
8. Major supply/demand zones treated any wick as mitigation, discarding every
   level that had ever been tested and rejected

---

## Operating notes

- At a 90 threshold, quiet periods are normal and expected. If you are seeing
  nothing for hours, that is the filter working, not a fault. Use `/scan` to
  confirm the engine is reading charts correctly.
- If `/report` shows a healthy win rate but poor R, your losers are bigger
  than your winners — raise `/minrr`.
- If almost everything expires unfilled, your entry zones are too tight for
  current volatility, or you are signalling into moves that have already gone.
  Check the fill rate line at the bottom of `/report`.
- `/pause` stops scanning but keeps tracking anything already live, so you
  never lose sight of an open trade.

This is software, not financial advice. Signals are opinions produced by
pattern-matching code, and no threshold makes them reliable.
