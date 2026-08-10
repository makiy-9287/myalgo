"""
Telegram message construction.

Everything is HTML (not MarkdownV2) because HTML has exactly four special
characters and one escape function, whereas MarkdownV2 has eighteen and a
context-dependent escaping rule that breaks on the first price containing a
dot. Every interpolated value goes through esc().

The signal card deliberately omits the confirmation list. The engine still
computes and scores all of them — they are just not printed, because a
fifteen-line rationale buries the numbers that actually matter. /why <SYMBOL>
prints them on demand.
"""
from __future__ import annotations

import html
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from core.models import Outcome, Signal, TakeProfit

MAX_LEN = 4000

SIDE_EMOJI = {"LONG": "🟢", "SHORT": "🔴"}
MODE_EMOJI = {"DAY": "•", "SWING": "•"}
STRATEGY_LABEL = {
    "SWEEP_MSS": "Liquidity Sweep + MSS",
    "OB_FVG": "Order Block + FVG",
}


# --------------------------------------------------------------------------- #
def esc(text) -> str:
    return html.escape(str(text), quote=False)


def bar(value: float, maximum: float = 100.0, width: int = 10) -> str:
    filled = int(round(max(0.0, min(1.0, value / maximum)) * width))
    return "█" * filled + "░" * (width - filled)


def split_message(text: str, limit: int = MAX_LEN) -> List[str]:
    """Split on line boundaries so an HTML tag is never cut in half."""
    if len(text) <= limit:
        return [text]
    chunks, current = [], ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > limit:
            if current:
                chunks.append(current.rstrip())
            # a single line longer than the limit is hard-split as a last resort
            while len(line) > limit:
                chunks.append(line[:limit])
                line = line[limit:]
            current = line + "\n"
        else:
            current += line + "\n"
    if current.strip():
        chunks.append(current.rstrip())
    return chunks


def _fmt(price: float) -> str:
    """Price formatting that survives both BTC and shitcoins."""
    p = abs(price)
    if p >= 1000:
        return f"{price:,.2f}"
    if p >= 10:
        return f"{price:.3f}"
    if p >= 0.1:
        return f"{price:.4f}"
    if p >= 0.001:
        return f"{price:.6f}"
    return f"{price:.8f}"


def _dur(minutes: float) -> str:
    minutes = max(0, int(minutes))
    if minutes < 60:
        return f"{minutes}m"
    h, m = divmod(minutes, 60)
    if h < 24:
        return f"{h}h {m:02d}m" if m else f"{h}h"
    d, h = divmod(h, 24)
    return f"{d}d {h}h" if h else f"{d}d"


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M UTC")


# --------------------------------------------------------------------------- #
# the signal card
# --------------------------------------------------------------------------- #
def format_signal(sig: Signal, tick: Optional[float] = None) -> str:
    """
    The signal card, stripped to what a trader acts on.

    Coin, direction, entry, targets, stop, mode, strategy. Nothing else. The
    confirmations, timeframe reads and volume statistics all still exist and
    are still scored — they live behind /why, where they can be audited
    without being in the way every single time a setup fires.
    """
    arrow = "LONG" if sig.is_long else "SHORT"
    strat = STRATEGY_LABEL.get(sig.strategy, sig.strategy)

    lines = [
        f"<b>{esc(sig.symbol)}</b>  ·  <b>{arrow}</b>",
        f"{esc(sig.mode)} · {esc(strat)}",
        "",
        f"Entry   <code>{_fmt(sig.entry_low)} – {_fmt(sig.entry_high)}</code>",
    ]
    for tp in sig.take_profits:
        lines.append(f"TP{tp.level}     <code>{_fmt(tp.price)}</code>"
                     f"   <i>{tp.rr:.1f}R</i>")
    lines.append(f"SL      <code>{_fmt(sig.stop_loss)}</code>"
                 f"   <i>{sig.risk_pct:.2f}%</i>")
    return "\n".join(lines)


def format_fill(sig: Signal, fill: float, price: float) -> str:
    """Entry filled — same restraint as the signal card."""
    arrow = "LONG" if sig.is_long else "SHORT"
    tp1 = sig.take_profits[0] if sig.take_profits else None
    lines = [
        f"<b>FILLED</b>  ·  <b>{esc(sig.symbol)}</b>  {arrow}",
        f"{esc(sig.mode)} · {esc(STRATEGY_LABEL.get(sig.strategy, sig.strategy))}",
        "",
        f"Entry   <code>{_fmt(fill)}</code>",
    ]
    if tp1:
        lines.append(f"TP1     <code>{_fmt(tp1.price)}</code>   <i>{tp1.rr:.1f}R</i>")
    lines.append(f"SL      <code>{_fmt(sig.stop_loss)}</code>"
                 f"   <i>{sig.risk_pct:.2f}%</i>")
    return "\n".join(lines)


def format_expired(sig: Signal) -> str:
    """Expired — one line, because this fires often."""
    arrow = "LONG" if sig.is_long else "SHORT"
    return (f"<b>EXPIRED</b>  ·  {esc(sig.symbol)}  {arrow}  "
            f"<i>({esc(sig.mode)})</i>\n"
            f"Entry never reached.")


def format_tp_hit(sig: Signal, tp: TakeProfit, price: float) -> str:
    entry = sig.fill_price or sig.entry
    pct = abs(tp.price - entry) / entry * 100 if entry else 0
    remaining = [t for t in sig.take_profits if not t.hit]
    held = sum(t.allocation for t in remaining) * 100

    lines = [
        f"🎯 <b>TP{tp.level} HIT</b> · {esc(sig.side)} <b>{esc(sig.symbol)}</b>",
        "",
        f"💰 <code>{_fmt(tp.price)}</code>  <b>+{pct:.2f}%</b>  ({tp.rr:.2f}R)",
        f"📤 Suggested close: <b>{tp.allocation * 100:.0f}%</b> of the position",
    ]
    if remaining:
        nxt = remaining[0]
        lines.append(f"🎯 Next: <b>TP{nxt.level}</b> at <code>{_fmt(nxt.price)}</code> "
                     f"({nxt.rr:.2f}R) · <b>{held:.0f}%</b> still running")
        if tp.level == 1:
            lines.append("🛡 <i>Consider moving the stop to breakeven.</i>")
    else:
        lines.append("🏁 <b>Full target ladder complete.</b>")
    lines.append("")
    lines.append(f"<code>ID {esc(sig.signal_id)}</code> · {_dur((time.time() - sig.filled_at) / 60)} in trade")
    return "\n".join(lines)


def format_closed(sig: Signal, outcome: Outcome) -> str:
    won = outcome.is_win
    head = "🏆 <b>CLOSED — WIN</b>" if won else "🛑 <b>CLOSED — LOSS</b>"
    sign = "+" if outcome.pct >= 0 else ""
    duration = (outcome.closed_at - outcome.filled_at) / 60 if outcome.filled_at else 0

    lines = [
        f"{head} · {esc(sig.side)} <b>{esc(sig.symbol)}</b>",
        f"<i>{esc(sig.mode)} · {esc(STRATEGY_LABEL.get(sig.strategy, sig.strategy))}</i>",
        "",
        f"📥 Entry <code>{_fmt(outcome.fill_price)}</code>",
        f"📤 Exit  <code>{_fmt(outcome.exit_price)}</code>  ({esc(outcome.result)})",
        f"📊 <b>{sign}{outcome.pct:.2f}%</b> · <b>{outcome.r_multiple:+.2f}R</b>",
        f"🎯 Targets reached: <b>{outcome.tp_hits}/{len(sig.take_profits)}</b>",
        f"⏱ Duration: <b>{_dur(duration)}</b>",
        "",
        f"<code>ID {esc(sig.signal_id)}</code>",
    ]
    return "\n".join(lines)


def format_pnl(rows: List[Dict]) -> str:
    if not rows:
        return ("💤 <b>NO ACTIVE TRADES</b>\n\n"
                "Nothing has filled yet. Use /pending to see signals still "
                "waiting for their entry zone.")

    total_r = sum(r["r"] for r in rows)
    winners = sum(1 for r in rows if r["r"] > 0)

    lines = [
        f"📊 <b>ACTIVE TRADES</b> ({len(rows)})",
        f"<i>Live progress from entry toward target or stop</i>",
        "",
    ]
    for r in rows:
        sig: Signal = r["signal"]
        emoji = SIDE_EMOJI.get(sig.side, "⚪️")
        trend = "🟩" if r["r"] > 0 else ("🟥" if r["r"] < 0 else "⬜️")
        sign = "+" if r["pct"] >= 0 else ""

        lines.append(f"{trend} {emoji} <b>{esc(sig.symbol)}</b> "
                     f"<i>{esc(sig.mode)}</i> · <b>{sign}{r['pct']:.2f}%</b> "
                     f"({r['r']:+.2f}R)")
        lines.append(f"     entry <code>{_fmt(sig.fill_price or sig.entry)}</code> "
                     f"→ now <code>{_fmt(r['price'])}</code>")

        nxt: Optional[TakeProfit] = r["next_tp"]
        if nxt:
            prog = max(0.0, min(100.0, r["to_target"]))
            lines.append(f"     <code>{bar(prog, width=8)}</code> "
                         f"{prog:.0f}% to TP{nxt.level} "
                         f"<code>{_fmt(nxt.price)}</code>")
        lines.append(f"     🛑 <code>{_fmt(sig.stop_loss)}</code> · "
                     f"🎯 {r['tp_hits']}/{len(sig.take_profits)} hit · "
                     f"⏱ {_dur(r['age_min'])}")
        lines.append("")

    lines.append("<b>━━━━━━━━━━━━━━━━━━━━</b>")
    lines.append(f"📈 Aggregate: <b>{total_r:+.2f}R</b> across {len(rows)} "
                 f"trade(s) · {winners} in profit")
    return "\n".join(lines)


def format_pending(sigs: List[Signal]) -> str:
    if not sigs:
        return "💤 <b>NO PENDING SIGNALS</b>\n\nNothing is waiting for an entry."
    lines = [f"⏳ <b>PENDING ENTRIES</b> ({len(sigs)})", ""]
    for sig in sigs:
        emoji = SIDE_EMOJI.get(sig.side, "⚪️")
        left = (sig.expires_at - time.time()) / 60
        lines.append(f"{emoji} <b>{esc(sig.symbol)}</b> {esc(sig.side)} "
                     f"<i>{esc(sig.mode)}</i> · score {sig.score:.0f}")
        lines.append(f"     zone <code>{_fmt(sig.entry_low)}</code>–"
                     f"<code>{_fmt(sig.entry_high)}</code> · "
                     f"expires in <b>{_dur(left)}</b>")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# /report
# --------------------------------------------------------------------------- #
def format_report(rep: Dict, scope: str = "ALL") -> str:
    total = rep["total"]
    if not total:
        return ("<b>PERFORMANCE REPORT</b>\n\n"
                f"No completed trades yet.\n\n"
                f"Published <b>{rep['published']}</b> · "
                f"Filled <b>{rep['filled']}</b> · "
                f"Expired <b>{rep['expired']}</b>\n\n"
                "<i>Statistics appear once filled signals reach a target or "
                "their stop.</i>")

    wr = rep["win_rate"]
    fill_rate = (rep["filled"] / rep["published"] * 100) if rep["published"] else 0
    rev = rep.get("reversed_after_tp1", 0)
    stalled = rep.get("stalled_at_tp2", 0)
    follow = rep.get("followed_through", 0.0)

    lines = [
        f"<b>PERFORMANCE REPORT</b>  <i>({esc(scope)})</i>",
        "",
        f"Win rate  <code>{bar(wr)}</code>  <b>{wr:.1f}%</b>",
        f"<i>{rep['wins']}W / {rep['losses']}L across {total} completed</i>",
        "",
        "<b>TARGETS REACHED</b>",
        f"TP1        <b>{rep['tp1']}</b>   ({rep['tp1'] / total * 100:.0f}%)",
        f"TP2        <b>{rep['tp2']}</b>   ({rep['tp2'] / total * 100:.0f}%)",
        f"TP3 full   <b>{rep['tp3']}</b>   ({rep['tp3'] / total * 100:.0f}%)",
        f"Stopped    <b>{rep['sl']}</b>   ({rep['sl'] / total * 100:.0f}%)",
        f"   └ <i>{rep['clean_sl']} never reached a target</i>",
        "",
        "<b>FOLLOW-THROUGH</b>",
        f"Reversed after TP1   <b>{rev}</b>"
        + (f"   ({rev / rep['tp1'] * 100:.0f}% of TP1 hits)" if rep["tp1"] else ""),
        f"Stalled after TP2    <b>{stalled}</b>",
        f"TP1 → TP2 conversion <b>{follow:.0f}%</b>",
    ]

    if rep["tp1"] and rev / rep["tp1"] > 0.4:
        lines.append("<i>More than 40% of winners are reversing straight after "
                     "TP1 — the first target is being placed where price turns, "
                     "not where liquidity sits.</i>")

    lines += [
        "",
        "<b>R MULTIPLES</b>",
        f"Total <b>{rep['total_r']:+.2f}R</b> · Average <b>{rep['avg_r']:+.2f}R</b>",
        f"Best <b>{rep['best_r']:+.2f}R</b> · Worst <b>{rep['worst_r']:+.2f}R</b>",
    ]

    if rep["by_mode"]:
        lines += ["", "<b>BY MODE</b>"]
        for mode, m in sorted(rep["by_mode"].items()):
            rate = m["w"] / m["n"] * 100 if m["n"] else 0
            lines.append(f"{esc(mode)} — {m['w']}/{m['n']} (<b>{rate:.0f}%</b>)")

    if rep["by_strategy"]:
        lines += ["", "<b>BY STRATEGY</b>"]
        for strat, m in sorted(rep["by_strategy"].items()):
            rate = m["w"] / m["n"] * 100 if m["n"] else 0
            label = STRATEGY_LABEL.get(strat, strat)
            lines.append(f"{esc(label)} — {m['w']}/{m['n']} (<b>{rate:.0f}%</b>)")

    lines += [
        "",
        "<b>SIGNAL FLOW</b>",
        f"Published <b>{rep['published']}</b> · "
        f"Filled <b>{rep['filled']}</b> ({fill_rate:.0f}%) · "
        f"Expired <b>{rep['expired']}</b>",
        "",
        "<i>A trade counts as a win if it reached TP1 before its stop.</i>",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# status / misc
# --------------------------------------------------------------------------- #
def format_status(data: Dict) -> str:
    rt = data["runtime"]
    modes = " ".join(f"{MODE_EMOJI.get(m, '•')}{m[:2]}"
                     for m, on in rt["modes_enabled"].items() if on) or "none"
    strategies = sum(1 for v in rt["strategies_enabled"].values() if v)
    state = "⏸ PAUSED" if rt["paused"] else "🟢 RUNNING"

    lines = [
        "🤖 <b>ENGINE STATUS</b>",
        "",
        f"State: <b>{state}</b> · Signals: "
        f"<b>{'ON' if rt['signals_enabled'] else 'OFF'}</b>",
        f"Uptime: <b>{_dur(data['uptime_min'])}</b>",
        "",
        "<b>━━━ SCANNING ━━━</b>",
        f"🎯 Modes: <b>{esc(modes)}</b> · Strategies active: <b>{strategies}/2</b>",
        f"🪙 Universe: <b>{data['universe']}</b> symbols "
        f"(&gt;${data['min_volume'] / 1e6:.0f}M volume)",
        f"⏱ Last scan: <b>{data['last_scan']}</b> "
        f"({data['scan_duration']:.1f}s)",
        f"📊 Min score: <b>"
        + " · ".join(f"{m[:2]} {v:.0f}" for m, v in rt["min_score"].items())
        + "</b>",
        "",
        "<b>━━━ TRACKING ━━━</b>",
        f"⏳ Pending entries: <b>{data['pending']}</b>",
        f"📈 Active trades: <b>{data['live']}</b>",
        f"📋 Completed: <b>{data['completed']}</b> · "
        f"Win rate: <b>{data['win_rate']:.1f}%</b>",
        "",
        "<b>━━━ HEALTH ━━━</b>",
        f"⚖️ API weight: <b>{data['weight']}</b>/2400",
        f"📨 Telegram queue: <b>{data['queue']}</b>",
    ]
    if data.get("weekend"):
        lines.append("🗓 <b>Weekend rules active</b> — extra confirmations required")
    return "\n".join(lines)


HELP_TEXT = """<b>BINANCE FUTURES SIGNAL ENGINE</b>
<i>Signal-only. No API keys, no orders, no execution risk.</i>

<b>STRATEGIES</b>
Liquidity Sweep + MSS — reversal, traded after a stop hunt
Order Block + FVG — continuation, traded from unfilled orders

<b>MONITORING</b>
/status — engine state and health
/pnl — live PnL of active trades
/pending — signals awaiting entry
/report — win rate, targets, follow-through
/why &lt;SYMBOL&gt; — confirmations behind a signal

<b>ANALYSIS</b>
/scan &lt;SYMBOL&gt; — force a full MTF scan
/top [n] — best current candidates
/symbols — active universe
/refresh — rebuild the universe now

<b>SETTINGS</b>
/settings — interactive control panel
/mode day|swing on|off
/strategy sweep|obfvg on|off
/score &lt;mode&gt; &lt;value&gt;
/minrr &lt;value&gt;
/weekend on|off — strict Fri 23:00–Mon 20:00 window
/signals on|off · /pause · /resume

<b>SYSTEM</b>
/ping · /log [n] · /help

<i>Signals fire only when every timeframe agrees, the setup clears the score
threshold and the ladder pays at least 6R. Fewer, better.</i>"""
