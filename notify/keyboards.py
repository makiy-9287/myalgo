"""
Interactive settings panel.

Every runtime setting is reachable from an inline keyboard, so nothing has to
be typed on a phone. The panel edits itself in place (editMessageText) rather
than posting a new message per tap, which keeps the chat clean.

Callback data is `verb:domain:key[:value]` and must stay under Telegram's
64-byte limit — the longest string here is 32 bytes, leaving plenty of room.

    nav:<panel>                 navigate to a panel
    tog:<domain>:<key>          flip a boolean
    adj:<domain>:<key>:<delta>  nudge a number
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from config.settings import RuntimeConfig
from notify.formatter import MODE_EMOJI, STRATEGY_LABEL, bar, esc

ON, OFF = "✅", "⛔️"


def _btn(text: str, data: str) -> Dict[str, str]:
    return {"text": text, "callback_data": data}


def _kb(rows: List[List[Dict]]) -> Dict[str, Any]:
    return {"inline_keyboard": rows}


def _mark(flag: bool) -> str:
    return ON if flag else OFF


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def panel_main(rt: RuntimeConfig) -> Tuple[str, Dict]:
    modes_on = sum(1 for v in rt.modes_enabled.values() if v)
    strat_on = sum(1 for v in rt.strategies_enabled.values() if v)
    scores = " · ".join(f"{m[:2]} {v:.0f}" for m, v in rt.min_score.items())
    state = "⏸ PAUSED" if rt.paused else "🟢 RUNNING"

    text = "\n".join([
        "⚙️ <b>CONTROL PANEL</b>",
        "",
        f"Engine: <b>{state}</b>",
        f"Signals: <b>{'ON' if rt.signals_enabled else 'OFF'}</b>",
        f"Modes active: <b>{modes_on}/3</b> · Strategies: <b>{strat_on}/2</b>",
        f"Min score: <b>{esc(scores)}</b> · Min RR: <b>{rt.min_rr:.1f}</b>",
        f"Close/reopen rules: <b>{'ON' if rt.danger_window_enabled else 'OFF'}</b>",
        "",
        "<i>Tap a section to change it. Everything saves instantly and "
        "survives a restart.</i>",
    ])
    kb = _kb([
        [_btn("⚡ Modes", "nav:modes"), _btn("🧠 Strategies", "nav:strat")],
        [_btn("🎯 Score & RR", "nav:score"), _btn("🔔 Alerts", "nav:alerts")],
        [_btn("🗓 Close/Reopen", "nav:weekend"), _btn("🔧 Engine", "nav:engine")],
        [_btn("🔄 Refresh", "nav:main"), _btn("✖️ Close", "nav:close")],
    ])
    return text, kb


# --------------------------------------------------------------------------- #
def panel_modes(rt: RuntimeConfig) -> Tuple[str, Dict]:
    from config.modes import MODES

    lines = ["⚡ <b>TRADING MODES</b>", ""]
    rows = []
    for name, spec in MODES.items():
        on = rt.modes_enabled.get(name, False)
        lines.append(f"{_mark(on)} <b>{esc(name)}</b> — "
                     f"{esc(spec.bias_tf)}→{esc(spec.structure_tf)}→"
                     f"{esc(spec.setup_tf)}→{esc(spec.trigger_tf)}")
        lines.append(f"     <i>entry valid {spec.expiry_minutes // 60}h"
                     f"{spec.expiry_minutes % 60 or ''}"
                     f"{'m' if spec.expiry_minutes % 60 else ''} · "
                     f"min {spec.min_tp3_rr:.1f}R to TP3</i>")
        rows.append([_btn(f"{_mark(on)} {MODE_EMOJI.get(name, '•')} {name}",
                          f"tog:modes:{name}")])

    lines.append("")
    lines.append("<i>Each mode scans independently with its own timeframe "
                 "stack and expiry.</i>")
    rows.append([_btn("◀️ Back", "nav:main")])
    return "\n".join(lines), _kb(rows)


# --------------------------------------------------------------------------- #
def panel_strategies(rt: RuntimeConfig) -> Tuple[str, Dict]:
    keys = ["SWEEP_MSS", "OB_FVG"]
    short = {"SWEEP_MSS": "sweep", "OB_FVG": "obfvg"}

    lines = ["🧠 <b>STRATEGIES</b>", ""]
    rows = []
    for key in keys:
        on = rt.strategies_enabled.get(key, False)
        lines.append(f"{_mark(on)} <b>{esc(STRATEGY_LABEL.get(key, key))}</b>")
        rows.append([_btn(f"{_mark(on)} {short[key].upper()}", f"tog:strat:{key}")])

    lines += [
        "",
        "<i>Both run independently and label their signals separately, so you "
        "can see which engine is actually performing before turning the other "
        "off.</i>",
    ]
    rows.append([_btn("◀️ Back", "nav:main")])
    return "\n".join(lines), _kb(rows)


# --------------------------------------------------------------------------- #
def panel_score(rt: RuntimeConfig) -> Tuple[str, Dict]:
    lines = ["🎯 <b>SCORE &amp; RISK THRESHOLDS</b>", ""]
    rows = []
    for mode in ("DAY", "SWING"):
        val = rt.min_score.get(mode, 90.0)
        lines.append(f"{MODE_EMOJI.get(mode, '•')} <b>{esc(mode)}</b>  "
                     f"<code>{bar(val, width=8)}</code>  <b>{val:.0f}</b>/100")
        rows.append([
            _btn("−5", f"adj:score:{mode}:-5"),
            _btn(f"{mode[:2]} {val:.0f}", "nav:score"),
            _btn("+5", f"adj:score:{mode}:5"),
        ])

    lines += [
        "",
        f"⚖️ <b>Minimum RR</b>: <b>{rt.min_rr:.1f}</b>",
        f"📤 <b>Max signals/cycle</b>: <b>{rt.max_signals_per_cycle}</b>",
        f"❄️ <b>Cooldown</b>: <b>{rt.signal_cooldown_min}m</b> per symbol",
        "",
        "<i>90 is deliberately brutal: it demands nearly every confirmation "
        "in the 100-point budget. Combined with a 6R floor, expect a handful "
        "of setups a week, not a handful an hour.</i>",
    ]
    rows += [
        [_btn("RR −0.2", "adj:num:min_rr:-2"),
         _btn(f"RR {rt.min_rr:.1f}", "nav:score"),
         _btn("RR +0.2", "adj:num:min_rr:2")],
        [_btn("Max −1", "adj:num:max_signals_per_cycle:-1"),
         _btn(f"Max {rt.max_signals_per_cycle}", "nav:score"),
         _btn("Max +1", "adj:num:max_signals_per_cycle:1")],
        [_btn("Cool −15", "adj:num:signal_cooldown_min:-15"),
         _btn(f"{rt.signal_cooldown_min}m", "nav:score"),
         _btn("Cool +15", "adj:num:signal_cooldown_min:15")],
        [_btn("◀️ Back", "nav:main")],
    ]
    return "\n".join(lines), _kb(rows)


# --------------------------------------------------------------------------- #
def panel_alerts(rt: RuntimeConfig) -> Tuple[str, Dict]:
    items = [
        ("alert_on_fill", "Entry filled"),
        ("alert_on_tp", "Target hit"),
        ("alert_on_sl", "Stop hit"),
        ("alert_on_expiry", "Signal expired"),
        ("delete_expired_message", "Delete expired card"),
    ]
    lines = ["🔔 <b>ALERTS</b>", ""]
    rows = []
    for key, label in items:
        on = bool(getattr(rt, key))
        lines.append(f"{_mark(on)} {esc(label)}")
        rows.append([_btn(f"{_mark(on)} {label}", f"tog:flag:{key}")])

    lines += [
        "",
        "<i>“Delete expired card” removes the original signal message from "
        "the chat when its entry never triggered, so only live setups remain "
        "visible.</i>",
    ]
    rows.append([_btn("◀️ Back", "nav:main")])
    return "\n".join(lines), _kb(rows)


# --------------------------------------------------------------------------- #
def panel_weekend(rt: RuntimeConfig) -> Tuple[str, Dict]:
    on = rt.danger_window_enabled
    lines = [
        "🗓 <b>CLOSE / REOPEN RULES</b>",
        "",
        f"{_mark(on)} Strict mode Fri 23:00 → Mon 20:00 UTC",
        f"📊 Score bonus: <b>+{rt.danger_score_bonus:.0f}</b>",
        f"✔️ Extra confirmations: <b>+{rt.danger_extra_confirmations}</b>",
        "",
        "<i>Desks flatten into the Friday close and reopen through Monday. "
        "Books thin out, breaks fail and levels that held all week give way "
        "for no reason that survives Monday. With this on, signals in that "
        "window must also span all four confirmation categories — volume, "
        "bias, liquidity and momentum — not merely clear the score.</i>",
    ]
    rows = [
        [_btn(f"{_mark(on)} Strict window", "tog:flag:danger_window_enabled")],
        [_btn("Bonus −2", "adj:num:danger_score_bonus:-2"),
         _btn(f"+{rt.danger_score_bonus:.0f}", "nav:weekend"),
         _btn("Bonus +2", "adj:num:danger_score_bonus:2")],
        [_btn("Confirm −1", "adj:num:danger_extra_confirmations:-1"),
         _btn(f"+{rt.danger_extra_confirmations}", "nav:weekend"),
         _btn("Confirm +1", "adj:num:danger_extra_confirmations:1")],
        [_btn("◀️ Back", "nav:main")],
    ]
    return "\n".join(lines), _kb(rows)


# --------------------------------------------------------------------------- #
def panel_engine(rt: RuntimeConfig) -> Tuple[str, Dict]:
    lines = [
        "🔧 <b>ENGINE</b>",
        "",
        f"{_mark(rt.signals_enabled)} Signal delivery",
        f"{_mark(not rt.paused)} Scanning active",
        "",
        f"📉 ATR band: <b>{rt.min_atr_pct:.2f}%</b> – <b>{rt.max_atr_pct:.1f}%</b>",
        f"💰 Funding block: <b>±{rt.funding_extreme * 100:.3f}%</b>",
        f"📏 Max spread: <b>{rt.max_spread_pct:.2f}%</b>",
        "",
        "<i>Pausing stops scanning but keeps tracking anything already live, "
        "so you never lose sight of an open trade.</i>",
    ]
    rows = [
        [_btn(f"{_mark(rt.signals_enabled)} Signals", "tog:flag:signals_enabled")],
        [_btn("▶️ Resume" if rt.paused else "⏸ Pause", "tog:flag:paused")],
        [_btn("ATR min −0.02", "adj:num:min_atr_pct:-2"),
         _btn(f"{rt.min_atr_pct:.2f}%", "nav:engine"),
         _btn("ATR min +0.02", "adj:num:min_atr_pct:2")],
        [_btn("◀️ Back", "nav:main")],
    ]
    return "\n".join(lines), _kb(rows)


# --------------------------------------------------------------------------- #
PANELS: Dict[str, Callable[[RuntimeConfig], Tuple[str, Dict]]] = {
    "main": panel_main,
    "modes": panel_modes,
    "strat": panel_strategies,
    "score": panel_score,
    "alerts": panel_alerts,
    "weekend": panel_weekend,
    "engine": panel_engine,
}

# numeric nudges: attribute -> (delta multiplier, low, high)
# The multiplier lets a callback carry an integer while the setting moves in
# fractional steps (min_rr moves 0.2 per tap from a payload of "2").
NUMERIC: Dict[str, Tuple[float, float, float]] = {
    "min_rr": (0.1, 1.0, 15.0),
    "max_signals_per_cycle": (1, 1, 30),
    "signal_cooldown_min": (1, 0, 720),
    "danger_score_bonus": (1, 0, 30),
    "danger_extra_confirmations": (1, 0, 8),
    "min_atr_pct": (0.01, 0.0, 5.0),
    "max_atr_pct": (0.5, 1.0, 30.0),
}


def render(panel: str, rt: RuntimeConfig) -> Tuple[str, Dict]:
    fn = PANELS.get(panel, panel_main)
    return fn(rt)
