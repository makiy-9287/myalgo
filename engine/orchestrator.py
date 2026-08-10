"""
Application wiring.

Four loops:
    scan_loop        every ANALYSIS_INTERVAL   — find setups
    tracker_loop     every TRACKER_INTERVAL    — fills, targets, stops
    universe_loop    hourly                    — rebuild the symbol list
    housekeeping     every 5 min               — expiries, cooldowns, persistence

Plus the Telegram command surface and the interactive settings panel.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from config.modes import MODES
from config.settings import LOG_DIR, RUNTIME, SETTINGS, RuntimeConfig, Settings
from core.data_manager import DataManager
from core.exchange import BinanceFutures
from core.models import Signal
from core.state import StateStore
from engine.scanner import Scanner
from engine.tracker import SignalTracker
from notify import keyboards
from notify.formatter import (HELP_TEXT, MODE_EMOJI, STRATEGY_LABEL, esc,
                              format_pending, format_pnl, format_report,
                              format_signal, format_status)
from notify.telegram_bot import TelegramBot
from utils.helpers import danger_window_label, in_danger_window
from utils.logger import get_logger

log = get_logger("app")

STRATEGY_ALIASES = {"sweep": "SWEEP_MSS", "mss": "SWEEP_MSS",
                    "reversal": "SWEEP_MSS", "obfvg": "OB_FVG",
                    "ob": "OB_FVG", "fvg": "OB_FVG", "continuation": "OB_FVG"}


class Application:
    def __init__(self, settings: Settings = SETTINGS,
                 runtime: RuntimeConfig = RUNTIME):
        self.s = settings
        self.rt = runtime
        self.started_at = time.time()

        self.exchange = BinanceFutures(settings)
        self.state = StateStore()
        self.data = DataManager(self.exchange, settings)
        self.scanner = Scanner(self.data, self.state, settings, runtime)
        self.bot = TelegramBot(settings.telegram_token, settings.telegram_chat_id,
                               settings.admin_ids, settings.telegram_poll_timeout)
        self.tracker = SignalTracker(self.exchange, self.state, runtime,
                                     notify=self._notify,
                                     delete_message=self.bot.delete_message)
        self._tasks: List[asyncio.Task] = []
        self._running = False

    # ------------------------------------------------------------------ #
    async def _notify(self, text: str, silent: bool = False) -> None:
        if not self.rt.signals_enabled:
            return
        await self.bot.send(text, silent=silent)

    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        await self.exchange.start()
        await self.bot.start()
        self._register_commands()
        self.bot.register_callback(self.on_callback)

        await self.data.refresh_universe()
        self._running = True

        self._tasks = [
            asyncio.create_task(self._guard(self.bot.sender_loop(), "sender")),
            asyncio.create_task(self._guard(self.bot.polling_loop(), "polling")),
            asyncio.create_task(self._guard(self.scan_loop(), "scan")),
            asyncio.create_task(self._guard(self.tracker_loop(), "tracker")),
            asyncio.create_task(self._guard(self.universe_loop(), "universe")),
            asyncio.create_task(self._guard(self.housekeeping_loop(), "housekeeping")),
        ]

        await self.bot.set_my_commands([
            ("status", "Engine state and health"),
            ("pnl", "Live PnL of active trades"),
            ("pending", "Signals awaiting entry"),
            ("report", "Win rate and TP/SL breakdown"),
            ("settings", "Interactive control panel"),
            ("scan", "Force a scan of one symbol"),
            ("top", "Best current candidates"),
            ("why", "Confirmations behind a signal"),
            ("pause", "Stop scanning"),
            ("resume", "Resume scanning"),
            ("help", "All commands"),
        ])
        await self.bot.send(self._boot_message())
        log.info("Application started")

    async def _guard(self, coro, name: str) -> None:
        """Keep a loop alive across unexpected exceptions."""
        while self._running:
            try:
                await coro
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:                        # noqa: BLE001
                log.error("%s loop crashed: %s — restarting in 5s", name, exc,
                          exc_info=True)
                await asyncio.sleep(5)
                return

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self.state.save()
        try:
            await self.bot.send("🛑 <b>Engine stopped.</b>")
            await asyncio.sleep(0.6)
        except Exception:                                   # noqa: BLE001
            pass
        await self.bot.stop()
        await self.exchange.close()
        log.info("Application stopped")

    async def run_forever(self) -> None:
        await self.start()
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------ #
    # loops
    # ------------------------------------------------------------------ #
    async def scan_loop(self) -> None:
        while self._running:
            try:
                if not self.rt.paused:
                    await self.run_scan()
            except Exception as exc:                        # noqa: BLE001
                log.error("scan cycle failed: %s", exc, exc_info=True)
            await asyncio.sleep(self.s.analysis_interval)

    async def run_scan(self) -> Dict[str, List[Signal]]:
        results = await self.scanner.scan_all()
        for _mode, signals in results.items():
            for sig in signals:
                await self._dispatch(sig)
        return results

    async def _dispatch(self, sig: Signal) -> None:
        """Publish a signal and start tracking it for an entry fill."""
        if not self.rt.signals_enabled:
            return
        card = format_signal(sig)
        message_id = await self.bot.send(card, wait=True)
        sig.message_id = message_id
        self.state.publish(sig)
        log.info("SIGNAL %s %s %s score=%.0f rr=%.2f",
                 sig.symbol, sig.side, sig.mode, sig.score, sig.rr_total)

    async def tracker_loop(self) -> None:
        while self._running:
            try:
                await self.tracker.run_once()
            except Exception as exc:                        # noqa: BLE001
                log.error("tracker failed: %s", exc, exc_info=True)
            await asyncio.sleep(self.s.tracker_interval)

    async def universe_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.s.universe_refresh_interval)
            try:
                await self.data.refresh_universe()
            except Exception as exc:                        # noqa: BLE001
                log.error("universe refresh failed: %s", exc)

    async def housekeeping_loop(self) -> None:
        while self._running:
            await asyncio.sleep(60)
            try:
                await self.tracker.handle_expiries()
                self.state.purge_cooldowns()
                self.data.prune_cache()
                self.state.save()
            except Exception as exc:                        # noqa: BLE001
                log.error("housekeeping failed: %s", exc)

    # ------------------------------------------------------------------ #
    def _boot_message(self) -> str:
        modes = ", ".join(m for m, on in self.rt.modes_enabled.items() if on)
        scores = " · ".join(f"{m[:2]} {v:.0f}" for m, v in self.rt.min_score.items())
        return "\n".join([
            "🚀 <b>SIGNAL ENGINE ONLINE</b>",
            "",
            f"🪙 Universe: <b>{len(self.data.universe)}</b> symbols "
            f"(&gt;${self.s.min_volume_usdt / 1e6:.0f}M 24h volume)",
            f"🎯 Modes: <b>{esc(modes or 'none')}</b>",
            f"📊 Min score: <b>{esc(scores)}</b>",
            f"⏱ Deep scan every <b>{self.s.analysis_interval}s</b> · "
            f"tracking every <b>{self.s.tracker_interval}s</b>",
            "",
            "<i>Signal-only build — no API keys, no orders, nothing to "
            "liquidate.</i>",
            "",
            "Send /settings for the control panel or /help for commands.",
        ])

    # ------------------------------------------------------------------ #
    def _register_commands(self) -> None:
        r = self.bot.register
        r("start", self.cmd_help)
        r("help", self.cmd_help)
        r("status", self.cmd_status)
        r("pnl", self.cmd_pnl)
        r("pending", self.cmd_pending)
        r("report", self.cmd_report)
        r("why", self.cmd_why)
        r("settings", self.cmd_settings)
        r("signals", self.cmd_signals)
        r("pause", self.cmd_pause)
        r("resume", self.cmd_resume)
        r("mode", self.cmd_mode)
        r("strategy", self.cmd_strategy)
        r("score", self.cmd_score)
        r("minrr", self.cmd_minrr)
        r("weekend", self.cmd_weekend)
        r("scan", self.cmd_scan)
        r("top", self.cmd_top)
        r("symbols", self.cmd_symbols)
        r("refresh", self.cmd_refresh)
        r("ping", self.cmd_ping)
        r("log", self.cmd_log)

    # ------------------------------------------------------------------ #
    # interactive panel
    # ------------------------------------------------------------------ #
    async def cmd_settings(self, args, chat) -> Tuple[str, Dict]:
        return keyboards.render("main", self.rt)

    async def on_callback(self, data: str, chat: str,
                          message_id: int) -> Optional[Tuple]:
        """
        Handle a button tap.

        Returns (toast, text, markup). The panel is re-rendered from live
        config after every mutation, so what you see is always the real state
        rather than an optimistic guess.
        """
        parts = (data or "").split(":")
        verb = parts[0] if parts else ""
        toast = ""

        if verb == "nav":
            target = parts[1] if len(parts) > 1 else "main"
            if target == "close":
                await self.bot.delete_message(message_id, chat_id=chat)
                return None
            text, markup = keyboards.render(target, self.rt)
            return "", text, markup

        if verb == "tog" and len(parts) >= 3:
            domain, key = parts[1], parts[2]
            if domain == "modes":
                cur = self.rt.modes_enabled.get(key, True)
                self.rt.modes_enabled[key] = not cur
                self.rt.save()
                toast = f"{key} {'enabled' if not cur else 'disabled'}"
                text, markup = keyboards.render("modes", self.rt)
            elif domain == "strat":
                cur = self.rt.strategies_enabled.get(key, True)
                self.rt.strategies_enabled[key] = not cur
                self.rt.save()
                toast = f"{'Enabled' if not cur else 'Disabled'}"
                text, markup = keyboards.render("strat", self.rt)
            elif domain == "flag" and hasattr(self.rt, key):
                new = self.rt.toggle(key)
                toast = f"{'ON' if new else 'OFF'}"
                panel = "engine" if key in ("signals_enabled", "paused") else (
                    "weekend" if key.startswith("weekend") else "alerts")
                text, markup = keyboards.render(panel, self.rt)
            else:
                return "Unknown option", None, None
            return toast, text, markup

        if verb == "adj" and len(parts) >= 4:
            domain, key, raw = parts[1], parts[2], parts[3]
            try:
                delta = float(raw)
            except ValueError:
                return "Bad value", None, None

            if domain == "score":
                cur = float(self.rt.min_score.get(key, 90.0))
                new = max(40.0, min(99.0, cur + delta))
                self.rt.min_score[key] = new
                self.rt.save()
                toast = f"{key} min score {new:.0f}"
                text, markup = keyboards.render("score", self.rt)
                return toast, text, markup

            if domain == "num" and key in keyboards.NUMERIC:
                step, low, high = keyboards.NUMERIC[key]
                cur = float(getattr(self.rt, key))
                new = max(low, min(high, cur + delta * step))
                if isinstance(getattr(self.rt, key), int) and step == 1:
                    new = int(round(new))
                self.rt.set(key, new)
                toast = f"{key.replace('_', ' ')} = {new:g}"
                panel = ("weekend" if key.startswith("weekend")
                         else "engine" if "atr" in key else "score")
                text, markup = keyboards.render(panel, self.rt)
                return toast, text, markup

        return "", None, None

    # ------------------------------------------------------------------ #
    # commands
    # ------------------------------------------------------------------ #
    async def cmd_help(self, args, chat) -> str:
        return HELP_TEXT

    async def cmd_status(self, args, chat) -> str:
        rep = self.state.report()
        last = (datetime.fromtimestamp(self.scanner.last_scan_at, tz=timezone.utc)
                .strftime("%H:%M:%S UTC") if self.scanner.last_scan_at else "never")
        return format_status({
            "runtime": self.rt.to_dict(),
            "uptime_min": (time.time() - self.started_at) / 60,
            "universe": len(self.data.universe),
            "min_volume": self.s.min_volume_usdt,
            "last_scan": last,
            "scan_duration": self.scanner.last_scan_duration,
            "pending": self.state.pending_count,
            "live": self.state.live_count,
            "completed": rep["total"],
            "win_rate": rep["win_rate"],
            "weight": self.exchange.weight_used,
            "queue": self.bot.queue_size,
            "weekend": in_danger_window(),
        })

    async def cmd_pnl(self, args, chat) -> str:
        rows = await self.tracker.snapshot()
        return format_pnl(rows)

    async def cmd_pending(self, args, chat) -> str:
        return format_pending(self.state.pending_signals())

    async def cmd_report(self, args, chat) -> str:
        mode = strategy = None
        scope = "ALL"
        if args:
            token = args[0].upper()
            if token in MODES:
                mode, scope = token, token
            elif args[0].lower() in STRATEGY_ALIASES:
                strategy = STRATEGY_ALIASES[args[0].lower()]
                scope = STRATEGY_LABEL.get(strategy, strategy)
        return format_report(self.state.report(mode=mode, strategy=strategy), scope)

    async def cmd_why(self, args, chat) -> str:
        """
        The confirmation list, on demand.

        It is kept out of the signal card because a fifteen-line rationale
        buries the numbers that matter, but the reasoning is still there when
        you want to audit a setup.
        """
        if not args:
            return ("Usage: <code>/why BTCUSDT</code>\n\n"
                    "Shows every confirmation behind a tracked signal.")
        symbol = args[0].upper()
        if not symbol.endswith(self.s.quote_asset):
            symbol += self.s.quote_asset

        matches = self.state.find(symbol)
        if not matches:
            return (f"No tracked signal for <b>{esc(symbol)}</b>.\n"
                    f"Use <code>/scan {esc(symbol)}</code> to analyse it now.")

        out = []
        for sig in matches:
            head = (f"🔍 <b>{esc(sig.symbol)}</b> {esc(sig.side)} "
                    f"<i>{esc(sig.mode)}</i> · score <b>{sig.score:.0f}</b>/100 "
                    f"· <i>{esc(sig.status)}</i>")
            lines = [head, ""]
            if not sig.confirmations:
                lines.append("<i>Confirmations were not retained for this "
                             "signal (restored from disk).</i>")
            for c in sorted(sig.confirmations, key=lambda x: x.weight, reverse=True):
                tf = f"{esc(c.timeframe)} " if c.timeframe else ""
                lines.append(f"• <b>{esc(c.name)}</b> <code>+{c.weight:.1f}</code>")
                lines.append(f"     └ <i>{tf}{esc(c.detail)}</i>")
            out.append("\n".join(lines))
        return "\n\n".join(out)

    async def cmd_signals(self, args, chat) -> str:
        if args and args[0].lower() in ("on", "off"):
            self.rt.set("signals_enabled", args[0].lower() == "on")
        return (f"📢 Signal delivery: "
                f"<b>{'ON' if self.rt.signals_enabled else 'OFF'}</b>")

    async def cmd_pause(self, args, chat) -> str:
        self.rt.set("paused", True)
        return ("⏸ <b>Scanning paused.</b>\n"
                "<i>Anything already live is still being tracked.</i>")

    async def cmd_resume(self, args, chat) -> str:
        self.rt.set("paused", False)
        return "▶️ <b>Scanning resumed.</b>"

    async def cmd_mode(self, args, chat) -> str:
        if len(args) < 2:
            rows = "\n".join(
                f"{MODE_EMOJI.get(m, '•')} <b>{esc(m)}</b>: "
                f"{'ON' if on else 'OFF'}"
                for m, on in self.rt.modes_enabled.items())
            return (f"🎯 <b>MODES</b>\n\n{rows}\n\n"
                    f"Usage: <code>/mode scalp on</code>")
        name = args[0].upper()
        if name not in MODES:
            return f"Unknown mode <code>{esc(args[0])}</code>. Use scalp, day or swing."
        if args[1].lower() not in ("on", "off"):
            return "Second argument must be <code>on</code> or <code>off</code>."
        self.rt.modes_enabled[name] = args[1].lower() == "on"
        self.rt.save()
        return f"{MODE_EMOJI.get(name, '•')} <b>{esc(name)}</b> → {args[1].upper()}"

    async def cmd_strategy(self, args, chat) -> str:
        if len(args) < 2:
            rows = "\n".join(
                f"• <b>{esc(STRATEGY_LABEL.get(k, k))}</b>: {'ON' if v else 'OFF'}"
                for k, v in self.rt.strategies_enabled.items())
            return (f"🧠 <b>STRATEGIES</b>\n\n{rows}\n\n"
                    f"Usage: <code>/strategy smc off</code>")
        key = STRATEGY_ALIASES.get(args[0].lower())
        if not key:
            return "Unknown strategy. Use <code>smc</code> or <code>trend</code>."
        if args[1].lower() not in ("on", "off"):
            return "Second argument must be <code>on</code> or <code>off</code>."
        self.rt.strategies_enabled[key] = args[1].lower() == "on"
        self.rt.save()
        return f"<b>{esc(STRATEGY_LABEL.get(key, key))}</b> → {args[1].upper()}"

    async def cmd_score(self, args, chat) -> str:
        if len(args) < 2:
            rows = "\n".join(f"{MODE_EMOJI.get(m, '•')} <b>{esc(m)}</b>: {v:.0f}"
                             for m, v in self.rt.min_score.items())
            return (f"📊 <b>MINIMUM SCORE</b>\n\n{rows}\n\n"
                    f"Usage: <code>/score day 90</code>")
        name = args[0].upper()
        if name not in MODES:
            return f"Unknown mode <code>{esc(args[0])}</code>."
        try:
            val = float(args[1])
        except ValueError:
            return "Score must be a number."
        if not 40 <= val <= 99:
            return "Score must be between 40 and 99."
        self.rt.min_score[name] = val
        self.rt.save()
        note = ("\n<i>Below 85 you will start seeing setups that are merely "
                "good rather than exceptional.</i>" if val < 85 else "")
        return f"📊 <b>{esc(name)}</b> minimum score → <b>{val:.0f}</b>{note}"

    async def cmd_minrr(self, args, chat) -> str:
        if not args:
            return f"⚖️ Minimum RR: <b>{self.rt.min_rr:.2f}</b>"
        try:
            val = float(args[0])
        except ValueError:
            return "RR must be a number."
        if not 0.5 <= val <= 10:
            return "RR must be between 0.5 and 10."
        self.rt.set("min_rr", val)
        return f"⚖️ Minimum RR → <b>{val:.2f}</b>"

    async def cmd_weekend(self, args, chat) -> str:
        if args and args[0].lower() in ("on", "off"):
            self.rt.set("danger_window_enabled", args[0].lower() == "on")
        state = "ON" if self.rt.danger_window_enabled else "OFF"
        return (f"Strict close/reopen window: <b>{state}</b>\n"
                f"Score bonus <b>+{self.rt.danger_score_bonus:.0f}</b> · "
                f"extra confirmations <b>+{self.rt.danger_extra_confirmations}</b>")

    async def cmd_scan(self, args, chat) -> str:
        if not args:
            return "Usage: <code>/scan BTCUSDT</code>"
        signals, notes = await self.scanner.scan_single(args[0])
        if signals:
            for sig in signals:
                await self._dispatch(sig)
            return f"✅ Found <b>{len(signals)}</b> signal(s) — sent above."
        head = f"🔍 <b>{esc(args[0].upper())}</b> — no valid setup right now\n"
        return head + "\n".join(f"• <i>{esc(n)}</i>" for n in notes[:12])

    async def cmd_top(self, args, chat) -> str:
        try:
            n = min(20, max(1, int(args[0]))) if args else 8
        except ValueError:
            n = 8
        rows = self.state.pending_signals()[:n]
        if not rows:
            return ("No pending candidates.\n"
                    "<i>At a 90 threshold, quiet periods are expected.</i>")
        return format_pending(rows)

    async def cmd_symbols(self, args, chat) -> str:
        uni = self.data.universe
        if not uni:
            return "Universe is empty — try /refresh."
        head = (f"🪙 <b>UNIVERSE</b> — {len(uni)} symbols "
                f"(&gt;${self.s.min_volume_usdt / 1e6:.0f}M)\n\n")
        return head + esc(", ".join(s.replace(self.s.quote_asset, "")
                                    for s in uni[:120]))

    async def cmd_refresh(self, args, chat) -> str:
        await self.data.refresh_universe()
        return f"🔄 Universe rebuilt: <b>{len(self.data.universe)}</b> symbols."

    async def cmd_ping(self, args, chat) -> str:
        t0 = time.time()
        try:
            await self.exchange.ping()
            latency = (time.time() - t0) * 1000
            return (f"🏓 <b>pong</b> — Binance {latency:.0f}ms · "
                    f"weight {self.exchange.weight_used}/2400")
        except Exception as exc:                            # noqa: BLE001
            return f"⚠️ Binance unreachable: <code>{esc(str(exc)[:200])}</code>"

    async def cmd_log(self, args, chat) -> str:
        try:
            n = min(50, max(1, int(args[0]))) if args else 15
        except ValueError:
            n = 15
        path = LOG_DIR / "bot.log"
        if not path.exists():
            return "No log file yet."
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return f"Could not read log: <code>{esc(str(exc))}</code>"
        tail = "\n".join(lines[-n:])[-3500:]
        return f"📄 <b>LOG</b> (last {n})\n\n<pre>{esc(tail)}</pre>"
