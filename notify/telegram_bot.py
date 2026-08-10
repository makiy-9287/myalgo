"""
Telegram interface.

Raw Bot API over aiohttp — no python-telegram-bot, whose major versions keep
breaking handler signatures. That removes an entire class of runtime errors.

Supports the full interactive surface:
  * inline keyboards (reply_markup) for the settings panel
  * callback_query handling with answerCallbackQuery, so buttons never spin
  * editMessageText for in-place panel navigation (no chat spam)
  * deleteMessage so expired signal cards remove themselves

Reliability details that matter in production:
  * an outbound queue with a paced worker (Telegram throttles ~30 msg/s and
    ~20 msg/min to the same group)
  * automatic 429 handling using the `retry_after` the API gives back
  * messages split on line boundaries at 4000 chars so HTML tags never break
  * an HTML fallback: if the API rejects entities, the same text is re-sent as
    plain text rather than being lost
"""
from __future__ import annotations

import asyncio
import html
import re
import time
from typing import Any, Callable, Dict, List, Optional

import aiohttp

from notify.formatter import esc, split_message
from utils.logger import get_logger

log = get_logger("telegram")

API = "https://api.telegram.org/bot{token}/{method}"


class TelegramBot:
    def __init__(self, token: str, chat_id: str, admin_ids: set[str],
                 poll_timeout: int = 25):
        self.token = token
        self.chat_id = str(chat_id)
        self.admin_ids = {str(a) for a in admin_ids}
        self.poll_timeout = poll_timeout

        self._session: Optional[aiohttp.ClientSession] = None
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._offset = 0
        self._running = False
        self._handlers: Dict[str, Callable] = {}
        self._callback_handler: Optional[Callable] = None
        self._start_time = time.time()
        self.enabled = bool(token and chat_id)

    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        if not self.enabled:
            log.warning("Telegram disabled - token or chat id missing")
            return
        timeout = aiohttp.ClientTimeout(total=self.poll_timeout + 20)
        self._session = aiohttp.ClientSession(timeout=timeout)
        self._running = True
        try:
            me = await self._call("getMe")
            log.info("Telegram connected as @%s", (me or {}).get("username", "?"))
        except Exception as exc:                           # noqa: BLE001
            log.error("Telegram getMe failed: %s", exc)

    async def stop(self) -> None:
        self._running = False
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    def register(self, command: str, handler: Callable) -> None:
        self._handlers[command.lower().lstrip("/")] = handler

    def register_callback(self, handler: Callable) -> None:
        """handler(data: str, chat: str, message_id: int) -> (text, markup) | None"""
        self._callback_handler = handler

    @property
    def commands(self) -> List[str]:
        return sorted(self._handlers)

    # ------------------------------------------------------------------ #
    async def _call(self, method: str, payload: Dict | None = None,
                    timeout: int | None = None) -> Any:
        if self._session is None or self._session.closed:
            return None
        url = API.format(token=self.token, method=method)
        kwargs: Dict[str, Any] = {"json": payload or {}}
        if timeout:
            kwargs["timeout"] = aiohttp.ClientTimeout(total=timeout)
        async with self._session.post(url, **kwargs) as resp:
            data = await resp.json(content_type=None)
            if not data.get("ok"):
                desc = data.get("description", "")
                params = data.get("parameters") or {}
                if resp.status == 429 or "Too Many Requests" in desc:
                    wait = float(params.get("retry_after", 3))
                    log.warning("Telegram rate limited, waiting %.1fs", wait)
                    await asyncio.sleep(wait + 0.5)
                    raise RuntimeError("rate_limited")
                raise RuntimeError(f"{method}: {desc}")
            return data.get("result")

    # ------------------------------------------------------------------ #
    # outbound
    # ------------------------------------------------------------------ #
    async def send(self, text: str, chat_id: str | None = None,
                   parse_mode: str = "HTML", silent: bool = False,
                   markup: Dict | None = None,
                   wait: bool = False) -> Optional[int]:
        """
        Queue a message. Never raises — delivery failures are logged only.

        With wait=True the call blocks until the message is actually delivered
        and returns its message_id, which is what lets a signal card be deleted
        later when it expires.
        """
        if not self.enabled or not text:
            return None

        chunks = split_message(text)
        result_id: Optional[int] = None

        for i, chunk in enumerate(chunks):
            payload = {
                "chat_id": chat_id or self.chat_id,
                "text": chunk,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
                "disable_notification": silent,
            }
            # the keyboard belongs on the final chunk only
            if markup and i == len(chunks) - 1:
                payload["reply_markup"] = markup

            fut: Optional[asyncio.Future] = None
            if wait:
                fut = asyncio.get_running_loop().create_future()

            try:
                self._queue.put_nowait({"payload": payload, "future": fut})
            except asyncio.QueueFull:
                log.error("Telegram queue full - dropping message")
                if fut and not fut.done():
                    fut.set_result(None)
                continue

            if fut is not None:
                try:
                    mid = await asyncio.wait_for(fut, timeout=45)
                    if i == 0:
                        result_id = mid
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    log.warning("Timed out waiting for message id")

        return result_id

    async def sender_loop(self) -> None:
        """Paced worker so we never trip Telegram's flood limits."""
        while self._running:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except Exception:                              # noqa: BLE001
                continue

            payload = item["payload"]
            future = item.get("future")
            message_id: Optional[int] = None

            for attempt in range(4):
                try:
                    result = await self._call("sendMessage", payload)
                    if isinstance(result, dict):
                        message_id = result.get("message_id")
                    break
                except RuntimeError as exc:
                    msg = str(exc)
                    if "rate_limited" in msg:
                        continue
                    if "can't parse entities" in msg.lower() or "entity" in msg.lower():
                        payload = {**payload, "text": _strip_html(payload["text"])}
                        payload.pop("parse_mode", None)
                        log.warning("HTML parse rejected, retrying as plain text")
                        continue
                    log.error("sendMessage failed: %s", msg)
                    await asyncio.sleep(1.5 * (attempt + 1))
                except Exception as exc:                   # noqa: BLE001
                    log.error("sendMessage error: %s", exc)
                    await asyncio.sleep(1.5 * (attempt + 1))

            if message_id is None:
                log.error("Dropping undeliverable Telegram message")
            if future is not None and not future.done():
                future.set_result(message_id)

            self._queue.task_done()
            await asyncio.sleep(0.35)                      # ~3 msg/s ceiling

    # ------------------------------------------------------------------ #
    async def delete_message(self, message_id: int,
                             chat_id: str | None = None) -> bool:
        """Telegram only allows deleting messages under 48h old."""
        if not self.enabled or not message_id:
            return False
        try:
            await self._call("deleteMessage", {
                "chat_id": chat_id or self.chat_id,
                "message_id": int(message_id),
            })
            return True
        except Exception as exc:                           # noqa: BLE001
            log.debug("deleteMessage %s failed: %s", message_id, exc)
            return False

    async def edit_message(self, message_id: int, text: str,
                           chat_id: str | None = None,
                           markup: Dict | None = None) -> bool:
        if not self.enabled or not message_id:
            return False
        payload: Dict[str, Any] = {
            "chat_id": chat_id or self.chat_id,
            "message_id": int(message_id),
            "text": text[:4000],
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        if markup is not None:
            payload["reply_markup"] = markup
        try:
            await self._call("editMessageText", payload)
            return True
        except RuntimeError as exc:
            # Telegram errors when the new content is byte-identical; harmless.
            if "not modified" in str(exc).lower():
                return True
            log.debug("editMessageText failed: %s", exc)
            return False
        except Exception as exc:                           # noqa: BLE001
            log.debug("editMessageText error: %s", exc)
            return False

    async def answer_callback(self, callback_id: str, text: str = "",
                              alert: bool = False) -> None:
        try:
            await self._call("answerCallbackQuery", {
                "callback_query_id": callback_id,
                "text": text[:200],
                "show_alert": alert,
            })
        except Exception as exc:                           # noqa: BLE001
            log.debug("answerCallbackQuery failed: %s", exc)

    async def set_my_commands(self, commands: List[tuple]) -> None:
        """Populates the blue menu button in the Telegram client."""
        try:
            await self._call("setMyCommands", {
                "commands": [{"command": c, "description": d[:250]}
                             for c, d in commands][:100]
            })
        except Exception as exc:                           # noqa: BLE001
            log.debug("setMyCommands failed: %s", exc)

    # ------------------------------------------------------------------ #
    # inbound
    # ------------------------------------------------------------------ #
    async def polling_loop(self) -> None:
        if not self.enabled:
            return
        while self._running:
            try:
                updates = await self._call("getUpdates", {
                    "offset": self._offset,
                    "timeout": self.poll_timeout,
                    "allowed_updates": ["message", "callback_query"],
                }, timeout=self.poll_timeout + 15)
            except asyncio.TimeoutError:
                continue
            except RuntimeError as exc:
                if "rate_limited" not in str(exc):
                    log.warning("getUpdates: %s", exc)
                await asyncio.sleep(2)
                continue
            except Exception as exc:                       # noqa: BLE001
                log.warning("getUpdates error: %s", exc)
                await asyncio.sleep(3)
                continue

            for update in updates or []:
                self._offset = max(self._offset, int(update.get("update_id", 0)) + 1)
                try:
                    if "callback_query" in update:
                        await self._handle_callback(update["callback_query"])
                    else:
                        await self._handle_message(update)
                except Exception as exc:                   # noqa: BLE001
                    log.error("update handling failed: %s", exc, exc_info=True)

    def _authorised(self, chat: str, user: str) -> bool:
        if not self.admin_ids:
            return True
        return chat in self.admin_ids or user in self.admin_ids

    async def _handle_message(self, update: Dict) -> None:
        msg = update.get("message") or {}
        text = (msg.get("text") or "").strip()
        chat = str((msg.get("chat") or {}).get("id", ""))
        user = str((msg.get("from") or {}).get("id", ""))

        if not text.startswith("/"):
            return

        if not self._authorised(chat, user):
            log.warning("Ignored command from unauthorised chat %s / user %s", chat, user)
            await self.send("⛔️ Not authorised.", chat_id=chat)
            return

        parts = text.split()
        cmd = parts[0][1:].lower()
        if "@" in cmd:                                     # /status@MyBot in groups
            cmd = cmd.split("@")[0]
        args = parts[1:]

        handler = self._handlers.get(cmd)
        if not handler:
            await self.send(
                f"❓ Unknown command <code>/{esc(cmd)}</code>. Send /help.",
                chat_id=chat)
            return

        try:
            reply = await handler(args, chat)
        except Exception as exc:                           # noqa: BLE001
            log.error("Command /%s failed: %s", cmd, exc, exc_info=True)
            reply = f"⚠️ <b>/{esc(cmd)} failed:</b> <code>{esc(str(exc)[:300])}</code>"

        # a handler may return plain text, or (text, inline_keyboard)
        if isinstance(reply, tuple):
            body, markup = reply
            if body:
                await self.send(body, chat_id=chat, markup=markup)
        elif reply:
            await self.send(reply, chat_id=chat)

    async def _handle_callback(self, cq: Dict) -> None:
        cb_id = cq.get("id", "")
        data = cq.get("data") or ""
        msg = cq.get("message") or {}
        chat = str((msg.get("chat") or {}).get("id", ""))
        user = str((cq.get("from") or {}).get("id", ""))
        message_id = msg.get("message_id")

        if not self._authorised(chat, user):
            await self.answer_callback(cb_id, "Not authorised", alert=True)
            return

        if not self._callback_handler:
            await self.answer_callback(cb_id)
            return

        try:
            result = await self._callback_handler(data, chat, message_id)
        except Exception as exc:                           # noqa: BLE001
            log.error("callback %r failed: %s", data, exc, exc_info=True)
            await self.answer_callback(cb_id, "Something went wrong", alert=True)
            return

        toast, text, markup = "", None, None
        if isinstance(result, tuple):
            if len(result) == 3:
                toast, text, markup = result
            elif len(result) == 2:
                text, markup = result

        await self.answer_callback(cb_id, toast or "")
        if text is not None and message_id:
            await self.edit_message(message_id, text, chat_id=chat, markup=markup)

    # ------------------------------------------------------------------ #
    @property
    def queue_size(self) -> int:
        return self._queue.qsize()


def _strip_html(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text))
