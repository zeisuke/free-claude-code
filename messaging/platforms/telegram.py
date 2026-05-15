"""
Telegram Platform Adapter

Implements MessagingPlatform for Telegram using python-telegram-bot.
"""

import asyncio
import contextlib
import os
import tempfile
from pathlib import Path

import httpx

# Opt-in to future behavior for python-telegram-bot (retry_after as timedelta)
# This must be set BEFORE importing telegram.error
os.environ["PTB_TIMEDELTA"] = "1"

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from loguru import logger

from core.anthropic import format_user_error_preview

if TYPE_CHECKING:
    from telegram import Update
    from telegram.ext import ContextTypes

from ..models import IncomingMessage
from ..rendering.telegram_markdown import escape_md_v2, format_status
from ..voice import PendingVoiceRegistry, VoiceTranscriptionService
from .base import MessagingPlatform

# Optional import - python-telegram-bot may not be installed
try:
    from telegram import Update
    from telegram.error import NetworkError, RetryAfter, TelegramError
    from telegram.ext import (
        Application,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )
    from telegram.request import HTTPXRequest

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False


class TelegramPlatform(MessagingPlatform):
    """
    Telegram messaging platform adapter.

    Uses python-telegram-bot (BoT API) for Telegram access.
    Requires a Bot Token from @BotFather.
    """

    name = "telegram"

    def __init__(
        self,
        bot_token: str | None = None,
        allowed_user_id: str | None = None,
        *,
        voice_note_enabled: bool = True,
        whisper_model: str = "base",
        whisper_device: str = "cpu",
        hf_token: str = "",
        nvidia_nim_api_key: str = "",
        messaging_rate_limit: int = 1,
        messaging_rate_window: float = 1.0,
        log_raw_messaging_content: bool = False,
        log_api_error_tracebacks: bool = False,
    ):
        if not TELEGRAM_AVAILABLE:
            raise ImportError(
                "python-telegram-bot is required. Install with: pip install python-telegram-bot"
            )

        self.bot_token = bot_token
        self.allowed_user_id = allowed_user_id

        if not self.bot_token:
            # We don't raise here to allow instantiation for testing/conditional logic,
            # but start() will fail.
            logger.warning("TELEGRAM_BOT_TOKEN not set")

        self._application: Application | None = None
        self._message_handler: Callable[[IncomingMessage], Awaitable[None]] | None = (
            None
        )
        self._connected = False
        self._limiter: Any | None = None  # Will be MessagingRateLimiter
        # Pending voice transcriptions: (chat_id, msg_id) -> (voice_msg_id, status_msg_id)
        self._pending_voice = PendingVoiceRegistry()
        self._voice_transcription = VoiceTranscriptionService(
            hf_token=hf_token,
            nvidia_nim_api_key=nvidia_nim_api_key,
        )
        self._voice_note_enabled = voice_note_enabled
        self._whisper_model = whisper_model
        self._whisper_device = whisper_device
        self._messaging_rate_limit = messaging_rate_limit
        self._messaging_rate_window = messaging_rate_window
        self._log_raw_messaging_content = log_raw_messaging_content
        self._log_api_error_tracebacks = log_api_error_tracebacks

    async def _register_pending_voice(
        self, chat_id: str, voice_msg_id: str, status_msg_id: str
    ) -> None:
        """Register a voice note as pending transcription (for /clear reply during transcription)."""
        await self._pending_voice.register(chat_id, voice_msg_id, status_msg_id)

    async def cancel_pending_voice(
        self, chat_id: str, reply_id: str
    ) -> tuple[str, str] | None:
        """Cancel a pending voice transcription. Returns (voice_msg_id, status_msg_id) if found."""
        return await self._pending_voice.cancel(chat_id, reply_id)

    async def _is_voice_still_pending(self, chat_id: str, voice_msg_id: str) -> bool:
        """Check if a voice note is still pending (not cancelled)."""
        return await self._pending_voice.is_pending(chat_id, voice_msg_id)

    async def start(self) -> None:
        """Initialize and connect to Telegram."""
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")

        # Configure request with longer timeouts
        request = HTTPXRequest(
            connection_pool_size=8, connect_timeout=30.0, read_timeout=30.0
        )

        # Build Application
        builder = Application.builder().token(self.bot_token).request(request)
        self._application = builder.build()

        # Register Internal Handlers
        # We catch ALL text messages and commands to forward them
        self._application.add_handler(
            MessageHandler(filters.TEXT & (~filters.COMMAND), self._on_telegram_message)
        )
        self._application.add_handler(CommandHandler("start", self._on_start_command))
        # Catch-all for other commands if needed, or let them fall through
        self._application.add_handler(
            MessageHandler(filters.COMMAND, self._on_telegram_message)
        )
        # Voice note handler
        self._application.add_handler(
            MessageHandler(filters.VOICE, self._on_telegram_voice)
        )
        # Photo handler
        self._application.add_handler(
            MessageHandler(filters.PHOTO, self._on_telegram_photo)
        )
        # Video handler (includes video notes)
        self._application.add_handler(
            MessageHandler(filters.VIDEO | filters.VIDEO_NOTE, self._on_telegram_video)
        )
        # Audio file handler
        self._application.add_handler(
            MessageHandler(filters.AUDIO, self._on_telegram_audio)
        )
        # Document handler (PDFs and other files)
        self._application.add_handler(
            MessageHandler(filters.Document.ALL, self._on_telegram_document)
        )

        # Initialize internal components with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await self._application.initialize()
                await self._application.start()

                # Start polling (non-blocking way for integration)
                if self._application.updater:
                    await self._application.updater.start_polling(
                        drop_pending_updates=False
                    )

                self._connected = True
                break
            except (NetworkError, Exception) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 * (attempt + 1)
                    logger.warning(
                        f"Connection failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to connect after {max_retries} attempts")
                    raise

        # Initialize rate limiter
        from ..limiter import MessagingRateLimiter

        self._limiter = await MessagingRateLimiter.get_instance(
            rate_limit=self._messaging_rate_limit,
            rate_window=self._messaging_rate_window,
        )

        # Send startup notification
        try:
            target = self.allowed_user_id
            if target:
                startup_text = (
                    f"🚀 *{escape_md_v2('Claude Code Proxy is online!')}* "
                    f"{escape_md_v2('(Bot API)')}"
                )
                await self.send_message(
                    target,
                    startup_text,
                )
        except Exception as e:
            if self._log_api_error_tracebacks:
                logger.warning("Could not send startup message: {}", e)
            else:
                logger.warning(
                    "Could not send startup message: exc_type={}",
                    type(e).__name__,
                )

        logger.info("Telegram platform started (Bot API)")

    async def stop(self) -> None:
        """Stop the bot."""
        if self._application and self._application.updater:
            await self._application.updater.stop()
            await self._application.stop()
            await self._application.shutdown()

        self._connected = False
        logger.info("Telegram platform stopped")

    async def _with_retry(
        self, func: Callable[..., Awaitable[Any]], *args, **kwargs
    ) -> Any:
        """Helper to execute a function with exponential backoff on network errors."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except (TimeoutError, NetworkError) as e:
                if "Message is not modified" in str(e):
                    return None
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # 1s, 2s, 4s
                    logger.warning(
                        f"Telegram API network error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Telegram API failed after {max_retries} attempts: {e}"
                    )
                    raise
            except RetryAfter as e:
                # Telegram explicitly tells us to wait (PTB_TIMEDELTA: retry_after is timedelta)
                from datetime import timedelta

                retry_after = e.retry_after
                if isinstance(retry_after, timedelta):
                    wait_secs = retry_after.total_seconds()
                else:
                    wait_secs = float(retry_after)

                logger.warning(f"Rate limited by Telegram, waiting {wait_secs}s...")
                await asyncio.sleep(wait_secs)
                # We don't increment attempt here, as this is a specific instruction
                return await func(*args, **kwargs)
            except TelegramError as e:
                # Non-network Telegram errors
                err_lower = str(e).lower()
                if "message is not modified" in err_lower:
                    return None
                # Best-effort no-op cases (common during chat cleanup / /clear).
                if any(
                    x in err_lower
                    for x in [
                        "message to edit not found",
                        "message to delete not found",
                        "message can't be deleted",
                        "message can't be edited",
                        "not enough rights to delete",
                    ]
                ):
                    return None
                if "Can't parse entities" in str(e) and kwargs.get("parse_mode"):
                    logger.warning("Markdown failed, retrying without parse_mode")
                    kwargs["parse_mode"] = None
                    return await func(*args, **kwargs)
                raise

    async def send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        parse_mode: str | None = "MarkdownV2",
        message_thread_id: str | None = None,
    ) -> str:
        """Send a message to a chat."""
        app = self._application
        if not app or not app.bot:
            raise RuntimeError("Telegram application or bot not initialized")

        async def _do_send(parse_mode=parse_mode):
            bot = app.bot
            kwargs: dict[str, Any] = {
                "chat_id": chat_id,
                "text": text,
                "reply_to_message_id": int(reply_to) if reply_to else None,
                "parse_mode": parse_mode,
            }
            if message_thread_id is not None:
                kwargs["message_thread_id"] = int(message_thread_id)
            msg = await bot.send_message(**kwargs)
            return str(msg.message_id)

        return await self._with_retry(_do_send, parse_mode=parse_mode)

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: str | None = "MarkdownV2",
    ) -> None:
        """Edit an existing message."""
        app = self._application
        if not app or not app.bot:
            raise RuntimeError("Telegram application or bot not initialized")

        async def _do_edit(parse_mode=parse_mode):
            bot = app.bot
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=int(message_id),
                text=text,
                parse_mode=parse_mode,
            )

        await self._with_retry(_do_edit, parse_mode=parse_mode)

    async def delete_message(
        self,
        chat_id: str,
        message_id: str,
    ) -> None:
        """Delete a message from a chat."""
        app = self._application
        if not app or not app.bot:
            raise RuntimeError("Telegram application or bot not initialized")

        async def _do_delete():
            bot = app.bot
            await bot.delete_message(chat_id=chat_id, message_id=int(message_id))

        await self._with_retry(_do_delete)

    async def delete_messages(self, chat_id: str, message_ids: list[str]) -> None:
        """Delete multiple messages (best-effort)."""
        if not message_ids:
            return
        app = self._application
        if not app or not app.bot:
            raise RuntimeError("Telegram application or bot not initialized")

        # PTB supports bulk deletion via delete_messages; fall back to per-message.
        bot = app.bot
        if hasattr(bot, "delete_messages"):

            async def _do_bulk():
                mids = []
                for mid in message_ids:
                    try:
                        mids.append(int(mid))
                    except Exception:
                        continue
                if not mids:
                    return None
                # delete_messages accepts a sequence of ints (up to 100).
                await bot.delete_messages(chat_id=chat_id, message_ids=mids)

            await self._with_retry(_do_bulk)
            return

        for mid in message_ids:
            await self.delete_message(chat_id, mid)

    async def queue_send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        parse_mode: str | None = "MarkdownV2",
        fire_and_forget: bool = True,
        message_thread_id: str | None = None,
    ) -> str | None:
        """Enqueue a message to be sent (using limiter)."""
        # Note: Bot API handles limits better, but we still use our limiter for nice queuing
        if not self._limiter:
            return await self.send_message(
                chat_id, text, reply_to, parse_mode, message_thread_id
            )

        async def _send():
            return await self.send_message(
                chat_id, text, reply_to, parse_mode, message_thread_id
            )

        if fire_and_forget:
            self._limiter.fire_and_forget(_send)
            return None
        else:
            return await self._limiter.enqueue(_send)

    async def queue_edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: str | None = "MarkdownV2",
        fire_and_forget: bool = True,
    ) -> None:
        """Enqueue a message edit."""
        if not self._limiter:
            return await self.edit_message(chat_id, message_id, text, parse_mode)

        async def _edit():
            return await self.edit_message(chat_id, message_id, text, parse_mode)

        dedup_key = f"edit:{chat_id}:{message_id}"
        if fire_and_forget:
            self._limiter.fire_and_forget(_edit, dedup_key=dedup_key)
        else:
            await self._limiter.enqueue(_edit, dedup_key=dedup_key)

    async def queue_delete_message(
        self,
        chat_id: str,
        message_id: str,
        fire_and_forget: bool = True,
    ) -> None:
        """Enqueue a message delete."""
        if not self._limiter:
            return await self.delete_message(chat_id, message_id)

        async def _delete():
            return await self.delete_message(chat_id, message_id)

        dedup_key = f"del:{chat_id}:{message_id}"
        if fire_and_forget:
            self._limiter.fire_and_forget(_delete, dedup_key=dedup_key)
        else:
            await self._limiter.enqueue(_delete, dedup_key=dedup_key)

    async def queue_delete_messages(
        self,
        chat_id: str,
        message_ids: list[str],
        fire_and_forget: bool = True,
    ) -> None:
        """Enqueue a bulk delete (if supported) or a sequence of deletes."""
        if not message_ids:
            return

        if not self._limiter:
            return await self.delete_messages(chat_id, message_ids)

        async def _bulk():
            return await self.delete_messages(chat_id, message_ids)

        # Dedup by the chunk content; okay to be coarse here.
        dedup_key = f"del_bulk:{chat_id}:{hash(tuple(message_ids))}"
        if fire_and_forget:
            self._limiter.fire_and_forget(_bulk, dedup_key=dedup_key)
        else:
            await self._limiter.enqueue(_bulk, dedup_key=dedup_key)

    def fire_and_forget(self, task: Awaitable[Any]) -> None:
        """Execute a coroutine without awaiting it."""
        if asyncio.iscoroutine(task):
            _ = asyncio.create_task(task)
        else:
            _ = asyncio.ensure_future(task)

    async def _dispatch_to_execute(self, incoming: IncomingMessage) -> None:
        """Send status msg then fire-and-forget POST /execute. search_api replies directly."""
        status_msg_id = await self.queue_send_message(
            incoming.chat_id,
            "🔄 Processing\\.\\.\\.".replace("\\.", "\\."),
            reply_to=incoming.message_id,
            parse_mode="MarkdownV2",
            fire_and_forget=False,
            message_thread_id=incoming.message_thread_id,
        )
        try:
            async with httpx.AsyncClient(timeout=10.0) as _http:
                await _http.post(
                    "http://localhost:8765/execute",
                    json={
                        "task": incoming.text,
                        "from": f"{incoming.platform}:{incoming.user_id}",
                        "reply_chat_id": incoming.chat_id,
                        "status_message_id": str(status_msg_id) if status_msg_id else None,
                    },
                )
        except Exception as e:
            logger.warning("dispatch_to_execute failed chat={}: {}", incoming.chat_id, e)
            if status_msg_id:
                await self.queue_edit_message(
                    incoming.chat_id, str(status_msg_id),
                    "❌ Service unavailable\\.",
                    parse_mode="MarkdownV2",
                )

    def on_message(
        self,
        handler: Callable[[IncomingMessage], Awaitable[None]],
    ) -> None:
        """Register a message handler callback."""
        self._message_handler = handler

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    async def _on_start_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /start command."""
        if update.message:
            await update.message.reply_text("👋 Hello! I am the Claude Code Proxy Bot.")
        # We can also treat this as a message if we want it to trigger something
        await self._on_telegram_message(update, context)

    async def _on_telegram_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming updates."""
        if (
            not update.message
            or not update.message.text
            or not update.effective_user
            or not update.effective_chat
        ):
            return

        user_id = str(update.effective_user.id)
        chat_id = str(update.effective_chat.id)

        # Security check
        if self.allowed_user_id and user_id != str(self.allowed_user_id).strip():
            logger.warning(f"Unauthorized access attempt from {user_id}")
            return

        message_id = str(update.message.message_id)
        reply_to = (
            str(update.message.reply_to_message.message_id)
            if update.message.reply_to_message
            else None
        )
        thread_id = (
            str(update.message.message_thread_id)
            if getattr(update.message, "message_thread_id", None) is not None
            else None
        )
        raw_text = update.message.text or ""
        if self._log_raw_messaging_content:
            text_preview = raw_text[:80]
            if len(raw_text) > 80:
                text_preview += "..."
            logger.info(
                "TELEGRAM_MSG: chat_id={} message_id={} reply_to={} text_preview={!r}",
                chat_id,
                message_id,
                reply_to,
                text_preview,
            )
        else:
            logger.info(
                "TELEGRAM_MSG: chat_id={} message_id={} reply_to={} text_len={}",
                chat_id,
                message_id,
                reply_to,
                len(raw_text),
            )

        incoming = IncomingMessage(
            text=update.message.text,
            chat_id=chat_id,
            user_id=user_id,
            message_id=message_id,
            platform="telegram",
            reply_to_message_id=reply_to,
            message_thread_id=thread_id,
            raw_event=update,
        )

        try:
            await self._dispatch_to_execute(incoming)
        except Exception as e:
            if self._log_api_error_tracebacks:
                logger.error("Error dispatching message: {}", e)
            else:
                logger.error("Error dispatching message: exc_type={}", type(e).__name__)
            with contextlib.suppress(Exception):
                await self.send_message(
                    chat_id,
                    f"❌ *{escape_md_v2('Error:')}* {escape_md_v2(format_user_error_preview(e))}",
                    reply_to=incoming.message_id,
                    message_thread_id=thread_id,
                    parse_mode="MarkdownV2",
                )

    async def _on_telegram_voice(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming voice messages."""
        if (
            not update.message
            or not update.message.voice
            or not update.effective_user
            or not update.effective_chat
        ):
            return

        if not self._voice_note_enabled:
            await update.message.reply_text("Voice notes are disabled.")
            return

        user_id = str(update.effective_user.id)
        chat_id = str(update.effective_chat.id)

        if self.allowed_user_id and user_id != str(self.allowed_user_id).strip():
            logger.warning(f"Unauthorized voice access attempt from {user_id}")
            return

        thread_id = (
            str(update.message.message_thread_id)
            if getattr(update.message, "message_thread_id", None) is not None
            else None
        )
        status_msg_id = await self.queue_send_message(
            chat_id,
            format_status("⏳", "Transcribing voice note..."),
            reply_to=str(update.message.message_id),
            parse_mode="MarkdownV2",
            fire_and_forget=False,
            message_thread_id=thread_id,
        )

        message_id = str(update.message.message_id)
        await self._register_pending_voice(chat_id, message_id, str(status_msg_id))
        reply_to = (
            str(update.message.reply_to_message.message_id)
            if update.message.reply_to_message
            else None
        )

        voice = update.message.voice
        suffix = ".ogg"
        if voice.mime_type and "mpeg" in voice.mime_type:
            suffix = ".mp3"
        elif voice.mime_type and "mp4" in voice.mime_type:
            suffix = ".mp4"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            tg_file = await context.bot.get_file(voice.file_id)
            await tg_file.download_to_drive(custom_path=str(tmp_path))

            transcribed = await self._voice_transcription.transcribe(
                tmp_path,
                voice.mime_type or "audio/ogg",
                whisper_model=self._whisper_model,
                whisper_device=self._whisper_device,
            )

            if not await self._is_voice_still_pending(chat_id, message_id):
                await self.queue_delete_message(chat_id, str(status_msg_id))
                return

            await self._pending_voice.complete(chat_id, message_id, str(status_msg_id))

            incoming = IncomingMessage(
                text=transcribed,
                chat_id=chat_id,
                user_id=user_id,
                message_id=message_id,
                platform="telegram",
                reply_to_message_id=reply_to,
                message_thread_id=thread_id,
                raw_event=update,
                status_message_id=status_msg_id,
            )

            if self._log_raw_messaging_content:
                logger.info(
                    "TELEGRAM_VOICE: chat_id={} message_id={} transcribed={!r}",
                    chat_id,
                    message_id,
                    (
                        transcribed[:80] + "..."
                        if len(transcribed) > 80
                        else transcribed
                    ),
                )
            else:
                logger.info(
                    "TELEGRAM_VOICE: chat_id={} message_id={} transcribed_len={}",
                    chat_id,
                    message_id,
                    len(transcribed),
                )

            await self._dispatch_to_execute(incoming)
        except ValueError as e:
            await update.message.reply_text(format_user_error_preview(e))
        except ImportError as e:
            await update.message.reply_text(format_user_error_preview(e))
        except Exception as e:
            if self._log_api_error_tracebacks:
                logger.error("Voice transcription failed: {}", e)
            else:
                logger.error(
                    "Voice transcription failed: exc_type={}", type(e).__name__
                )
            await update.message.reply_text(
                "Could not transcribe voice note. Please try again or send text."
            )
        finally:
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)

    def _check_auth(self, update: "Update") -> tuple[str, str] | None:
        """Return (user_id, chat_id) if authorized, else None (and log warning)."""
        if not update.effective_user or not update.effective_chat:
            return None
        user_id = str(update.effective_user.id)
        chat_id = str(update.effective_chat.id)
        if self.allowed_user_id and user_id != str(self.allowed_user_id).strip():
            logger.warning("Unauthorized media access attempt from {}", user_id)
            return None
        return user_id, chat_id

    def _thread_id(self, update: "Update") -> str | None:
        return (
            str(update.message.message_thread_id)
            if update.message and getattr(update.message, "message_thread_id", None) is not None
            else None
        )

    async def _on_telegram_photo(
        self, update: "Update", context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming photos — analyse via Ollama vision and dispatch as text."""
        if not update.message or not update.message.photo:
            return
        auth = self._check_auth(update)
        if not auth:
            return
        user_id, chat_id = auth
        thread_id = self._thread_id(update)
        message_id = str(update.message.message_id)
        caption = update.message.caption or ""

        status_mid = await self.queue_send_message(
            chat_id,
            format_status("🖼️", "Analysing image\\.\\.\\."),
            reply_to=message_id,
            parse_mode="MarkdownV2",
            fire_and_forget=False,
            message_thread_id=thread_id,
        )

        from ..media_helpers import analyze_image_ollama

        # Use the largest available photo
        photo = update.message.photo[-1]
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            tg_file = await context.bot.get_file(photo.file_id)
            await tg_file.download_to_drive(custom_path=str(tmp_path))

            prompt = caption.strip() or "Describe this image in detail."
            description = await analyze_image_ollama(tmp_path, prompt)

            task_text = f"[Image from user]\n{description}"
            if caption:
                task_text += f"\n\nUser caption: {caption}"

            incoming = IncomingMessage(
                text=task_text,
                chat_id=chat_id,
                user_id=user_id,
                message_id=message_id,
                platform="telegram",
                message_thread_id=thread_id,
                raw_event=update,
                status_message_id=status_mid,
            )
            await self._dispatch_to_execute(incoming)
        except Exception as exc:
            logger.error("Photo handler failed: {}", exc)
            with contextlib.suppress(Exception):
                await self.queue_edit_message(
                    chat_id, str(status_mid),
                    "❌ Image analysis failed\\.",
                    parse_mode="MarkdownV2",
                )
        finally:
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)

    async def _on_telegram_video(
        self, update: "Update", context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming videos — extract frames, analyse via vision, dispatch as text."""
        if not update.message:
            return
        video = update.message.video or update.message.video_note
        if not video:
            return
        auth = self._check_auth(update)
        if not auth:
            return
        user_id, chat_id = auth
        thread_id = self._thread_id(update)
        message_id = str(update.message.message_id)
        caption = getattr(update.message, "caption", None) or ""

        status_mid = await self.queue_send_message(
            chat_id,
            format_status("🎬", "Analysing video frames\\.\\.\\."),
            reply_to=message_id,
            parse_mode="MarkdownV2",
            fire_and_forget=False,
            message_thread_id=thread_id,
        )

        from ..media_helpers import analyze_video_frames

        suffix = ".mp4"
        if hasattr(video, "mime_type") and video.mime_type:
            if "webm" in video.mime_type:
                suffix = ".webm"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            tg_file = await context.bot.get_file(video.file_id)
            await tg_file.download_to_drive(custom_path=str(tmp_path))

            prompt = caption.strip() or "Describe this video in detail."
            analysis = await analyze_video_frames(tmp_path, prompt)

            task_text = f"[Video from user]\n{analysis}"
            if caption:
                task_text += f"\n\nUser caption: {caption}"

            incoming = IncomingMessage(
                text=task_text,
                chat_id=chat_id,
                user_id=user_id,
                message_id=message_id,
                platform="telegram",
                message_thread_id=thread_id,
                raw_event=update,
                status_message_id=status_mid,
            )
            await self._dispatch_to_execute(incoming)
        except Exception as exc:
            logger.error("Video handler failed: {}", exc)
            with contextlib.suppress(Exception):
                await self.queue_edit_message(
                    chat_id, str(status_mid),
                    "❌ Video analysis failed\\.",
                    parse_mode="MarkdownV2",
                )
        finally:
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)

    async def _on_telegram_audio(
        self, update: "Update", context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle audio files — transcribe via STT and dispatch as text."""
        if not update.message or not update.message.audio:
            return
        auth = self._check_auth(update)
        if not auth:
            return
        user_id, chat_id = auth
        thread_id = self._thread_id(update)
        message_id = str(update.message.message_id)
        caption = update.message.caption or ""

        status_mid = await self.queue_send_message(
            chat_id,
            format_status("🎵", "Transcribing audio\\.\\.\\."),
            reply_to=message_id,
            parse_mode="MarkdownV2",
            fire_and_forget=False,
            message_thread_id=thread_id,
        )

        audio = update.message.audio
        mime = audio.mime_type or "audio/mpeg"
        suffix_map = {"audio/ogg": ".ogg", "audio/mp4": ".mp4", "audio/x-wav": ".wav",
                      "audio/wav": ".wav", "audio/mpeg": ".mp3", "audio/mp3": ".mp3"}
        suffix = suffix_map.get(mime, ".mp3")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            tg_file = await context.bot.get_file(audio.file_id)
            await tg_file.download_to_drive(custom_path=str(tmp_path))

            transcribed = await self._voice_transcription.transcribe(
                tmp_path, mime,
                whisper_model=self._whisper_model,
                whisper_device=self._whisper_device,
            )

            task_text = transcribed
            if caption:
                task_text = f"{caption}\n\n[Audio transcript: {transcribed}]"

            incoming = IncomingMessage(
                text=task_text,
                chat_id=chat_id,
                user_id=user_id,
                message_id=message_id,
                platform="telegram",
                message_thread_id=thread_id,
                raw_event=update,
                status_message_id=status_mid,
            )
            await self._dispatch_to_execute(incoming)
        except Exception as exc:
            logger.error("Audio handler failed: {}", exc)
            with contextlib.suppress(Exception):
                await self.queue_edit_message(
                    chat_id, str(status_mid),
                    "❌ Audio transcription failed\\. Please send text instead\\.",
                    parse_mode="MarkdownV2",
                )
        finally:
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)

    async def _on_telegram_document(
        self, update: "Update", context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle document uploads — extract text for PDFs, pass path for others."""
        if not update.message or not update.message.document:
            return
        auth = self._check_auth(update)
        if not auth:
            return
        user_id, chat_id = auth
        thread_id = self._thread_id(update)
        message_id = str(update.message.message_id)
        caption = update.message.caption or ""
        doc = update.message.document
        mime = doc.mime_type or ""
        file_name = doc.file_name or "document"

        status_mid = await self.queue_send_message(
            chat_id,
            format_status("📄", "Processing document\\.\\.\\."),
            reply_to=message_id,
            parse_mode="MarkdownV2",
            fire_and_forget=False,
            message_thread_id=thread_id,
        )

        suffix = Path(file_name).suffix or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            tg_file = await context.bot.get_file(doc.file_id)
            await tg_file.download_to_drive(custom_path=str(tmp_path))

            if mime == "application/pdf" or suffix.lower() == ".pdf":
                from ..media_helpers import extract_pdf_text
                extracted = extract_pdf_text(tmp_path)
                task_text = (
                    f"[PDF document: '{file_name}']\n\n"
                    f"Extracted text:\n{extracted}"
                )
            elif mime.startswith("text/") or suffix.lower() in (
                ".txt", ".md", ".csv", ".log", ".json", ".xml",
                ".yaml", ".yml", ".toml", ".ini", ".cfg",
            ):
                try:
                    extracted = tmp_path.read_text(encoding="utf-8", errors="replace")[:8000]
                    task_text = f"[Text document: '{file_name}']\n\n{extracted}"
                except Exception:
                    task_text = f"[Document: '{file_name}' saved at {tmp_path}. Ask user what to do with it.]"
            else:
                task_text = (
                    f"[Document: '{file_name}' (type: {mime or 'unknown'}) "
                    f"saved at {tmp_path}. Ask the user what they'd like to do with it.]"
                )

            if caption:
                task_text = f"{caption}\n\n{task_text}"

            incoming = IncomingMessage(
                text=task_text,
                chat_id=chat_id,
                user_id=user_id,
                message_id=message_id,
                platform="telegram",
                message_thread_id=thread_id,
                raw_event=update,
                status_message_id=status_mid,
            )
            await self._dispatch_to_execute(incoming)
        except Exception as exc:
            logger.error("Document handler failed: {}", exc)
            with contextlib.suppress(Exception):
                await self.queue_edit_message(
                    chat_id, str(status_mid),
                    "❌ Document processing failed\\.",
                    parse_mode="MarkdownV2",
                )
