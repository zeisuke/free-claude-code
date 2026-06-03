"""Application services for the Claude-compatible API."""

from __future__ import annotations

import asyncio
import json
import traceback
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from config.settings import Settings
from core import circuit_breaker as _cb
from core.anthropic import get_token_count, get_user_facing_error_message
from core.anthropic.sse import ANTHROPIC_SSE_RESPONSE_HEADERS
from core.trace import api_messages_request_snapshot, trace_event, traced_async_stream
from providers.base import BaseProvider
from providers.circuit_breaker import ModelPool
from providers.exceptions import InvalidRequestError, ProviderError
from providers.pool_stream import stream_with_pool
from providers.registry import ProviderRegistry

from .model_router import ModelRouter
from .models.anthropic import MessagesRequest, TokenCountRequest
from .models.responses import TokenCountResponse
from .optimization_handlers import try_optimizations
from .web_tools.egress import WebFetchEgressPolicy
from .web_tools.request import (
    is_web_server_tool_request,
    openai_chat_upstream_server_tool_error,
)
from .web_tools.streaming import stream_web_server_tool_response

TokenCounter = Callable[[list[Any], str | list[Any] | None, list[Any] | None], int]

ProviderGetter = Callable[[str], BaseProvider]

# Providers that use ``/chat/completions`` + Anthropic-to-OpenAI conversion (not native Messages).
_OPENAI_CHAT_UPSTREAM_IDS = frozenset({"nvidia_nim", "opencode", "opencode_go"})


async def _accumulate_sse_to_json(stream: AsyncIterator[str]):
    """Accumulate SSE stream into a non-streaming Anthropic JSON Message response."""
    from fastapi.responses import JSONResponse
    import json as _json

    msg_id = None
    model = None
    stop_reason = "end_turn"
    text_parts: list[str] = []
    input_tokens = 0
    output_tokens = 0

    async for chunk in stream:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")
        for line in chunk.splitlines():
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data in ("", "[DONE]"):
                continue
            try:
                ev = _json.loads(data)
            except Exception:
                continue
            t = ev.get("type", "")
            if t == "message_start":
                msg = ev.get("message", {})
                msg_id = msg.get("id")
                model = msg.get("model")
                usage = msg.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
            elif t == "content_block_delta":
                delta = ev.get("delta", {})
                if delta.get("type") == "text_delta":
                    text_parts.append(delta.get("text", ""))
            elif t == "message_delta":
                delta = ev.get("delta", {})
                stop_reason = delta.get("stop_reason", stop_reason)
                usage = ev.get("usage", {})
                output_tokens = usage.get("output_tokens", 0)

    body = {
        "id": msg_id or f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "".join(text_parts)}],
        "model": model or "unknown",
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }
    return JSONResponse(content=body)


def anthropic_sse_streaming_response(
    body: AsyncIterator[str],
) -> StreamingResponse:
    """Return a :class:`StreamingResponse` for Anthropic-style SSE streams."""
    return StreamingResponse(
        body,
        media_type="text/event-stream",
        headers=ANTHROPIC_SSE_RESPONSE_HEADERS,
    )


def _http_status_for_unexpected_service_exception(_exc: BaseException) -> int:
    """HTTP status for uncaught non-provider failures (stable client contract)."""
    return 500


def _log_unexpected_service_exception(
    settings: Settings,
    exc: BaseException,
    *,
    context: str,
    request_id: str | None = None,
) -> None:
    """Log service-layer failures without echoing exception text unless opted in."""
    if settings.log_api_error_tracebacks:
        if request_id is not None:
            logger.error("{} request_id={}: {}", context, request_id, exc)
        else:
            logger.error("{}: {}", context, exc)
        logger.error(traceback.format_exc())
        return
    if request_id is not None:
        logger.error(
            "{} request_id={} exc_type={}",
            context,
            request_id,
            type(exc).__name__,
        )
    else:
        logger.error("{} exc_type={}", context, type(exc).__name__)


def _require_non_empty_messages(messages: list[Any]) -> None:
    if not messages:
        raise InvalidRequestError("messages cannot be empty")


_ALL_FAILED_SSE = (
    'event: error\ndata: {"type":"error","error":{"type":"overloaded_error",'
    '"message":"All pool models unavailable — retry later."}}\n\n'
)


async def _pool_stream(
    pool: list[str],
    base_request: MessagesRequest,
    input_tokens: int,
    request_id: str,
    provider_getter: ProviderGetter,
    settings: Settings,
) -> AsyncIterator[str]:
    """Try each hot pool model in order; fall back on hang or error SSE.

    A model is penalised (5 min cooldown) when:
    - No first SSE token within ``model_pool_first_token_timeout`` seconds, OR
    - The first token is an error-type SSE event (model returned an error).

    A 429 error SSE triggers a 15-min cooldown.
    """
    timeout = settings.model_pool_first_token_timeout
    ordered = _cb.pick_models(pool)
    # Respect the model router's decision: if the resolved model is in the pool, try it first.
    _req_model = base_request.model
    _pool_match = next((m for m in pool if Settings.parse_model_name(m) == _req_model), None)
    if _pool_match and ordered and ordered[0] != _pool_match:
        ordered = [_pool_match] + [m for m in ordered if m != _pool_match]
    last_err_chunk: str | None = None

    for model_ref in ordered:
        provider_id = Settings.parse_provider_type(model_ref)
        provider_model = Settings.parse_model_name(model_ref)
        try:
            provider = provider_getter(provider_id)
        except Exception:
            continue

        routed = base_request.model_copy(update={"model": provider_model}, deep=True)
        thinking = settings.resolve_thinking(base_request.model)
        gen = provider.stream_response(
            routed,
            input_tokens=input_tokens,
            request_id=request_id,
            thinking_enabled=thinking,
        )

        # --- probe: first token with timeout ---
        try:
            first = await asyncio.wait_for(gen.__anext__(), timeout=timeout)
        except StopAsyncIteration:
            _cb.mark_used(model_ref)
            return
        except asyncio.TimeoutError:
            logger.warning("CB: {} first-token timeout ({}s) → penalising", model_ref, timeout)
            _cb.penalise(model_ref, is_429=False)
            try:
                await gen.aclose()
            except Exception:
                pass
            continue
        except Exception as exc:
            logger.warning("CB: {} error before first token: {} → penalising", model_ref, exc)
            _cb.penalise(model_ref, is_429=False)
            try:
                await gen.aclose()
            except Exception:
                pass
            continue

        # --- inspect first token for error SSE ---
        is_error = "event: error" in first or (
            '"type":"error"' in first and '"type":"message"' not in first
        )
        if is_error:
            is_429 = "rate_limit" in first
            logger.warning("CB: {} returned error SSE (429={}) → penalising", model_ref, is_429)
            _cb.penalise(model_ref, is_429=is_429)
            last_err_chunk = first
            # drain and discard remaining error events
            try:
                async for _ in gen:
                    pass
            except Exception:
                pass
            continue

        # --- success: stream the rest ---
        _cb.mark_used(model_ref)
        logger.info("CB: {} selected for request_id={}", model_ref, request_id)
        yield first
        async for chunk in gen:
            yield chunk
        return

    # All pool models failed/cooling. Bypass pool: route directly through
    # claude_subprocess provider which has _is_fcc_active() → Ollama logic:
    #   FCC active   → Ollama (llama3.1:8b text / moondream vision)
    #   FCC inactive → claude -p → Ollama fallback on subprocess failure
    logger.warning(
        "CB: all pool models failed — bypassing pool, routing via claude_subprocess"
    )
    try:
        bypass_provider = provider_getter("claude_subprocess")
        raw_model = settings.model
        bypass_model = Settings.parse_model_name(raw_model) if "/" in raw_model else raw_model
        bypass_request = base_request.model_copy(update={"model": bypass_model}, deep=True)
        async for chunk in bypass_provider.stream_response(
            bypass_request, input_tokens=input_tokens, request_id=request_id
        ):
            yield chunk
        return
    except Exception as bypass_exc:
        logger.error("CB: claude_subprocess bypass failed: {}", bypass_exc)

    # all models failed — emit the last error or a generic one
    yield last_err_chunk or _ALL_FAILED_SSE


class ClaudeProxyService:
    """Coordinate request optimization, model routing, token count, and providers."""

    def __init__(
        self,
        settings: Settings,
        provider_getter: ProviderGetter,
        model_router: ModelRouter | None = None,
        token_counter: TokenCounter = get_token_count,
        pool: ModelPool | None = None,
        provider_registry: ProviderRegistry | None = None,
    ):
        self._settings = settings
        self._provider_getter = provider_getter
        self._model_router = model_router or ModelRouter(settings)
        self._token_counter = token_counter
        self._pool = pool
        self._provider_registry = provider_registry

    def create_message(self, request_data: MessagesRequest) -> object:
        """Create a message response or streaming response."""
        try:
            _require_non_empty_messages(request_data.messages)

            routed = self._model_router.resolve_messages_request(request_data)
            if routed.resolved.provider_id in _OPENAI_CHAT_UPSTREAM_IDS:
                tool_err = openai_chat_upstream_server_tool_error(
                    routed.request,
                    web_tools_enabled=self._settings.enable_web_server_tools,
                )
                if tool_err is not None:
                    raise InvalidRequestError(tool_err)

            if self._settings.enable_web_server_tools and is_web_server_tool_request(
                routed.request
            ):
                input_tokens = self._token_counter(
                    routed.request.messages, routed.request.system, routed.request.tools
                )
                trace_event(
                    stage="routing",
                    event="api.optimization.web_server_tool",
                    source="api",
                    model=routed.request.model,
                )
                egress = WebFetchEgressPolicy(
                    allow_private_network_targets=self._settings.web_fetch_allow_private_networks,
                    allowed_schemes=self._settings.web_fetch_allowed_scheme_set(),
                )
                return anthropic_sse_streaming_response(
                    stream_web_server_tool_response(
                        routed.request,
                        input_tokens=input_tokens,
                        web_fetch_egress=egress,
                        verbose_client_errors=self._settings.log_api_error_tracebacks,
                    ),
                )

            optimized = try_optimizations(routed.request, self._settings)
            if optimized is not None:
                trace_event(
                    stage="routing",
                    event="api.optimization.short_circuit",
                    source="api",
                    model=routed.request.model,
                )
                return optimized
            logger.debug("No optimization matched, routing to provider")

            provider = self._provider_getter(routed.resolved.provider_id)
            provider.preflight_stream(
                routed.request,
                thinking_enabled=routed.resolved.thinking_enabled,
            )

            trace_event(
                stage="routing",
                event="api.route.resolved",
                source="api",
                provider_id=routed.resolved.provider_id,
                provider_model=routed.resolved.provider_model,
                provider_model_ref=routed.resolved.provider_model_ref,
                gateway_model=routed.request.model,
                thinking_enabled=routed.resolved.thinking_enabled,
            )

            request_id = f"req_{uuid.uuid4().hex[:12]}"
            with logger.contextualize(request_id=request_id):
                trace_event(
                    stage="ingress",
                    event="api.request.received",
                    source="api",
                    message_count=len(routed.request.messages),
                    snapshot=api_messages_request_snapshot(routed.request),
                )

            input_tokens = self._token_counter(
                routed.request.messages, routed.request.system, routed.request.tools
            )

            if (
                self._pool is not None
                and self._provider_registry is not None
                and routed.resolved.provider_id == "open_router"
            ):
                stream = stream_with_pool(
                    routed.request,
                    self._pool,
                    self._provider_registry,
                    self._settings,
                    request_id=request_id,
                )
            else:
                settings_pool = self._settings.model_pool
                if settings_pool and routed.resolved.provider_id not in _OPENAI_CHAT_UPSTREAM_IDS:
                    stream = _pool_stream(
                        settings_pool,
                        routed.request,
                        input_tokens,
                        request_id,
                        self._provider_getter,
                        self._settings,
                    )
                else:
                    stream = provider.stream_response(
                        routed.request,
                        input_tokens=input_tokens,
                        request_id=request_id,
                        thinking_enabled=routed.resolved.thinking_enabled,
                    )
            return anthropic_sse_streaming_response(stream)

        except ProviderError:
            raise
        except Exception as e:
            _log_unexpected_service_exception(
                self._settings, e, context="CREATE_MESSAGE_ERROR"
            )
            raise HTTPException(
                status_code=_http_status_for_unexpected_service_exception(e),
                detail=get_user_facing_error_message(e),
            ) from e

    def count_tokens(self, request_data: TokenCountRequest) -> TokenCountResponse:
        """Count tokens for a request after applying configured model routing."""
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        with logger.contextualize(request_id=request_id):
            try:
                _require_non_empty_messages(request_data.messages)
                routed = self._model_router.resolve_token_count_request(request_data)
                tokens = self._token_counter(
                    routed.request.messages, routed.request.system, routed.request.tools
                )
                trace_event(
                    stage="routing",
                    event="api.route.resolved",
                    source="api",
                    kind="count_tokens",
                    provider_id=routed.resolved.provider_id,
                    provider_model=routed.resolved.provider_model,
                    provider_model_ref=routed.resolved.provider_model_ref,
                    gateway_model=routed.request.model,
                )
                trace_event(
                    stage="ingress",
                    event="api.count_tokens.completed",
                    source="api",
                    message_count=len(routed.request.messages),
                    input_tokens=tokens,
                    snapshot=api_messages_request_snapshot(routed.request),
                )
                return TokenCountResponse(input_tokens=tokens)
            except ProviderError:
                raise
            except Exception as e:
                _log_unexpected_service_exception(
                    self._settings,
                    e,
                    context="COUNT_TOKENS_ERROR",
                    request_id=request_id,
                )
                raise HTTPException(
                    status_code=_http_status_for_unexpected_service_exception(e),
                    detail=get_user_facing_error_message(e),
                ) from e
