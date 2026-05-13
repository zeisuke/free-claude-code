"""Streaming async generator with circuit-breaker pool fallback.

Pool state is sourced from the search_api service (http://localhost:8765) when
reachable; the local ModelPool is used transparently as fallback.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
from loguru import logger

from config.settings import Settings
from providers.circuit_breaker import ModelPool
from providers.exceptions import RateLimitError, ServiceUnavailableError
from providers.registry import ProviderRegistry

_SEARCH_API_BASE = "http://localhost:8765"
_SEARCH_API_TIMEOUT = httpx.Timeout(connect=0.5, read=1.0, write=1.0, pool=1.0)

# The local ModelPool uses "open_router/<model>" refs; search_api uses bare "<model>" refs.
_POOL_PREFIX = "open_router/"


def _strip_prefix(model_ref: str) -> str:
    """Strip the open_router/ prefix for search_api calls."""
    if model_ref.startswith(_POOL_PREFIX):
        return model_ref[len(_POOL_PREFIX):]
    return model_ref


def _add_prefix(model_id: str) -> str:
    """Re-add the open_router/ prefix when mapping search_api responses back."""
    if not model_id.startswith(_POOL_PREFIX):
        return _POOL_PREFIX + model_id
    return model_id


async def _pick_models(pool: ModelPool) -> list[str]:
    """Return ordered model list from search_api, falling back to pool.pick()."""
    try:
        async with httpx.AsyncClient(timeout=_SEARCH_API_TIMEOUT) as client:
            resp = await client.get(f"{_SEARCH_API_BASE}/openrouter/pool")
            resp.raise_for_status()
            data = resp.json()
            models = data.get("models", [])
            if models:
                return [_add_prefix(m) for m in models]
    except Exception as exc:
        logger.debug("pool_stream: search_api pick unavailable ({}), using local pool", type(exc).__name__)
    return pool.pick()


async def _penalise(pool: ModelPool, model_ref: str, *, is_429: bool = False, is_timeout: bool = False) -> None:
    """Penalise via search_api AND local pool (both always updated)."""
    try:
        async with httpx.AsyncClient(timeout=_SEARCH_API_TIMEOUT) as client:
            resp = await client.post(
                f"{_SEARCH_API_BASE}/openrouter/penalize",
                json={"model": _strip_prefix(model_ref), "is_429": is_429, "is_timeout": is_timeout},
            )
            resp.raise_for_status()
    except Exception as exc:
        logger.debug("pool_stream: search_api penalise unavailable ({})", type(exc).__name__)
    kwargs: dict[str, bool] = {}
    if is_429:
        kwargs["is_429"] = True
    if is_timeout:
        kwargs["is_timeout"] = True
    pool.penalise(model_ref, **kwargs)


async def _mark_used(pool: ModelPool, model_ref: str) -> None:
    """Mark model used via search_api AND local pool (both always updated)."""
    try:
        async with httpx.AsyncClient(timeout=_SEARCH_API_TIMEOUT) as client:
            resp = await client.post(
                f"{_SEARCH_API_BASE}/openrouter/mark_used",
                json={"model": _strip_prefix(model_ref)},
            )
            resp.raise_for_status()
    except Exception as exc:
        logger.debug("pool_stream: search_api mark_used unavailable ({})", type(exc).__name__)
    pool.mark_used(model_ref)


async def stream_with_pool(
    request: Any,
    pool: ModelPool,
    provider_registry: ProviderRegistry,
    settings: Settings,
    *,
    request_id: str | None = None,
) -> AsyncIterator[str]:
    """Try each model in *pool* in priority order; fall back on timeout or error.

    Algorithm:
    1. Call ``_pick_models()`` — tries search_api GET /openrouter/pool, falls back to pool.pick().
    2. For each ``model_ref`` (e.g. ``"open_router/minimax/minimax-m2.5:free"``):
       a. Parse ``provider_id`` and ``provider_model``.
       b. Build a copy of ``request`` with ``model`` set to ``provider_model``.
       c. Get the provider; skip on any exception.
       d. Probe first chunk with ``model_pool_first_token_timeout``-second deadline.
       e. On timeout: penalise(is_timeout=True), close stream, try next model.
       f. On RateLimitError: penalise(is_429=True), try next model.
       g. On any other exception: penalise generically, try next model.
       h. On success: mark_used, yield first chunk, stream remainder, return.
    3. If all models exhausted: raise ServiceUnavailableError.
    """
    ordered = await _pick_models(pool)

    for model_ref in ordered:
        # Parse "provider_id/rest/of/model"
        provider_id, provider_model = model_ref.split("/", 1)

        # Build request copy with model set to the provider-specific name
        model_request = request.model_copy(update={"model": provider_model}, deep=True)

        # Resolve provider
        try:
            provider = provider_registry.get(provider_id, settings)
        except Exception as exc:
            logger.warning(
                "pool_stream: skipping {} — provider lookup failed: {}",
                model_ref,
                type(exc).__name__,
            )
            continue

        thinking = settings.resolve_thinking(request.model)
        stream = provider.stream_response(
            model_request,
            request_id=request_id,
            thinking_enabled=thinking,
        )

        # Probe: await first chunk with timeout
        try:
            first_chunk = await asyncio.wait_for(
                stream.__anext__(),
                timeout=settings.model_pool_first_token_timeout,
            )
        except StopAsyncIteration:
            # Empty stream counts as success
            await _mark_used(pool, model_ref)
            logger.info(
                "pool_stream: {} — empty stream (mark_used)", model_ref
            )
            return
        except asyncio.TimeoutError:
            logger.warning(
                "pool_stream: {} first-token timeout ({}s) → penalise(is_timeout=True)",
                model_ref,
                settings.model_pool_first_token_timeout,
            )
            await _penalise(pool, model_ref, is_timeout=True)
            try:
                await stream.aclose()
            except Exception:
                pass
            continue
        except RateLimitError as exc:
            logger.warning(
                "pool_stream: {} RateLimitError before first token → penalise(is_429=True): {}",
                model_ref,
                exc,
            )
            await _penalise(pool, model_ref, is_429=True)
            continue
        except Exception as exc:
            logger.warning(
                "pool_stream: {} error before first token → penalise: {}",
                model_ref,
                exc,
            )
            await _penalise(pool, model_ref)
            try:
                await stream.aclose()
            except Exception:
                pass
            continue

        # Success path — stream the rest
        await _mark_used(pool, model_ref)
        logger.info(
            "pool_stream: {} selected request_id={}",
            model_ref,
            request_id,
        )
        yield first_chunk
        async for chunk in stream:
            yield chunk
        return

    raise ServiceUnavailableError("all pool models exhausted")
