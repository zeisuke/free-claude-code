"""Tests for providers.pool_stream.stream_with_pool."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from providers.circuit_breaker import ModelPool
from providers.exceptions import RateLimitError, ServiceUnavailableError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(timeout: float = 5.0) -> Any:
    """Return a minimal settings-like object."""
    settings = MagicMock()
    settings.model_pool_first_token_timeout = timeout
    settings.resolve_thinking.return_value = False
    return settings


def _make_pool(models: list[str]) -> ModelPool:
    return ModelPool(models)


def _make_registry(providers: dict[str, Any]) -> Any:
    """Return a mock ProviderRegistry that returns providers by id."""
    registry = MagicMock()

    def _get(provider_id: str, settings: Any) -> Any:
        if provider_id not in providers:
            raise KeyError(f"unknown provider: {provider_id}")
        return providers[provider_id]

    registry.get.side_effect = _get
    return registry


def _make_request(model: str = "claude-3-5-sonnet-20241022") -> Any:
    """Return a minimal Pydantic-like request mock with model_copy support."""
    req = MagicMock()
    req.model = model

    def _model_copy(update: dict, deep: bool = False) -> Any:
        copy = MagicMock()
        copy.model = update.get("model", model)
        copy.model_copy = req.model_copy
        return copy

    req.model_copy.side_effect = _model_copy
    return req


async def _immediate_gen(*chunks: str) -> AsyncIterator[str]:
    """Async generator that yields chunks immediately."""
    for chunk in chunks:
        yield chunk


async def _hanging_gen() -> AsyncIterator[str]:
    """Async generator that hangs indefinitely (simulates first-token timeout)."""
    await asyncio.sleep(9999)
    yield "never"  # pragma: no cover


async def _collect(gen: AsyncIterator[str]) -> list[str]:
    """Collect all chunks from an async generator."""
    return [chunk async for chunk in gen]


# ---------------------------------------------------------------------------
# 1. Happy path — first model responds immediately
# ---------------------------------------------------------------------------

class TestHappyPath:
    @pytest.mark.asyncio
    async def test_first_model_selected_when_immediate(self):
        model_ref = "open_router/minimax/minimax-m2.5:free"
        pool = _make_pool([model_ref])

        provider = MagicMock()
        provider.stream_response.return_value = _immediate_gen("chunk1", "chunk2")
        registry = _make_registry({"open_router": provider})
        settings = _make_settings()
        request = _make_request()

        from providers.pool_stream import stream_with_pool

        chunks = await _collect(stream_with_pool(request, pool, registry, settings))
        assert chunks == ["chunk1", "chunk2"]

    @pytest.mark.asyncio
    async def test_mark_used_called_on_success(self):
        model_ref = "open_router/minimax/minimax-m2.5:free"
        pool = _make_pool([model_ref])

        provider = MagicMock()
        provider.stream_response.return_value = _immediate_gen("ok")
        registry = _make_registry({"open_router": provider})
        settings = _make_settings()

        with patch.object(pool, "mark_used") as mock_mark:
            from providers.pool_stream import stream_with_pool
            await _collect(stream_with_pool(_make_request(), pool, registry, settings))
            mock_mark.assert_called_once_with(model_ref)

    @pytest.mark.asyncio
    async def test_penalise_not_called_on_success(self):
        model_ref = "open_router/google/gemma-4-31b-it:free"
        pool = _make_pool([model_ref])

        provider = MagicMock()
        provider.stream_response.return_value = _immediate_gen("hello")
        registry = _make_registry({"open_router": provider})
        settings = _make_settings()

        with patch.object(pool, "penalise") as mock_pen:
            from providers.pool_stream import stream_with_pool
            await _collect(stream_with_pool(_make_request(), pool, registry, settings))
            mock_pen.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Timeout fallback — first model hangs, second model succeeds
# ---------------------------------------------------------------------------

class TestTimeoutFallback:
    @pytest.mark.asyncio
    async def test_timeout_falls_back_to_next(self):
        model_a = "open_router/slow/model:free"
        model_b = "open_router/fast/model:free"
        pool = _make_pool([model_a, model_b])

        slow_provider = MagicMock()
        slow_provider.stream_response.return_value = _hanging_gen()
        fast_provider = MagicMock()
        fast_provider.stream_response.return_value = _immediate_gen("fast_chunk")

        registry = MagicMock()
        call_count = {"n": 0}

        def _get(provider_id: str, settings: Any) -> Any:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return slow_provider
            return fast_provider

        registry.get.side_effect = _get
        settings = _make_settings(timeout=0.05)  # 50ms timeout

        from providers.pool_stream import stream_with_pool
        chunks = await _collect(stream_with_pool(_make_request(), pool, registry, settings))
        assert chunks == ["fast_chunk"]

    @pytest.mark.asyncio
    async def test_penalise_is_timeout_called_on_hang(self):
        model_a = "open_router/slow/model:free"
        model_b = "open_router/fast/model:free"
        pool = _make_pool([model_a, model_b])

        slow_provider = MagicMock()
        slow_provider.stream_response.return_value = _hanging_gen()
        fast_provider = MagicMock()
        fast_provider.stream_response.return_value = _immediate_gen("ok")

        registry = MagicMock()
        call_count = {"n": 0}

        def _get(provider_id: str, settings: Any) -> Any:
            call_count["n"] += 1
            return slow_provider if call_count["n"] == 1 else fast_provider

        registry.get.side_effect = _get
        settings = _make_settings(timeout=0.05)

        with patch.object(pool, "penalise") as mock_pen:
            from providers.pool_stream import stream_with_pool
            await _collect(stream_with_pool(_make_request(), pool, registry, settings))
            mock_pen.assert_called_once_with(model_a, is_timeout=True)


# ---------------------------------------------------------------------------
# 3. 429 fallback — first model raises RateLimitError
# ---------------------------------------------------------------------------

class TestRateLimitFallback:
    @pytest.mark.asyncio
    async def test_rate_limit_falls_back_to_next(self):
        model_a = "open_router/limited/model:free"
        model_b = "open_router/available/model:free"
        pool = _make_pool([model_a, model_b])

        async def _rl_gen() -> AsyncIterator[str]:
            raise RateLimitError("rate limited")
            yield  # pragma: no cover

        rate_limited_provider = MagicMock()
        rate_limited_provider.stream_response.return_value = _rl_gen()
        ok_provider = MagicMock()
        ok_provider.stream_response.return_value = _immediate_gen("success")

        registry = MagicMock()
        call_count = {"n": 0}

        def _get(provider_id: str, settings: Any) -> Any:
            call_count["n"] += 1
            return rate_limited_provider if call_count["n"] == 1 else ok_provider

        registry.get.side_effect = _get
        settings = _make_settings()

        from providers.pool_stream import stream_with_pool
        chunks = await _collect(stream_with_pool(_make_request(), pool, registry, settings))
        assert chunks == ["success"]

    @pytest.mark.asyncio
    async def test_penalise_is_429_called_on_rate_limit(self):
        model_a = "open_router/limited/model:free"
        model_b = "open_router/available/model:free"
        pool = _make_pool([model_a, model_b])

        async def _rl_gen() -> AsyncIterator[str]:
            raise RateLimitError("rate limited")
            yield  # pragma: no cover

        rate_limited_provider = MagicMock()
        rate_limited_provider.stream_response.return_value = _rl_gen()
        ok_provider = MagicMock()
        ok_provider.stream_response.return_value = _immediate_gen("ok")

        registry = MagicMock()
        call_count = {"n": 0}

        def _get(provider_id: str, settings: Any) -> Any:
            call_count["n"] += 1
            return rate_limited_provider if call_count["n"] == 1 else ok_provider

        registry.get.side_effect = _get
        settings = _make_settings()

        with patch.object(pool, "penalise") as mock_pen:
            from providers.pool_stream import stream_with_pool
            await _collect(stream_with_pool(_make_request(), pool, registry, settings))
            mock_pen.assert_called_once_with(model_a, is_429=True)


# ---------------------------------------------------------------------------
# 4. All exhausted — raise ServiceUnavailableError
# ---------------------------------------------------------------------------

class TestAllExhausted:
    @pytest.mark.asyncio
    async def test_raises_service_unavailable_when_all_fail(self):
        models = [
            "open_router/model-a:free",
            "open_router/model-b:free",
        ]
        pool = _make_pool(models)

        async def _err_gen() -> AsyncIterator[str]:
            raise RateLimitError("all gone")
            yield  # pragma: no cover

        provider = MagicMock()
        provider.stream_response.return_value = _err_gen()

        registry = MagicMock()
        registry.get.return_value = provider
        settings = _make_settings()

        from providers.pool_stream import stream_with_pool

        with pytest.raises(ServiceUnavailableError, match="all pool models exhausted"):
            # Need to re-create generator each time because provider.stream_response
            # returns a different generator per call
            async def _rl_gen_factory() -> AsyncIterator[str]:
                raise RateLimitError("all gone")
                yield  # pragma: no cover

            provider.stream_response.side_effect = lambda *a, **kw: _rl_gen_factory()
            await _collect(stream_with_pool(_make_request(), pool, registry, settings))

    @pytest.mark.asyncio
    async def test_raises_service_unavailable_all_timeout(self):
        models = ["open_router/slow-a:free", "open_router/slow-b:free"]
        pool = _make_pool(models)

        provider = MagicMock()
        provider.stream_response.side_effect = lambda *a, **kw: _hanging_gen()

        registry = MagicMock()
        registry.get.return_value = provider
        settings = _make_settings(timeout=0.05)

        from providers.pool_stream import stream_with_pool

        with pytest.raises(ServiceUnavailableError, match="all pool models exhausted"):
            await _collect(stream_with_pool(_make_request(), pool, registry, settings))


# ---------------------------------------------------------------------------
# 5. Pool ordering respected — penalised model tried last, not skipped
# ---------------------------------------------------------------------------

class TestPoolOrdering:
    @pytest.mark.asyncio
    async def test_penalised_model_tried_last_not_skipped(self):
        import time

        model_a = "open_router/model-a:free"
        model_b = "open_router/model-b:free"
        pool = _make_pool([model_a, model_b])

        # Penalise model_a so it goes to end of pick() list
        pool.penalise(model_a, is_timeout=True)

        tried_order: list[str] = []

        provider = MagicMock()

        call_count = {"n": 0}

        def _stream_side_effect(*args: Any, **kwargs: Any) -> AsyncIterator[str]:
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First call should be model_b (model_a is penalised)
                return _immediate_gen("from_b")
            return _immediate_gen("from_a")

        provider.stream_response.side_effect = _stream_side_effect

        registry = MagicMock()
        call_registry = {"n": 0, "models_tried": []}

        def _get(provider_id: str, settings: Any) -> Any:
            return provider

        registry.get.side_effect = _get

        # Track which model was used by intercepting model_copy
        request = _make_request()
        models_in_stream_calls: list[str] = []

        original_model_copy = request.model_copy.side_effect

        def _model_copy_tracking(update: dict, deep: bool = False) -> Any:
            models_in_stream_calls.append(update.get("model", ""))
            return original_model_copy(update, deep=deep)

        request.model_copy.side_effect = _model_copy_tracking

        settings = _make_settings()

        from providers.pool_stream import stream_with_pool
        chunks = await _collect(stream_with_pool(request, pool, registry, settings))

        # Should have succeeded with model_b first (model_a was penalised)
        assert chunks == ["from_b"]
        # model_b's model part should have been tried first
        assert len(models_in_stream_calls) >= 1
        # The first model attempted should be model_b's model part ("model-b:free")
        assert models_in_stream_calls[0] == "model-b:free"

    @pytest.mark.asyncio
    async def test_penalised_model_still_tried_when_others_fail(self):
        """Penalised model at end is still tried when all hot models fail."""
        model_hot = "open_router/hot/model:free"
        model_cold = "open_router/cold/model:free"
        pool = _make_pool([model_hot, model_cold])
        pool.penalise(model_cold, is_timeout=True)

        call_count = {"n": 0}

        def _stream_side_effect(*args: Any, **kwargs: Any) -> AsyncIterator[str]:
            call_count["n"] += 1
            if call_count["n"] == 1:
                # hot model fails
                return _hanging_gen()
            # cold (penalised) model succeeds
            return _immediate_gen("cold_success")

        provider = MagicMock()
        provider.stream_response.side_effect = _stream_side_effect
        registry = MagicMock()
        registry.get.return_value = provider
        settings = _make_settings(timeout=0.05)

        from providers.pool_stream import stream_with_pool
        chunks = await _collect(stream_with_pool(_make_request(), pool, registry, settings))
        assert chunks == ["cold_success"]
        # Two models were attempted
        assert call_count["n"] == 2
