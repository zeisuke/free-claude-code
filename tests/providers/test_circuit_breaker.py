"""Tests for providers.circuit_breaker.ModelPool."""

import threading
import time

import pytest

from providers.circuit_breaker import (
    DEFAULT_POOL,
    ModelPool,
    _COOLDOWN_429,
    _COOLDOWN_5XX,
    _COOLDOWN_TIMEOUT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pool(count: int = 3) -> ModelPool:
    """Return a small pool for tests that don't need specific model names."""
    models = [f"provider/model-{i}:free" for i in range(count)]
    return ModelPool(models)


def _models(count: int = 3) -> list[str]:
    return [f"provider/model-{i}:free" for i in range(count)]


# ---------------------------------------------------------------------------
# 1. pick() returns all models when none are cooling
# ---------------------------------------------------------------------------

class TestPickAllHot:
    def test_returns_all_models(self):
        pool = ModelPool(DEFAULT_POOL)
        result = pool.pick()
        assert sorted(result) == sorted(DEFAULT_POOL)

    def test_returns_correct_count(self):
        models = _models(5)
        pool = ModelPool(models)
        assert len(pool.pick()) == 5

    def test_hot_order_matches_rank(self):
        """When all models are hot, order must match insertion rank."""
        models = _models(4)
        pool = ModelPool(models)
        assert pool.pick() == models


# ---------------------------------------------------------------------------
# 2. Hot models appear before cooling models in pick()
# ---------------------------------------------------------------------------

class TestPickHotBeforeCooling:
    def test_penalised_model_moves_to_end(self, monkeypatch):
        models = _models(3)
        pool = ModelPool(models)
        pool.penalise(models[0])  # generic 5xx penalty

        result = pool.pick()
        # The penalised model must appear after the hot ones
        hot_part = result[: len(models) - 1]
        cooling_part = result[len(models) - 1 :]
        assert models[0] not in hot_part
        assert models[0] in cooling_part

    def test_multiple_hot_before_multiple_cooling(self, monkeypatch):
        models = _models(4)
        pool = ModelPool(models)
        pool.penalise(models[0])
        pool.penalise(models[2])

        result = pool.pick()
        hot_indices = [result.index(m) for m in [models[1], models[3]]]
        cooling_indices = [result.index(m) for m in [models[0], models[2]]]
        assert max(hot_indices) < min(cooling_indices)


# ---------------------------------------------------------------------------
# 3. Penalised model still appears in pick() (not removed)
# ---------------------------------------------------------------------------

class TestPenalisedModelNotRemoved:
    def test_model_still_in_pick_after_penalise(self):
        models = _models(3)
        pool = ModelPool(models)
        pool.penalise(models[1], is_429=True)

        result = pool.pick()
        assert models[1] in result
        assert len(result) == 3

    def test_all_models_present_even_if_all_cooling(self):
        models = _models(3)
        pool = ModelPool(models)
        for m in models:
            pool.penalise(m, is_timeout=True)

        result = pool.pick()
        assert sorted(result) == sorted(models)


# ---------------------------------------------------------------------------
# 4. After cooldown expires, model returns to hot in pick()
# ---------------------------------------------------------------------------

class TestCooldownExpiry:
    def test_model_becomes_hot_after_cooldown(self, monkeypatch):
        now = 1_000_000.0
        monkeypatch.setattr(time, "time", lambda: now)

        models = _models(2)
        pool = ModelPool(models)
        pool.penalise(models[0], is_timeout=True)  # 3-min cooldown

        # Still cooling at T+1
        monkeypatch.setattr(time, "time", lambda: now + 1)
        result = pool.pick()
        assert result.index(models[0]) > result.index(models[1])

        # Hot again after cooldown expires
        monkeypatch.setattr(time, "time", lambda: now + _COOLDOWN_TIMEOUT + 1)
        result = pool.pick()
        assert result[0] == models[0]  # back to rank-0 position


# ---------------------------------------------------------------------------
# 5. is_429 cooldown longer than is_timeout cooldown
# ---------------------------------------------------------------------------

class TestCooldownDurations:
    def test_429_longer_than_timeout(self):
        assert _COOLDOWN_429 > _COOLDOWN_TIMEOUT

    def test_429_longer_than_5xx(self):
        assert _COOLDOWN_429 > _COOLDOWN_5XX

    def test_penalise_429_sets_longer_cooldown(self, monkeypatch):
        now = 2_000_000.0
        monkeypatch.setattr(time, "time", lambda: now)

        models = _models(2)
        pool_429 = ModelPool(models)
        pool_to = ModelPool(models)

        pool_429.penalise(models[0], is_429=True)
        pool_to.penalise(models[0], is_timeout=True)

        status_429 = pool_429.get_status()
        status_to = pool_to.get_status()

        remaining_429 = next(
            e["cooldown_remaining_s"] for e in status_429["model_pool"] if e["model"] == models[0]
        )
        remaining_to = next(
            e["cooldown_remaining_s"] for e in status_to["model_pool"] if e["model"] == models[0]
        )
        assert remaining_429 > remaining_to

    def test_generic_5xx_longer_than_timeout(self, monkeypatch):
        """Generic (5xx) cooldown (_COOLDOWN_5XX=5min) is longer than is_timeout (3min)."""
        now = 2_000_000.0
        monkeypatch.setattr(time, "time", lambda: now)

        models = _models(2)
        pool_to = ModelPool(models)
        pool_5xx = ModelPool(models)

        pool_to.penalise(models[0], is_timeout=True)
        pool_5xx.penalise(models[0])  # generic = _COOLDOWN_5XX (300s)

        status_to = pool_to.get_status()
        status_5xx = pool_5xx.get_status()

        remaining_to = next(
            e["cooldown_remaining_s"] for e in status_to["model_pool"] if e["model"] == models[0]
        )
        remaining_5xx = next(
            e["cooldown_remaining_s"] for e in status_5xx["model_pool"] if e["model"] == models[0]
        )
        # _COOLDOWN_5XX (300s) > _COOLDOWN_TIMEOUT (180s)
        assert remaining_5xx > remaining_to


# ---------------------------------------------------------------------------
# 6. mark_used clears any existing cooldown
# ---------------------------------------------------------------------------

class TestMarkUsed:
    def test_mark_used_clears_cooldown(self, monkeypatch):
        now = 3_000_000.0
        monkeypatch.setattr(time, "time", lambda: now)

        models = _models(3)
        pool = ModelPool(models)
        pool.penalise(models[1], is_429=True)

        # Verify it's cooling
        result_before = pool.pick()
        assert result_before[-1] == models[1] or result_before.index(models[1]) > result_before.index(models[2])

        pool.mark_used(models[1])

        # Should be hot again, back at rank-1 position
        result_after = pool.pick()
        assert result_after[1] == models[1]

    def test_mark_used_sets_last_used(self, monkeypatch):
        now = 3_000_000.0
        monkeypatch.setattr(time, "time", lambda: now)

        models = _models(2)
        pool = ModelPool(models)
        pool.mark_used(models[0])

        status = pool.get_status()
        assert status["last_used"] == now

    def test_mark_used_on_already_hot_model_is_safe(self):
        models = _models(2)
        pool = ModelPool(models)
        pool.mark_used(models[0])  # no prior penalty — should not raise

        result = pool.pick()
        assert models[0] in result


# ---------------------------------------------------------------------------
# 7. get_status() hot_count + cooling_count == total models
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_counts_sum_to_total(self):
        models = _models(5)
        pool = ModelPool(models)
        pool.penalise(models[0])
        pool.penalise(models[3], is_429=True)

        status = pool.get_status()
        assert status["hot_count"] + status["cooling_count"] == len(models)

    def test_all_hot_initially(self):
        models = _models(4)
        pool = ModelPool(models)
        status = pool.get_status()
        assert status["hot_count"] == 4
        assert status["cooling_count"] == 0

    def test_per_model_dict_fields(self):
        models = _models(2)
        pool = ModelPool(models)
        status = pool.get_status()
        entry = status["model_pool"][0]
        assert "model" in entry
        assert "rank" in entry
        assert "status" in entry
        assert entry["status"] in ("hot", "cooling")
        assert "cooldown_remaining_s" in entry
        assert "cooldown_until" in entry

    def test_status_hot_string(self):
        models = _models(2)
        pool = ModelPool(models)
        status = pool.get_status()
        for entry in status["model_pool"]:
            assert entry["status"] == "hot"

    def test_status_cooling_string_after_penalise(self, monkeypatch):
        now = 4_000_000.0
        monkeypatch.setattr(time, "time", lambda: now)

        models = _models(2)
        pool = ModelPool(models)
        pool.penalise(models[0], is_429=True)

        status = pool.get_status()
        entry = next(e for e in status["model_pool"] if e["model"] == models[0])
        assert entry["status"] == "cooling"
        assert entry["cooldown_remaining_s"] > 0

    def test_last_used_reflects_mark_used(self, monkeypatch):
        now = 4_000_000.0
        monkeypatch.setattr(time, "time", lambda: now)

        models = _models(2)
        pool = ModelPool(models)
        pool.mark_used(models[1])

        status = pool.get_status()
        assert status["last_used"] == now


# ---------------------------------------------------------------------------
# 8. Pool with single model still works after penalise
# ---------------------------------------------------------------------------

class TestSingleModelPool:
    def test_pick_returns_single_model(self):
        pool = ModelPool(["solo/model:free"])
        assert pool.pick() == ["solo/model:free"]

    def test_penalise_single_model_still_in_pick(self):
        pool = ModelPool(["solo/model:free"])
        pool.penalise("solo/model:free", is_429=True)
        result = pool.pick()
        assert result == ["solo/model:free"]

    def test_mark_used_single_model(self):
        pool = ModelPool(["solo/model:free"])
        pool.penalise("solo/model:free", is_timeout=True)
        pool.mark_used("solo/model:free")
        status = pool.get_status()
        assert status["hot_count"] == 1
        assert status["cooling_count"] == 0

    def test_get_status_single_model(self):
        pool = ModelPool(["solo/model:free"])
        status = pool.get_status()
        assert len(status["model_pool"]) == 1
        assert status["hot_count"] + status["cooling_count"] == 1


# ---------------------------------------------------------------------------
# 9. Thread safety: concurrent penalise calls don't crash
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_penalise_no_crash(self):
        models = _models(5)
        pool = ModelPool(models)
        errors: list[Exception] = []

        def worker(model: str) -> None:
            try:
                for _ in range(50):
                    pool.penalise(model, is_429=True)
                    pool.mark_used(model)
                    pool.pick()
                    pool.get_status()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(m,)) for m in models]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_concurrent_penalise_and_pick_no_crash(self):
        models = _models(7)
        pool = ModelPool(models)
        errors: list[Exception] = []

        def penaliser() -> None:
            try:
                for m in models * 20:
                    pool.penalise(m, is_timeout=True)
            except Exception as exc:
                errors.append(exc)

        def picker() -> None:
            try:
                for _ in range(100):
                    pool.pick()
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=penaliser),
            threading.Thread(target=penaliser),
            threading.Thread(target=picker),
            threading.Thread(target=picker),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_concurrent_mark_used_no_crash(self):
        models = _models(4)
        pool = ModelPool(models)
        errors: list[Exception] = []

        def marker() -> None:
            try:
                for m in models * 30:
                    pool.mark_used(m)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=marker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
