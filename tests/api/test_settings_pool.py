"""Tests for model pool configuration fields on Settings."""

import pytest

from config.settings import Settings


def _settings(**kwargs) -> Settings:
    """Construct a Settings instance via model_construct to bypass env-file loading."""
    defaults = dict(
        model="nvidia_nim/z-ai/glm4.7",
        model_opus=None,
        model_sonnet=None,
        model_haiku=None,
        model_blocklist_raw="",
        model_pool_raw="",
        model_pool_first_token_timeout=8.0,
        anthropic_auth_token="",
    )
    defaults.update(kwargs)
    return Settings.model_construct(**defaults)


def test_model_pool_empty_when_not_set():
    s = _settings()
    assert s.model_pool == []


def test_model_pool_parses_comma_separated_string():
    s = _settings(
        model_pool_raw="open_router/minimax/minimax-m2.5:free,open_router/google/gemma-4-31b-it:free"
    )
    assert s.model_pool == [
        "open_router/minimax/minimax-m2.5:free",
        "open_router/google/gemma-4-31b-it:free",
    ]


def test_model_pool_strips_whitespace():
    s = _settings(
        model_pool_raw="  open_router/minimax/minimax-m2.5:free , open_router/google/gemma-4-31b-it:free  "
    )
    assert s.model_pool == [
        "open_router/minimax/minimax-m2.5:free",
        "open_router/google/gemma-4-31b-it:free",
    ]


def test_model_pool_ignores_empty_entries_from_trailing_comma():
    s = _settings(model_pool_raw="open_router/minimax/minimax-m2.5:free,")
    assert s.model_pool == ["open_router/minimax/minimax-m2.5:free"]


def test_model_pool_first_token_timeout_defaults_to_8():
    s = _settings()
    assert s.model_pool_first_token_timeout == 8.0


def test_model_pool_first_token_timeout_overridable():
    s = _settings(model_pool_first_token_timeout=15.5)
    assert s.model_pool_first_token_timeout == 15.5


def test_model_pool_env_var_construction(monkeypatch):
    """Verify that MODEL_POOL env var is picked up by Settings()."""
    monkeypatch.setenv("MODEL_POOL", "open_router/x/model-a:free,open_router/y/model-b:free")
    monkeypatch.setenv("MODEL_POOL_FIRST_TOKEN_TIMEOUT", "12.0")
    s = Settings()
    assert s.model_pool == ["open_router/x/model-a:free", "open_router/y/model-b:free"]
    assert s.model_pool_first_token_timeout == 12.0
