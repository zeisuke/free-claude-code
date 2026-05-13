"""Tests for the /pool/status endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from providers.circuit_breaker import ModelPool


def _make_client_with_pool(pool=None):
    """Create a TestClient with optional pool set on app.state."""
    app = create_app()
    with (
        patch(
            "providers.registry.ProviderRegistry.validate_configured_models",
            new_callable=AsyncMock,
        ),
        patch("providers.registry.ProviderRegistry.start_model_list_refresh"),
        TestClient(app) as client,
    ):
        if pool is not None:
            app.state.pool = pool
        yield client, app


@pytest.fixture
def client_no_pool():
    """TestClient with no pool on app.state."""
    for client, _app in _make_client_with_pool(pool=None):
        yield client


@pytest.fixture
def client_with_pool():
    """TestClient with a real ModelPool on app.state."""
    pool = ModelPool(["open_router/model-a:free", "open_router/model-b:free"])
    for client, _app in _make_client_with_pool(pool=pool):
        yield client


AUTH = {"Authorization": "Bearer freecc"}


def test_pool_status_no_pool_returns_disabled(client_no_pool):
    response = client_no_pool.get("/pool/status", headers=AUTH)
    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is False
    assert data["model_pool"] == []
    assert data["hot_count"] == 0
    assert data["cooling_count"] == 0
    assert data["last_used"] is None


def test_pool_status_with_pool_returns_enabled(client_with_pool):
    response = client_with_pool.get("/pool/status", headers=AUTH)
    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is True


def test_pool_status_with_pool_includes_required_fields(client_with_pool):
    response = client_with_pool.get("/pool/status", headers=AUTH)
    assert response.status_code == 200
    data = response.json()
    assert "model_pool" in data
    assert "hot_count" in data
    assert "cooling_count" in data
    assert "last_used" in data


def test_pool_status_with_pool_model_pool_structure(client_with_pool):
    response = client_with_pool.get("/pool/status", headers=AUTH)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["model_pool"], list)
    assert len(data["model_pool"]) == 2
    for entry in data["model_pool"]:
        assert "model" in entry
        assert "rank" in entry
        assert "status" in entry
        assert entry["status"] in ("hot", "cooling")
        assert "cooldown_remaining_s" in entry


def test_pool_status_with_pool_hot_count_correct(client_with_pool):
    response = client_with_pool.get("/pool/status", headers=AUTH)
    assert response.status_code == 200
    data = response.json()
    # Fresh pool: all models are hot (no cooldowns set)
    assert data["hot_count"] == 2
    assert data["cooling_count"] == 0


def test_pool_status_requires_auth():
    app = create_app()
    with (
        patch(
            "providers.registry.ProviderRegistry.validate_configured_models",
            new_callable=AsyncMock,
        ),
        patch("providers.registry.ProviderRegistry.start_model_list_refresh"),
        TestClient(app) as client,
    ):
        # Patch settings to enforce auth
        from api.dependencies import get_settings
        from config.settings import Settings

        settings_with_auth = Settings.model_construct(
            model="nvidia_nim/z-ai/glm4.7",
            model_opus=None,
            model_sonnet=None,
            model_haiku=None,
            model_blocklist_raw="",
            model_pool_raw="",
            model_pool_first_token_timeout=8.0,
            anthropic_auth_token="freecc",
        )
        app.dependency_overrides[get_settings] = lambda: settings_with_auth

        response = client.get("/pool/status")
        assert response.status_code == 401

        app.dependency_overrides.clear()


def test_pool_status_with_valid_auth_passes():
    app = create_app()
    with (
        patch(
            "providers.registry.ProviderRegistry.validate_configured_models",
            new_callable=AsyncMock,
        ),
        patch("providers.registry.ProviderRegistry.start_model_list_refresh"),
        TestClient(app) as client,
    ):
        from api.dependencies import get_settings
        from config.settings import Settings

        settings_with_auth = Settings.model_construct(
            model="nvidia_nim/z-ai/glm4.7",
            model_opus=None,
            model_sonnet=None,
            model_haiku=None,
            model_blocklist_raw="",
            model_pool_raw="",
            model_pool_first_token_timeout=8.0,
            anthropic_auth_token="freecc",
        )
        app.dependency_overrides[get_settings] = lambda: settings_with_auth

        response = client.get("/pool/status", headers=AUTH)
        assert response.status_code == 200

        app.dependency_overrides.clear()
