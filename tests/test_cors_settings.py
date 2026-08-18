"""Verify browser-origin settings accepted by local and production deployments."""

import pytest

from horde_model_reference import HordeModelReferenceSettings


def test_comma_separated_cors_origins_are_usable_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept the documented comma-separated environment form as individual origins."""
    monkeypatch.setenv(
        "HORDE_MODEL_REFERENCE_CORS_ALLOWED_ORIGINS",
        "http://localhost:4200, http://127.0.0.1:4200",
    )

    settings = HordeModelReferenceSettings(_env_file=None)

    assert settings.cors_allowed_origins == ["http://localhost:4200", "http://127.0.0.1:4200"]


def test_json_cors_origin_array_remains_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept the JSON-array form used by list-aware deployment configuration."""
    monkeypatch.setenv(
        "HORDE_MODEL_REFERENCE_CORS_ALLOWED_ORIGINS",
        '["https://models.example", "https://admin.example"]',
    )

    settings = HordeModelReferenceSettings(_env_file=None)

    assert settings.cors_allowed_origins == ["https://models.example", "https://admin.example"]
