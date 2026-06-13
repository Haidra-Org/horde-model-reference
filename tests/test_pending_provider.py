"""Tests for :class:`PendingModelProvider` (the REPLICA-side beta model source)."""

from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from horde_model_reference import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import ImageGenerationModelRecord
from horde_model_reference.providers.pending_provider import PENDING_SOURCE_ID, PendingModelProvider

_PRIMARY_URL = "https://primary.example/api"
_CATEGORY = MODEL_REFERENCE_CATEGORY.image_generation
_PENDING_URL = f"{_PRIMARY_URL}/model_references/v2/{_CATEGORY.value}/pending"


def _provider(**kwargs: object) -> PendingModelProvider:
    params: dict[str, object] = {
        "primary_api_url": _PRIMARY_URL,
        "apikey": "secret-key",
        "categories": {_CATEGORY},
        "retry_max_attempts": 1,
    }
    params.update(kwargs)
    return PendingModelProvider(**params)  # type: ignore[arg-type]


def test_source_id_and_categories() -> None:
    """The provider advertises the ``"pending"`` source and its configured categories."""
    provider = _provider()
    assert provider.source_id == PENDING_SOURCE_ID
    assert provider.provided_categories() == {_CATEGORY}


def test_fetch_category_returns_validated_records_and_sends_apikey(httpx_mock: HTTPXMock) -> None:
    """A 200 response is parsed into records, with the model name injected and apikey sent."""
    httpx_mock.add_response(
        url=_PENDING_URL,
        json={"beta_model": {"baseline": "stable_diffusion_xl", "nsfw": True}},
    )
    provider = _provider()

    records = provider.fetch_category(_CATEGORY)

    assert records is not None
    record = records["beta_model"]
    assert isinstance(record, ImageGenerationModelRecord)
    assert record.name == "beta_model"
    assert record.nsfw is True

    request = httpx_mock.get_requests()[0]
    assert request.headers["apikey"] == "secret-key"


def test_fetch_category_skips_invalid_records(httpx_mock: HTTPXMock) -> None:
    """Records that fail validation are skipped; valid ones are still returned."""
    httpx_mock.add_response(
        url=_PENDING_URL,
        json={
            "good": {"baseline": "stable_diffusion_1", "nsfw": False},
            "bad": {"baseline": "NotARealBaseline"},
        },
    )
    provider = _provider()

    records = provider.fetch_category(_CATEGORY)

    assert records is not None
    assert set(records) == {"good"}


def test_fetch_category_non_200_returns_none(httpx_mock: HTTPXMock) -> None:
    """A non-retryable error status yields ``None`` rather than raising."""
    httpx_mock.add_response(url=_PENDING_URL, status_code=403)
    provider = _provider()
    assert provider.fetch_category(_CATEGORY) is None


def test_fetch_unserved_category_returns_none_without_request() -> None:
    """A category the provider does not serve returns ``None`` and makes no HTTP call."""
    provider = _provider()
    # No httpx_mock response registered: a request here would raise.
    assert provider.fetch_category(MODEL_REFERENCE_CATEGORY.text_generation) is None


def test_fetch_category_caches_within_ttl(httpx_mock: HTTPXMock) -> None:
    """A second fetch within the TTL is served from cache (no second request)."""
    httpx_mock.add_response(
        url=_PENDING_URL,
        json={"beta_model": {"baseline": "stable_diffusion_1", "nsfw": False}},
    )
    provider = _provider(cache_ttl_seconds=300)

    first = provider.fetch_category(_CATEGORY)
    second = provider.fetch_category(_CATEGORY)

    assert first is not None
    assert second is not None
    assert set(first) == set(second) == {"beta_model"}
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.asyncio
async def test_fetch_category_async_parity(httpx_mock: HTTPXMock) -> None:
    """The async fetch path returns the same validated records."""
    httpx_mock.add_response(
        url=_PENDING_URL,
        json={"beta_model": {"baseline": "stable_diffusion_1", "nsfw": False}},
    )
    provider = _provider()

    records = await provider.fetch_category_async(_CATEGORY)

    assert records is not None
    assert set(records) == {"beta_model"}
