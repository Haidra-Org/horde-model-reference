from pathlib import Path
from typing import Any, cast, override

import httpx
import pytest

from horde_model_reference import ReplicateMode
from horde_model_reference.backends.github_backend import GitHubBackend
from horde_model_reference.backends.http_backend import HTTPBackend
from horde_model_reference.backends.replica_backend_base import ReplicaBackendBase
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


class StubGitHubBackend(ReplicaBackendBase):
    """GitHub backend stub wired for HTTP backend tests."""

    def __init__(
        self,
        responses: dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None],
        legacy_responses: dict[MODEL_REFERENCE_CATEGORY, tuple[dict[str, Any] | None, str | None]] | None = None,
    ) -> None:
        """Initialize with a mapping of category to response data."""
        super().__init__(mode=ReplicateMode.REPLICA, cache_ttl_seconds=None)
        self._responses = responses
        self._legacy_responses = legacy_responses or {}
        self.sync_calls = 0
        self.async_calls = 0
        self.legacy_json_calls = 0
        self.legacy_json_string_calls = 0

    @override
    def fetch_category(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        self.sync_calls += 1
        return self._responses.get(category)

    @override
    def fetch_all_categories(
        self,
        *,
        force_refresh: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        return {
            category: self.fetch_category(category, force_refresh=force_refresh)
            for category in MODEL_REFERENCE_CATEGORY
        }

    @override
    async def fetch_category_async(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        self.async_calls += 1
        return self._responses.get(category)

    @override
    async def fetch_all_categories_async(
        self,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        force_refresh: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        return {
            category: await self.fetch_category_async(
                category,
                httpx_client=httpx_client,
                force_refresh=force_refresh,
            )
            for category in MODEL_REFERENCE_CATEGORY
        }

    @override
    def get_category_file_path(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        return None

    @override
    def get_all_category_file_paths(self) -> dict[MODEL_REFERENCE_CATEGORY, Path | None]:
        return {category: None for category in MODEL_REFERENCE_CATEGORY}

    @override
    def get_legacy_json(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        redownload: bool = False,
    ) -> dict[str, Any] | None:
        self.legacy_json_calls += 1
        legacy_data = self._legacy_responses.get(category)
        return legacy_data[0] if legacy_data else None

    @override
    def get_legacy_json_string(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        redownload: bool = False,
    ) -> str | None:
        self.legacy_json_string_calls += 1
        legacy_data = self._legacy_responses.get(category)
        return legacy_data[1] if legacy_data else None


class DummyResponse:
    """Minimal httpx.Response stub."""

    def __init__(self, status_code: int, payload: dict[str, Any] | None, text: str | None = None) -> None:
        """Initialize with a status code and optional JSON payload."""
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or ("{" if payload else "")

    def json(self) -> dict[str, Any]:
        """Return the preset payload."""
        return self._payload


def test_http_backend_primary_success_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure PRIMARY hits are cached and counted once."""
    category = MODEL_REFERENCE_CATEGORY.image_generation
    github_stub = StubGitHubBackend({})
    backend = HTTPBackend(
        primary_api_url="https://primary",
        github_backend=cast(GitHubBackend, github_stub),
        cache_ttl_seconds=60,
    )
    current_time = 1_000.0

    def fake_time() -> float:
        return current_time

    monkeypatch.setattr("horde_model_reference.backends.replica_backend_base.time.time", fake_time)

    calls = {"count": 0}

    def fake_get(url: str, timeout: float) -> DummyResponse:
        calls["count"] += 1
        return DummyResponse(200, {"source": "primary"})

    monkeypatch.setattr("horde_model_reference.backends.http_backend.httpx.get", fake_get)

    first = backend.fetch_category(category)
    second = backend.fetch_category(category)

    assert first == {"source": "primary"}
    assert second == {"source": "primary"}
    assert calls["count"] == 1
    assert backend.get_statistics()["primary_hits"] == 1
    assert github_stub.sync_calls == 0


def test_http_backend_fallback_when_primary_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallback should defer to GitHub backend when PRIMARY errors."""
    category = MODEL_REFERENCE_CATEGORY.image_generation
    fallback_payload = {"source": "github"}
    github_stub = StubGitHubBackend({category: fallback_payload})
    backend = HTTPBackend(
        primary_api_url="https://primary",
        github_backend=cast(GitHubBackend, github_stub),
        cache_ttl_seconds=60,
    )

    def fake_get(url: str, timeout: float) -> DummyResponse:
        return DummyResponse(500, None)

    monkeypatch.setattr("horde_model_reference.backends.http_backend.httpx.get", fake_get)

    result = backend.fetch_category(category)

    assert result == fallback_payload
    assert github_stub.sync_calls == 1
    assert backend.get_statistics()["github_fallbacks"] == 1


def test_http_backend_disable_fallback_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disabling fallback should surface PRIMARY failures."""
    category = MODEL_REFERENCE_CATEGORY.image_generation
    github_stub = StubGitHubBackend({category: {"source": "github"}})
    backend = HTTPBackend(
        primary_api_url="https://primary",
        github_backend=cast(GitHubBackend, github_stub),
        cache_ttl_seconds=60,
        enable_github_fallback=False,
    )

    def fake_get(url: str, timeout: float) -> DummyResponse:
        return DummyResponse(503, None)

    monkeypatch.setattr("horde_model_reference.backends.http_backend.httpx.get", fake_get)

    assert backend.fetch_category(category) is None
    assert github_stub.sync_calls == 0


def test_http_backend_force_refresh_triggers_refetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """force_refresh should bypass cache and re-query PRIMARY."""
    category = MODEL_REFERENCE_CATEGORY.image_generation
    github_stub = StubGitHubBackend({})
    backend = HTTPBackend(
        primary_api_url="https://primary",
        github_backend=cast(GitHubBackend, github_stub),
        cache_ttl_seconds=60,
    )
    current_time = 2_000.0

    def fake_time() -> float:
        return current_time

    monkeypatch.setattr("horde_model_reference.backends.replica_backend_base.time.time", fake_time)

    payloads = [{"version": 1}, {"version": 2}]
    calls = {"index": 0}

    def fake_get(url: str, timeout: float) -> DummyResponse:
        response = payloads[calls["index"]]
        calls["index"] += 1
        return DummyResponse(200, response)

    monkeypatch.setattr("horde_model_reference.backends.http_backend.httpx.get", fake_get)

    first = backend.fetch_category(category)
    cached = backend.fetch_category(category)
    refreshed = backend.fetch_category(category, force_refresh=True)

    assert first == {"version": 1}
    assert cached == {"version": 1}
    assert refreshed == {"version": 2}
    assert calls["index"] == 2


@pytest.mark.asyncio
async def test_http_backend_async_caches_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """Async fetches should populate cache and skip redundant PRIMARY calls."""
    category = MODEL_REFERENCE_CATEGORY.image_generation
    github_stub = StubGitHubBackend({})
    backend = HTTPBackend(
        primary_api_url="https://primary",
        github_backend=cast(GitHubBackend, github_stub),
        cache_ttl_seconds=60,
    )
    current_time = 3_000.0

    def fake_time() -> float:
        return current_time

    monkeypatch.setattr("horde_model_reference.backends.replica_backend_base.time.time", fake_time)

    calls = {"count": 0}

    async def fake_fetch_async(
        self: HTTPBackend,
        category: MODEL_REFERENCE_CATEGORY,
        client: httpx.AsyncClient,
    ) -> dict[str, Any] | None:
        calls["count"] += 1
        return {"async": calls["count"]}

    monkeypatch.setattr(HTTPBackend, "_fetch_from_primary_async", fake_fetch_async)

    dummy_client = cast(httpx.AsyncClient, object())
    first = await backend.fetch_category_async(category, httpx_client=dummy_client)
    second = await backend.fetch_category_async(category, httpx_client=dummy_client)

    assert first == {"async": 1}
    assert second == {"async": 1}
    assert calls["count"] == 1
    assert github_stub.async_calls == 0


def test_http_backend_legacy_json_primary_success_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy JSON from PRIMARY should be cached and counted."""
    category = MODEL_REFERENCE_CATEGORY.image_generation
    github_stub = StubGitHubBackend({})
    backend = HTTPBackend(
        primary_api_url="https://primary",
        github_backend=cast(GitHubBackend, github_stub),
        cache_ttl_seconds=60,
    )
    current_time = 4_000.0

    def fake_time() -> float:
        return current_time

    monkeypatch.setattr("horde_model_reference.backends.replica_backend_base.time.time", fake_time)

    calls = {"count": 0}
    legacy_dict = {"legacy": "data"}
    legacy_string = '{"legacy": "data"}'

    def fake_get(url: str, timeout: float) -> DummyResponse:
        calls["count"] += 1
        return DummyResponse(200, legacy_dict, legacy_string)

    monkeypatch.setattr("horde_model_reference.backends.http_backend.httpx.get", fake_get)

    first = backend.get_legacy_json(category)
    second = backend.get_legacy_json(category)
    string_result = backend.get_legacy_json_string(category)

    assert first == legacy_dict
    assert second == legacy_dict
    assert string_result == legacy_string
    assert calls["count"] == 1  # Only one call due to caching
    assert backend.get_statistics()["primary_hits"] == 1
    assert github_stub.legacy_json_calls == 0


def test_http_backend_legacy_json_fallback_to_github(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy JSON should fallback to GitHub when PRIMARY fails."""
    category = MODEL_REFERENCE_CATEGORY.image_generation
    legacy_dict = {"github": "legacy"}
    legacy_string = '{"github": "legacy"}'
    github_stub = StubGitHubBackend({}, {category: (legacy_dict, legacy_string)})
    backend = HTTPBackend(
        primary_api_url="https://primary",
        github_backend=cast(GitHubBackend, github_stub),
        cache_ttl_seconds=60,
    )

    def fake_get(url: str, timeout: float) -> DummyResponse:
        return DummyResponse(500, None)

    monkeypatch.setattr("horde_model_reference.backends.http_backend.httpx.get", fake_get)

    result = backend.get_legacy_json(category)
    string_result = backend.get_legacy_json_string(category)

    assert result == legacy_dict
    assert string_result == legacy_string
    assert github_stub.legacy_json_calls == 1
    # String result comes from cache after first fetch, so only 1 call
    assert github_stub.legacy_json_string_calls == 1
    # Only 1 fallback because string is cached from first call
    assert backend.get_statistics()["github_fallbacks"] == 1


def test_http_backend_legacy_json_redownload_bypasses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """redownload=True should bypass cache for legacy JSON."""
    category = MODEL_REFERENCE_CATEGORY.image_generation
    github_stub = StubGitHubBackend({})
    backend = HTTPBackend(
        primary_api_url="https://primary",
        github_backend=cast(GitHubBackend, github_stub),
        cache_ttl_seconds=60,
    )
    current_time = 5_000.0

    def fake_time() -> float:
        return current_time

    monkeypatch.setattr("horde_model_reference.backends.replica_backend_base.time.time", fake_time)

    payloads = [{"version": 1}, {"version": 2}]
    calls = {"index": 0}

    def fake_get(url: str, timeout: float) -> DummyResponse:
        response = payloads[calls["index"]]
        calls["index"] += 1
        return DummyResponse(200, response, str(response))

    monkeypatch.setattr("horde_model_reference.backends.http_backend.httpx.get", fake_get)

    first = backend.get_legacy_json(category)
    cached = backend.get_legacy_json(category)
    refreshed = backend.get_legacy_json(category, redownload=True)

    assert first == {"version": 1}
    assert cached == {"version": 1}
    assert refreshed == {"version": 2}
    assert calls["index"] == 2


def test_http_backend_legacy_json_disable_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disabling fallback should return None for legacy JSON when PRIMARY fails."""
    category = MODEL_REFERENCE_CATEGORY.image_generation
    github_stub = StubGitHubBackend({}, {category: ({"github": "data"}, '{"github": "data"}')})
    backend = HTTPBackend(
        primary_api_url="https://primary",
        github_backend=cast(GitHubBackend, github_stub),
        cache_ttl_seconds=60,
        enable_github_fallback=False,
    )

    def fake_get(url: str, timeout: float) -> DummyResponse:
        return DummyResponse(503, None)

    monkeypatch.setattr("horde_model_reference.backends.http_backend.httpx.get", fake_get)

    result = backend.get_legacy_json(category)
    string_result = backend.get_legacy_json_string(category)

    assert result is None
    assert string_result is None
    assert github_stub.legacy_json_calls == 0
    assert github_stub.legacy_json_string_calls == 0
