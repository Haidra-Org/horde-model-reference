from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pytest_httpx import HTTPXMock
from typing_extensions import override

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
        return dict.fromkeys(MODEL_REFERENCE_CATEGORY)

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


def test_http_backend_primary_success_and_cache(
    monkeypatch: pytest.MonkeyPatch,
    httpx_mock: HTTPXMock,
) -> None:
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

    # Mock the PRIMARY API response
    httpx_mock.add_response(
        url="https://primary/model_references/v2/image_generation",
        json={"source": "primary"},
    )

    first = backend.fetch_category(category)
    second = backend.fetch_category(category)

    assert first == {"source": "primary"}
    assert second == {"source": "primary"}
    # Verify only one request was made (second was cached)
    assert len(httpx_mock.get_requests()) == 1
    assert backend.get_statistics()["primary_hits"] == 1
    assert github_stub.sync_calls == 0


def test_http_backend_fallback_when_primary_fails(httpx_mock: HTTPXMock) -> None:
    """Fallback should defer to GitHub backend when PRIMARY errors."""
    category = MODEL_REFERENCE_CATEGORY.image_generation
    fallback_payload = {"source": "github"}
    github_stub = StubGitHubBackend({category: fallback_payload})
    backend = HTTPBackend(
        primary_api_url="https://primary",
        github_backend=cast(GitHubBackend, github_stub),
        cache_ttl_seconds=60,
    )

    # Mock PRIMARY API to return 500 error
    # HTTPBackend will retry up to 3 times, so register multiple responses
    httpx_mock.add_response(
        url="https://primary/model_references/v2/image_generation",
        status_code=500,
        is_optional=False,  # First attempt must be made
    )
    httpx_mock.add_response(
        url="https://primary/model_references/v2/image_generation",
        status_code=500,
        is_optional=True,  # Subsequent retries are optional
    )
    httpx_mock.add_response(
        url="https://primary/model_references/v2/image_generation",
        status_code=500,
        is_optional=True,
    )

    result = backend.fetch_category(category)

    assert result == fallback_payload
    assert github_stub.sync_calls == 1
    assert backend.get_statistics()["github_fallbacks"] == 1


def test_http_backend_disable_fallback_returns_none(httpx_mock: HTTPXMock) -> None:
    """Disabling fallback should surface PRIMARY failures."""
    category = MODEL_REFERENCE_CATEGORY.image_generation
    github_stub = StubGitHubBackend({category: {"source": "github"}})
    backend = HTTPBackend(
        primary_api_url="https://primary",
        github_backend=cast(GitHubBackend, github_stub),
        cache_ttl_seconds=60,
        enable_github_fallback=False,
    )

    # Mock PRIMARY API to return 503 error
    # Using is_optional since the backend retries up to 3 times
    httpx_mock.add_response(
        url="https://primary/model_references/v2/image_generation",
        status_code=503,
        is_optional=False,  # First attempt must be made
    )
    httpx_mock.add_response(
        url="https://primary/model_references/v2/image_generation",
        status_code=503,
        is_optional=True,  # Subsequent retries are optional depending on timing
    )
    httpx_mock.add_response(
        url="https://primary/model_references/v2/image_generation",
        status_code=503,
        is_optional=True,
    )

    assert backend.fetch_category(category) is None
    assert github_stub.sync_calls == 0


def test_http_backend_force_refresh_triggers_refetch(
    monkeypatch: pytest.MonkeyPatch,
    httpx_mock: HTTPXMock,
) -> None:
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

    # Mock two different responses
    httpx_mock.add_response(
        url="https://primary/model_references/v2/image_generation",
        json={"version": 1},
    )
    httpx_mock.add_response(
        url="https://primary/model_references/v2/image_generation",
        json={"version": 2},
    )

    first = backend.fetch_category(category)
    cached = backend.fetch_category(category)
    refreshed = backend.fetch_category(category, force_refresh=True)

    assert first == {"version": 1}
    assert cached == {"version": 1}
    assert refreshed == {"version": 2}
    # Verify two requests were made (first and refreshed)
    assert len(httpx_mock.get_requests()) == 2


@pytest.mark.asyncio
async def test_http_backend_async_caches_results(
    monkeypatch: pytest.MonkeyPatch,
    httpx_mock: HTTPXMock,
) -> None:
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

    # Mock the PRIMARY API response
    httpx_mock.add_response(
        url="https://primary/model_references/v2/image_generation",
        json={"async": 1},
    )

    # Create actual async client for the test
    async with httpx.AsyncClient() as client:
        first = await backend.fetch_category_async(category, httpx_client=client)
        second = await backend.fetch_category_async(category, httpx_client=client)

    assert first == {"async": 1}
    assert second == {"async": 1}
    # Verify only one request was made (second was cached)
    assert len(httpx_mock.get_requests()) == 1
    assert github_stub.async_calls == 0


def test_http_backend_legacy_json_primary_success_and_cache(
    monkeypatch: pytest.MonkeyPatch,
    httpx_mock: HTTPXMock,
) -> None:
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

    legacy_dict = {"legacy": "data"}
    legacy_string = '{"legacy": "data"}'

    # Mock the legacy PRIMARY API response
    httpx_mock.add_response(
        url="https://primary/model_references/v1/image_generation",
        json=legacy_dict,
        text=legacy_string,
    )

    first = backend.get_legacy_json(category)
    second = backend.get_legacy_json(category)
    string_result = backend.get_legacy_json_string(category)

    assert first == legacy_dict
    assert second == legacy_dict
    assert string_result == legacy_string
    # Verify only one request was made due to caching
    assert len(httpx_mock.get_requests()) == 1
    assert backend.get_statistics()["primary_hits"] == 1
    assert github_stub.legacy_json_calls == 0


def test_http_backend_legacy_json_fallback_to_github(httpx_mock: HTTPXMock) -> None:
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

    # Mock PRIMARY API to return 500 error
    httpx_mock.add_response(
        url="https://primary/model_references/v1/image_generation",
        status_code=500,
        is_optional=False,
    )
    httpx_mock.add_response(
        url="https://primary/model_references/v1/image_generation",
        status_code=500,
        is_optional=True,
    )
    httpx_mock.add_response(
        url="https://primary/model_references/v1/image_generation",
        status_code=500,
        is_optional=True,
    )

    result = backend.get_legacy_json(category)
    string_result = backend.get_legacy_json_string(category)

    assert result == legacy_dict
    assert string_result == legacy_string
    assert github_stub.legacy_json_calls == 1
    # String result comes from cache after first fetch, so only 1 call
    assert github_stub.legacy_json_string_calls == 1
    # Only 1 fallback because string is cached from first call
    assert backend.get_statistics()["github_fallbacks"] == 1


def test_http_backend_legacy_json_redownload_bypasses_cache(
    monkeypatch: pytest.MonkeyPatch,
    httpx_mock: HTTPXMock,
) -> None:
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

    # Mock two different responses
    httpx_mock.add_response(
        url="https://primary/model_references/v1/image_generation",
        json={"version": 1},
        text='{"version": 1}',
    )
    httpx_mock.add_response(
        url="https://primary/model_references/v1/image_generation",
        json={"version": 2},
        text='{"version": 2}',
    )

    first = backend.get_legacy_json(category)
    cached = backend.get_legacy_json(category)
    refreshed = backend.get_legacy_json(category, redownload=True)

    assert first == {"version": 1}
    assert cached == {"version": 1}
    assert refreshed == {"version": 2}
    # Verify two requests were made (first and redownload)
    assert len(httpx_mock.get_requests()) == 2


def test_http_backend_legacy_json_disable_fallback(httpx_mock: HTTPXMock) -> None:
    """Disabling fallback should return None for legacy JSON when PRIMARY fails."""
    category = MODEL_REFERENCE_CATEGORY.image_generation
    github_stub = StubGitHubBackend({}, {category: ({"github": "data"}, '{"github": "data"}')})
    backend = HTTPBackend(
        primary_api_url="https://primary",
        github_backend=cast(GitHubBackend, github_stub),
        cache_ttl_seconds=60,
        enable_github_fallback=False,
    )

    # Mock PRIMARY API to return 503 error
    # get_legacy_json will try 3 times, then get_legacy_json_string will try 3 more times
    # because they both fail and there's no cache hit
    for _ in range(6):  # Both methods will retry when fallback is disabled
        httpx_mock.add_response(
            url="https://primary/model_references/v1/image_generation",
            status_code=503,
            is_optional=True,  # All attempts are part of retry logic
        )

    result = backend.get_legacy_json(category)
    string_result = backend.get_legacy_json_string(category)

    assert result is None
    assert string_result is None
    assert github_stub.legacy_json_calls == 0
    assert github_stub.legacy_json_string_calls == 0
