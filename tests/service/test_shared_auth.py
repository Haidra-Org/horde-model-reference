"""Tests for auth error handling and httpx client lifecycle (SH-3, SH-4, AQ-5)."""

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException

from horde_model_reference.service import shared
from horde_model_reference.service.shared import auth_against_horde


@pytest.fixture(autouse=True)
def _reset_fallback_flags() -> Generator[None]:
    """Reset module-level log-once flags after each test."""
    original_req = shared._requestor_fallback_logged
    original_app = shared._approver_fallback_logged
    yield
    shared._requestor_fallback_logged = original_req
    shared._approver_fallback_logged = original_app


class TestAuthAgainstHordeErrorHandling:
    """Tests for httpx error handling in auth_against_horde (AQ-5, SH-3)."""

    @pytest.mark.asyncio
    async def test_timeout_raises_503(self) -> None:
        """Verify that httpx timeout produces a 503 response, not a 500."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.TimeoutException("Connection timed out")

        with pytest.raises(HTTPException) as exc_info:
            await auth_against_horde("test-key", mock_client)

        assert exc_info.value.status_code == 503
        assert "timed out" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_connection_error_raises_503(self) -> None:
        """Verify that httpx connection errors produce a 503, not a 500."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(HTTPException) as exc_info:
            await auth_against_horde("test-key", mock_client)

        assert exc_info.value.status_code == 503
        assert "unavailable" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_successful_auth_returns_context(self) -> None:
        """Verify that successful auth returns a HordeUserContext."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"username": "TestUser#42"}
        mock_client.get.return_value = mock_response

        result = await auth_against_horde("test-key", mock_client)

        assert result is not None
        assert result.user_id == "42"
        assert result.username == "TestUser#42"

    @pytest.mark.asyncio
    async def test_unauthorized_response_returns_none(self) -> None:
        """Verify that a non-200 response returns None."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_client.get.return_value = mock_response

        result = await auth_against_horde("test-key", mock_client)

        assert result is None

    @pytest.mark.asyncio
    async def test_user_not_in_allowlist_returns_none(self) -> None:
        """Verify that a user not in the allowlist returns None."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"username": "TestUser#42"}
        mock_client.get.return_value = mock_response

        result = await auth_against_horde("test-key", mock_client, allowed_user_ids={"99"})

        assert result is None


class TestHttpxClientConfiguration:
    """Tests for httpx client timeout configuration (SH-3/CR-6)."""

    def test_httpx_client_has_timeout(self) -> None:
        """Verify the module-level httpx client has a timeout set."""
        assert shared.httpx_client.timeout is not None
        assert shared.httpx_client.timeout.connect is not None
        assert isinstance(shared.httpx_client.timeout.connect, (int, float))
        assert shared.httpx_client.timeout.connect > 0
