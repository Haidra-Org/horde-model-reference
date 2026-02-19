from collections.abc import Generator

import pytest
from httpx import AsyncClient

from horde_model_reference import horde_model_reference_settings
from horde_model_reference.service import shared
from horde_model_reference.service.shared import HordeUserContext


@pytest.fixture(autouse=True)
def restore_pending_queue_settings() -> Generator[None, None, None]:
    """Reset pending queue allowlists after each test."""
    settings = horde_model_reference_settings.pending_queue
    original_requestors = list(settings.requestor_ids)
    original_approvers = list(settings.approver_ids)
    yield
    settings.requestor_ids = original_requestors
    settings.approver_ids = original_approvers


def test_queue_requestor_allowlist_combines_configured_ids() -> None:
    """Ensure requestor allowlist merges requestor and approver IDs."""
    settings = horde_model_reference_settings.pending_queue
    settings.requestor_ids = ["111"]
    settings.approver_ids = ["222"]

    allowlist = shared._queue_requestor_allowlist()

    assert allowlist == {"111", "222"}


def test_queue_requestor_allowlist_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify requestor allowlist falls back when no IDs are configured."""
    settings = horde_model_reference_settings.pending_queue
    settings.requestor_ids = []
    settings.approver_ids = []

    monkeypatch.setattr(shared, "allowed_users", ["fallback"])
    monkeypatch.setattr(shared, "_requestor_fallback_logged", False)

    allowlist = shared._queue_requestor_allowlist()

    assert allowlist == {"fallback"}


def test_queue_approver_allowlist_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify approver allowlist falls back when no IDs are configured."""
    settings = horde_model_reference_settings.pending_queue
    settings.approver_ids = []

    monkeypatch.setattr(shared, "allowed_users", ["fallback"])
    monkeypatch.setattr(shared, "_approver_fallback_logged", False)

    allowlist = shared._queue_approver_allowlist()

    assert allowlist == {"fallback"}


@pytest.mark.asyncio
async def test_authenticate_queue_approver_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure approver authentication succeeds with fallback allowlist."""
    settings = horde_model_reference_settings.pending_queue
    settings.requestor_ids = []
    settings.approver_ids = []

    monkeypatch.setattr(shared, "allowed_users", ["fallback"])
    monkeypatch.setattr(shared, "_approver_fallback_logged", False)

    async def _mock_auth(
        apikey: str,
        client: AsyncClient,
        *,
        allowed_user_ids: set[str],
    ) -> HordeUserContext | None:
        assert allowed_user_ids == {"fallback"}
        return HordeUserContext(user_id="fallback", username="tester#fallback")

    monkeypatch.setattr(shared, "auth_against_horde", _mock_auth)

    result = await shared.authenticate_queue_approver("dummy")

    assert result.user_id == "fallback"


@pytest.mark.asyncio
async def test_authenticate_queue_requestor_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure requestor authentication succeeds with fallback allowlist."""
    settings = horde_model_reference_settings.pending_queue
    settings.requestor_ids = []
    settings.approver_ids = []

    monkeypatch.setattr(shared, "allowed_users", ["fallback"])
    monkeypatch.setattr(shared, "_requestor_fallback_logged", False)

    async def _mock_auth(
        apikey: str,
        client: AsyncClient,
        *,
        allowed_user_ids: set[str],
    ) -> HordeUserContext | None:
        assert allowed_user_ids == {"fallback"}
        return HordeUserContext(user_id="fallback", username="tester#fallback")

    monkeypatch.setattr(shared, "auth_against_horde", _mock_auth)

    result = await shared.authenticate_queue_requestor("dummy")

    assert result.user_id == "fallback"


@pytest.mark.asyncio
async def test_get_user_roles_returns_approver_role(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure get_user_roles returns approver role when user is in approver allowlist."""
    settings = horde_model_reference_settings.pending_queue
    settings.requestor_ids = []
    settings.approver_ids = ["123"]

    async def _mock_auth(
        apikey: str,
        client: AsyncClient,
        *,
        allowed_user_ids: set[str] | None,
    ) -> HordeUserContext | None:
        return HordeUserContext(user_id="123", username="tester#123")

    monkeypatch.setattr(shared, "auth_against_horde", _mock_auth)

    context, roles = await shared.get_user_roles("dummy")

    assert context is not None
    assert context.user_id == "123"
    assert "approver" in roles
    assert "requestor" in roles  # Approvers are also requestors


@pytest.mark.asyncio
async def test_get_user_roles_returns_requestor_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure get_user_roles returns only requestor role when user is only in requestor allowlist."""
    settings = horde_model_reference_settings.pending_queue
    settings.requestor_ids = ["456"]
    settings.approver_ids = ["789"]  # Different user

    async def _mock_auth(
        apikey: str,
        client: AsyncClient,
        *,
        allowed_user_ids: set[str] | None,
    ) -> HordeUserContext | None:
        return HordeUserContext(user_id="456", username="tester#456")

    monkeypatch.setattr(shared, "auth_against_horde", _mock_auth)

    context, roles = await shared.get_user_roles("dummy")

    assert context is not None
    assert context.user_id == "456"
    assert "requestor" in roles
    assert "approver" not in roles


@pytest.mark.asyncio
async def test_get_user_roles_returns_no_roles_for_regular_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure get_user_roles returns empty roles for user not in any allowlist."""
    settings = horde_model_reference_settings.pending_queue
    settings.requestor_ids = ["111"]
    settings.approver_ids = ["222"]

    async def _mock_auth(
        apikey: str,
        client: AsyncClient,
        *,
        allowed_user_ids: set[str] | None,
    ) -> HordeUserContext | None:
        return HordeUserContext(user_id="999", username="regular#999")

    monkeypatch.setattr(shared, "auth_against_horde", _mock_auth)

    context, roles = await shared.get_user_roles("dummy")

    assert context is not None
    assert context.user_id == "999"
    assert len(roles) == 0


@pytest.mark.asyncio
async def test_get_user_roles_returns_none_for_invalid_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure get_user_roles returns None context for invalid API key."""

    async def _mock_auth(
        apikey: str,
        client: AsyncClient,
        *,
        allowed_user_ids: set[str] | None,
    ) -> HordeUserContext | None:
        return None

    monkeypatch.setattr(shared, "auth_against_horde", _mock_auth)

    context, roles = await shared.get_user_roles("invalid")

    assert context is None
    assert len(roles) == 0
