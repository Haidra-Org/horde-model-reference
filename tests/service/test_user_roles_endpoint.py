"""Regression tests for the user roles endpoint."""

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.usefixtures("mock_auth_success")


def test_me_roles_uses_static_route(api_client: TestClient) -> None:
    """Ensure /me/roles resolves to the user router instead of category routes."""
    response = api_client.get(
        "/api/model_references/v2/me/roles",
        headers={"apikey": "test-key"},
    )

    assert response.status_code == 200

    body = response.json()
    assert body == {
        "user_id": "test-user-id",
        "username": "tester#test-user-id",
        "roles": ["approver", "requestor"],
        "is_approver": True,
        "is_requestor": True,
    }
