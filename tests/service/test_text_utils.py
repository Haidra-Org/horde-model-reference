"""Tests for v2 text model group utility endpoints."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from horde_model_reference import (
    MODEL_REFERENCE_CATEGORY,
    ModelReferenceManager,
)
from horde_model_reference.service.shared import get_model_reference_manager

_V2 = "/model_references/v2"


def _text_record(name: str, parameters: int, *, baseline: str = "qwen3", **kwargs: object) -> dict[str, object]:
    """Build a minimal text generation model record dict."""
    record: dict[str, object] = {
        "name": name,
        "record_type": "text_generation",
        "model_classification": {"domain": "text", "purpose": "generation"},
        "parameters": parameters,
        "baseline": baseline,
        "nsfw": False,
        **kwargs,
    }
    return record


@pytest.fixture
def text_group_manager(
    primary_manager_override_factory: Callable[[Callable[[], ModelReferenceManager]], ModelReferenceManager],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[ModelReferenceManager]:
    """PRIMARY manager seeded with a text model group for testing."""
    from horde_model_reference import CanonicalFormat, horde_model_reference_settings

    monkeypatch.setattr(horde_model_reference_settings, "canonical_format", CanonicalFormat.v2)
    manager = primary_manager_override_factory(get_model_reference_manager)
    backend = manager.backend

    # Seed a "Qwen3" group with multiple sizes
    backend.update_model(
        MODEL_REFERENCE_CATEGORY.text_generation,
        "Qwen3-0.6B",
        _text_record(
            "Qwen3-0.6B",
            600_000_000,
            text_model_group="Qwen3",
            description="Small Qwen3",
            url="https://example.com/qwen3",
        ),
    )
    backend.update_model(
        MODEL_REFERENCE_CATEGORY.text_generation,
        "Qwen3-4B",
        _text_record(
            "Qwen3-4B",
            4_000_000_000,
            text_model_group="Qwen3",
            description="Small Qwen3",
            url="https://example.com/qwen3",
        ),
    )
    backend.update_model(
        MODEL_REFERENCE_CATEGORY.text_generation,
        "Qwen3-8B-Instruct",
        _text_record(
            "Qwen3-8B-Instruct",
            8_000_000_000,
            text_model_group="Qwen3",
            description="Small Qwen3",
            url="https://example.com/qwen3",
        ),
    )
    # A different group
    backend.update_model(
        MODEL_REFERENCE_CATEGORY.text_generation,
        "Llama-3-8B",
        _text_record(
            "Llama-3-8B",
            8_000_000_000,
            text_model_group="Llama-3",
            baseline="llama3",
            description="Llama 3",
        ),
    )

    manager._invalidate_cache()
    yield manager


class TestParseName:
    """Tests for the parse_name endpoint."""

    def test_parse_basic(self, api_client: TestClient, text_group_manager: ModelReferenceManager) -> None:
        """Test parsing a complex name with all components (base, size, variant, quantization)."""
        resp = api_client.get(f"{_V2}/text_generation/parse_name", params={"name": "Qwen3-0.6B-Instruct-Q4_K_M"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["original_name"] == "Qwen3-0.6B-Instruct-Q4_K_M"
        assert data["base_name"] == "Qwen3"
        assert data["size"] == "0.6B"
        assert data["variant"] == "Instruct"
        assert data["quant"] == "Q4_K_M"
        assert data["suggested_group"] == "Qwen3"

    def test_parse_simple_name(self, api_client: TestClient, text_group_manager: ModelReferenceManager) -> None:
        """Test parsing a simple name with only a base component."""
        resp = api_client.get(f"{_V2}/text_generation/parse_name", params={"name": "Mistral-7B"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["base_name"] == "Mistral"
        assert data["size"] == "7B"

    def test_parse_no_size(self, api_client: TestClient, text_group_manager: ModelReferenceManager) -> None:
        """Test parsing a name that doesn't include a size component."""
        resp = api_client.get(f"{_V2}/text_generation/parse_name", params={"name": "GPT-4"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["size"] is None


class TestGetGroup:
    """Tests for the group members endpoint."""

    def test_get_group_members(self, api_client: TestClient, text_group_manager: ModelReferenceManager) -> None:
        """Test retrieving all members of a text model group and their details."""
        resp = api_client.get(f"{_V2}/text_generation/group/Qwen3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["group_name"] == "Qwen3"
        assert data["canonical_count"] == 3
        assert data["backend_duplicate_count"] == 0
        assert len(data["members"]) == 3

    def test_group_common_fields(self, api_client: TestClient, text_group_manager: ModelReferenceManager) -> None:
        """Test that common fields across group members are correctly identified and returned."""
        resp = api_client.get(f"{_V2}/text_generation/group/Qwen3")
        data = resp.json()
        common = data["common_fields"]
        # All canonical members share baseline="qwen3" and description
        assert common.get("baseline") == "qwen3"
        assert common.get("description") == "Small Qwen3"

    def test_group_available_sizes(self, api_client: TestClient, text_group_manager: ModelReferenceManager) -> None:
        """Test that the available sizes within a text model group are correctly identified and returned."""
        resp = api_client.get(f"{_V2}/text_generation/group/Qwen3")
        data = resp.json()
        sizes = data["available_sizes"]
        assert "0.6B" in sizes
        assert "4B" in sizes
        assert "8B" in sizes

    def test_group_usage_counts(self, api_client: TestClient, text_group_manager: ModelReferenceManager) -> None:
        """Test that usage counts by size, variant, and quant are correct and returned for a text model group."""
        resp = api_client.get(f"{_V2}/text_generation/group/Qwen3")
        assert resp.status_code == 200
        data = resp.json()

        assert data["size_usage"] == {
            "0.6B": 1,
            "4B": 1,
            "8B": 1,
        }
        assert data["variant_usage"] == {
            "Instruct": 1,
        }
        assert data["quant_usage"] == {}

    def test_group_not_found(self, api_client: TestClient, text_group_manager: ModelReferenceManager) -> None:
        """Test requesting a text model group that doesn't exist returns a 404 error."""
        resp = api_client.get(f"{_V2}/text_generation/group/NonExistent")
        assert resp.status_code == 404

    def test_group_parsed_info(self, api_client: TestClient, text_group_manager: ModelReferenceManager) -> None:
        """Test that parsed info for group members is included and correct."""
        resp = api_client.get(f"{_V2}/text_generation/group/Qwen3")
        data = resp.json()
        # Find the Instruct variant
        instruct_members = [m for m in data["members"] if "Instruct" in m["name"]]
        assert len(instruct_members) >= 1
        member = instruct_members[0]
        assert member["parsed"]["variant"] == "Instruct"
        assert member["parsed"]["size"] == "8B"

    def test_group_no_backend_duplicates_in_v2(
        self, api_client: TestClient, text_group_manager: ModelReferenceManager
    ) -> None:
        """Backend duplicates are not stored in v2 format — only canonical models."""
        resp = api_client.get(f"{_V2}/text_generation/group/Qwen3")
        data = resp.json()
        dups = [m for m in data["members"] if m["is_backend_duplicate"]]
        assert len(dups) == 0


class TestComposeName:
    """Tests for the name composition endpoint."""

    def test_compose_basic(self, api_client: TestClient, text_group_manager: ModelReferenceManager) -> None:
        """Test composing a name from basic components (base and size)."""
        resp = api_client.post(
            f"{_V2}/text_generation/compose_name",
            json={"base_name": "Qwen3", "size": "14B"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["composed_name"] == "Qwen3-14B"
        assert data["already_exists"] is False
        assert data["suggested_group"] == "Qwen3"

    def test_compose_with_variant_and_quant(
        self, api_client: TestClient, text_group_manager: ModelReferenceManager
    ) -> None:
        """Test composing a name with variant and quantization components."""
        resp = api_client.post(
            f"{_V2}/text_generation/compose_name",
            json={"base_name": "Qwen3", "size": "8B", "variant": "Instruct", "quant": "Q4_K_M"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["composed_name"] == "Qwen3-8B-Instruct-Q4_K_M"

    def test_compose_with_author(self, api_client: TestClient, text_group_manager: ModelReferenceManager) -> None:
        """Test composing a name with an author component."""
        resp = api_client.post(
            f"{_V2}/text_generation/compose_name",
            json={"author": "Qwen", "base_name": "Qwen3", "size": "14B"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["composed_name"] == "Qwen/Qwen3-14B"

    def test_compose_collision_detected(
        self, api_client: TestClient, text_group_manager: ModelReferenceManager
    ) -> None:
        """Test that composing a name that already exists in the backend returns the correct flag."""
        resp = api_client.post(
            f"{_V2}/text_generation/compose_name",
            json={"base_name": "Qwen3", "size": "0.6B"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["composed_name"] == "Qwen3-0.6B"
        assert data["already_exists"] is True


class TestDistinctBaselines:
    """Tests for the distinct baseline values endpoint."""

    def test_distinct_baselines(self, api_client: TestClient, text_group_manager: ModelReferenceManager) -> None:
        """Test that the endpoint returns the correct list of distinct baseline values for text generation models."""
        resp = api_client.get(f"{_V2}/text_generation/distinct_baselines")
        assert resp.status_code == 200
        data = resp.json()
        assert data["baselines"] == ["llama3", "qwen3"]
