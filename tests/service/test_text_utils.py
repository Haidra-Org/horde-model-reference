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
        """Backend duplicates are not stored in v2 format - only canonical models."""
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


class TestListGroups:
    """Tests for the group names list endpoint."""

    def test_list_groups(self, api_client: TestClient, text_group_manager: ModelReferenceManager) -> None:
        """Test listing all distinct text model group names."""
        resp = api_client.get(f"{_V2}/text_generation/groups")
        assert resp.status_code == 200
        data = resp.json()
        assert "Qwen3" in data["groups"]
        assert "Llama-3" in data["groups"]

    def test_list_groups_sorted(self, api_client: TestClient, text_group_manager: ModelReferenceManager) -> None:
        """Test that group names are returned in sorted order."""
        resp = api_client.get(f"{_V2}/text_generation/groups")
        data = resp.json()
        assert data["groups"] == sorted(data["groups"])


class TestGroupNameSchema:
    """Tests for group naming schema CRUD endpoints."""

    def test_get_inferred_schema(self, api_client: TestClient, text_group_manager: ModelReferenceManager) -> None:
        """Test that an inferred schema is returned when no custom schema exists."""
        # Ensure no custom schema is persisted from other tests
        store = text_group_manager.group_schema_store
        if store:
            store.delete("Qwen3")

        resp = api_client.get(f"{_V2}/text_generation/group/Qwen3/name_schema")
        assert resp.status_code == 200
        data = resp.json()
        assert data["group_name"] == "Qwen3"
        assert data["is_custom"] is False
        assert data["name_schema"]["separator"] in ("-", "_")
        assert isinstance(data["name_schema"]["part_order"], list)

    def test_get_schema_group_not_found(
        self, api_client: TestClient, text_group_manager: ModelReferenceManager
    ) -> None:
        """Test that requesting a schema for a non-existent group returns 404."""
        resp = api_client.get(f"{_V2}/text_generation/group/NonExistent/name_schema")
        assert resp.status_code == 404

    def test_put_custom_schema(
        self, api_client: TestClient, text_group_manager: ModelReferenceManager, mock_auth_success: None
    ) -> None:
        """Test saving a custom naming schema for a group."""
        resp = api_client.put(
            f"{_V2}/text_generation/group/Qwen3/name_schema",
            json={"separator": "_", "author_included": False},
            headers={"apikey": "test-key"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_custom"] is True
        assert data["name_schema"]["separator"] == "_"
        assert data["name_schema"]["author_included"] is False

    def test_get_custom_schema_after_save(
        self, api_client: TestClient, text_group_manager: ModelReferenceManager, mock_auth_success: None
    ) -> None:
        """Test that a saved custom schema is returned on subsequent GET."""
        # Save a custom schema
        api_client.put(
            f"{_V2}/text_generation/group/Qwen3/name_schema",
            json={"separator": "_", "common_author": "Qwen"},
            headers={"apikey": "test-key"},
        )
        # Retrieve it
        resp = api_client.get(f"{_V2}/text_generation/group/Qwen3/name_schema")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_custom"] is True
        assert data["name_schema"]["separator"] == "_"
        assert data["name_schema"]["common_author"] == "Qwen"

    def test_delete_custom_schema(
        self, api_client: TestClient, text_group_manager: ModelReferenceManager, mock_auth_success: None
    ) -> None:
        """Test deleting a custom schema reverts to inferred."""
        # Save then delete
        api_client.put(
            f"{_V2}/text_generation/group/Qwen3/name_schema",
            json={"separator": "_"},
            headers={"apikey": "test-key"},
        )
        resp = api_client.delete(
            f"{_V2}/text_generation/group/Qwen3/name_schema",
            headers={"apikey": "test-key"},
        )
        assert resp.status_code == 204

        # Verify it's back to inferred
        resp = api_client.get(f"{_V2}/text_generation/group/Qwen3/name_schema")
        assert resp.json()["is_custom"] is False

    def test_delete_nonexistent_schema(
        self, api_client: TestClient, text_group_manager: ModelReferenceManager, mock_auth_success: None
    ) -> None:
        """Test deleting a schema that doesn't exist returns 404."""
        resp = api_client.delete(
            f"{_V2}/text_generation/group/Qwen3/name_schema",
            headers={"apikey": "test-key"},
        )
        assert resp.status_code == 404

    def test_group_response_includes_schema_custom_flag(
        self, api_client: TestClient, text_group_manager: ModelReferenceManager, mock_auth_success: None
    ) -> None:
        """Test that the group endpoint reflects custom schema state."""
        # Before saving: inferred
        resp = api_client.get(f"{_V2}/text_generation/group/Qwen3")
        assert resp.json()["name_schema_is_custom"] is False

        # After saving custom schema
        api_client.put(
            f"{_V2}/text_generation/group/Qwen3/name_schema",
            json={"separator": "_"},
            headers={"apikey": "test-key"},
        )
        resp = api_client.get(f"{_V2}/text_generation/group/Qwen3")
        data = resp.json()
        assert data["name_schema_is_custom"] is True
        assert data["name_format"]["separator"] == "_"


class TestNameException:
    """Tests for the name_schema_exception field endpoints."""

    def test_set_name_exception(
        self, api_client: TestClient, text_group_manager: ModelReferenceManager, mock_auth_success: None
    ) -> None:
        """Test setting a name exception flag on a model creates a pending change."""
        resp = api_client.put(
            f"{_V2}/text_generation/Qwen3-0.6B/name_exception",
            json={"reason": "Uses non-standard naming"},
            headers={"apikey": "test-key"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "pending_change_id" in data

    def test_set_name_exception_model_not_found(
        self, api_client: TestClient, text_group_manager: ModelReferenceManager, mock_auth_success: None
    ) -> None:
        """Test setting exception on a non-existent model returns 404."""
        resp = api_client.put(
            f"{_V2}/text_generation/NonExistent/name_exception",
            json={"reason": "test"},
            headers={"apikey": "test-key"},
        )
        assert resp.status_code == 404

    def test_group_response_includes_exception_members(
        self, api_client: TestClient, text_group_manager: ModelReferenceManager
    ) -> None:
        """Test that exception members appear in the group response after direct backend update."""
        backend = text_group_manager.backend
        all_models = backend.fetch_category(MODEL_REFERENCE_CATEGORY.text_generation)
        record_data = dict(all_models["Qwen3-0.6B"])  # type: ignore[index]
        record_data["name_schema_exception"] = "Historical naming"
        backend.update_model(MODEL_REFERENCE_CATEGORY.text_generation, "Qwen3-0.6B", record_data)
        text_group_manager._invalidate_cache()

        resp = api_client.get(f"{_V2}/text_generation/group/Qwen3")
        data = resp.json()
        exceptions = data["exception_members"]
        assert len(exceptions) == 1
        assert exceptions[0]["name"] == "Qwen3-0.6B"
        assert exceptions[0]["reason"] == "Historical naming"


class TestParseNameExtras:
    """Tests for extras information in the parse_name response."""

    def test_parse_date_suffix_returns_extras(
        self, api_client: TestClient, text_group_manager: ModelReferenceManager
    ) -> None:
        """A date suffix like 08-2024 should appear in the extras array."""
        resp = api_client.get(
            f"{_V2}/text_generation/parse_name",
            params={"name": "c4ai-command-r-08-2024"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["base_name"] == "c4ai-command-r"
        extras = data["extras"]
        assert len(extras) >= 1
        date_extras = [e for e in extras if e["inferred_type"] == "DATE"]
        assert len(date_extras) == 1
        assert date_extras[0]["value"] == "08-2024"

    def test_parse_standard_name_no_extras(
        self, api_client: TestClient, text_group_manager: ModelReferenceManager
    ) -> None:
        """A standard name like Qwen3-8B-Instruct should have no extras."""
        resp = api_client.get(
            f"{_V2}/text_generation/parse_name",
            params={"name": "Qwen3-8B-Instruct"},
        )
        assert resp.status_code == 200
        assert resp.json()["extras"] == []

    def test_parse_alias_resolved_suggested_group(
        self, api_client: TestClient, text_group_manager: ModelReferenceManager
    ) -> None:
        """When aliases exist, suggested_group should resolve through them."""
        alias_store = text_group_manager.group_alias_store
        assert alias_store is not None
        alias_store.set_aliases("c4ai-command-r", ["c4ai-command-r-plus"])

        resp = api_client.get(
            f"{_V2}/text_generation/parse_name",
            params={"name": "c4ai-command-r-plus-08-2024"},
        )
        assert resp.status_code == 200
        assert resp.json()["suggested_group"] == "c4ai-command-r"

        # Clean up
        alias_store.delete("c4ai-command-r")


class TestGroupAliases:
    """Tests for group alias CRUD endpoints."""

    def test_list_aliases_empty(self, api_client: TestClient, text_group_manager: ModelReferenceManager) -> None:
        """An empty alias store should return an empty list."""
        resp = api_client.get(f"{_V2}/text_generation/aliases")
        assert resp.status_code == 200
        assert resp.json()["entries"] == []

    def test_set_and_list_aliases(
        self, api_client: TestClient, text_group_manager: ModelReferenceManager, mock_auth_success: None
    ) -> None:
        """Setting aliases via PUT should make them visible in the list."""
        resp = api_client.put(
            f"{_V2}/text_generation/aliases/c4ai-command-r",
            json={"aliases": ["c4ai-command-r-plus", "c4ai-command-a"]},
            headers={"apikey": "test-key"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["canonical"] == "c4ai-command-r"
        assert set(data["aliases"]) == {"c4ai-command-r-plus", "c4ai-command-a"}

        # Verify via list endpoint
        resp = api_client.get(f"{_V2}/text_generation/aliases")
        entries = resp.json()["entries"]
        assert len(entries) >= 1
        cr_entry = [e for e in entries if e["canonical"] == "c4ai-command-r"]
        assert len(cr_entry) == 1

        # Clean up
        api_client.delete(
            f"{_V2}/text_generation/aliases/c4ai-command-r",
            headers={"apikey": "test-key"},
        )

    def test_get_alias(
        self, api_client: TestClient, text_group_manager: ModelReferenceManager, mock_auth_success: None
    ) -> None:
        """GET for a specific canonical group should return its aliases."""
        api_client.put(
            f"{_V2}/text_generation/aliases/Broken-Tutu",
            json={"aliases": ["Broken-Tutu-Unslop"]},
            headers={"apikey": "test-key"},
        )

        resp = api_client.get(f"{_V2}/text_generation/aliases/Broken-Tutu")
        assert resp.status_code == 200
        assert resp.json()["aliases"] == ["Broken-Tutu-Unslop"]

        # Clean up
        api_client.delete(
            f"{_V2}/text_generation/aliases/Broken-Tutu",
            headers={"apikey": "test-key"},
        )

    def test_get_alias_not_found(self, api_client: TestClient, text_group_manager: ModelReferenceManager) -> None:
        """GET for a non-existent canonical should return 404."""
        resp = api_client.get(f"{_V2}/text_generation/aliases/nonexistent")
        assert resp.status_code == 404

    def test_add_single_alias(
        self, api_client: TestClient, text_group_manager: ModelReferenceManager, mock_auth_success: None
    ) -> None:
        """POST add should append a single alias to an entry."""
        api_client.put(
            f"{_V2}/text_generation/aliases/Broken-Tutu",
            json={"aliases": ["Broken-Tutu-Unslop"]},
            headers={"apikey": "test-key"},
        )

        resp = api_client.post(
            f"{_V2}/text_generation/aliases/Broken-Tutu/add",
            json={"alias": "Broken-Tutu-Transgression"},
            headers={"apikey": "test-key"},
        )
        assert resp.status_code == 200
        assert "Broken-Tutu-Transgression" in resp.json()["aliases"]
        assert "Broken-Tutu-Unslop" in resp.json()["aliases"]

        # Clean up
        api_client.delete(
            f"{_V2}/text_generation/aliases/Broken-Tutu",
            headers={"apikey": "test-key"},
        )

    def test_remove_single_alias(
        self, api_client: TestClient, text_group_manager: ModelReferenceManager, mock_auth_success: None
    ) -> None:
        """POST remove should remove one alias and return the updated entry."""
        api_client.put(
            f"{_V2}/text_generation/aliases/Broken-Tutu",
            json={"aliases": ["alias-a", "alias-b"]},
            headers={"apikey": "test-key"},
        )

        resp = api_client.post(
            f"{_V2}/text_generation/aliases/Broken-Tutu/remove",
            json={"alias": "alias-a"},
            headers={"apikey": "test-key"},
        )
        assert resp.status_code == 200
        assert resp.json()["aliases"] == ["alias-b"]

        # Clean up
        api_client.delete(
            f"{_V2}/text_generation/aliases/Broken-Tutu",
            headers={"apikey": "test-key"},
        )

    def test_remove_nonexistent_alias(
        self, api_client: TestClient, text_group_manager: ModelReferenceManager, mock_auth_success: None
    ) -> None:
        """Removing a non-existent alias returns 404."""
        api_client.put(
            f"{_V2}/text_generation/aliases/group-x",
            json={"aliases": ["alias-y"]},
            headers={"apikey": "test-key"},
        )

        resp = api_client.post(
            f"{_V2}/text_generation/aliases/group-x/remove",
            json={"alias": "not-registered"},
            headers={"apikey": "test-key"},
        )
        assert resp.status_code == 404

        # Clean up
        api_client.delete(
            f"{_V2}/text_generation/aliases/group-x",
            headers={"apikey": "test-key"},
        )

    def test_delete_aliases(
        self, api_client: TestClient, text_group_manager: ModelReferenceManager, mock_auth_success: None
    ) -> None:
        """DELETE should remove the entire alias entry."""
        api_client.put(
            f"{_V2}/text_generation/aliases/to-delete",
            json={"aliases": ["del-alias"]},
            headers={"apikey": "test-key"},
        )

        resp = api_client.delete(
            f"{_V2}/text_generation/aliases/to-delete",
            headers={"apikey": "test-key"},
        )
        assert resp.status_code == 204

        resp = api_client.get(f"{_V2}/text_generation/aliases/to-delete")
        assert resp.status_code == 404

    def test_delete_nonexistent(
        self, api_client: TestClient, text_group_manager: ModelReferenceManager, mock_auth_success: None
    ) -> None:
        """Deleting a non-existent canonical returns 404."""
        resp = api_client.delete(
            f"{_V2}/text_generation/aliases/nope",
            headers={"apikey": "test-key"},
        )
        assert resp.status_code == 404

    def test_conflict_on_duplicate_alias(
        self, api_client: TestClient, text_group_manager: ModelReferenceManager, mock_auth_success: None
    ) -> None:
        """Setting an alias already claimed by another canonical returns 409."""
        api_client.put(
            f"{_V2}/text_generation/aliases/group-a",
            json={"aliases": ["shared-alias"]},
            headers={"apikey": "test-key"},
        )

        resp = api_client.put(
            f"{_V2}/text_generation/aliases/group-b",
            json={"aliases": ["shared-alias"]},
            headers={"apikey": "test-key"},
        )
        assert resp.status_code == 409

        # Clean up
        api_client.delete(
            f"{_V2}/text_generation/aliases/group-a",
            headers={"apikey": "test-key"},
        )


class TestGroupsSummary:
    """Tests for the enriched groups summary endpoint."""

    def test_summary_returns_all_groups(
        self, api_client: TestClient, text_group_manager: ModelReferenceManager
    ) -> None:
        """GET /groups/summary should return an entry for each known group."""
        resp = api_client.get(f"{_V2}/text_generation/groups/summary")
        assert resp.status_code == 200
        data = resp.json()
        group_names = [g["group_name"] for g in data["groups"]]
        assert "Qwen3" in group_names
        assert "Llama-3" in group_names
        assert data["total_groups"] == len(data["groups"])

    def test_summary_member_counts(self, api_client: TestClient, text_group_manager: ModelReferenceManager) -> None:
        """Summary entries should report correct canonical and duplicate counts."""
        resp = api_client.get(f"{_V2}/text_generation/groups/summary")
        data = resp.json()
        qwen = next(g for g in data["groups"] if g["group_name"] == "Qwen3")
        assert qwen["canonical_count"] == 3
        assert qwen["backend_duplicate_count"] == 0

    def test_summary_available_sizes(self, api_client: TestClient, text_group_manager: ModelReferenceManager) -> None:
        """Summary should include parsed sizes from model names."""
        resp = api_client.get(f"{_V2}/text_generation/groups/summary")
        data = resp.json()
        qwen = next(g for g in data["groups"] if g["group_name"] == "Qwen3")
        assert "0.6B" in qwen["available_sizes"]
        assert "4B" in qwen["available_sizes"]

    def test_summary_aggregate_totals(self, api_client: TestClient, text_group_manager: ModelReferenceManager) -> None:
        """Aggregate fields should tally correctly."""
        resp = api_client.get(f"{_V2}/text_generation/groups/summary")
        data = resp.json()
        assert data["total_models"] == sum(g["canonical_count"] + g["backend_duplicate_count"] for g in data["groups"])

    def test_summary_reflects_family(
        self,
        api_client: TestClient,
        text_group_manager: ModelReferenceManager,
        mock_auth_success: None,
    ) -> None:
        """After creating a family, summary should report the family_name."""
        # Create a family containing Qwen3
        api_client.put(
            f"{_V2}/text_generation/families/qwen-family",
            json={"members": ["Qwen3"]},
            headers={"apikey": "test-key"},
        )

        resp = api_client.get(f"{_V2}/text_generation/groups/summary")
        data = resp.json()
        qwen = next(g for g in data["groups"] if g["group_name"] == "Qwen3")
        assert qwen["family_name"] == "qwen-family"
        assert data["groups_with_families"] >= 1

        # Clean up
        api_client.delete(
            f"{_V2}/text_generation/families/qwen-family",
            headers={"apikey": "test-key"},
        )

    def test_summary_reflects_aliases(
        self,
        api_client: TestClient,
        text_group_manager: ModelReferenceManager,
        mock_auth_success: None,
    ) -> None:
        """After setting aliases, summary should report them."""
        api_client.put(
            f"{_V2}/text_generation/aliases/Qwen3",
            json={"aliases": ["qwen-three"]},
            headers={"apikey": "test-key"},
        )

        resp = api_client.get(f"{_V2}/text_generation/groups/summary")
        data = resp.json()
        qwen = next(g for g in data["groups"] if g["group_name"] == "Qwen3")
        assert "qwen-three" in qwen["aliases"]
        assert data["groups_with_aliases"] >= 1

        # Clean up
        api_client.delete(
            f"{_V2}/text_generation/aliases/Qwen3",
            headers={"apikey": "test-key"},
        )


class TestGroupsHealth:
    """Tests for the group health check endpoint."""

    def test_health_returns_issues(self, api_client: TestClient, text_group_manager: ModelReferenceManager) -> None:
        """GET /groups/health should return an issue list and aggregate counts."""
        resp = api_client.get(f"{_V2}/text_generation/groups/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "issues" in data
        assert "total_groups_checked" in data
        assert "groups_with_issues" in data
        assert "issue_counts_by_type" in data
        assert data["total_groups_checked"] >= 2

    def test_singleton_group_flagged(self, api_client: TestClient, text_group_manager: ModelReferenceManager) -> None:
        """A group with only one canonical member should have a 'singleton_group' issue."""
        resp = api_client.get(f"{_V2}/text_generation/groups/health")
        data = resp.json()
        llama_issues = [i for i in data["issues"] if i["group_name"] == "Llama-3"]
        issue_types = [i["issue_type"] for i in llama_issues]
        assert "singleton_group" in issue_types

    def test_healthy_group_no_issues(self, api_client: TestClient, text_group_manager: ModelReferenceManager) -> None:
        """A multi-member group with consistent data should have no issues or only info-level ones."""
        resp = api_client.get(f"{_V2}/text_generation/groups/health")
        data = resp.json()
        qwen_issues = [i for i in data["issues"] if i["group_name"] == "Qwen3"]
        warning_issues = [i for i in qwen_issues if i.get("severity", "warning") == "warning"]
        # Qwen3 has 3 consistent members - no warnings expected
        assert len(warning_issues) == 0
