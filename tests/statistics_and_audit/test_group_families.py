"""Tests for related-group families: store CRUD, detection heuristics, and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from horde_model_reference.group_families import GroupFamily, GroupFamilyStore, detect_families


class TestGroupFamily:
    """Tests for the GroupFamily Pydantic model."""

    def test_basic_construction(self) -> None:
        """Family should hold a name and a list of members."""
        family = GroupFamily(family_name="Mistral", members=["Mistral-Large", "Mistral-Small"])
        assert family.family_name == "Mistral"
        assert family.members == ["Mistral-Large", "Mistral-Small"]

    def test_empty_members_default(self) -> None:
        """Members should default to an empty list."""
        family = GroupFamily(family_name="Empty")
        assert family.members == []


class TestGroupFamilyStore:
    """Tests for GroupFamilyStore CRUD operations."""

    @pytest.fixture()
    def store_path(self, tmp_path: Path) -> Path:
        """Return a temporary file path for the family store."""
        return tmp_path / "families.json"

    @pytest.fixture()
    def store(self, store_path: Path) -> GroupFamilyStore:
        """Return a fresh GroupFamilyStore instance."""
        return GroupFamilyStore(file_path=store_path)

    # -- set_family / get_family --

    def test_set_and_get_family(self, store: GroupFamilyStore) -> None:
        """Setting a family should make it retrievable by name."""
        store.set_family("Mistral", ["Mistral-Large", "Mistral-Small", "Mistral-Nemo"])

        family = store.get_family("Mistral")
        assert family is not None
        assert family.family_name == "Mistral"
        assert set(family.members) == {"Mistral-Large", "Mistral-Small", "Mistral-Nemo"}

    def test_get_nonexistent_family(self, store: GroupFamilyStore) -> None:
        """Getting a family that doesn't exist should return None."""
        assert store.get_family("Nonexistent") is None

    def test_set_family_empty_members_raises(self, store: GroupFamilyStore) -> None:
        """Setting a family with no members should raise ValueError."""
        with pytest.raises(ValueError, match="at least one member"):
            store.set_family("Empty", [])

    def test_set_family_replaces_previous(self, store: GroupFamilyStore) -> None:
        """Re-setting a family should replace its members."""
        store.set_family("Llama", ["Llama-3", "Llama-3.1"])
        store.set_family("Llama", ["Llama-3", "Llama-3.2", "Llama-3.3"])

        family = store.get_family("Llama")
        assert family is not None
        assert "Llama-3.1" not in family.members
        assert "Llama-3.2" in family.members
        assert "Llama-3.3" in family.members

    def test_set_family_releases_old_members(self, store: GroupFamilyStore) -> None:
        """When a family is re-set, old members should be released from the reverse index."""
        store.set_family("Llama", ["Llama-3", "Llama-3.1"])
        store.set_family("Llama", ["Llama-3"])

        # Llama-3.1 should now be unaffiliated
        assert store.get_family_for_group("Llama-3.1") is None

    # -- Conflict detection --

    def test_member_conflict_across_families(self, store: GroupFamilyStore) -> None:
        """A group already in one family cannot be assigned to another."""
        store.set_family("Mistral", ["Mistral-Large", "Mistral-Small"])

        with pytest.raises(ValueError, match="already belongs to family"):
            store.set_family("OtherFamily", ["Mistral-Large", "SomeGroup"])

    def test_same_member_same_family_is_fine(self, store: GroupFamilyStore) -> None:
        """Re-assigning a member to its own family should not raise."""
        store.set_family("Mistral", ["Mistral-Large", "Mistral-Small"])
        store.set_family("Mistral", ["Mistral-Large", "Mistral-Small", "Mistral-Nemo"])

        family = store.get_family("Mistral")
        assert family is not None
        assert len(family.members) == 3

    # -- get_family_for_group --

    def test_get_family_for_group(self, store: GroupFamilyStore) -> None:
        """Reverse lookup: group name to its family."""
        store.set_family("GPT-Neo", ["GPT-Neo-AID", "GPT-Neo-Picard"])

        family = store.get_family_for_group("GPT-Neo-AID")
        assert family is not None
        assert family.family_name == "GPT-Neo"

    def test_get_family_for_unaffiliated_group(self, store: GroupFamilyStore) -> None:
        """A group with no family should return None."""
        store.set_family("GPT-Neo", ["GPT-Neo-AID"])
        assert store.get_family_for_group("SomeOther") is None

    # -- add_member --

    def test_add_member(self, store: GroupFamilyStore) -> None:
        """Adding a member to an existing family should work."""
        store.set_family("Mistral", ["Mistral-Large"])
        store.add_member("Mistral", "Mistral-Small")

        family = store.get_family("Mistral")
        assert family is not None
        assert "Mistral-Small" in family.members

    def test_add_member_nonexistent_family(self, store: GroupFamilyStore) -> None:
        """Adding to a nonexistent family should raise KeyError."""
        with pytest.raises(KeyError, match="does not exist"):
            store.add_member("Nonexistent", "SomeGroup")

    def test_add_member_conflict(self, store: GroupFamilyStore) -> None:
        """Adding a member that belongs to another family should raise ValueError."""
        store.set_family("Mistral", ["Mistral-Large"])
        store.set_family("Other", ["SomeGroup"])

        with pytest.raises(ValueError, match="already belongs"):
            store.add_member("Mistral", "SomeGroup")

    def test_add_member_idempotent(self, store: GroupFamilyStore) -> None:
        """Adding an already-present member should be a no-op."""
        store.set_family("Mistral", ["Mistral-Large", "Mistral-Small"])
        store.add_member("Mistral", "Mistral-Large")

        family = store.get_family("Mistral")
        assert family is not None
        assert family.members.count("Mistral-Large") == 1

    # -- remove_member --

    def test_remove_member(self, store: GroupFamilyStore) -> None:
        """Removing a member should update the family."""
        store.set_family("Mistral", ["Mistral-Large", "Mistral-Small"])
        removed = store.remove_member("Mistral", "Mistral-Small")

        assert removed is True
        family = store.get_family("Mistral")
        assert family is not None
        assert "Mistral-Small" not in family.members
        assert store.get_family_for_group("Mistral-Small") is None

    def test_remove_member_deletes_empty_family(self, store: GroupFamilyStore) -> None:
        """Removing the last member should delete the family entirely."""
        store.set_family("Solo", ["OnlyMember"])
        removed = store.remove_member("Solo", "OnlyMember")

        assert removed is True
        assert store.get_family("Solo") is None

    def test_remove_member_not_found(self, store: GroupFamilyStore) -> None:
        """Removing a non-member should return False."""
        store.set_family("Mistral", ["Mistral-Large"])
        assert store.remove_member("Mistral", "NotHere") is False

    def test_remove_member_nonexistent_family(self, store: GroupFamilyStore) -> None:
        """Removing from a nonexistent family should return False."""
        assert store.remove_member("Nonexistent", "SomeGroup") is False

    # -- delete --

    def test_delete_family(self, store: GroupFamilyStore) -> None:
        """Deleting a family should remove it and release all members."""
        store.set_family("Mistral", ["Mistral-Large", "Mistral-Small"])
        deleted = store.delete("Mistral")

        assert deleted is True
        assert store.get_family("Mistral") is None
        assert store.get_family_for_group("Mistral-Large") is None
        assert store.get_family_for_group("Mistral-Small") is None

    def test_delete_nonexistent_family(self, store: GroupFamilyStore) -> None:
        """Deleting a nonexistent family should return False."""
        assert store.delete("Nonexistent") is False

    # -- list_all --

    def test_list_all(self, store: GroupFamilyStore) -> None:
        """list_all should return all configured families."""
        store.set_family("Mistral", ["Mistral-Large", "Mistral-Small"])
        store.set_family("Llama", ["Llama-3", "Llama-3.1"])

        all_families = store.list_all()
        assert len(all_families) == 2
        assert "Mistral" in all_families
        assert "Llama" in all_families

    def test_list_all_empty(self, store: GroupFamilyStore) -> None:
        """list_all on an empty store should return an empty dict."""
        assert store.list_all() == {}

    def test_list_all_returns_copies(self, store: GroupFamilyStore) -> None:
        """Modifying the returned dict should not affect the store."""
        store.set_family("Mistral", ["Mistral-Large"])
        families = store.list_all()
        families["Mistral"].members.append("INJECTED")

        original = store.get_family("Mistral")
        assert original is not None
        assert "INJECTED" not in original.members

    # -- Persistence --

    def test_persistence_round_trip(self, store_path: Path) -> None:
        """Data should survive a store reload from disk."""
        store1 = GroupFamilyStore(file_path=store_path)
        store1.set_family("Mistral", ["Mistral-Large", "Mistral-Small"])
        store1.set_family("Llama", ["Llama-3"])

        store2 = GroupFamilyStore(file_path=store_path)
        assert store2.get_family("Mistral") is not None
        assert store2.get_family("Llama") is not None
        assert store2.get_family_for_group("Mistral-Large") is not None

    def test_persistence_file_format(self, store_path: Path) -> None:
        """The persisted JSON should be a dict of family_name → family data."""
        store = GroupFamilyStore(file_path=store_path)
        store.set_family("Mistral", ["Mistral-Large", "Mistral-Small"])

        raw = json.loads(store_path.read_text(encoding="utf-8"))
        assert "Mistral" in raw
        assert raw["Mistral"]["family_name"] == "Mistral"
        assert set(raw["Mistral"]["members"]) == {"Mistral-Large", "Mistral-Small"}

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        """Loading from a nonexistent file should produce an empty store."""
        store = GroupFamilyStore(file_path=tmp_path / "nonexistent.json")
        assert store.list_all() == {}

    def test_load_corrupt_file(self, tmp_path: Path) -> None:
        """Loading from a corrupt file should gracefully result in an empty store."""
        corrupt_path = tmp_path / "corrupt.json"
        corrupt_path.write_text("not valid json{{{", encoding="utf-8")

        store = GroupFamilyStore(file_path=corrupt_path)
        assert store.list_all() == {}


class TestDetectFamilies:
    """Tests for the detect_families heuristic function."""

    def test_basic_prefix_detection(self) -> None:
        """Groups sharing a prefix should be detected as a family."""
        groups = ["Mistral-Large", "Mistral-Small", "Mistral-Nemo", "Llama-3"]
        families = detect_families(groups)

        assert "Mistral" in families
        assert set(families["Mistral"]) == {"Mistral-Large", "Mistral-Small", "Mistral-Nemo"}

    def test_single_group_no_family(self) -> None:
        """A prefix matching only one group should not form a family."""
        groups = ["Mistral-Large", "Llama-3", "Qwen-7B"]
        families = detect_families(groups)

        assert "Mistral" not in families
        assert "Llama" not in families

    def test_empty_input(self) -> None:
        """Empty input should produce no families."""
        assert detect_families([]) == {}

    def test_no_hyphenated_groups(self) -> None:
        """Groups without hyphens should produce no families."""
        groups = ["Mistral", "Llama", "Qwen"]
        assert detect_families(groups) == {}

    def test_longer_prefix_wins(self) -> None:
        """When both 'DeepSeek' and 'DeepSeek-R1-Distill' match, the longer prefix claims its members."""
        groups = [
            "DeepSeek-R1-Distill-Llama",
            "DeepSeek-R1-Distill-Qwen",
            "DeepSeek-V2",
            "DeepSeek-V3",
        ]
        families = detect_families(groups)

        # DeepSeek-R1-Distill should be its own family
        assert "DeepSeek-R1-Distill" in families
        assert set(families["DeepSeek-R1-Distill"]) == {
            "DeepSeek-R1-Distill-Llama",
            "DeepSeek-R1-Distill-Qwen",
        }
        # DeepSeek-V2 and DeepSeek-V3 should form the broader DeepSeek family
        assert "DeepSeek" in families
        assert set(families["DeepSeek"]) == {"DeepSeek-V2", "DeepSeek-V3"}

    def test_min_prefix_length(self) -> None:
        """Short prefixes below min_prefix_length should be excluded."""
        groups = ["AB-One", "AB-Two", "Long-One", "Long-Two"]
        families = detect_families(groups, min_prefix_length=3)

        assert "AB" not in families
        assert "Long" in families

    def test_min_family_size(self) -> None:
        """Families below min_family_size should be excluded."""
        groups = ["Mistral-Large", "Mistral-Small", "Llama-One", "Llama-Two", "Solo-Only"]
        families = detect_families(groups, min_family_size=3)

        # Mistral has only 2 members, should be excluded with min_family_size=3
        assert "Mistral" not in families
        assert "Llama" not in families

    def test_members_are_sorted(self) -> None:
        """Family members should be returned in sorted order."""
        groups = ["Zeta-Two", "Zeta-One", "Zeta-Three"]
        families = detect_families(groups)

        assert families["Zeta"] == ["Zeta-One", "Zeta-Three", "Zeta-Two"]

    def test_no_double_claiming(self) -> None:
        """A group claimed by a longer prefix should not also appear in a shorter one."""
        groups = [
            "X-Y-A",
            "X-Y-B",
            "X-Z-One",
            "X-Z-Two",
        ]
        families = detect_families(groups)

        all_members = []
        for members in families.values():
            all_members.extend(members)
        # No duplicates
        assert len(all_members) == len(set(all_members))

    def test_real_world_mistral_family(self) -> None:
        """Realistic Mistral-like group names should form a single family."""
        groups = [
            "Mistral-Erebus",
            "Mistral-Large",
            "Mistral-Nemo",
            "Mistral-Small",
            "Mixtral",  # Different base — should NOT be in Mistral family
        ]
        families = detect_families(groups)

        assert "Mistral" in families
        assert "Mixtral" not in families.get("Mistral", [])

    def test_result_keys_are_sorted(self) -> None:
        """Result dict should have keys in sorted order."""
        groups = ["Zeta-A", "Zeta-B", "Alpha-A", "Alpha-B"]
        families = detect_families(groups)

        keys = list(families.keys())
        assert keys == sorted(keys)
