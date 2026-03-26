"""Tests for PendingQueueStore, including corruption recovery (AQ-4)."""

from __future__ import annotations

import json
from pathlib import Path

from horde_model_reference.audit.events import AuditOperation
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.pending_queue.models import PendingChangeRecord, PendingChangeStatus
from horde_model_reference.pending_queue.store import PendingQueueStore


def _make_record(change_id: int, batch_id: int | None = None) -> dict:
    return PendingChangeRecord(
        change_id=change_id,
        category=MODEL_REFERENCE_CATEGORY.image_generation,
        model_name=f"test-model-{change_id}",
        operation=AuditOperation.CREATE,
        requested_by="user1",
        requested_username="User One",
        status=PendingChangeStatus.APPROVED,
        batch_id=batch_id,
    ).model_dump(mode="json", exclude_none=True)


class TestStoreCorruptionRecovery:
    """AQ-4: Store recovers IDs from changes.json when index.json is corrupt."""

    def test_corrupt_index_recovers_from_changes(self, tmp_path: Path) -> None:
        """If index.json is corrupt, store should recover last_change_id and last_batch_id from changes.json."""
        root = tmp_path / "queue"
        root.mkdir()

        changes = [_make_record(5, batch_id=2), _make_record(10, batch_id=3)]
        (root / "changes.json").write_text(json.dumps(changes))
        (root / "index.json").write_text("NOT VALID JSON{{{")

        store = PendingQueueStore(root_path=root)

        # Should have recovered the highest change_id and batch_id
        new_record = PendingChangeRecord(
            change_id=0,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
            model_name="new-model",
            operation=AuditOperation.CREATE,
            requested_by="user2",
            requested_username="User Two",
        )
        result = store.enqueue_change(new_record)
        assert result.change_id == 11  # 10 + 1

    def test_corrupt_index_recovers_batch_id(self, tmp_path: Path) -> None:
        """If index.json is corrupt, store should recover last_batch_id from changes.json."""
        root = tmp_path / "queue"
        root.mkdir()

        changes = [_make_record(3, batch_id=7)]
        (root / "changes.json").write_text(json.dumps(changes))
        (root / "index.json").write_text("{broken")

        store = PendingQueueStore(root_path=root)
        batch_id = store.get_or_create_pending_batch_id()
        # batch_id 7 is already on an APPROVED change, so it should be reused
        assert batch_id == 7

    def test_corrupt_index_no_changes_starts_at_zero(self, tmp_path: Path) -> None:
        """If index.json is corrupt and there are no changes, store should start IDs at zero."""
        root = tmp_path / "queue"
        root.mkdir()

        (root / "index.json").write_text("corrupt!")
        # No changes.json

        store = PendingQueueStore(root_path=root)
        new_record = PendingChangeRecord(
            change_id=0,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
            model_name="model",
            operation=AuditOperation.CREATE,
            requested_by="user",
            requested_username="User",
        )
        result = store.enqueue_change(new_record)
        assert result.change_id == 1  # starts from 0

    def test_missing_index_is_not_corruption(self, tmp_path: Path) -> None:
        """If index.json is missing, store should work normally."""
        root = tmp_path / "queue"
        root.mkdir()

        # Fresh store with no files — should work normally
        store = PendingQueueStore(root_path=root)
        new_record = PendingChangeRecord(
            change_id=0,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
            model_name="model",
            operation=AuditOperation.CREATE,
            requested_by="user",
            requested_username="User",
        )
        result = store.enqueue_change(new_record)
        assert result.change_id == 1

    def test_recovery_persists_repaired_state(self, tmp_path: Path) -> None:
        """After recovery, index.json should be rewritten with correct values."""
        root = tmp_path / "queue"
        root.mkdir()

        changes = [_make_record(8, batch_id=4)]
        (root / "changes.json").write_text(json.dumps(changes))
        (root / "index.json").write_text("{{garbage}}")

        PendingQueueStore(root_path=root)

        # Verify index.json was rewritten correctly
        repaired = json.loads((root / "index.json").read_text())
        assert repaired["last_change_id"] == 8
        assert repaired["last_batch_id"] == 4


class TestStoreApplyingState:
    """AQ-1/CR-7: APPROVED → APPLYING → APPLIED state transitions."""

    def _make_store_with_approved(self, tmp_path: Path) -> tuple[PendingQueueStore, int]:
        """Create a store with one APPROVED record and return (store, change_id)."""
        store = PendingQueueStore(root_path=tmp_path / "queue")
        record = PendingChangeRecord(
            change_id=0,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
            model_name="test-model",
            operation=AuditOperation.CREATE,
            requested_by="user",
            requested_username="User",
            status=PendingChangeStatus.APPROVED,
            batch_id=1,
        )
        persisted = store.enqueue_change(record)
        # Manually set to APPROVED (enqueue creates as whatever status is passed)
        return store, persisted.change_id

    def test_reserve_transitions_to_applying(self, tmp_path: Path) -> None:
        """reserve_for_apply sets status to APPLYING."""
        store, change_id = self._make_store_with_approved(tmp_path)
        reserved = store.reserve_for_apply(change_id=change_id, reservation_id="job-1")

        assert reserved.status == PendingChangeStatus.APPLYING
        assert reserved.applied_job_id == "job-1"

        # Confirm persisted state
        reloaded = store.get_change(change_id)
        assert reloaded is not None
        assert reloaded.status == PendingChangeStatus.APPLYING

    def test_clear_reservation_reverts_applying_to_approved(self, tmp_path: Path) -> None:
        """Clearing reservation on APPLYING record reverts to APPROVED."""
        store, change_id = self._make_store_with_approved(tmp_path)
        store.reserve_for_apply(change_id=change_id, reservation_id="job-1")
        store.clear_reservation_if_matches(change_id=change_id, reservation_id="job-1")

        record = store.get_change(change_id)
        assert record is not None
        assert record.status == PendingChangeStatus.APPROVED
        assert record.applied_job_id is None

    def test_get_applying_records(self, tmp_path: Path) -> None:
        """get_applying_records returns only APPLYING records."""
        store, change_id = self._make_store_with_approved(tmp_path)
        assert store.get_applying_records() == []

        store.reserve_for_apply(change_id=change_id, reservation_id="job-1")
        applying = store.get_applying_records()
        assert len(applying) == 1
        assert applying[0].change_id == change_id

    def test_revert_applying_to_approved(self, tmp_path: Path) -> None:
        """revert_applying_to_approved resets status and clears reservation."""
        store, change_id = self._make_store_with_approved(tmp_path)
        store.reserve_for_apply(change_id=change_id, reservation_id="job-1")
        reverted = store.revert_applying_to_approved(change_id)

        assert reverted.status == PendingChangeStatus.APPROVED
        assert reverted.applied_job_id is None

    def test_revert_applying_rejects_non_applying(self, tmp_path: Path) -> None:
        """revert_applying_to_approved raises on non-APPLYING record."""
        import pytest

        store, change_id = self._make_store_with_approved(tmp_path)
        with pytest.raises(ValueError, match="not in APPLYING state"):
            store.revert_applying_to_approved(change_id)
