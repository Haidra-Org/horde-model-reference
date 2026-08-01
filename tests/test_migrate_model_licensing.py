"""Test behavioral guarantees of the one-time model licensing migration."""

import json
from pathlib import Path

import httpx
import pytest

import scripts.migrate_model_licensing as migration
from horde_model_reference.licensing import ModelLicensing, PermissionStatus


def _known_assignment() -> ModelLicensing:
    """Return a reviewed conclusion used to distinguish audited records from fallback records."""
    return ModelLicensing(
        license_expression="MIT",
        license_ids=("MIT",),
        commercial_use=PermissionStatus.ALLOWED,
        redistribution=PermissionStatus.ALLOWED_WITH_CONDITIONS,
        reviewed_by="migration-test",
    )


def test_migration_is_complete_fail_closed_and_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify every PRIMARY record is emitted and API ordering cannot change the review artifact."""
    primary_records = {
        "miscellaneous": {
            "z-unaudited": {"name": "z-unaudited", "record_type": "miscellaneous"},
            "a-audited": {"name": "a-audited", "record_type": "miscellaneous"},
        },
        "clip": {
            "middle": {"name": "middle", "record_type": "clip"},
        },
    }
    monkeypatch.setattr(migration, "_load_primary_records", lambda _url: primary_records)
    monkeypatch.setattr(
        migration,
        "_audit_assignments",
        lambda _path: {("miscellaneous", "a-audited"): _known_assignment()},
    )
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"

    assert (
        migration.main(
            ["--primary-api-url", "https://primary.invalid/api", "--output", str(first_output)],
        )
        == 0
    )
    reversed_records = {
        category: dict(reversed(list(category_records.items())))
        for category, category_records in reversed(list(primary_records.items()))
    }
    monkeypatch.setattr(migration, "_load_primary_records", lambda _url: reversed_records)
    assert (
        migration.main(
            ["--primary-api-url", "https://primary.invalid/api", "--output", str(second_output)],
        )
        == 0
    )

    first_payload = json.loads(first_output.read_text(encoding="utf-8"))
    assert first_output.read_bytes() == second_output.read_bytes()
    assert [(record["category"], record["model_name"]) for record in first_payload["records"]] == [
        ("clip", "middle"),
        ("miscellaneous", "a-audited"),
        ("miscellaneous", "z-unaudited"),
    ]
    by_name = {record["model_name"]: record["record"]["licensing"] for record in first_payload["records"]}
    assert by_name["a-audited"]["license_expression"] == "MIT"
    assert by_name["middle"]["license_expression"] == "NOASSERTION"
    assert by_name["z-unaudited"]["commercial_use"] == "unknown"
    assert by_name["z-unaudited"]["redistribution"] == "unknown"


def test_submit_routes_backfill_through_the_review_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify migration submission proposes canonical model updates instead of mutating another store."""
    monkeypatch.setattr(
        migration,
        "_load_primary_records",
        lambda _url: {
            "miscellaneous": {
                "submitted-model": {"name": "submitted-model", "record_type": "miscellaneous"},
            },
        },
    )
    monkeypatch.setattr(
        migration,
        "_audit_assignments",
        lambda _path: {("miscellaneous", "submitted-model"): _known_assignment()},
    )
    submissions: list[tuple[str, dict[str, str], dict[str, object]]] = []

    def _record_submission(
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
        timeout: float,
    ) -> httpx.Response:
        assert timeout == 60.0
        submissions.append((url, headers, json))
        return httpx.Response(202, request=httpx.Request("PUT", url))

    monkeypatch.setattr(migration.httpx, "put", _record_submission)
    output_path = tmp_path / "submitted.json"

    result = migration.main(
        [
            "--primary-api-url",
            "https://primary.invalid/api",
            "--output",
            str(output_path),
            "--submit",
            "--apikey",
            "queue-requestor-key",
        ],
    )

    assert result == 0
    assert len(submissions) == 1
    submission_url, headers, submitted_record = submissions[0]
    assert submission_url.endswith("/model_references/v2/miscellaneous/model/submitted-model")
    assert headers == {"apikey": "queue-requestor-key"}
    submitted_licensing = submitted_record["licensing"]
    assert isinstance(submitted_licensing, dict)
    assert submitted_licensing["license_expression"] == "MIT"
    assert output_path.is_file()
