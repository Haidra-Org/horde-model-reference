from __future__ import annotations

import copy
from pathlib import Path
from random import Random
from typing import Any

from horde_model_reference.audit import (
    AuditDomain,
    AuditPayload,
    AuditReplayer,
    AuditTrailReader,
    AuditTrailWriter,
)
from horde_model_reference.audit.events import AuditOperation

LEGACY_DOMAIN = AuditDomain("legacy")
CREATE_OPERATION = AuditOperation("create")
UPDATE_OPERATION = AuditOperation("update")
DELETE_OPERATION = AuditOperation("delete")
CATEGORY_NAME = "image_generation"


def _writer(tmp_path: Path) -> AuditTrailWriter:
    return AuditTrailWriter(root_path=tmp_path / "audit")


def _build_snapshot(name: str, *, revision: int, extra_seed: int) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"{name}-description-{revision}",
        "revision": revision,
        "tags": [f"tag-{revision % 3}", f"group-{len(name)}"],
        "metadata": {"score": extra_seed % 100},
    }


def test_audit_trail_reader_applies_filters(tmp_path: Path) -> None:
    """Audit trail reader should support filtering by category and event id."""
    writer = _writer(tmp_path)

    for index in range(5):
        payload = AuditPayload.from_create({"name": f"model-{index}"})
        writer.append_event(
            domain=LEGACY_DOMAIN,
            category="image_generation" if index < 4 else "text_generation",
            model_name=f"model-{index}",
            operation=CREATE_OPERATION,
            logical_user_id="u-1",
            payload=payload,
        )

    reader = AuditTrailReader(root_path=tmp_path / "audit")
    events = list(
        reader.iter_events(
            domains={LEGACY_DOMAIN},
            categories={CATEGORY_NAME},
            min_event_id=2,
            max_event_id=4,
        )
    )

    assert [event.event_id for event in events] == [2, 3, 4]
    assert {event.model_name for event in events} == {"model-1", "model-2", "model-3"}


def test_audit_replayer_reconstructs_state(tmp_path: Path) -> None:
    """Replayer should rebuild model state using audit events."""
    writer = _writer(tmp_path)

    create_payload = {"name": "model-a", "description": "initial", "version": 1}
    writer.append_event(
        domain=LEGACY_DOMAIN,
        category=CATEGORY_NAME,
        model_name="model-a",
        operation=CREATE_OPERATION,
        logical_user_id="u-1",
        payload=AuditPayload.from_create(create_payload),
    )

    writer.append_event(
        domain=LEGACY_DOMAIN,
        category=CATEGORY_NAME,
        model_name="model-a",
        operation=UPDATE_OPERATION,
        logical_user_id="u-1",
        payload=AuditPayload.from_update(
            create_payload,
            {"name": "model-a", "description": "updated", "version": 2},
        ),
    )

    writer.append_event(
        domain=LEGACY_DOMAIN,
        category=CATEGORY_NAME,
        model_name="model-b",
        operation=CREATE_OPERATION,
        logical_user_id="u-2",
        payload=AuditPayload.from_create({"name": "model-b", "description": "temp"}),
    )

    writer.append_event(
        domain=LEGACY_DOMAIN,
        category=CATEGORY_NAME,
        model_name="model-b",
        operation=DELETE_OPERATION,
        logical_user_id="u-2",
    )

    reader = AuditTrailReader(root_path=tmp_path / "audit")
    replayer = AuditReplayer(reader=reader)
    result = replayer.reconstruct_state(domain=LEGACY_DOMAIN, category=CATEGORY_NAME)

    assert result.applied_events == 4
    assert result.last_event_id == 4
    assert set(result.state.keys()) == {"model-a"}
    assert result.state["model-a"]["description"] == "updated"
    assert result.state["model-a"]["version"] == 2


def test_audit_replayer_handles_complex_mixed_sequences(tmp_path: Path) -> None:
    """Mixed sequences of create/update/delete should replay to the exact final state."""
    rng = Random(1337)
    writer = AuditTrailWriter(root_path=tmp_path / "audit", max_file_size_bytes=512)
    expected_state: dict[str, dict[str, Any]] = {}
    emitted_events = 0

    for _ in range(250):
        model_name = f"model-{rng.randint(0, 25)}"
        action = rng.choice([CREATE_OPERATION, UPDATE_OPERATION, DELETE_OPERATION])

        if action is CREATE_OPERATION:
            if model_name in expected_state:
                continue
            snapshot = _build_snapshot(model_name, revision=0, extra_seed=rng.randint(0, 10_000))
            writer.append_event(
                domain=LEGACY_DOMAIN,
                category=CATEGORY_NAME,
                model_name=model_name,
                operation=CREATE_OPERATION,
                logical_user_id="complex-user",
                payload=AuditPayload.from_create(snapshot),
            )
            expected_state[model_name] = snapshot
            emitted_events += 1
            continue

        if action is UPDATE_OPERATION:
            if model_name not in expected_state:
                continue
            before = copy.deepcopy(expected_state[model_name])
            after = copy.deepcopy(before)
            after["revision"] = before.get("revision", 0) + 1
            after["description"] = f"{model_name}-description-{after['revision']}"
            after["metadata"]["score"] = (after["metadata"].get("score", 0) + 7) % 100
            writer.append_event(
                domain=LEGACY_DOMAIN,
                category=CATEGORY_NAME,
                model_name=model_name,
                operation=UPDATE_OPERATION,
                logical_user_id="complex-user",
                payload=AuditPayload.from_update(before, after),
            )
            expected_state[model_name] = after
            emitted_events += 1
            continue

        if model_name not in expected_state:
            continue
        removed_snapshot = expected_state.pop(model_name)
        writer.append_event(
            domain=LEGACY_DOMAIN,
            category=CATEGORY_NAME,
            model_name=model_name,
            operation=DELETE_OPERATION,
            logical_user_id="complex-user",
            payload=AuditPayload.from_delete(removed_snapshot),
        )
        emitted_events += 1

    reader = AuditTrailReader(root_path=tmp_path / "audit")
    events = list(
        reader.iter_events(
            domains={LEGACY_DOMAIN},
            categories={CATEGORY_NAME},
        )
    )

    assert len(events) == emitted_events
    assert [event.event_id for event in events] == list(range(1, emitted_events + 1))

    replayer = AuditReplayer(reader=reader)
    result = replayer.reconstruct_state(domain=LEGACY_DOMAIN, category=CATEGORY_NAME)
    assert result.applied_events == emitted_events
    assert result.state == expected_state


def test_audit_replayer_regression_fixed_sequence(tmp_path: Path) -> None:
    """Deterministic regression sequence should produce a fixed final state."""
    writer = AuditTrailWriter(root_path=tmp_path / "audit", max_file_size_bytes=256)

    alpha_v1 = {"name": "model-alpha", "description": "alpha v1", "revision": 1, "metadata": {"score": 10}}
    alpha_v2 = {"name": "model-alpha", "description": "alpha v2", "revision": 2, "metadata": {"score": 20}}
    alpha_v3 = {"name": "model-alpha", "description": "alpha v3", "revision": 3, "metadata": {"score": 25}}
    alpha_v4 = {"name": "model-alpha", "description": "alpha reboot", "revision": 1, "metadata": {"score": 5}}

    beta_v1 = {"name": "model-beta", "description": "beta v1", "revision": 1}

    gamma_v1 = {"name": "model-gamma", "description": "gamma v1", "revision": 4, "metadata": {"score": 44}}
    gamma_v2 = {"name": "model-gamma", "description": "gamma stabilized", "revision": 5, "metadata": {"score": 50}}

    sequence = [
        ("model-alpha", CREATE_OPERATION, AuditPayload.from_create(alpha_v1)),
        ("model-beta", CREATE_OPERATION, AuditPayload.from_create(beta_v1)),
        ("model-alpha", UPDATE_OPERATION, AuditPayload.from_update(alpha_v1, alpha_v2)),
        ("model-beta", DELETE_OPERATION, AuditPayload.from_delete(beta_v1)),
        ("model-gamma", CREATE_OPERATION, AuditPayload.from_create(gamma_v1)),
        ("model-alpha", UPDATE_OPERATION, AuditPayload.from_update(alpha_v2, alpha_v3)),
        ("model-alpha", DELETE_OPERATION, AuditPayload.from_delete(alpha_v3)),
        ("model-alpha", CREATE_OPERATION, AuditPayload.from_create(alpha_v4)),
        ("model-gamma", UPDATE_OPERATION, AuditPayload.from_update(gamma_v1, gamma_v2)),
    ]

    for model_name, operation, payload in sequence:
        writer.append_event(
            domain=LEGACY_DOMAIN,
            category=CATEGORY_NAME,
            model_name=model_name,
            operation=operation,
            logical_user_id="regression-user",
            payload=payload,
        )

    reader = AuditTrailReader(root_path=tmp_path / "audit")
    events = list(reader.iter_events(domains={LEGACY_DOMAIN}, categories={CATEGORY_NAME}))

    expected_operations = [entry[1] for entry in sequence]
    assert [event.operation for event in events] == expected_operations
    assert [event.event_id for event in events] == list(range(1, len(sequence) + 1))

    replayer = AuditReplayer(reader=reader)
    result = replayer.reconstruct_state(domain=LEGACY_DOMAIN, category=CATEGORY_NAME)

    assert result.applied_events == len(sequence)
    assert result.state == {
        "model-alpha": alpha_v4,
        "model-gamma": gamma_v2,
    }
