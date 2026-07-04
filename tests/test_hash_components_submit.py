"""Tests for the offline pass's v2 submit dispatch (dry-run guard and request shape), without real network."""

from __future__ import annotations

from typing import Any

import pytest

from horde_model_reference import ModelClassification
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecord,
    GenericModelRecordConfig,
)
from scripts.hash_components import _submit_record


class _StubResponse:
    """A minimal stand-in for a requests.Response."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.text = ""


class _RecordingSession:
    """Records put/post calls instead of sending them."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def put(self, url: str, *, json: dict[str, Any], headers: dict[str, str], timeout: float) -> _StubResponse:
        self.calls.append(("PUT", url, json))
        return _StubResponse(200)

    def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str], timeout: float) -> _StubResponse:
        self.calls.append(("POST", url, json))
        return _StubResponse(201)


def _record() -> GenericModelRecord:
    return GenericModelRecord(
        record_type="image_generation",
        name="MyModel",
        model_classification=ModelClassification(domain="image", purpose="generation"),
        config=GenericModelRecordConfig(
            download=[
                DownloadRecord(
                    file_name="ae.safetensors",
                    file_url="https://example.invalid/ae",
                    file_purpose="vae",
                    content_hash="a" * 64,
                ),
            ],
        ),
    )


def _submit(session: _RecordingSession, *, dry_run: bool) -> None:
    _submit_record(
        _record(),
        "MyModel",
        MODEL_REFERENCE_CATEGORY.image_generation,
        api_version="v2",
        base_url="https://primary.invalid/",
        apikey="key",
        session=session,  # type: ignore[arg-type]
        dry_run=dry_run,
        update=False,
        timeout=1.0,
    )


def test_v2_dry_run_sends_nothing() -> None:
    """A dry-run submit constructs the payload but never calls the session."""
    session = _RecordingSession()
    _submit(session, dry_run=True)
    assert session.calls == []


def test_v2_real_puts_record_with_hash() -> None:
    """A real v2 submit PUTs to the per-model URL with the populated record as the body."""
    session = _RecordingSession()
    _submit(session, dry_run=False)
    assert len(session.calls) == 1
    method, url, payload = session.calls[0]
    assert method == "PUT"
    assert url == "https://primary.invalid/api/model_references/v2/image_generation/MyModel"
    assert payload["name"] == "MyModel"
    assert payload["config"]["download"][0]["content_hash"] == "a" * 64


@pytest.fixture
def _stub_legacy_converter(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the v2->legacy converter with a stub, so the v1 dispatch is tested independently of it."""
    legacy_payload = {"name": "MyModel", "baseline": "stable diffusion 1"}
    monkeypatch.setattr(
        "horde_model_reference.legacy.classes.legacy_converters.image_generation_record_to_legacy_dict",
        lambda record: legacy_payload,
    )
    return legacy_payload


def _submit_v1(session: _RecordingSession, *, dry_run: bool, update: bool) -> None:
    _submit_record(
        _record(),
        "MyModel",
        MODEL_REFERENCE_CATEGORY.image_generation,
        api_version="v1",
        base_url="https://primary.invalid/",
        apikey="key",
        session=session,  # type: ignore[arg-type]
        dry_run=dry_run,
        update=update,
        timeout=1.0,
    )


def test_v1_dry_run_sends_nothing(_stub_legacy_converter: dict[str, Any]) -> None:
    """A v1 dry-run builds the legacy payload but never calls the session."""
    session = _RecordingSession()
    _submit_v1(session, dry_run=True, update=False)
    assert session.calls == []


def test_v1_create_posts_legacy_payload(_stub_legacy_converter: dict[str, Any]) -> None:
    """A v1 create POSTs the legacy-shaped payload to the category endpoint."""
    session = _RecordingSession()
    _submit_v1(session, dry_run=False, update=False)
    method, url, payload = session.calls[0]
    assert method == "POST"
    assert url == "https://primary.invalid/api/model_references/v1/image_generation"
    assert payload is _stub_legacy_converter


def test_v1_update_puts_legacy_payload(_stub_legacy_converter: dict[str, Any]) -> None:
    """A v1 update with --update PUTs instead of POSTs."""
    session = _RecordingSession()
    _submit_v1(session, dry_run=False, update=True)
    assert session.calls[0][0] == "PUT"
