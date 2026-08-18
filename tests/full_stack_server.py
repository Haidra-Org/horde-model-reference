"""Serve an isolated authenticated API for opt-in browser lifecycle tests.

This module is never imported by the production application. Start it explicitly
with uvicorn while ``HMR_FULL_STACK_DATA_ROOT`` points at a disposable directory.
The browser test runner is a different origin, so the served app must allow it via
``HORDE_MODEL_REFERENCE_CORS_ALLOWED_ORIGINS`` (for example ``http://127.0.0.1:4400``);
without it the browser discards every API response and the UI reports a read-only backend.
"""

from __future__ import annotations

import os
from collections.abc import Collection
from pathlib import Path
from typing import Any

from horde_model_reference import (
    CanonicalFormat,
    ModelReferenceManager,
    PrefetchStrategy,
    ReplicateMode,
    horde_model_reference_settings,
)
from horde_model_reference.backends.filesystem_backend import FileSystemBackend
from horde_model_reference.service import shared
from horde_model_reference.service.app import app
from horde_model_reference.service.shared import HordeUserContext, get_model_reference_manager

_TEST_API_KEY = "full-stack-test-key"
_TEST_USER_ID = "full-stack-maintainer"
_TEST_USERNAME = "full-stack-maintainer#1"

_data_root_value = os.environ.get("HMR_FULL_STACK_DATA_ROOT")
if not _data_root_value:
    raise RuntimeError("HMR_FULL_STACK_DATA_ROOT must point at a disposable test directory")

_data_root = Path(_data_root_value).resolve()
_data_root.mkdir(parents=True, exist_ok=True)
_queue_root = _data_root / "pending_queue"
_queue_root.mkdir(parents=True, exist_ok=True)

horde_model_reference_settings.canonical_format = CanonicalFormat.LEGACY
horde_model_reference_settings.pending_queue.root_path_override = str(_queue_root)
horde_model_reference_settings.pending_queue.requestor_ids = [_TEST_USER_ID]
horde_model_reference_settings.pending_queue.approver_ids = [_TEST_USER_ID]
horde_model_reference_settings.licensing.editor_ids = [_TEST_USER_ID]

_backend = FileSystemBackend(
    base_path=_data_root,
    cache_ttl_seconds=0,
    replicate_mode=ReplicateMode.PRIMARY,
)
ModelReferenceManager._instance = None
_manager = ModelReferenceManager(
    backend=_backend,
    prefetch_strategy=PrefetchStrategy.LAZY,
    replicate_mode=ReplicateMode.PRIMARY,
)


async def _authenticate_test_user(
    apikey: str,
    client: Any,  # noqa: ANN401
    *,
    allowed_user_ids: Collection[str] | None = None,
) -> HordeUserContext | None:
    """Return the fixed test maintainer only for the fixture API key."""
    del client
    if apikey != _TEST_API_KEY:
        return None
    if allowed_user_ids is not None and _TEST_USER_ID not in allowed_user_ids:
        return None
    return HordeUserContext(user_id=_TEST_USER_ID, username=_TEST_USERNAME)


shared.auth_against_horde = _authenticate_test_user
app.dependency_overrides[get_model_reference_manager] = lambda: _manager
