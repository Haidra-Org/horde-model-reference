"""Shared utility functions for the horde_model_reference package."""

import json
import os
import re
import time
from pathlib import Path
from threading import Lock

THROTTLED_LOG_INTERVAL_SECONDS = 30.0
"""Default width of the time box used by `throttled_log_level`, in seconds."""

_throttled_log_last_emitted: dict[str, float] = {}
_throttled_log_lock = Lock()


def model_name_to_showcase_folder_name(model_name: str) -> str:
    """Convert a model name to a lowercase, standardized and sanitized showcase folder name.

    Args:
        model_name (str): The model name to convert.

    Returns:
        str: This is a lowercase, sanitized version of the model name.

    """
    model_name = model_name.lower()
    model_name = model_name.replace("'", "")
    return re.sub(r"[^a-z0-9]", "_", model_name)


def throttled_log_level(
    key: str,
    interval_seconds: float,
    *,
    normal_level: str = "DEBUG",
    suppressed_level: str = "TRACE",
    now: float | None = None,
) -> str:
    """Return the level a repetitive log call should use for this emission.

    The first call for *key*, and the first call after *interval_seconds* have elapsed, get
    *normal_level*. Everything in between gets *suppressed_level* so that high-frequency
    per-request lines stay available at a lower level instead of being discarded. Callers pass
    the result to `logger.log(level, ...)`.

    Args:
        key: Identifier for the call site, usually a site name combined with a category.
        interval_seconds: Width of the time box. Values at or below zero disable throttling.
        normal_level: Level used for the emission that opens a time box.
        suppressed_level: Level used for emissions inside an open time box.
        now: Reading of a monotonic clock, in seconds. Defaults to `time.monotonic()`.

    Returns:
        str: The loguru level name to log this emission at.

    """
    if interval_seconds <= 0:
        return normal_level

    current = time.monotonic() if now is None else now

    with _throttled_log_lock:
        last_emitted = _throttled_log_last_emitted.get(key)
        if last_emitted is not None and current - last_emitted < interval_seconds:
            return suppressed_level

        _throttled_log_last_emitted[key] = current

    return normal_level


def reset_throttled_log_state(key: str | None = None) -> None:
    """Forget recorded emissions so the next call re-opens a time box.

    Args:
        key: The single key to forget. All keys are forgotten when omitted.

    """
    with _throttled_log_lock:
        if key is None:
            _throttled_log_last_emitted.clear()
        else:
            _throttled_log_last_emitted.pop(key, None)


def atomic_write_json(path: Path, payload: object, *, ensure_ascii: bool = True) -> None:
    """Atomically write JSON content to *path* using tmp + fsync + rename.

    Args:
        path: Target file path.
        payload: JSON-serializable object.
        ensure_ascii: Whether to escape non-ASCII characters.

    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=ensure_ascii)
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)
