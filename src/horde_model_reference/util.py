"""Shared utility functions for the horde_model_reference package."""

import json
import os
import re
from pathlib import Path


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
