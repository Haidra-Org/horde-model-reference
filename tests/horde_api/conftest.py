"""Pytest configuration for Horde API integration tests."""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, Any]:
    """Configure VCR for HTTP interaction recording in audit tests.

    Returns:
        Configuration dict for pytest-recording
    """
    return {
        # Filter sensitive headers
        "filter_headers": [
            ("authorization", "REDACTED"),
            ("x-api-key", "REDACTED"),
        ],
        # Decode compressed responses for readability
        "decode_compressed_response": True,
        # Use cassette if exists, record if missing (good for local dev)
        "record_mode": "once",
        # Match requests on these criteria
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
        # Store cassettes in this directory
        "cassette_library_dir": "tests/horde_api/cassettes",
        # Allow playback to be repeated
        "allow_playback_repeats": True,
    }
