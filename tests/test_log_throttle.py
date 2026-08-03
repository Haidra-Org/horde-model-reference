"""Tests for the time-boxed log level helper in `horde_model_reference.util`."""

import pytest

from horde_model_reference.util import reset_throttled_log_state, throttled_log_level


@pytest.fixture(autouse=True)
def _clean_throttle_state() -> None:
    """Ensure each test starts from an empty throttle ledger."""
    reset_throttled_log_state()


def test_first_emission_uses_the_normal_level() -> None:
    """An unseen key is always allowed through at the normal level."""
    assert throttled_log_level("alpha", 30.0, now=100.0) == "DEBUG"


def test_repeat_within_the_interval_is_suppressed() -> None:
    """Repeats inside the interval are demoted rather than dropped."""
    assert throttled_log_level("alpha", 30.0, now=100.0) == "DEBUG"
    assert throttled_log_level("alpha", 30.0, now=100.5) == "TRACE"
    assert throttled_log_level("alpha", 30.0, now=129.9) == "TRACE"


def test_normal_level_returns_once_the_interval_elapses() -> None:
    """The interval boundary re-arms the normal level and restarts the box."""
    assert throttled_log_level("alpha", 30.0, now=100.0) == "DEBUG"
    assert throttled_log_level("alpha", 30.0, now=130.0) == "DEBUG"
    assert throttled_log_level("alpha", 30.0, now=131.0) == "TRACE"
    assert throttled_log_level("alpha", 30.0, now=160.0) == "DEBUG"


def test_keys_are_throttled_independently() -> None:
    """Traffic on one key never consumes another key's budget."""
    assert throttled_log_level("alpha", 30.0, now=100.0) == "DEBUG"
    assert throttled_log_level("beta", 30.0, now=100.0) == "DEBUG"
    assert throttled_log_level("alpha", 30.0, now=101.0) == "TRACE"
    assert throttled_log_level("beta", 30.0, now=101.0) == "TRACE"


def test_custom_levels_are_honoured() -> None:
    """Callers may pick the pair of levels used for allowed and suppressed calls."""
    assert throttled_log_level("alpha", 30.0, normal_level="INFO", suppressed_level="DEBUG", now=100.0) == "INFO"
    assert throttled_log_level("alpha", 30.0, normal_level="INFO", suppressed_level="DEBUG", now=100.1) == "DEBUG"


def test_reset_accepts_a_single_key() -> None:
    """Resetting one key leaves the rest of the ledger intact."""
    throttled_log_level("alpha", 30.0, now=100.0)
    throttled_log_level("beta", 30.0, now=100.0)

    reset_throttled_log_state("alpha")

    assert throttled_log_level("alpha", 30.0, now=100.1) == "DEBUG"
    assert throttled_log_level("beta", 30.0, now=100.1) == "TRACE"


def test_reset_without_a_key_clears_everything() -> None:
    """Resetting without a key drops all recorded emissions."""
    throttled_log_level("alpha", 30.0, now=100.0)
    throttled_log_level("beta", 30.0, now=100.0)

    reset_throttled_log_state()

    assert throttled_log_level("alpha", 30.0, now=100.1) == "DEBUG"
    assert throttled_log_level("beta", 30.0, now=100.1) == "DEBUG"


def test_non_positive_interval_never_suppresses() -> None:
    """A zero or negative interval disables throttling entirely."""
    assert throttled_log_level("alpha", 0.0, now=100.0) == "DEBUG"
    assert throttled_log_level("alpha", 0.0, now=100.0) == "DEBUG"


def test_monotonic_clock_is_used_when_now_is_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting `now` reads the process monotonic clock."""
    current = 500.0

    monkeypatch.setattr("horde_model_reference.util.time.monotonic", lambda: current)

    assert throttled_log_level("alpha", 30.0) == "DEBUG"
    assert throttled_log_level("alpha", 30.0) == "TRACE"

    current += 30.0
    assert throttled_log_level("alpha", 30.0) == "DEBUG"
