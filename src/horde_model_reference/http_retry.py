"""Shared HTTP retry utilities and circuit breaker state for the AI Horde integration.

Uses tenacity for retry logic with full jitter exponential backoff. Provides both
sync and async retry decorator factories, structured retry logging, and a circuit
breaker that tracks degraded connectivity to the external AI Horde API.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

import httpx
from loguru import logger
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

TRANSIENT_HTTP_EXCEPTIONS: tuple[type[Exception], ...] = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
)
"""Network-level exceptions that are always worth retrying."""


def is_retryable_status_code(status_code: int) -> bool:
    """Return True if the HTTP status code suggests a transient server-side issue."""
    return status_code >= 500 or status_code == 429


class RetryableHTTPStatusError(Exception):
    """Raised when an HTTP response has a retryable status code (5xx, 429).

    Wraps the original httpx response so callers can inspect it after retries are exhausted.
    """

    def __init__(self, response: httpx.Response) -> None:
        """Wrap an httpx response with a retryable status code."""
        self.response = response
        super().__init__(f"HTTP {response.status_code} from {response.url}")


def _log_retry(retry_state: RetryCallState) -> None:
    """Emit a structured log line before each retry attempt."""
    outcome = retry_state.outcome
    exc = outcome.exception() if outcome else None
    wait = retry_state.next_action.sleep if retry_state.next_action else 0

    logger.warning(
        "HTTP retry | attempt={attempt} | wait={wait:.2f}s | error={error}",
        attempt=retry_state.attempt_number,
        wait=wait,
        error=str(exc) if exc else "unknown",
    )


def http_retry_sync(
    *,
    max_attempts: int = 3,
    min_wait: float = 0.5,
    max_wait: float = 10.0,
    extra_exceptions: tuple[type[Exception], ...] = (),
) -> Retrying:
    """Create a synchronous tenacity Retrying context manager.

    Usage::

        for attempt in http_retry_sync():
            with attempt:
                response = httpx.get(url)
                if is_retryable_status_code(response.status_code):
                    raise RetryableHTTPStatusError(response)

    Args:
        max_attempts: Maximum number of attempts before giving up.
        min_wait: Minimum wait time for jittered exponential backoff.
        max_wait: Maximum wait time for jittered exponential backoff.
        extra_exceptions: Additional exception types to retry on beyond the defaults.

    """
    return Retrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(multiplier=0.5, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(TRANSIENT_HTTP_EXCEPTIONS + extra_exceptions + (RetryableHTTPStatusError,)),
        before_sleep=_log_retry,
        reraise=True,
    )


def http_retry_async(
    *,
    max_attempts: int = 3,
    min_wait: float = 0.5,
    max_wait: float = 10.0,
    extra_exceptions: tuple[type[Exception], ...] = (),
) -> AsyncRetrying:
    """Create an asynchronous tenacity AsyncRetrying context manager.

    Usage::

        async for attempt in http_retry_async():
            with attempt:
                response = await client.get(url)
                if is_retryable_status_code(response.status_code):
                    raise RetryableHTTPStatusError(response)

    Args:
        max_attempts: Maximum number of attempts before giving up.
        min_wait: Minimum wait time for jittered exponential backoff.
        max_wait: Maximum wait time for jittered exponential backoff.
        extra_exceptions: Additional exception types to retry on beyond the defaults.

    """
    return AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(multiplier=0.5, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(TRANSIENT_HTTP_EXCEPTIONS + extra_exceptions + (RetryableHTTPStatusError,)),
        before_sleep=_log_retry,
        reraise=True,
    )


@dataclass
class _CircuitState:
    """Internal mutable state for the circuit breaker."""

    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    is_open: bool = False
    lock: RLock = field(default_factory=RLock)


class HordeAPICircuitBreaker:
    """Lightweight circuit breaker for the external AI Horde API.

    States:
        CLOSED  - normal operation, requests go through.
        OPEN    - too many consecutive failures; requests are short-circuited
                  for ``cooldown_seconds``. After cooldown, a single probe
                  request is allowed (half-open). On success the circuit
                  closes; on failure it stays open.

    The breaker exposes ``is_degraded`` and ``seconds_until_retry`` for the
    ``/heartbeat`` endpoint and log messages on hot paths.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        cooldown_seconds: float = 120.0,
    ) -> None:
        """Initialize circuit breaker with failure threshold and cooldown."""
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._state = _CircuitState()

    @property
    def is_degraded(self) -> bool:
        """True when the circuit is open (AI Horde unreachable)."""
        with self._state.lock:
            if not self._state.is_open:
                return False
            # Auto-transition to half-open after cooldown
            if self._cooldown_elapsed():
                return True  # still degraded, but will allow a probe
            return True

    @property
    def seconds_until_retry(self) -> float | None:
        """Seconds remaining before the next probe attempt, or None if not degraded."""
        with self._state.lock:
            if not self._state.is_open:
                return None
            remaining = self._cooldown_seconds - (time.monotonic() - self._state.last_failure_time)
            return max(0.0, remaining)

    @property
    def consecutive_failures(self) -> int:
        """Number of consecutive failures recorded."""
        with self._state.lock:
            return self._state.consecutive_failures

    def should_allow_request(self) -> bool:
        """Return True if a request should proceed (circuit closed or half-open probe)."""
        with self._state.lock:
            if not self._state.is_open:
                return True
            if self._cooldown_elapsed():
                logger.info(
                    "AI Horde circuit breaker: cooldown elapsed, allowing probe request "
                    f"(failures={self._state.consecutive_failures})"
                )
                return True
            return False

    def record_success(self) -> None:
        """Record a successful request, closing the circuit if it was open."""
        with self._state.lock:
            was_open = self._state.is_open
            self._state.consecutive_failures = 0
            self._state.is_open = False
            self._state.last_success_time = time.monotonic()
        if was_open:
            logger.info("AI Horde circuit breaker: CLOSED (service recovered)")

    def record_failure(self) -> None:
        """Record a failed request, potentially opening the circuit."""
        with self._state.lock:
            self._state.consecutive_failures += 1
            self._state.last_failure_time = time.monotonic()

            if not self._state.is_open and self._state.consecutive_failures >= self._failure_threshold:
                self._state.is_open = True
                logger.error(
                    f"AI Horde circuit breaker: OPEN after {self._state.consecutive_failures} consecutive failures. "
                    f"Requests will be short-circuited for {self._cooldown_seconds:.0f}s."
                )

    def get_status_dict(self) -> dict[str, Any]:
        """Return a dict suitable for inclusion in the /heartbeat response."""
        with self._state.lock:
            return {
                "degraded": self._state.is_open,
                "consecutive_failures": self._state.consecutive_failures,
                "seconds_until_retry": round(self.seconds_until_retry, 1) if self.seconds_until_retry else None,
            }

    def _cooldown_elapsed(self) -> bool:
        return (time.monotonic() - self._state.last_failure_time) >= self._cooldown_seconds


# Module-level singleton
horde_api_circuit_breaker = HordeAPICircuitBreaker()
"""Global circuit breaker for the external AI Horde API.

Import this from any module that calls the Horde API to check/update state.
"""
