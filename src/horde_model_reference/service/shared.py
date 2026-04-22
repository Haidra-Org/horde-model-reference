import urllib.parse
from collections.abc import Collection
from dataclasses import dataclass
from enum import auto
from typing import Literal

import httpx
from fastapi import HTTPException
from fastapi.security import APIKeyHeader
from loguru import logger
from pydantic import BaseModel
from strenum import StrEnum

from horde_model_reference import (
    CanonicalFormat,
    ModelReferenceManager,
    ai_horde_worker_settings,
    horde_model_reference_settings,
)

header_auth_scheme = APIKeyHeader(name="apikey")

DEFAULT_AUTH_TIMEOUT_SECONDS = 10.0

httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(DEFAULT_AUTH_TIMEOUT_SECONDS))

v1_prefix = "/model_references/v1"
v2_prefix = "/model_references/v2"
statistics_prefix = "/model_references/statistics"


class PathVariables(StrEnum):
    """Path variables used in the router."""

    model_category_name = auto()
    model_name = auto()


class RouteNames(StrEnum):
    """Route names used in the router."""

    get_reference_info = auto()
    get_reference_names = auto()
    get_reference_by_category = auto()
    get_text_generation_reference = auto()
    get_single_model = auto()
    image_generation_model = auto()
    text_generation_model = auto()
    blip_model = auto()
    clip_model = auto()
    codeformer_model = auto()
    controlnet_model = auto()
    esrgan_model = auto()
    gfpgan_model = auto()
    safety_checker_model = auto()
    miscellaneous_model = auto()
    create_model = auto()
    update_model = auto()
    update_image_generation_model = auto()
    update_text_generation_model = auto()
    update_controlnet_model = auto()
    delete_model = auto()
    get_models_with_stats = auto()
    get_category_statistics = auto()
    get_category_deletion_risk = auto()

    # V1 metadata routes
    get_legacy_last_updated = auto()
    get_legacy_category_last_updated = auto()
    get_all_legacy_metadata = auto()
    get_legacy_category_metadata = auto()

    # V2 metadata routes
    get_v2_last_updated = auto()
    get_v2_category_last_updated = auto()
    get_all_v2_metadata = auto()
    get_v2_category_metadata = auto()


class RouteRegistry:
    """Registry for routes.

    Routes are stored with a composite key of (prefix, route_name) to support
    multiple API versions (v1, v2) with overlapping route names.
    """

    _routes: dict[tuple[str, RouteNames], str]

    def __init__(self) -> None:
        """Initialize the route registry."""
        self._routes = {}

    def register_route(self, prefix: str, route_name: RouteNames, path: str) -> None:
        """Register a route with its path.

        Args:
            prefix (str): The prefix to use for the route (e.g., "/model_references/v1").
            route_name (RouteNames): The name of the route to register.
            path (str): The path of the route.
        """
        key = (prefix, route_name)
        self._routes[key] = f"{prefix}{path}"

    def url_for(self, route_name: RouteNames, path_params: dict[str, str], prefix: str) -> str:
        """Get the URL for a registered route.

        Args:
            route_name (RouteNames): The name of the route to get the URL for.
            path_params (dict[str, str]): The path parameters to include in the URL.
            prefix (str | None): The API prefix (e.g., "/model_references/v2").
                If None, defaults to v2_prefix for backward compatibility.

        Returns:
            str: The complete URL path with parameters substituted.

        Raises:
            ValueError: If the route is not registered.
        """
        key = (prefix, route_name)
        path = self._routes.get(key)
        if path is None:
            raise ValueError(f"Route {route_name} with prefix {prefix} is not registered.")

        for key_name, value in path_params.items():
            path = path.replace(f"{{{key_name}}}", value)
        return path


route_registry = RouteRegistry()


class Operation(StrEnum):
    """CRUD operation types."""

    create = "create"
    update = "update"
    delete = "delete"


_requestor_fallback_logged = False
_approver_fallback_logged = False


@dataclass(frozen=True)
class HordeUserContext:
    """Immutable details about the authenticated Horde user."""

    user_id: str
    username: str


async def auth_against_horde(
    apikey: str,
    client: httpx.AsyncClient,
    *,
    allowed_user_ids: Collection[str] | None = None,
) -> HordeUserContext | None:
    """Authenticate the provided API key against the AI-Horde.

    This uses the endpoint defined by AI_HORDE_URL by AIHordeClientSettings in haidra_core.

    Args:
        apikey (str): The API key to authenticate.
        client (httpx.AsyncClient): The HTTP client to use for the request.
        allowed_user_ids (Collection[str] | None): Optional allowlist of Horde user IDs permitted for the caller.

    Returns:
        HordeUserContext | None: User details if authentication is successful, None otherwise.

    Raises:
        HTTPException: 503 if the Horde auth service is unreachable or times out.
    """
    find_user_subpath = "v2/find_user"
    url = urllib.parse.urljoin(
        str(ai_horde_worker_settings.ai_horde_url),
        find_user_subpath,
    )

    try:
        response = await client.get(
            url,
            headers={"apikey": f"{apikey}"},
        )
    except httpx.TimeoutException:
        logger.warning("Horde auth service timed out")
        raise HTTPException(status_code=503, detail="Auth service timed out") from None
    except httpx.HTTPError as exc:
        logger.warning(f"Horde auth service unreachable: {exc}")
        raise HTTPException(status_code=503, detail="Auth service unavailable") from None

    if response.status_code != 200:
        return None

    user_data = response.json()
    user_name = user_data.get("username", "")

    if "#" not in user_name:
        logger.warning(f"Unknown apikey: {user_data}")
        return None

    user_id = user_name.split("#")[-1]

    if allowed_user_ids and user_id not in allowed_user_ids:
        logger.warning(f"Unauthorized user ID: {user_id}")
        return None

    return HordeUserContext(user_id=user_id, username=user_name)


def _normalize_ids(values: Collection[str]) -> set[str]:
    return {value.strip() for value in values if value and value.strip()}


def _fallback_allowed_users(context: Literal["requestor", "approver"]) -> set[str]:
    """Return an empty set and log a warning when no allowlist is configured.

    Fails closed: if no allowlist is configured, no users are authorized.
    """
    global _requestor_fallback_logged, _approver_fallback_logged
    already_logged = _requestor_fallback_logged if context == "requestor" else _approver_fallback_logged
    if not already_logged:
        logger.warning(
            f"Pending queue {context} allowlist is not configured; all {context} requests will be rejected",
        )
        if context == "requestor":
            _requestor_fallback_logged = True
        else:
            _approver_fallback_logged = True

    return set()


def _queue_requestor_allowlist() -> set[str]:
    settings = horde_model_reference_settings.pending_queue

    logger.debug("Building requestor allowlist from settings")

    allowlist = _normalize_ids(settings.requestor_ids)
    allowlist.update(_normalize_ids(settings.approver_ids))

    logger.debug(f"Combined requestor allowlist (including approvers): {allowlist}")

    if allowlist:
        return allowlist
    return _fallback_allowed_users("requestor")


def _queue_approver_allowlist() -> set[str]:
    settings = horde_model_reference_settings.pending_queue

    logger.debug("Building approver allowlist from settings")

    allowlist = _normalize_ids(settings.approver_ids)

    logger.debug(f"Approver allowlist: {allowlist}")

    if allowlist:
        return allowlist
    return _fallback_allowed_users("approver")


async def authenticate_queue_requestor(apikey: str) -> HordeUserContext:
    """Authenticate a queue requestor using the configured allowlist.

    Raises:
        APIKeyInvalidException: If no allowlist is configured or the user is not authorized.
        HTTPException: 503 if the Horde auth service is unreachable.
    """
    allowlist = _queue_requestor_allowlist()
    if not allowlist:
        raise APIKeyInvalidException()

    context = await auth_against_horde(apikey, httpx_client, allowed_user_ids=allowlist)
    if context is None:
        raise APIKeyInvalidException()
    return context


async def authenticate_queue_approver(apikey: str) -> HordeUserContext:
    """Authenticate a queue approver using the configured allowlist.

    Raises:
        APIKeyInvalidException: If no allowlist is configured or the user is not authorized.
        HTTPException: 503 if the Horde auth service is unreachable.
    """
    allowlist = _queue_approver_allowlist()
    if not allowlist:
        raise APIKeyInvalidException()

    context = await auth_against_horde(apikey, httpx_client, allowed_user_ids=allowlist)
    if context is not None:
        logger.debug(f"Approver authenticated: user_id={context.user_id}")
    if context is None:
        raise APIKeyInvalidException()
    return context


async def get_user_roles(apikey: str) -> tuple[HordeUserContext | None, set[str]]:
    """Authenticate a user and determine their roles based on configured allowlists.

    This function authenticates the user without enforcing any specific role requirement,
    then checks which roles the user has been granted.

    Args:
        apikey: The API key to authenticate.

    Returns:
        A tuple of (user_context, roles) where:
        - user_context: The authenticated user details, or None if authentication failed.
        - roles: A set of role names the user has (e.g., {'approver', 'requestor'}).
    """
    # Authenticate without any allowlist restriction first
    context = await auth_against_horde(apikey, httpx_client, allowed_user_ids=None)
    if context is None:
        return None, set()

    roles: set[str] = set()

    # Check approver status
    approver_allowlist = _queue_approver_allowlist()
    if context.user_id in approver_allowlist:
        roles.add("approver")

    # Check requestor status
    requestor_allowlist = _queue_requestor_allowlist()
    if context.user_id in requestor_allowlist:
        roles.add("requestor")

    return context, roles


class APIKeyInvalidException(HTTPException):
    """Exception raised when an API key is invalid."""

    def __init__(self, headers: dict[str, str] | None = None) -> None:
        """Initialize the exception."""
        super().__init__(
            status_code=401,
            detail="Invalid API key",
            headers=headers,
        )


class ErrorDetail(BaseModel):
    """Detail about a specific error."""

    loc: list[str | int] | None = None
    """Location of the error (for validation errors)."""
    msg: str
    """Error message."""
    type: str | None = None
    """Error type."""


class ErrorResponse(BaseModel):
    """Standardized error response."""

    detail: str | list[ErrorDetail]
    """Error details - either a string message or list of validation errors."""


_INVALID_MODEL_NAME_CHARS = frozenset("\\")


def validate_model_name(model_name: str) -> None:
    """Reject model names that are empty, whitespace-only, or contain path separators.

    Raises:
        HTTPException: 422 if the model name is invalid.
    """
    if not model_name or not model_name.strip():
        raise HTTPException(
            status_code=422,
            detail="Model name must not be empty or whitespace-only.",
        )
    if _INVALID_MODEL_NAME_CHARS & set(model_name):
        raise HTTPException(
            status_code=422,
            detail=f"Model name must not contain invalid characters {''.join(_INVALID_MODEL_NAME_CHARS)}: "
            f"'{model_name}'",
        )


def get_model_reference_manager() -> ModelReferenceManager:
    """Dependency helper that returns the singleton model reference manager."""
    return ModelReferenceManager()


def assert_primary_mode(manager: ModelReferenceManager) -> None:
    """Ensure the backend is in PRIMARY mode (supports writes), regardless of canonical format.

    Use for metadata-only write operations (schemas, aliases, families) that don't
    modify model records and are not gated by the canonical format setting.
    """
    if not manager.backend.supports_writes():
        raise HTTPException(
            status_code=503,
            detail=(
                "This instance is in REPLICA mode and does not support write operations. "
                "Only PRIMARY instances can modify data."
            ),
        )


def assert_canonical_write_enabled(
    manager: ModelReferenceManager,
    *,
    canonical_format: CanonicalFormat,
) -> None:
    """Ensure that writes are attempted only when the canonical format allows them."""
    assert_primary_mode(manager)

    backend = manager.backend
    expected_format = horde_model_reference_settings.canonical_format
    if canonical_format == CanonicalFormat.v2:
        if expected_format != canonical_format:
            raise HTTPException(
                status_code=503,
                detail=(
                    "This deployment does not expose write operations for this API. "
                    f"Expected canonical_format='{canonical_format}', got '{expected_format}'."
                ),
            )
        return

    if not backend.supports_legacy_writes():
        raise HTTPException(
            status_code=503,
            detail=(
                "This instance cannot process legacy writes. PRIMARY deployments with legacy canonical format "
                "must enable legacy write support."
            ),
        )

    if expected_format != canonical_format:
        raise HTTPException(
            status_code=503,
            detail=(
                "This deployment does not expose write operations for this API. "
                f"Expected canonical_format='{canonical_format}', got '{expected_format}'."
            ),
        )


def assert_pending_queue_write_enabled(manager: ModelReferenceManager) -> None:
    """Ensure pending-queue operations are allowed for the active canonical format."""
    backend = manager.backend
    canonical_format = horde_model_reference_settings.canonical_format

    if canonical_format == CanonicalFormat.v2:
        if not backend.supports_writes():
            raise HTTPException(
                status_code=503,
                detail=(
                    "This instance is in REPLICA mode and does not support write operations. "
                    "Only PRIMARY instances can queue model changes."
                ),
            )
        return

    if canonical_format == CanonicalFormat.LEGACY:
        if not backend.supports_legacy_writes():
            raise HTTPException(
                status_code=503,
                detail=(
                    "This instance cannot process legacy writes. PRIMARY deployments with legacy canonical "
                    "format must enable legacy write support."
                ),
            )
        return

    raise HTTPException(
        status_code=503,
        detail=(
            "Pending queue writes are not available for the configured canonical format. "
            f"canonical_format='{canonical_format}'"
        ),
    )
