import urllib.parse
from enum import auto

import httpx
from fastapi import HTTPException
from fastapi.security import APIKeyHeader
from loguru import logger
from pydantic import BaseModel
from strenum import StrEnum

from horde_model_reference import ModelReferenceManager, ai_horde_worker_settings, horde_model_reference_settings

header_auth_scheme = APIKeyHeader(name="apikey")

httpx_client = httpx.AsyncClient()

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
    delete_model = auto()
    get_models_with_stats = auto()
    get_category_statistics = auto()
    get_category_audit = auto()

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


# Full names, like Tazlin#6572, are unreliable because the user can change them.
# Instead, we use the immutable user ID for authentication allowlisting.
allowed_users = ["1", "6572"]


async def auth_against_horde(apikey: str, client: httpx.AsyncClient) -> bool:
    """Authenticate the provided API key against the AI-Horde.

    This uses the endpoint defined by AI_HORDE_URL by AIHordeClientSettings in haidra_core.

    Args:
        apikey (str): The API key to authenticate.
        client (httpx.AsyncClient): The HTTP client to use for the request.

    Returns:
        bool: True if authentication is successful, False otherwise.
    """
    find_user_subpath = "v2/find_user"
    url = urllib.parse.urljoin(
        str(ai_horde_worker_settings.ai_horde_url),
        find_user_subpath,
    )

    response = await client.get(
        url,
        headers={"apikey": f"{apikey}"},
    )

    if response.status_code == 200:
        user_data = response.json()
        user_name = user_data.get("username", "")

        if "#" not in user_name:
            logger.warning(f"Unknown apikey: {user_data}")
            return False

        user_id = user_name.split("#")[-1]

        if user_id in allowed_users:
            return True

        logger.warning(f"Unauthorized user ID: {user_id}")

    return False


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


def get_model_reference_manager() -> ModelReferenceManager:
    """Dependency helper that returns the singleton model reference manager."""
    return ModelReferenceManager()

