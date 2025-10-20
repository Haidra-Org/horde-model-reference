from enum import auto

from strenum import StrEnum

v1_prefix = "/model_references/v1"
v2_prefix = "/model_references/v2"


class PathVariables(StrEnum):
    """Path variables used in the router."""

    model_category_name = auto()
    model_name = auto()


class RouteNames(StrEnum):
    """Route names used in the router."""

    get_reference_info = auto()
    get_reference_names = auto()
    get_reference_by_category = auto()
    get_single_model = auto()
    create_model = auto()
    create_image_generation_model = auto()
    create_text_generation_model = auto()
    create_blip_model = auto()
    create_clip_model = auto()
    create_codeformer_model = auto()
    create_controlnet_model = auto()
    create_esrgan_model = auto()
    create_gfpgan_model = auto()
    create_safety_checker_model = auto()
    create_miscellaneous_model = auto()
    update_model = auto()
    delete_model = auto()


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
