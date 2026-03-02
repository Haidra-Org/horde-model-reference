"""User information router for v2 API."""

from typing import Annotated

from fastapi import APIRouter, Depends

from horde_model_reference.service.shared import (
    APIKeyInvalidException,
    get_user_roles,
    header_auth_scheme,
)
from horde_model_reference.service.v2.models import UserRolesResponse

router = APIRouter(
    responses={
        401: {"description": "Invalid API key"},
    },
)


@router.get(
    "/me/roles",
    response_model=UserRolesResponse,
    summary="Get current user roles",
    description=(
        "Returns the authenticated user's roles and permissions for the horde-model-reference service. "
        "This endpoint validates the provided API key against the AI Horde API and checks which "
        "roles (approver, requestor) the user has been granted."
    ),
)
async def get_current_user_roles(
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> UserRolesResponse:
    """Return the authenticated user's roles."""
    context, roles = await get_user_roles(apikey)

    if context is None:
        raise APIKeyInvalidException()

    return UserRolesResponse(
        user_id=context.user_id,
        username=context.username,
        roles=sorted(roles),
        is_approver="approver" in roles,
        is_requestor="requestor" in roles,
    )
