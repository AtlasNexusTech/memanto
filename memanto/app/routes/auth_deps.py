"""
Authentication Dependencies for V2 API

Shared authentication utilities to avoid circular imports.
"""

from fastapi import Depends, Header, HTTPException

from memanto.app.models.session import Session
from memanto.app.services.session_service import get_session_service
from memanto.app.utils.errors import (
    InvalidSessionTokenError,
    SessionExpiredError,
    SessionNotFoundError,
    map_error_to_http_exception,
)


def get_moorcheh_api_key(authorization: str | None = Header(None)) -> str:
    """
    Extract Moorcheh API key from Authorization header or use configured default

    Args:
        authorization: Authorization header (Bearer {api_key})

    Returns:
        API key

    Raises:
        HTTPException: If authorization header is invalid or no key is found
    """
    if authorization:
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Invalid authorization header. Use: Bearer {api_key}",
            )

        api_key = authorization.replace("Bearer ", "").strip()
        if api_key:
            return api_key

    from memanto.app.config import settings

    if settings.MOORCHEH_API_KEY:
        return settings.MOORCHEH_API_KEY

    raise HTTPException(
        status_code=401,
        detail="Missing API key in authorization header and no configured default",
    )


async def verify_moorcheh_api_key(api_key: str = Depends(get_moorcheh_api_key)) -> str:
    """
    Verify Moorcheh API key by making a lightweight request.

    Args:
        api_key: The API key to verify

    Returns:
        The verified API key

    Raises:
        HTTPException: If the API key is invalid or request fails
    """
    from moorcheh_sdk.exceptions import AuthenticationError, NamespaceNotFound

    from memanto.app.clients.moorcheh import get_async_moorcheh_client, _resolve_backend

    # In local backend mode there is nothing to verify against: accept the
    # key (any non-empty bearer token) without a network round-trip.
    if _resolve_backend() == "local":
        return api_key or "local"

    client = get_async_moorcheh_client(api_key)
    try:
        await client.documents.get(namespace_name="__memanto_auth_ping__", ids=["1"])
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid Moorcheh API key")
    except NamespaceNotFound:
        # Key is valid, the auth passed but the fake namespace doesn't exist
        pass
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to verify Moorcheh API key: {str(e)}"
        )
    return api_key


def get_current_session(
    authorization: str | None = Header(None), x_session_token: str | None = Header(None)
) -> Session:
    """
    Get and validate current session

    Args:
        authorization: Authorization header with Moorcheh API key
        x_session_token: Session token header

    Returns:
        Validated Session

    Raises:
        HTTPException: If session is invalid or expired
    """
    moorcheh_api_key = get_moorcheh_api_key(authorization)

    if not x_session_token:
        raise HTTPException(
            status_code=401, detail="Missing session token. Use X-Session-Token header."
        )

    session_service = get_session_service()

    try:
        token_payload = session_service.validate_session(
            x_session_token, moorcheh_api_key
        )

        # Get session from storage
        session = session_service.get_session(token_payload.agent_id)
        if not session:
            raise SessionNotFoundError(
                f"Session for agent {token_payload.agent_id} not found"
            )

        # Auto-renew session if near expiry
        renewed = session_service.check_and_auto_renew(
            agent_id=token_payload.agent_id,
            moorcheh_api_key=moorcheh_api_key,
        )
        if renewed:
            session = renewed

        return session

    except (SessionExpiredError, SessionNotFoundError, InvalidSessionTokenError) as e:
        raise map_error_to_http_exception(e)
