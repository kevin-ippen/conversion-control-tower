"""
Authentication helpers for Databricks Apps OBO (On-Behalf-Of) tokens.

In Databricks Apps, the user's token is forwarded via x-forwarded-access-token header.
This module provides utilities to extract and use that token for downstream calls.
"""

import os
import logging
import threading
from typing import Optional

from fastapi import Request, Depends, HTTPException
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)

# Lock to protect env var manipulation when creating OBO clients.
# The SDK reads DATABRICKS_CLIENT_ID/SECRET from env and would try OAuth M2M
# instead of using the explicitly-passed OBO token.
_auth_lock = threading.Lock()


def get_user_token(request: Request) -> Optional[str]:
    """Extract OBO token from request headers.

    In Databricks Apps, the user's OAuth token is forwarded via
    the x-forwarded-access-token header.
    """
    return request.headers.get("x-forwarded-access-token")


def get_user_email(request: Request) -> Optional[str]:
    """Extract user email from request headers."""
    return request.headers.get("x-forwarded-email")


def get_user_name(request: Request) -> Optional[str]:
    """Extract user display name from request headers."""
    return request.headers.get("x-forwarded-user")


def get_workspace_client(request: Request) -> WorkspaceClient:
    """Get WorkspaceClient configured with OBO token.

    Args:
        request: FastAPI request object

    Returns:
        WorkspaceClient configured for user-delegated access
    """
    token = get_user_token(request)
    host = os.environ.get("DATABRICKS_HOST", "")

    if not host.startswith("https://"):
        host = f"https://{host}"

    if token:
        # Use OBO token for user-delegated access.
        # Must temporarily hide SP env vars so the SDK doesn't try OAuth M2M.
        with _auth_lock:
            saved_client_id = os.environ.pop("DATABRICKS_CLIENT_ID", None)
            saved_client_secret = os.environ.pop("DATABRICKS_CLIENT_SECRET", None)
            try:
                client = WorkspaceClient(host=host, token=token)
            finally:
                if saved_client_id:
                    os.environ["DATABRICKS_CLIENT_ID"] = saved_client_id
                if saved_client_secret:
                    os.environ["DATABRICKS_CLIENT_SECRET"] = saved_client_secret
        return client
    else:
        # Fallback to default auth (service principal credentials from env)
        return WorkspaceClient(host=host)


def get_sql_connection_params(request: Request) -> dict:
    """Get parameters for SQL warehouse connection.

    Returns dict with host, warehouse_id, and token for databricks-sql-connector.
    """
    from .config import get_settings
    settings = get_settings()

    token = get_user_token(request)
    host = os.environ.get("DATABRICKS_HOST", "")

    if not host.startswith("https://"):
        host = f"https://{host}"

    return {
        "server_hostname": host.replace("https://", ""),
        "http_path": f"/sql/1.0/warehouses/{settings.databricks_warehouse_id}",
        "access_token": token or os.environ.get("DATABRICKS_TOKEN", "")
    }


class UserContext:
    """Container for user context extracted from request."""

    def __init__(self, request: Request):
        self.token = get_user_token(request)
        self.email = get_user_email(request)
        self.name = get_user_name(request)
        self.workspace_client = get_workspace_client(request)

    @property
    def user_id(self) -> str:
        """Get user identifier (email or 'anonymous')."""
        return self.email or "anonymous"


def get_user_context(request: Request) -> UserContext:
    """Dependency to get user context."""
    return UserContext(request)


def require_authenticated_user(request: Request) -> UserContext:
    """Dependency that requires an authenticated user.

    Raises HTTPException if no user token is present.
    """
    context = UserContext(request)
    if not context.token:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please access via Databricks Apps."
        )
    return context
