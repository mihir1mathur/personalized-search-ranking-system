"""
services.py  --  dependency providers for routers.
==================================================

Routers never construct a :class:`SearchService` themselves; they *ask* for one
via FastAPI's ``Depends``. The single instance is built once in the app
lifespan and stored on ``app.state.search_service``. This indirection is what
makes the API testable: a test can override :func:`get_search_service` to return
a lightweight fake, so unit tests run in milliseconds without loading any model.
"""

from __future__ import annotations

from fastapi import Depends, Request

from api.config.settings import Settings, get_settings
from api.services.exceptions import ModelNotReadyError
from api.services.search_service import SearchService


def get_settings_dep() -> Settings:
    """Provide the cached application settings."""
    return get_settings()


def get_search_service(request: Request) -> SearchService:
    """
    Return the process-wide SearchService held on app state.

    Raises 503 (via ModelNotReadyError) if the service was never attached --
    which only happens if startup was skipped and nothing was injected.
    """
    service = getattr(request.app.state, "search_service", None)
    if service is None:
        raise ModelNotReadyError("search service is not initialized")
    return service


# Convenience aliases used in route signatures for readability.
SettingsDep = Depends(get_settings_dep)
ServiceDep = Depends(get_search_service)
