"""FastAPI dependency-injection providers."""

from .services import get_search_service, get_settings_dep

__all__ = ["get_search_service", "get_settings_dep"]
