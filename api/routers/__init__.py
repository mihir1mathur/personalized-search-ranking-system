"""API routers -- one APIRouter per endpoint group, assembled in main.py."""

from . import health, hybrid, ltr, rerank, search

__all__ = ["health", "search", "hybrid", "rerank", "ltr"]
