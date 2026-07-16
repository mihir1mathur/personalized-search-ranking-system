"""Service layer: the SearchService (reuses Week 5), query cache, and errors."""

from .cache import QueryCache
from .exceptions import (
    EmptyQueryError,
    InvalidParameterError,
    ModelNotReadyError,
    QueryTooLongError,
    SearchServiceError,
    UnsupportedMethodError,
)
from .search_service import SearchService

__all__ = [
    "SearchService",
    "QueryCache",
    "SearchServiceError",
    "ModelNotReadyError",
    "EmptyQueryError",
    "QueryTooLongError",
    "InvalidParameterError",
    "UnsupportedMethodError",
]
