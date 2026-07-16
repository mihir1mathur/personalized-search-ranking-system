"""
exceptions.py  --  domain exceptions for the search service.
============================================================

The service raises these typed errors instead of returning ad-hoc strings or
letting raw library exceptions leak out. ``main.py`` maps each one to the right
HTTP status code and a uniform :class:`ErrorResponse` body, so the API's error
contract is consistent and testable.

    ModelNotReadyError    -> 503  (models/indexes still loading or missing)
    EmptyQueryError       -> 422  (blank query)
    QueryTooLongError     -> 422  (over the configured length limit)
    UnsupportedMethodError-> 422  (method not enabled)
    InvalidParameterError -> 422  (bad top_k / weights / depth)
    SearchServiceError    -> 500  (anything else that went wrong internally)
"""

from __future__ import annotations


class SearchServiceError(Exception):
    """Base class for all service-layer errors (maps to HTTP 500 by default)."""

    status_code = 500
    error_type = "SearchServiceError"

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class ModelNotReadyError(SearchServiceError):
    """A required model, index, or cache is missing or not yet loaded."""

    status_code = 503
    error_type = "ModelNotReady"


class EmptyQueryError(SearchServiceError):
    """The query was empty or whitespace only."""

    status_code = 422
    error_type = "EmptyQuery"


class QueryTooLongError(SearchServiceError):
    """The query exceeded the configured maximum length."""

    status_code = 422
    error_type = "QueryTooLong"


class UnsupportedMethodError(SearchServiceError):
    """The requested ranking method is not in the enabled set."""

    status_code = 422
    error_type = "UnsupportedMethod"


class InvalidParameterError(SearchServiceError):
    """A parameter (top_k, alpha/beta, candidate_depth) was out of range."""

    status_code = 422
    error_type = "InvalidParameter"
