"""
responses.py  --  Pydantic response models (the output side of the contract).
=============================================================================

Every endpoint returns a structured, documented JSON shape so clients can rely
on the schema (and FastAPI can render it in the Swagger UI). The core object is
:class:`SearchResultItem` -- one ranked product -- carrying:

    rank         1-based position (1 = best)
    product_id   the catalog id
    title        the product title (human readable)
    score        the raw ranking score from the chosen method (method-specific
                 scale; only the ORDER is meaningful across methods)
    confidence   a 0..1 monotonic transform of the score, comparable within a
                 response and easy for a UI to display (see search_service.py
                 for the exact per-method transform)

:class:`SearchResponse` wraps the list with the query, the ranking method used,
and the server-measured ``processing_time_ms`` so callers can see latency.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SearchResultItem(BaseModel):
    """One ranked product in a search response."""

    rank: int = Field(..., ge=1, description="1-based rank (1 = most relevant).")
    product_id: str = Field(..., description="Catalog product id.")
    title: str = Field(..., description="Product title (may be empty if unknown).")
    score: float = Field(..., description="Raw ranking score (method-specific scale).")
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Normalized 0..1 confidence derived monotonically from the score.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "rank": 1,
                "product_id": "B07XYZ1234",
                "title": "Wireless Noise Cancelling Over-Ear Headphones",
                "score": 8.7421,
                "confidence": 0.9998,
            }
        }
    }


class SearchResponse(BaseModel):
    """The envelope returned by every search endpoint."""

    query: str = Field(..., description="The (stripped) query that was searched.")
    ranking_method: str = Field(..., description="Which method produced this ranking.")
    top_k: int = Field(..., ge=0, description="Number of results requested/returned cap.")
    count: int = Field(..., ge=0, description="Number of results actually returned.")
    processing_time_ms: float = Field(
        ..., ge=0.0, description="Server-side processing time in milliseconds."
    )
    cached: bool = Field(
        default=False, description="True if served from the query cache."
    )
    results: List[SearchResultItem] = Field(
        default_factory=list, description="Ranked results, best first."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "wireless noise cancelling headphones",
                "ranking_method": "ltr",
                "top_k": 10,
                "count": 10,
                "processing_time_ms": 42.7,
                "cached": False,
                "results": [
                    {
                        "rank": 1,
                        "product_id": "B07XYZ1234",
                        "title": "Wireless Noise Cancelling Over-Ear Headphones",
                        "score": 8.7421,
                        "confidence": 0.9998,
                    }
                ],
            }
        }
    }


class ComponentStatus(BaseModel):
    """Readiness of one loaded pipeline component (for /health)."""

    name: str
    ready: bool
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    """Liveness/readiness payload for ``/health``."""

    status: str = Field(..., description="'ok' if ready to serve, else 'degraded'/'loading'.")
    ready: bool = Field(..., description="True when every required component is loaded.")
    environment: str = Field(..., description="Deployment environment.")
    version: str = Field(..., description="API version.")
    uptime_seconds: float = Field(..., ge=0.0, description="Seconds since startup.")
    corpus_size: int = Field(..., ge=0, description="Number of unique products indexed.")
    components: List[ComponentStatus] = Field(default_factory=list)


class VersionResponse(BaseModel):
    """Build / pipeline metadata for ``/version``."""

    name: str
    version: str
    environment: str
    pipeline: List[str] = Field(
        ..., description="Ordered ranking stages in the served pipeline."
    )
    embedding_model: str
    cross_encoder_model: str
    ltr_model: str
    supported_methods: List[str]


class ErrorResponse(BaseModel):
    """Uniform error body returned for every handled failure."""

    error: str = Field(..., description="Short machine-readable error type.")
    detail: str = Field(..., description="Human-readable explanation.")
    status_code: int = Field(..., description="HTTP status code.")
    request_id: Optional[str] = Field(
        default=None, description="Correlation id for tracing this request in logs."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "ValidationError",
                "detail": "top_k must be between 1 and 100",
                "status_code": 422,
                "request_id": "b3f1c2a4",
            }
        }
    }
