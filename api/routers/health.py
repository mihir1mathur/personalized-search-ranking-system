"""
health.py  --  /health and /version endpoints.
==============================================

Operational endpoints that carry no ranking logic:

  * ``GET /health``  -- liveness/readiness. Returns 200 with ``status: ok`` when
    every pipeline component is loaded, otherwise 503 with ``status: degraded``
    listing exactly which component is missing. Load balancers and container
    orchestrators poll this to decide whether to route traffic to the instance.
  * ``GET /version`` -- build + pipeline metadata (models, stages, methods).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from api.dependencies.services import get_search_service
from api.schemas.responses import HealthResponse, VersionResponse
from api.services.search_service import SearchService

router = APIRouter(tags=["operations"])


@router.get("/health", response_model=HealthResponse, summary="Liveness/readiness probe")
def health(response: Response,
           service: SearchService = Depends(get_search_service)) -> HealthResponse:
    payload = service.health()
    if not payload["ready"]:
        # Signal to orchestrators that this instance should not receive traffic.
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(**payload)


@router.get("/version", response_model=VersionResponse, summary="Build & pipeline metadata")
def version(service: SearchService = Depends(get_search_service)) -> VersionResponse:
    return VersionResponse(**service.version_info())
