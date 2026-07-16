"""
rerank.py  --  /rerank endpoint (Week 4 two-stage pipeline).
============================================================

Hybrid retrieval produces a Top-N candidate shortlist; the cross-encoder then
reads each (query, product) pair jointly and re-orders that shortlist into the
final Top-K. This is the classic "retrieve then re-rank" design. Because the
cross-encoder is the dominant serving cost, results are cached per query.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies.services import get_search_service
from api.schemas.requests import RerankRequest
from api.schemas.responses import SearchResponse
from api.services.search_service import SearchService

router = APIRouter(tags=["search"])


@router.post("/rerank", response_model=SearchResponse,
             summary="Hybrid retrieval + cross-encoder re-ranking")
def rerank(request: RerankRequest,
           service: SearchService = Depends(get_search_service)) -> SearchResponse:
    result = service.rerank(query=request.query, top_k=request.top_k,
                            candidate_depth=request.candidate_depth)
    return SearchResponse(**result)
