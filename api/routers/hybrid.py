"""
hybrid.py  --  /hybrid-search endpoint (Week 3 fusion).
=======================================================

Runs hybrid retrieval only: a weighted, min-max-normalized fusion of the BM25
(keyword) and embedding (semantic) signals. Optional per-request ``alpha`` and
``beta`` let a caller experiment with the mix; omitting them uses the Week 3/5
best weights from configuration.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies.services import get_search_service
from api.schemas.requests import HybridSearchRequest
from api.schemas.responses import SearchResponse
from api.services.search_service import SearchService

router = APIRouter(tags=["search"])


@router.post("/hybrid-search", response_model=SearchResponse,
             summary="Hybrid (BM25 + embedding) retrieval")
def hybrid_search(request: HybridSearchRequest,
                  service: SearchService = Depends(get_search_service)) -> SearchResponse:
    result = service.hybrid_search(query=request.query, top_k=request.top_k,
                                   alpha=request.alpha, beta=request.beta)
    return SearchResponse(**result)
