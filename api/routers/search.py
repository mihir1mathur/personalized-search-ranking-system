"""
search.py  --  /search endpoint (generic, method-selectable).
=============================================================

``/search`` is the front door: the caller names a ``method`` and gets back a
ranked list. It supports every stage of the pipeline via one contract:

    tfidf | bm25 | embedding | hybrid | rerank | ltr

Both POST (JSON body) and GET (query params) are provided -- POST is the
canonical form; GET is a convenience for quick curl/Swagger demos.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.dependencies.services import get_search_service
from api.schemas.requests import SearchRequest
from api.schemas.responses import SearchResponse
from api.services.search_service import SearchService

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse,
             summary="Search with a selectable ranking method")
def search(request: SearchRequest,
           service: SearchService = Depends(get_search_service)) -> SearchResponse:
    result = service.search(query=request.query, top_k=request.top_k,
                            method=request.method)
    return SearchResponse(**result)


@router.get("/search", response_model=SearchResponse,
            summary="Search (GET convenience form)")
def search_get(
    q: str = Query(..., description="The search query.", min_length=1),
    top_k: Optional[int] = Query(None, ge=1, description="Number of results."),
    method: str = Query("hybrid", description="tfidf|bm25|embedding|hybrid|rerank|ltr"),
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    result = service.search(query=q, top_k=top_k, method=method)
    return SearchResponse(**result)
