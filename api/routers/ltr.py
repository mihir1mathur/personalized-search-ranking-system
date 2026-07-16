"""
ltr.py  --  /ltr-search endpoint (Week 5 full pipeline, best method).
=====================================================================

The complete funnel: hybrid retrieval -> cross-encoder scoring -> a LightGBM
LambdaMART Learning-to-Rank model that combines every signal (cross-encoder
score + rank, hybrid/BM25/embedding scores, and cheap text features) into the
final ordering. This is the best-performing method from Week 5 (best NDCG@10,
MAP, MRR on the held-out test queries).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies.services import get_search_service
from api.schemas.requests import LTRSearchRequest
from api.schemas.responses import SearchResponse
from api.services.search_service import SearchService

router = APIRouter(tags=["search"])


@router.post("/ltr-search", response_model=SearchResponse,
             summary="Full pipeline with Learning-to-Rank (recommended)")
def ltr_search(request: LTRSearchRequest,
               service: SearchService = Depends(get_search_service)) -> SearchResponse:
    result = service.ltr_search(query=request.query, top_k=request.top_k,
                                candidate_depth=request.candidate_depth)
    return SearchResponse(**result)
