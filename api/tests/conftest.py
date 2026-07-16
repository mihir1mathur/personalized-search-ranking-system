"""
conftest.py  --  shared pytest fixtures.
========================================

The key idea that keeps these tests fast: we NEVER load the real ML models.
Instead we build the real FastAPI app in ``testing`` mode (so the lifespan does
not load anything) and inject a :class:`FakeSearchService`.

The fake SUBCLASSES the real :class:`SearchService`, overriding only the two
model-touching primitives (``_hybrid_candidates`` and ``_cross_encoder_scores``)
plus the retriever objects. Everything else -- input validation, the confidence
transform, the response envelope, caching, error mapping -- is the REAL code, so
the tests genuinely exercise the API/service behavior, just without a GPU-months
model download.
"""

from __future__ import annotations

import types
from typing import Dict, List, Tuple

import pytest
from fastapi.testclient import TestClient

from api.config.settings import Settings
from api.main import create_app
from api.services.search_service import SearchService


class _FakeRetriever:
    """Stands in for TfidfRetriever / BM25Retriever / EmbeddingRetriever."""

    def __init__(self, ids: List[str]) -> None:
        self._ids = ids

    def retrieve(self, query: str, top_k: int = 10):
        return [(pid, 1.0 / (i + 1), i + 1)
                for i, pid in enumerate(self._ids[:top_k])]


class _FakeLTR:
    """Stands in for the LightGBM LTRRanker."""

    def __init__(self, ids: List[str]) -> None:
        self._ids = ids

    def rank_candidates(self, candidate_ids, feature_matrix, top_k=10):
        return [(pid, float(len(candidate_ids) - i), i + 1)
                for i, pid in enumerate(candidate_ids[:top_k])]


class FakeSearchService(SearchService):
    """A SearchService with canned pipeline internals (no models)."""

    def __init__(self, settings: Settings, ready: bool = True) -> None:
        super().__init__(settings)
        self._ready = ready
        self.product_ids = [f"P{i}" for i in range(1, 6)]
        self.title_by_id = {p: f"Title for {p}" for p in self.product_ids}
        self.text_by_id = {p: f"searchable text for {p}" for p in self.product_ids}
        self.brand_by_id = {p: "BrandX" for p in self.product_ids}
        self.color_by_id = {p: "blue" for p in self.product_ids}
        self._component_status = {
            n: (ready, "fake") for n in
            ("data", "tfidf", "hybrid", "cross_encoder", "ltr")
        }
        self._simple_tokenize = lambda s: str(s).lower().split()
        self._build_feature_row = lambda **kw: [0.0]
        self._build_feature_matrix = lambda rows: rows
        self.tfidf = _FakeRetriever(self.product_ids)
        self.hybrid = types.SimpleNamespace(
            bm25=_FakeRetriever(self.product_ids),
            embedder=_FakeRetriever(self.product_ids),
        )
        self.reranker = object()
        self.ltr = _FakeLTR(self.product_ids)

    def _hybrid_candidates(self, query, depth, alpha, beta
                           ) -> Tuple[List[Tuple[str, float, int]], Dict]:
        ranked = [(p, 1.0 / (i + 1), i + 1)
                  for i, p in enumerate(self.product_ids[:depth])]
        norm = {p: (0.5, 0.5) for p in self.product_ids}
        return ranked, norm

    def _cross_encoder_scores(self, query, cand_ids) -> Dict[str, float]:
        return {p: float(len(cand_ids) - i) for i, p in enumerate(cand_ids)}


def _make_client(ready: bool = True) -> TestClient:
    settings = Settings(testing=True, log_to_file=False, cache_enabled=True)
    app = create_app(settings)
    app.state.search_service = FakeSearchService(settings, ready=ready)
    return TestClient(app)


@pytest.fixture
def client() -> TestClient:
    with _make_client(ready=True) as c:
        yield c


@pytest.fixture
def degraded_client() -> TestClient:
    with _make_client(ready=False) as c:
        yield c
