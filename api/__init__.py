"""
api  --  Week 6 production FastAPI backend for the search-ranking pipeline.
==========================================================================

This package turns the Week 0-5 retrieval/ranking pipeline
(TF-IDF -> BM25 -> Embeddings -> FAISS -> Hybrid -> CrossEncoder ->
Learning-to-Rank) into a production-quality REST API.

Nothing in this package changes the Week 0-5 algorithms, models, caches, or
evaluation outputs. It only *reuses* them behind a clean service layer and
exposes them over HTTP with configuration, logging, validation, error
handling, caching, and tests -- the software-engineering layer on top.

Layout (clean, layered architecture):

    api/
      main.py           FastAPI app factory, lifespan, exception handlers
      config/           all configurable values (no hardcoded constants)
      schemas/          Pydantic request/response models (the API contract)
      services/         SearchService (reuses Week 5), query cache, errors
      routers/          one APIRouter per endpoint group
      middleware/       request/latency logging middleware
      dependencies/     FastAPI dependency-injection providers
      utils/            logging setup + timing helpers
      tests/            unit tests (fast, fake-service based)
"""

__version__ = "6.0.0"
