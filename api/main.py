"""
main.py  --  FastAPI application factory for the Week 6 search backend.
=======================================================================

This wires the whole app together:

  * builds the :class:`SearchService` once during the lifespan startup (unless
    ``settings.testing`` is set, so unit tests can inject a fake),
  * registers every router (/search, /hybrid-search, /rerank, /ltr-search,
    /health, /version) plus a root banner,
  * installs the request-logging middleware and CORS,
  * registers exception handlers that turn any failure into a uniform
    :class:`ErrorResponse` JSON body with the request's correlation id.

Run locally with::

    uvicorn api.main:app --reload

Interactive API docs (Swagger UI) are served at ``/docs`` and ReDoc at
``/redoc`` -- FastAPI generates both from the Pydantic schemas automatically.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.config.settings import Settings, get_settings
from api.middleware.logging_middleware import RequestLoggingMiddleware
from api.routers import health, hybrid, ltr, rerank, search
from api.schemas.responses import ErrorResponse
from api.services.exceptions import SearchServiceError
from api.services.search_service import SearchService
from api.utils.logging_config import configure_logging, get_logger

logger = get_logger("main")


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _error_json(status_code: int, error: str, detail: str,
                request: Request) -> JSONResponse:
    body = ErrorResponse(error=error, detail=detail, status_code=status_code,
                         request_id=_request_id(request))
    return JSONResponse(status_code=status_code, content=body.model_dump())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the pipeline on startup; release references on shutdown."""
    settings: Settings = app.state.settings
    configure_logging(settings)
    logger.info("Starting %s v%s (env=%s)", settings.app_name,
                settings.app_version, settings.environment)

    if not settings.testing and getattr(app.state, "search_service", None) is None:
        service = SearchService(settings)
        service.load()  # never raises; degraded state is reported via /health
        app.state.search_service = service
    else:
        logger.info("Testing mode or pre-injected service: skipping model load.")

    yield

    logger.info("Shutting down %s", settings.app_name)
    app.state.search_service = None


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory (lets tests build an app with custom settings)."""
    settings = settings or get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production REST API over a multi-stage search-ranking pipeline: "
            "TF-IDF -> BM25 -> Embeddings -> FAISS -> Hybrid -> CrossEncoder -> "
            "Learning-to-Rank. Interactive docs at /docs."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.state.settings = settings

    # ---- Middleware --------------------------------------------------------
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # tighten per environment in production
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Routers -----------------------------------------------------------
    prefix = settings.api_prefix.rstrip("/")
    for module in (health, search, hybrid, rerank, ltr):
        app.include_router(module.router, prefix=prefix)

    # ---- Root banner -------------------------------------------------------
    @app.get("/", tags=["operations"], summary="Service banner")
    def root() -> dict:
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "docs": f"{prefix}/docs",
            "endpoints": [f"{prefix}{p}" for p in (
                "/search", "/hybrid-search", "/rerank", "/ltr-search",
                "/health", "/version")],
        }

    # ---- Exception handlers (uniform error contract) -----------------------
    @app.exception_handler(SearchServiceError)
    async def _service_error(request: Request, exc: SearchServiceError):
        return _error_json(exc.status_code, exc.error_type, exc.detail, request)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        # Summarize pydantic's error list into one readable message.
        pieces = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", []) if p != "body")
            pieces.append(f"{loc or 'body'}: {err.get('msg')}")
        detail = "; ".join(pieces) or "request validation failed"
        return _error_json(422, "ValidationError", detail, request)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return _error_json(exc.status_code, "HTTPError", detail, request)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.exception("Unhandled error: %s", exc)
        return _error_json(500, "InternalServerError",
                           "an unexpected error occurred", request)

    return app


# The ASGI application object uvicorn looks for: ``uvicorn api.main:app``.
app = create_app()
