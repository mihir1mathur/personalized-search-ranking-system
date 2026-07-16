"""
logging_middleware.py  --  per-request logging + latency + correlation id.
==========================================================================

Every request passes through this middleware, which:

  1. Assigns a short ``request_id`` (correlation id) and stores it on
     ``request.state`` so handlers and error responses can echo it -- essential
     for tracing one request across log lines in production.
  2. Logs the INCOMING request (method + path).
  3. Times the handler and logs the OUTCOME (status code + latency in ms).
  4. Adds ``X-Request-ID`` and ``X-Process-Time-ms`` response headers.
  5. Logs any unhandled exception (which the app's exception handlers then turn
     into a clean JSON error) so nothing fails silently.

The id is derived from a monotonic counter + time, avoiding any dependency and
staying readable in logs. Latency uses perf_counter for accuracy.
"""

from __future__ import annotations

import time
from itertools import count

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from api.utils.logging_config import get_logger

logger = get_logger("request")
_counter = count(1)


def _make_request_id() -> str:
    """A short, unique-enough id: base-36 counter + low bits of the clock."""
    n = next(_counter)
    stamp = int(time.time() * 1000) & 0xFFFFFF
    return f"{stamp:06x}-{n:x}"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs each request/response with a correlation id and latency."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = _make_request_id()
        request.state.request_id = request_id
        start = time.perf_counter()

        logger.info(
            "--> %s %s", request.method, request.url.path,
            extra={"request_id": request_id, "method": request.method,
                   "path": request.url.path},
        )

        try:
            response = await call_next(request)
        except Exception:
            latency_ms = (time.perf_counter() - start) * 1000.0
            # Log with stack trace; the registered exception handlers produce
            # the actual JSON error body returned to the client.
            logger.exception(
                "!!! %s %s failed after %.1f ms",
                request.method, request.url.path, latency_ms,
                extra={"request_id": request_id, "latency_ms": round(latency_ms, 2)},
            )
            raise

        latency_ms = (time.perf_counter() - start) * 1000.0
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-ms"] = f"{latency_ms:.2f}"

        log = logger.warning if response.status_code >= 500 else logger.info
        log(
            "<-- %s %s %s (%.1f ms)",
            request.method, request.url.path, response.status_code, latency_ms,
            extra={"request_id": request_id, "status_code": response.status_code,
                   "latency_ms": round(latency_ms, 2)},
        )
        return response
