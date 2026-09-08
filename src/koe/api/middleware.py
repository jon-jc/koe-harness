"""Request correlation, access logging, and metrics.

A voice session produces interleaved output from several concurrent tasks, and
a request that fails is usually reported by a user as "it broke at about
2pm" -- so the operational requirement is being able to reconstruct one
request's story from a log aggregator. That needs a correlation id present on
every line, which means it has to be established at the edge and carried, not
passed around by hand.

The id comes from the caller's ``X-Request-ID`` when present, so a trace that
starts at a load balancer or an upstream service stays one trace, and is
generated otherwise. It goes back on the response, because the fastest way to
debug a user's report is for them to be able to quote it.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from koe.telemetry.metrics import Metrics

logger = logging.getLogger("koe.access")

#: The current request's id, readable from anywhere in the handler's task.
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def current_request_id() -> str:
    return request_id_var.get()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request id, logs the access line, records latency."""

    def __init__(self, app: ASGIApp, *, metrics: Metrics) -> None:
        super().__init__(app)
        self.metrics = metrics

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id_var.set(request_id)
        started = time.perf_counter()

        # The route pattern, not the concrete path: /v1/sessions/{id} is one
        # metric, while the raw path would be one metric per session and blow
        # up cardinality in any metrics backend.
        route = request.scope.get("route")
        endpoint = getattr(route, "path", request.url.path)

        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.metrics.increment("http.errors")
            self.metrics.observe("http.latency_ms", elapsed_ms)
            logger.exception(
                "unhandled error",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(elapsed_ms, 2),
                },
            )
            # Never leak an internal traceback to a caller; the request id is
            # what connects their report to the logged detail.
            return JSONResponse(
                status_code=500,
                content={"detail": "internal error", "request_id": request_id},
                headers={"X-Request-ID": request_id},
            )
        finally:
            request_id_var.reset(token)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.headers["X-Request-ID"] = request_id

        self.metrics.increment("http.requests")
        self.metrics.observe("http.latency_ms", elapsed_ms)
        if response.status_code >= 500:
            self.metrics.increment("http.errors")

        logger.info(
            "%s %s %d",
            request.method,
            request.url.path,
            response.status_code,
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "endpoint": endpoint,
                "status": response.status_code,
                "duration_ms": round(elapsed_ms, 2),
            },
        )
        return response
