"""Security and optional API-key gate for HTTP requests."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.settings import api_key_set, security_headers_enabled


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Conservative headers suitable for a browser-facing API behind HTTPS."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        if security_headers_enabled():
            response.headers.setdefault("X-Content-Type-Options", "nosniff")
            response.headers.setdefault("X-Frame-Options", "DENY")
            response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
            response.headers.setdefault(
                "Permissions-Policy",
                "geolocation=(), microphone=(), camera=()",
            )
        return response


def _path_is_public_api(path: str) -> bool:
    return path == "/health" or path.startswith("/health/")


def _path_needs_api_key(path: str) -> bool:
    return path.startswith("/api/")


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    When API_KEYS is non-empty, require Authorization: Bearer <key> for /api/* routes.
    Skips OPTIONS (CORS preflight), /health*, and OpenAPI UI paths.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        keys = api_key_set()
        if not keys:
            return await call_next(request)

        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        if _path_is_public_api(path):
            return await call_next(request)
        if path.startswith("/docs") or path.startswith("/redoc") or path == "/openapi.json":
            return await call_next(request)
        if not _path_needs_api_key(path):
            return await call_next(request)

        auth = request.headers.get("Authorization")
        if not auth or not auth.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid Authorization header. Use: Bearer <api_key>"},
            )
        token = auth[7:].strip()
        if not token or token not in keys:
            return JSONResponse(status_code=403, content={"detail": "Invalid API key"})

        return await call_next(request)
