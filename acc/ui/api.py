"""Shared trainer API client for UI modules.

Every partial and action handler calls the trainer API through this module.
This is the single point of configuration for the trainer URL, HTTP client
settings, and the GET response cache.

Uses a persistent httpx.AsyncClient (connection pooling, keep-alive) so
concurrent HTMX polls don't create dozens of TCP connections per second.

GET requests are cached for 1 second at the module level. Since HTMX fires
many partials concurrently on page load, this eliminates redundant API calls
(e.g., 7 partials all hitting /health within the same render cycle). POST
requests are never cached, and `invalidate()` / `invalidate_all()` allow
action handlers to bust the cache after mutations.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import httpx

TRAINER_URL = os.environ.get("ACC_TRAINER_URL", "http://localhost:6060")

# Persistent client — reuses TCP connections across requests.
# Created lazily on first call, kept alive for the process lifetime.
# This eliminates per-request connection overhead that was hammering
# the trainer during HTMX polling.
_client: Optional[httpx.AsyncClient] = None

# Module-level GET cache: {path: (timestamp, data)}
# Shared across concurrent requests within the same ~1s window.
_cache: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 1.0  # seconds


def _get_client() -> httpx.AsyncClient:
    """Get or create the persistent HTTP client."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=TRAINER_URL,
            timeout=httpx.Timeout(3.0, connect=1.0),  # Short timeouts — fail fast
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )
    return _client


def invalidate(path: str) -> None:
    """Invalidate a specific cached GET path."""
    _cache.pop(path, None)


def invalidate_all() -> None:
    """Invalidate all cached data. Call after mutations."""
    _cache.clear()


async def call(path: str, method: str = "GET", json_data: dict = None,
               timeout: float | None = None) -> Optional[dict]:
    """Call the trainer API. Returns parsed JSON or None on error.

    GET requests are cached for _CACHE_TTL seconds. POST requests are
    never cached. Uses a persistent connection pool.
    """
    if method == "GET":
        cached = _cache.get(path)
        if cached:
            timestamp, data = cached
            if time.monotonic() - timestamp < _CACHE_TTL:
                return data

    try:
        client = _get_client()
        req_timeout = timeout if timeout is not None else (3.0 if method == "GET" else 30.0)
        if method == "GET":
            r = await client.get(path, timeout=req_timeout)
        else:
            r = await client.post(path, json=json_data, timeout=req_timeout)
        r.raise_for_status()
        result = r.json()
    except Exception as e:
        result = {"error": str(e)}

    if method == "GET":
        _cache[path] = (time.monotonic(), result)

    return result


def keep_existing():
    """Return a 204 No Content response. HTMX will not swap the DOM.

    Use this when an API call fails during polling — it preserves
    whatever the panel was showing instead of flashing to "empty".
    """
    from starlette.responses import Response
    return Response(status_code=204)


def is_error(data) -> bool:
    """Check if an API response represents an error.

    Handles all response shapes:
    - None (connection failure)
    - dict with "error" key (API error)
    - Non-dict types are NOT errors (e.g., list of tasks)

    This replaces 3+ inconsistent error-checking patterns scattered
    across partials and actions.
    """
    if data is None:
        return True
    if isinstance(data, dict) and "error" in data:
        return True
    return False
