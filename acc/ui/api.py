"""Shared trainer API client for UI modules.

Every partial and action handler calls the trainer API through this module.
This is the single point of configuration for the trainer URL, HTTP client
settings, and the GET response cache.

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

# Module-level GET cache: {path: (timestamp, data)}
# Shared across concurrent requests within the same ~1s window.
_cache: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 1.0  # seconds


def invalidate(path: str) -> None:
    """Invalidate a specific cached GET path."""
    _cache.pop(path, None)


def invalidate_all() -> None:
    """Invalidate all cached data. Call after mutations."""
    _cache.clear()


async def call(path: str, method: str = "GET", json_data: dict = None) -> Optional[dict]:
    """Call the trainer API. Returns parsed JSON or None on error.

    GET requests are cached for _CACHE_TTL seconds. POST requests are
    never cached.
    """
    if method == "GET":
        cached = _cache.get(path)
        if cached:
            timestamp, data = cached
            if time.monotonic() - timestamp < _CACHE_TTL:
                return data

    try:
        async with httpx.AsyncClient() as client:
            if method == "GET":
                r = await client.get(f"{TRAINER_URL}{path}", timeout=10.0)
            else:
                r = await client.post(f"{TRAINER_URL}{path}", json=json_data, timeout=30.0)
            r.raise_for_status()
            result = r.json()
    except Exception as e:
        result = {"error": str(e)}

    if method == "GET":
        _cache[path] = (time.monotonic(), result)

    return result


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
