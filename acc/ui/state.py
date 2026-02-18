"""DashboardState — cached async proxy to the trainer API.

The dashboard UI makes many redundant API calls to the trainer during a
single page render cycle. For example, /health is fetched 7 times by
independent partials, and /checkpoints/tree is fetched 3 times.

DashboardState provides a request-scoped cache: the first call to
state.health() hits the trainer API, subsequent calls return the cached
result. Call state.invalidate() or state.invalidate_all() to force
re-fetch (e.g., after an action mutates state).

Usage in a partial:
    async def partial_foo(request: Request):
        state = get_state(request)
        health = await state.health()
        if not health or not health.get("has_model"):
            return components.no_model_guard("Foo")
        ...

The state object is stored on the request via Starlette middleware or
a simple helper, so all partials within one HTTP request share the
same cache.
"""

import os
import time
from typing import Optional

import httpx

TRAINER_URL = os.environ.get("ACC_TRAINER_URL", "http://localhost:6060")

# Cache TTL in seconds. Within one browser page load cycle, multiple
# partials share the cache. Between requests, the cache expires.
_CACHE_TTL = 1.0


class DashboardState:
    """Request-scoped cached proxy to the trainer API.

    Each instance maintains a cache dict keyed by API path. Cache entries
    expire after _CACHE_TTL seconds.
    """

    def __init__(self):
        self._cache: dict[str, tuple[float, any]] = {}

    def invalidate(self, path: str) -> None:
        """Invalidate a specific cached API path."""
        self._cache.pop(path, None)

    def invalidate_all(self) -> None:
        """Invalidate all cached data. Call after mutations."""
        self._cache.clear()

    async def _api(self, path: str, method: str = "GET", json_data: dict = None) -> Optional[dict]:
        """Call the trainer API with caching for GET requests.

        POST requests are never cached. GET requests are cached for _CACHE_TTL.
        """
        # Only cache GET requests
        if method == "GET":
            cached = self._cache.get(path)
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

        # Cache GET results
        if method == "GET":
            self._cache[path] = (time.monotonic(), result)

        return result

    # ─── Typed convenience methods for frequently-used endpoints ───

    async def health(self) -> Optional[dict]:
        """Fetch /health (cached). Returns dict with 'has_model', 'status', etc."""
        return await self._api("/health")

    async def has_model(self) -> bool:
        """Check if a model is loaded."""
        h = await self.health()
        return bool(h and isinstance(h, dict) and h.get("has_model"))

    async def checkpoint_tree(self) -> Optional[dict]:
        """Fetch /checkpoints/tree (cached). Returns dict with 'nodes', 'current_id'."""
        return await self._api("/checkpoints/tree")

    async def current_checkpoint_id(self) -> Optional[str]:
        """Get the current checkpoint ID, or None."""
        tree = await self.checkpoint_tree()
        if tree and isinstance(tree, dict):
            return tree.get("current_id")
        return None

    async def datasets(self) -> Optional[list]:
        """Fetch /datasets (cached). Returns list of dataset dicts."""
        return await self._api("/datasets")

    async def current_job(self) -> Optional[dict]:
        """Fetch /jobs/current (cached). Returns current job dict or error."""
        return await self._api("/jobs/current")

    # ─── Passthrough for non-cached / POST calls ───

    async def api(self, path: str, method: str = "GET", json_data: dict = None) -> Optional[dict]:
        """Direct API call (goes through cache for GETs)."""
        return await self._api(path, method=method, json_data=json_data)


def get_state(request) -> DashboardState:
    """Get or create a DashboardState for this request.

    Stores the state on request.state so all partials within
    one request share the same cache.
    """
    if not hasattr(request.state, "dashboard"):
        request.state.dashboard = DashboardState()
    return request.state.dashboard
