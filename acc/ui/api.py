"""Shared trainer API client for UI modules.

Every partial and action handler calls the trainer API through this module.
This is the single point of configuration for the trainer URL and the
HTTP client settings.
"""

import os
from typing import Optional

import httpx

TRAINER_URL = os.environ.get("ACC_TRAINER_URL", "http://localhost:6060")


async def call(path: str, method: str = "GET", json_data: dict = None) -> Optional[dict]:
    """Call the trainer API. Returns parsed JSON or None on error."""
    try:
        async with httpx.AsyncClient() as client:
            if method == "GET":
                r = await client.get(f"{TRAINER_URL}{path}", timeout=10.0)
            else:
                r = await client.post(f"{TRAINER_URL}{path}", json=json_data, timeout=30.0)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        return {"error": str(e)}
