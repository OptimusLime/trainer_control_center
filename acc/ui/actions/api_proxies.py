"""API proxy endpoints for client-side JavaScript."""

import json

import httpx
from starlette.requests import Request
from starlette.responses import HTMLResponse, StreamingResponse

from acc.ui.api import call as _api, TRAINER_URL


async def sse_job(request: Request):
    """Proxy SSE stream from trainer to the browser."""
    job_id = request.path_params["job_id"]
    from_step = int(request.query_params.get("from_step", 0))

    async def event_generator():
        try:
            count = 0
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "GET",
                    f"{TRAINER_URL}/jobs/{job_id}/stream?from_step={from_step}",
                    timeout=None,
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            # Always forward "done" events; throttle the rest
                            if '"done"' in line:
                                yield f"{line}\n\n"
                            else:
                                count += 1
                                if count % 10 == 0:
                                    yield f"{line}\n\n"
        except Exception:
            yield 'data: {"done": true}\n\n'

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def api_jobs_current(request: Request):
    """Proxy for JS to check current job."""
    job = await _api("/jobs/current")
    return HTMLResponse(
        json.dumps(job),
        media_type="application/json",
    )


async def api_jobs_loss_history(request: Request):
    """Proxy for JS to fetch loss history for a job."""
    job_id = request.path_params["job_id"]
    data = await _api(f"/jobs/{job_id}/loss_history")
    return HTMLResponse(
        json.dumps(data),
        media_type="application/json",
    )


async def api_jobs_loss_summary(request: Request):
    """Proxy for JS to fetch loss summary for a job."""
    job_id = request.path_params["job_id"]
    data = await _api(f"/jobs/{job_id}/loss_summary")
    return HTMLResponse(
        json.dumps(data),
        media_type="application/json",
    )
