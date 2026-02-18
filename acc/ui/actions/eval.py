"""Eval action handlers."""

from starlette.requests import Request

from acc.ui.api import call as _api
from acc.ui.partials.eval import partial_eval, partial_eval_compare
from acc.ui.partials.health import partial_health


async def action_eval(request: Request):
    """Run eval and return the metrics table."""
    return await partial_eval(request)


async def action_eval_compare(request: Request):
    """Run eval comparison between current model and a checkpoint."""
    return await partial_eval_compare(request)


async def action_set_device(request: Request):
    """Change the active device via the trainer API."""
    form = await request.form()
    device = str(form.get("device", ""))
    if device:
        await _api("/device/set", method="POST", json_data={"device": device})
    return await partial_health(request)
