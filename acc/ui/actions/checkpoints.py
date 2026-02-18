"""Checkpoint action handlers."""

from starlette.requests import Request
from starlette.responses import HTMLResponse

from acc.ui.api import call as _api
from acc.ui import events as E


async def action_save_checkpoint(request: Request):
    await _api("/checkpoints/save", method="POST", json_data={"tag": "checkpoint"})
    return HTMLResponse(
        '<div class="panel"><h3>Checkpoint Tree</h3><div class="empty">Saving...</div></div>',
        headers={"HX-Trigger": E.CHECKPOINT_CHANGED},
    )


async def action_load_checkpoint(request: Request):
    cp_id = request.path_params["cp_id"]
    result = await _api("/checkpoints/load", method="POST", json_data={"id": cp_id})
    if result and isinstance(result, dict) and result.get("error"):
        return HTMLResponse(f'<div class="error">{result["error"]}</div>')
    # HX-Trigger fires checkpoint-changed -> all panels with that hx-trigger auto-refresh
    return HTMLResponse(
        '<div class="panel"><h3>Checkpoint Tree</h3><div class="empty">Loading checkpoint...</div></div>',
        headers={"HX-Trigger": E.CHECKPOINT_CHANGED},
    )


async def action_fork_checkpoint(request: Request):
    cp_id = request.path_params["cp_id"]
    await _api(
        "/checkpoints/fork", method="POST",
        json_data={"id": cp_id, "new_tag": f"fork_{cp_id[:6]}"}
    )
    return HTMLResponse(
        '<div class="panel"><h3>Checkpoint Tree</h3><div class="empty">Forking...</div></div>',
        headers={"HX-Trigger": E.CHECKPOINT_CHANGED},
    )
