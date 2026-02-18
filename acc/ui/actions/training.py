"""Training action handlers."""

from starlette.requests import Request
from starlette.responses import HTMLResponse

from acc.ui.api import call as _api, invalidate_all
from acc.ui import components as C


async def action_train(request: Request):
    form = await request.form()
    steps = int(form.get("train-steps", 500))
    lr = form.get("train-lr", "1e-3")
    try:
        lr_float = float(lr)
    except ValueError:
        lr_float = 1e-3

    result = await _api(
        "/train/start", method="POST", json_data={"steps": steps, "lr": lr_float}
    )
    if result and "id" in result:
        skeleton = C.training_panel_skeleton()
        return HTMLResponse(f"""
        <div class="panel">
            <h3>Loss Curves <span style="color:#f0883e;font-size:11px;">(training...)</span></h3>
            {skeleton}
        </div>
        <div class="panel" id="loss-summary-panel">
            <h3>Loss Summary</h3>
            <div id="loss-summary-content">{C.empty("Training in progress...")}</div>
        </div>
        <script>requestAnimationFrame(function() {{ initChart(); taskHealthState={{}}; startSSE('{result["id"]}'); }});</script>
        """)

    error = (
        result.get("error", "Unknown error")
        if result
        else "Failed to connect to trainer"
    )
    return HTMLResponse(
        f'<div class="panel"><h3>Training</h3><div class="error">{error}</div></div>'
    )


async def action_stop(request: Request):
    await _api("/train/stop", method="POST")
    return HTMLResponse("""
    <div class="panel" hx-get="/partial/training" hx-trigger="load">
        <h3>Training</h3><div class="empty">Stopped. Refreshing...</div>
    </div>
    """)
