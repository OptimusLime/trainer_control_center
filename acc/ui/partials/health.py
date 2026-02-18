"""Health indicator and device selector partial."""

from starlette.requests import Request
from starlette.responses import HTMLResponse

from acc.ui.api import call as _api


async def partial_health(request: Request):
    """Connection health indicator with device selector."""
    health = await _api("/health")
    connected = health and isinstance(health, dict) and health.get("status") == "ok"

    if connected:
        device = health.get("device", "?")
        n_tasks = health.get("num_tasks", 0)
        n_datasets = health.get("num_datasets", 0)

        device_info = await _api("/device")
        device_options = ""
        if device_info and "available" in device_info:
            for d in device_info["available"]:
                selected = "selected" if d == device_info.get("current") else ""
                device_options += f'<option value="{d}" {selected}>{d}</option>'

        device_selector = f"""
        <select id="device-select" style="background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:2px 6px;border-radius:3px;font-family:inherit;font-size:11px;"
            hx-post="/action/set_device"
            hx-include="this"
            hx-target="#trainer-status"
            hx-swap="innerHTML"
            hx-trigger="change"
            name="device">
            {device_options}
        </select>
        """ if device_options else ""

        return HTMLResponse(
            f'<span style="display:flex;align-items:center;gap:6px;">'
            f'<span style="color:#7ee787;font-size:11px;">&#9679; {n_tasks}T {n_datasets}D</span>'
            f'{device_selector}'
            f'</span>'
        )
    else:
        return HTMLResponse(
            f'<span style="color:#f85149;font-size:11px;">'
            f"&#9679; Disconnected</span>"
        )
