"""Task management action handlers."""

from starlette.requests import Request
from starlette.responses import HTMLResponse

from acc.ui.api import call as _api, is_error, invalidate_all
from acc.ui import events as E
from acc.ui.partials.model import partial_tasks


async def action_toggle_task(request: Request):
    task_name = request.path_params["name"]
    await _api(f"/tasks/{task_name}/toggle", method="POST")
    invalidate_all()
    resp = await partial_tasks(request)
    resp.headers["HX-Trigger"] = E.TASKS_CHANGED
    return resp


async def action_remove_task(request: Request):
    task_name = request.path_params["name"]
    await _api(f"/tasks/{task_name}/remove", method="POST")
    invalidate_all()
    resp = await partial_tasks(request)
    resp.headers["HX-Trigger"] = E.TASKS_CHANGED
    return resp


async def action_set_weight(request: Request):
    task_name = request.path_params["name"]
    form = await request.form()
    weight = float(form.get("weight", 1.0))
    await _api(f"/tasks/{task_name}/set_weight", method="POST", json_data={"weight": weight})
    invalidate_all()
    resp = await partial_tasks(request)
    resp.headers["HX-Trigger"] = E.TASKS_CHANGED
    return resp


async def action_add_task(request: Request):
    """Add a task from the dashboard form."""
    form = await request.form()
    class_name = str(form.get("class_name", ""))
    dataset_name = str(form.get("dataset_name", ""))
    task_name = str(form.get("task_name", ""))
    weight = float(form.get("weight", 1.0))
    latent_slice = str(form.get("latent_slice", ""))

    if not class_name or not dataset_name:
        return HTMLResponse(
            '<div class="panel"><h3>+ Task</h3><div class="error">Select a task class and dataset</div></div>'
        )

    # Auto-generate name if not provided
    if not task_name:
        task_name = f"{class_name.replace('Task', '').lower()}_{dataset_name}"

    json_data = {
        "class_name": class_name,
        "name": task_name,
        "dataset_name": dataset_name,
        "weight": weight,
    }
    if latent_slice:
        json_data["latent_slice"] = latent_slice

    result = await _api("/tasks/add", method="POST", json_data=json_data)

    if is_error(result):
        error = result.get("error", "Unknown error") if isinstance(result, dict) else "Failed"
        return HTMLResponse(
            f'<div class="panel"><h3>+ Task</h3><div class="error">{error}</div></div>'
        )

    # Success -- HX-Trigger fires tasks-changed -> tasks panel + add-task panel auto-refresh
    return HTMLResponse(
        '<div class="panel"><h3>+ Task</h3><div style="color:#7ee787;">Task added!</div></div>',
        headers={"HX-Trigger": E.TASKS_CHANGED},
    )
