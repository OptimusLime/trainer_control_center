"""Model and task management partials."""

from starlette.requests import Request
from starlette.responses import HTMLResponse

from acc.ui.api import call as _api, is_error, keep_existing
from acc.ui import components as C


async def partial_model(request: Request):
    data = await _api("/model/describe")
    if is_error(data):
        return keep_existing()

    return HTMLResponse(f"""
    <div class="panel">
        <h3>Model</h3>
        <div class="model-desc">{data["description"]}</div>
        <div style="margin-top:8px; font-size:11px; color:#8b949e;">
            Latent dim: {data["latent_dim"]} | Decoder: {"yes" if data["has_decoder"] else "no"}
        </div>
    </div>
    """)


async def partial_tasks(request: Request):
    """Enhanced task cards with weight adjustment and remove button."""
    tasks = await _api("/tasks")
    if is_error(tasks):
        return keep_existing()
    if not tasks:
        return HTMLResponse(
            '<div class="panel"><h3>Tasks</h3><div class="empty">No tasks attached</div></div>'
        )

    cards = ""
    for t in tasks:
        enabled = "on" if t.get("enabled") else "off"
        check = "checked" if t.get("enabled") else ""
        name = t["name"]
        task_type = t["type"]
        weight = t.get("weight", 1.0)
        dataset = t.get("dataset", "?")
        latent_slice = t.get("latent_slice")
        slice_str = f" | slice={latent_slice[0]}:{latent_slice[1]}" if latent_slice else ""

        cards += f"""
        <div class="task-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span class="name">{name}</span>
                <div style="display:flex;gap:4px;align-items:center;">
                    <input type="checkbox" {check}
                        hx-post="/action/toggle_task/{name}"
                        hx-target="#tasks-panel"
                        hx-swap="innerHTML">
                    <button class="btn btn-sm btn-danger"
                        hx-post="/action/remove_task/{name}"
                        hx-target="#tasks-panel"
                        hx-swap="innerHTML"
                        hx-confirm="Remove task {name}?">&times;</button>
                </div>
            </div>
            <div class="type">{task_type} | {dataset}{slice_str} | {enabled}</div>
            <div style="display:flex;gap:4px;align-items:center;margin-top:4px;">
                <label style="color:#8b949e;font-size:10px;">w:</label>
                <input type="number" step="0.1" min="0" value="{weight}"
                    class="weight-input"
                    id="weight-{name}"
                    hx-post="/action/set_weight/{name}"
                    hx-include="this"
                    hx-target="#tasks-panel"
                    hx-swap="innerHTML"
                    hx-trigger="change"
                    name="weight">
            </div>
        </div>"""

    return HTMLResponse(f"""
    <div class="panel">
        <h3>Tasks ({len(tasks)})</h3>
        {cards}
    </div>
    """)


async def partial_add_task(request: Request):
    """[+ Task] form — select class, dataset, name, weight, add."""
    task_classes = await _api("/registry/tasks")
    datasets = await _api("/datasets")

    if not task_classes or isinstance(task_classes, dict):
        task_options = '<option value="">No task classes available</option>'
    else:
        task_options = '<option value="">-- select task class --</option>'
        for tc in task_classes:
            desc = tc.get("description", "")
            task_options += f'<option value="{tc["class_name"]}" title="{desc}">{tc["class_name"]}</option>'

    if not datasets or isinstance(datasets, dict):
        ds_options = '<option value="">No datasets loaded</option>'
    else:
        ds_options = '<option value="">-- select dataset --</option>'
        for ds in datasets:
            ds_options += f'<option value="{ds["name"]}">{ds["name"]} ({ds["size"]} imgs)</option>'

    return HTMLResponse(f"""
    <div class="panel">
        <h3>+ Task</h3>
        <div class="form-group">
            <label>Task Class</label>
            <select name="class_name" id="add-task-class">{task_options}</select>
        </div>
        <div class="form-group">
            <label>Dataset</label>
            <select name="dataset_name" id="add-task-dataset">{ds_options}</select>
        </div>
        <div class="form-group">
            <label>Name (optional)</label>
            <input type="text" name="task_name" id="add-task-name" placeholder="auto-generated">
        </div>
        <div style="display:flex;gap:8px;">
            <div class="form-group" style="flex:1;">
                <label>Weight</label>
                <input type="number" step="0.1" min="0" value="1.0" name="weight" id="add-task-weight">
            </div>
            <div class="form-group" style="flex:1;">
                <label>Latent Slice</label>
                <input type="text" name="latent_slice" id="add-task-slice" placeholder="e.g. 0:4">
            </div>
        </div>
        <button class="btn btn-primary" style="width:100%;margin-top:4px;"
            hx-post="/action/add_task"
            hx-include="#add-task-class, #add-task-dataset, #add-task-name, #add-task-weight, #add-task-slice"
            hx-target="#add-task-panel"
            hx-swap="innerHTML">Add Task</button>
    </div>
    """)
