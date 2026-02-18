"""Dataset generation and recipe action handlers."""

from starlette.requests import Request
from starlette.responses import HTMLResponse

from acc.ui.api import call as _api
from acc.ui import events as E


async def action_generate_dataset(request: Request):
    """Generate a dataset from the dashboard form."""
    form = await request.form()
    gen_name = str(form.get("generator_name", ""))

    if not gen_name:
        return HTMLResponse(
            '<div class="panel"><h3>+ Dataset</h3><div class="error">Select a generator</div></div>'
        )

    # Collect param_ fields
    params = {}
    for key in form.keys():
        if key.startswith("param_"):
            pname = key[6:]  # strip "param_" prefix
            val = str(form.get(key, ""))
            # Try to parse as number
            try:
                params[pname] = int(val)
            except ValueError:
                try:
                    params[pname] = float(val)
                except ValueError:
                    params[pname] = val

    result = await _api("/generators/generate", method="POST", json_data={
        "generator_name": gen_name,
        "params": params,
    })

    if result and isinstance(result, dict) and result.get("error"):
        return HTMLResponse(
            f'<div class="panel"><h3>+ Dataset</h3><div class="error">{result["error"]}</div></div>'
        )

    ds_name = result.get("name", "?") if result else "?"
    ds_size = result.get("size", "?") if result else "?"
    # HX-Trigger fires datasets-changed -> datasets panel + add-task panel auto-refresh
    return HTMLResponse(
        f'<div class="panel"><h3>+ Dataset</h3><div style="color:#7ee787;">Generated {ds_name} ({ds_size} images)</div></div>',
        headers={"HX-Trigger": E.DATASETS_CHANGED},
    )


async def action_recipe_run(request: Request):
    form = await request.form()
    recipe_name = form.get("recipe-select", "")
    if not recipe_name:
        return HTMLResponse(
            '<div class="panel"><h3>Recipes</h3><div class="error">No recipe selected</div></div>'
        )
    result = await _api(f"/recipes/{recipe_name}/run", method="POST")
    if result and isinstance(result, dict) and result.get("error"):
        return HTMLResponse(
            f'<div class="panel"><h3>Recipes</h3><div class="error">{result["error"]}</div></div>'
        )
    return HTMLResponse(
        '<div hx-get="/partial/recipe" hx-trigger="load" hx-swap="innerHTML"></div>'
    )


async def action_recipe_stop(request: Request):
    await _api("/recipes/stop", method="POST")
    return HTMLResponse(
        '<div hx-get="/partial/recipe" hx-trigger="load" hx-swap="innerHTML"></div>'
    )
