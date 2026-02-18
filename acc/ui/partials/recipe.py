"""Recipe management partial."""

from starlette.requests import Request
from starlette.responses import HTMLResponse

from acc.ui.api import call as _api


async def partial_recipe(request: Request):
    """Recipe picker, run/stop button, and phase progress."""
    recipes = await _api("/recipes")
    current = await _api("/recipes/current")

    running = (
        current
        and isinstance(current, dict)
        and current.get("state") == "running"
    )

    if not recipes or isinstance(recipes, dict):
        options = '<option value="">No recipes available</option>'
    else:
        options = '<option value="">-- select recipe --</option>'
        for r in recipes:
            options += f'<option value="{r["name"]}">{r["name"]}</option>'

    progress_html = ""
    if current and isinstance(current, dict) and current.get("recipe_name"):
        state = current.get("state", "unknown")
        state_class = f"status-{state}"
        phase = current.get("current_phase", "")
        phases_done = current.get("phases_completed", [])

        phases_list = ""
        for p in phases_done:
            phases_list += f'<div class="phase-item phase-done">&#10003; {p}</div>'

        if state == "running":
            phases_list += f'<div class="phase-item phase-current">&#9654; {phase}</div>'
        elif state == "completed":
            phases_list += f'<div class="phase-item phase-done">&#10003; {phase}</div>'
        elif state == "failed":
            error = current.get("error", "Unknown error")
            phases_list += f'<div class="phase-item" style="color:#f85149;">&#10007; {phase}</div>'
            phases_list += f'<div style="color:#f85149;font-size:10px;margin-top:4px;">{error[:200]}</div>'

        progress_html = f"""
        <div class="phase-indicator">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                <span style="color:#f0f6fc;font-size:12px;">{current["recipe_name"]}</span>
                <span class="{state_class}" style="font-size:11px;">{state}</span>
            </div>
            {phases_list}
        </div>
        """

    if running:
        buttons = """
        <button class="btn btn-danger" style="width:100%;"
            hx-post="/action/recipe_stop"
            hx-target="#recipe-panel"
            hx-swap="innerHTML">Stop Recipe</button>
        """
    else:
        buttons = f"""
        <select id="recipe-select" name="recipe-select" class="recipe-select">{options}</select>
        <button class="btn btn-primary" style="width:100%;"
            hx-post="/action/recipe_run"
            hx-include="#recipe-select"
            hx-target="#recipe-panel"
            hx-swap="innerHTML">Run Recipe</button>
        """

    poll_attr = 'hx-get="/partial/recipe" hx-trigger="every 2s" hx-target="#recipe-panel" hx-swap="innerHTML"' if running else ''

    return HTMLResponse(f"""
    <div class="panel" {poll_attr}>
        <h3>Recipes</h3>
        {buttons}
        {progress_html}
    </div>
    """)
