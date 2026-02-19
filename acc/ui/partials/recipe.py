"""Recipe management partial."""

from starlette.requests import Request
from starlette.responses import HTMLResponse

from acc.ui.api import call as _api, is_error, keep_existing


def _phase_html(phase: str, state: str, is_current: bool) -> str:
    """Render a single phase item."""
    if is_current and state == "running":
        return f'<div class="phase-item phase-current">&#9654; {phase}</div>'
    elif is_current and state == "failed":
        return f'<div class="phase-item" style="color:#f85149;">&#10007; {phase}</div>'
    else:
        return f'<div class="phase-item phase-done">&#10003; {phase}</div>'


def _build_branch_progress(current: dict) -> str:
    """Build branch-grouped phase progress HTML."""
    state = current.get("state", "unknown")
    phases_done = current.get("phases_completed", [])
    current_phase = current.get("current_phase", "")
    branches = current.get("branches", [])
    branch_index = current.get("branch_index", 0)
    total_branches = current.get("total_branches", 0)
    branch_results = current.get("branch_results", {})

    all_phases = list(phases_done)
    # Include current phase in the rendered list (unless it's a terminal label)
    include_current = state in ("running", "failed") and current_phase not in ("complete", "initializing")

    if not branches:
        # No branch info — flat list (backward compat)
        html = ""
        for p in phases_done:
            html += _phase_html(p, "completed", False)
        if include_current:
            html += _phase_html(current_phase, state, True)
        elif state == "completed":
            html += _phase_html(current_phase, "completed", False)
        return html

    # Render phases grouped by branch
    html = ""

    # Phases before the first branch (setup phases)
    first_branch_start = branches[0].get("phase_start", 0) if branches else len(all_phases)
    for i in range(min(first_branch_start, len(all_phases))):
        html += _phase_html(all_phases[i], "completed", False)

    for bi, br in enumerate(branches):
        b_name = br.get("name", "?")
        b_desc = br.get("description", "")
        b_start = br.get("phase_start", 0)
        b_end = br.get("phase_end")  # None if branch is still active

        # Branch number (1-indexed)
        b_num = bi + 1
        total = total_branches or len(branches)

        # Branch state: done, active, or pending
        if b_end is not None:
            # Branch is finished
            branch_state_cls = "branch-done"
        elif state == "running" and bi == len(branches) - 1:
            # Last branch and recipe is running — this is the active branch
            branch_state_cls = "branch-active"
        else:
            branch_state_cls = "branch-done"

        desc_part = f' &mdash; {b_desc}' if b_desc else ''
        html += f'<div class="branch-group {branch_state_cls}">'
        html += f'<div class="branch-header">Branch {b_num} of {total}: {b_name}{desc_part}</div>'

        # Phases in this branch
        effective_end = b_end if b_end is not None else len(all_phases)
        for i in range(b_start, min(effective_end, len(all_phases))):
            html += _phase_html(all_phases[i], "completed", False)

        # If this is the active branch and recipe is running, show current phase
        if b_end is None and include_current:
            html += _phase_html(current_phase, state, True)

        html += '</div>'

    # Phases after the last branch (e.g., "Complete")
    if branches:
        last_end = branches[-1].get("phase_end")
        if last_end is not None:
            for i in range(last_end, len(all_phases)):
                html += _phase_html(all_phases[i], "completed", False)
            # Show current phase if it's after all branches
            if include_current:
                html += _phase_html(current_phase, state, True)
            elif state == "completed":
                html += _phase_html(current_phase, "completed", False)

    # Comparison summary when there are results
    if branch_results and state in ("completed", "stopped"):
        html += _build_comparison_table(branches, branch_results)

    return html


def _build_comparison_table(branches: list, branch_results: dict) -> str:
    """Build a comparison summary table from branch eval results."""
    if not branch_results:
        return ""

    # Collect all metric keys across all branches
    # Results structure: {branch_name: {task_name: {metric: value}}}
    all_tasks = set()
    all_metrics = set()
    for br_results in branch_results.values():
        if isinstance(br_results, dict):
            for task_name, metrics in br_results.items():
                all_tasks.add(task_name)
                if isinstance(metrics, dict):
                    all_metrics.update(metrics.keys())

    if not all_tasks:
        return ""

    sorted_tasks = sorted(all_tasks)
    sorted_metrics = sorted(all_metrics)
    branch_names = [br.get("name", "?") for br in branches if br.get("name") in branch_results]

    if len(branch_names) < 2:
        return ""

    html = '<div class="recipe-comparison">'
    html += '<div class="comparison-header">Comparison Summary</div>'
    html += '<table class="comparison-table"><thead><tr>'
    html += '<th>Task / Metric</th>'
    for name in branch_names:
        html += f'<th>{name}</th>'
    html += '</tr></thead><tbody>'

    for task in sorted_tasks:
        for metric in sorted_metrics:
            # Collect values across branches for this task+metric
            values = {}
            for bname in branch_names:
                br_res = branch_results.get(bname, {})
                task_res = br_res.get(task, {}) if isinstance(br_res, dict) else {}
                val = task_res.get(metric) if isinstance(task_res, dict) else None
                if val is not None:
                    values[bname] = val

            if not values:
                continue

            # Determine best value (lowest for loss-like, highest for accuracy-like)
            is_lower_better = any(kw in metric.lower() for kw in ("loss", "mse", "mae", "error", "kl", "l1", "l2"))
            numeric_vals = {k: v for k, v in values.items() if isinstance(v, (int, float))}
            best_branch = None
            if len(numeric_vals) >= 2:
                best_branch = min(numeric_vals, key=numeric_vals.get) if is_lower_better else max(numeric_vals, key=numeric_vals.get)

            html += f'<tr><td>{task} / {metric}</td>'
            for bname in branch_names:
                val = values.get(bname)
                if val is None:
                    html += '<td>-</td>'
                else:
                    fmt = f"{val:.4f}" if isinstance(val, float) else str(val)
                    cls = ' class="comparison-best"' if bname == best_branch else ''
                    html += f'<td{cls}>{fmt}</td>'
            html += '</tr>'

    html += '</tbody></table></div>'
    return html


async def partial_recipe(request: Request):
    """Recipe picker, run/stop button, and phase progress."""
    recipes = await _api("/recipes")
    current = await _api("/recipes/current")

    if is_error(recipes):
        return keep_existing()

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

        # Branch progress indicator (if running and has branch info)
        branch_info = ""
        if current.get("current_branch") and state == "running":
            bi = current.get("branch_index", 0)
            total = current.get("total_branches", 0)
            bname = current["current_branch"]
            if total > 0:
                branch_info = f'<div class="branch-indicator">Branch {bi} of {total}: {bname}</div>'

        # Error display
        error_html = ""
        if state == "failed":
            error = current.get("error", "Unknown error")
            error_html = f'<div style="color:#f85149;font-size:10px;margin-top:4px;white-space:pre-wrap;max-height:100px;overflow-y:auto;">{error[:500]}</div>'

        phases_html = _build_branch_progress(current)

        progress_html = f"""
        <div class="phase-indicator">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                <span style="color:#f0f6fc;font-size:12px;">{current["recipe_name"]}</span>
                <span class="{state_class}" style="font-size:11px;">{state}</span>
            </div>
            {branch_info}
            {phases_html}
            {error_html}
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

    poll_attr = 'hx-get="/partial/recipe" hx-trigger="every 10s" hx-target="#recipe-panel" hx-swap="innerHTML"' if running else ''

    return HTMLResponse(f"""
    <div class="panel" {poll_attr}>
        <h3>Recipes</h3>
        {buttons}
        {progress_html}
    </div>
    """)
