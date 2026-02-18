"""UI Dashboard — Starlette + HTMX + SSE on localhost:8081.

Stateless: all state comes from the trainer API (localhost:6060).
HTMX for partial page updates. SSE for live training loss streaming.

M2 dashboard: task management, per-task loss curves, reconstruction
comparison, eval metrics display, persistent training history.
"""

import json
import os
from pathlib import Path
from typing import Optional

import httpx
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, StreamingResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.requests import Request

from acc.ui import components as C
from acc.ui import events as E
from acc.ui.state import DashboardState, get_state

_STATIC_DIR = Path(__file__).parent / "static"

TRAINER_URL = os.environ.get("ACC_TRAINER_URL", "http://localhost:6060")


async def _api(path: str, method: str = "GET", json_data: dict = None) -> Optional[dict]:
    """Call the trainer API. Returns parsed JSON or None on error."""
    try:
        async with httpx.AsyncClient() as client:
            if method == "GET":
                r = await client.get(f"{TRAINER_URL}{path}", timeout=10.0)
            else:
                r = await client.post(f"{TRAINER_URL}{path}", json=json_data, timeout=30.0)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        return {"error": str(e)}


# ─── HTML Components ───


def _page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://unpkg.com/htmx.org@1.9.12"></script>
    <script src="https://unpkg.com/htmx.org@1.9.12/dist/ext/sse.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
    <link rel="stylesheet" href="/static/dashboard.css">
</head>
<body>
    <div class="header">
        <h1>ACC -- Autoencoder Control Center</h1>
        <div style="display:flex;align-items:center;gap:12px;">
            <span id="checkpoint-indicator" hx-get="/partial/checkpoint_indicator" hx-trigger="load, every 3s, checkpoint-changed from:body" style="font-weight:600;"></span>
            <span id="trainer-status" hx-get="/partial/health" hx-trigger="load, every 3s"></span>
            <span class="step" id="step-counter" hx-get="/partial/step" hx-trigger="every 2s">[step: -]</span>
        </div>
    </div>
    <div class="layout">
        <div class="sidebar" id="sidebar">
            {_sidebar_placeholder()}
        </div>
        <div class="main" id="main-content">
            {_main_placeholder()}
        </div>
    </div>
    <script src="/static/dashboard.js"></script>
</body>
</html>"""


def _sidebar_placeholder() -> str:
    return """
        <div id="recipe-panel" hx-get="/partial/recipe" hx-trigger="load">
            <div class="panel"><h3>Recipes</h3><div class="empty">Loading...</div></div>
        </div>
        <div id="model-panel" hx-get="/partial/model" hx-trigger="load, every 5s, checkpoint-changed from:body, model-changed from:body">
            <div class="panel"><h3>Model</h3><div class="empty">Loading...</div></div>
        </div>
        <div id="tasks-panel" hx-get="/partial/tasks" hx-trigger="load, every 3s, checkpoint-changed from:body, tasks-changed from:body">
            <div class="panel"><h3>Tasks</h3><div class="empty">Loading...</div></div>
        </div>
        <div id="add-task-panel" hx-get="/partial/add_task" hx-trigger="load, every 5s, tasks-changed from:body, datasets-changed from:body">
            <div class="panel"><h3>+ Task</h3><div class="empty">Loading...</div></div>
        </div>
        <div id="generate-panel" hx-get="/partial/generate" hx-trigger="load, every 5s, datasets-changed from:body">
            <div class="panel"><h3>+ Dataset</h3><div class="empty">Loading...</div></div>
        </div>
        <div id="checkpoints-panel" hx-get="/partial/checkpoints" hx-trigger="load, every 5s, checkpoint-changed from:body">
            <div class="panel"><h3>Checkpoint Tree</h3><div class="empty">Loading...</div></div>
        </div>
    """


def _main_placeholder() -> str:
    return """
        <div id="training-panel" hx-get="/partial/training" hx-trigger="load, checkpoint-changed from:body, training-done from:body">
            <div class="panel"><h3>Training</h3><div class="empty">Loading...</div></div>
        </div>
        <div id="recon-panel" hx-get="/partial/reconstructions" hx-trigger="load, every 10s, checkpoint-changed from:body, training-done from:body">
            <div class="panel"><h3>Reconstructions</h3><div class="empty">Loading...</div></div>
        </div>
        <div id="eval-panel" hx-get="/partial/eval" hx-trigger="load, checkpoint-changed from:body, training-done from:body, tasks-changed from:body">
            <div class="panel"><h3>Eval Metrics</h3><div class="empty">Loading...</div></div>
        </div>
        <div id="jobs-panel" hx-get="/partial/jobs_history" hx-trigger="load, every 5s, training-done from:body">
            <div class="panel"><h3>Job History</h3><div class="empty">Loading...</div></div>
        </div>
        <div id="traversal-panel" hx-get="/partial/traversals" hx-trigger="load, checkpoint-changed from:body, training-done from:body">
            <div class="panel"><h3>Latent Traversals</h3><div class="empty">Run eval to generate</div></div>
        </div>
        <div id="sort-panel" hx-get="/partial/sort_by_factor" hx-trigger="load, checkpoint-changed from:body, training-done from:body">
            <div class="panel"><h3>Sort by Factor</h3><div class="empty">Run eval to generate</div></div>
        </div>
        <div id="attention-panel" hx-get="/partial/attention_maps" hx-trigger="load, checkpoint-changed from:body, training-done from:body">
            <div class="panel"><h3>Attention Maps</h3><div class="empty">Run eval to generate</div></div>
        </div>
        <div id="datasets-panel" hx-get="/partial/datasets" hx-trigger="load, every 10s, datasets-changed from:body">
            <div class="panel"><h3>Datasets</h3><div class="empty">Loading...</div></div>
        </div>
    """


# _chart_js() — REMOVED: extracted to acc/ui/static/dashboard.js


# ─── HTMX Partial Endpoints ───


async def index(request: Request):
    return HTMLResponse(_page("ACC", ""))


async def partial_model(request: Request):
    data = await _api("/model/describe")
    if data is None or "error" in data:
        return HTMLResponse(C.no_model_guard("Model"))

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
    if not tasks or "error" in (tasks if isinstance(tasks, dict) else {}):
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


async def partial_generate(request: Request):
    """[+ Dataset] panel — pick generator, configure params, generate."""
    generators = await _api("/registry/generators")

    if not generators or isinstance(generators, dict):
        return HTMLResponse(
            '<div class="panel"><h3>+ Dataset</h3><div class="empty">No generators available</div></div>'
        )

    # Build generator selector and param forms
    gen_options = '<option value="">-- select generator --</option>'
    param_forms = ""
    for gen in generators:
        gen_name = gen["name"]
        gen_desc = gen.get("description", "")
        gen_options += f'<option value="{gen_name}" title="{gen_desc}">{gen_name}</option>'

        # Build param inputs for this generator (hidden until selected)
        params = gen.get("parameters", {})
        param_html = ""
        for pname, pinfo in params.items():
            ptype = pinfo.get("type", "text")
            pdefault = pinfo.get("default", "")
            pdesc = pinfo.get("description", "")
            input_type = "number" if ptype == "int" or ptype == "float" else "text"
            param_html += f"""
            <div class="form-group">
                <label>{pname}: {pdesc}</label>
                <input type="{input_type}" name="param_{pname}" value="{pdefault}">
            </div>"""

        param_forms += f'<div id="gen-params-{gen_name}" class="gen-params" style="display:none;">{param_html}</div>'

    return HTMLResponse(f"""
    <div class="panel">
        <h3>+ Dataset</h3>
        <div class="form-group">
            <label>Generator</label>
            <select name="generator_name" id="gen-select"
                onchange="document.querySelectorAll('.gen-params').forEach(e=>e.style.display='none'); var sel=this.value; if(sel) document.getElementById('gen-params-'+sel).style.display='block';">
                {gen_options}
            </select>
        </div>
        {param_forms}
        <button class="btn btn-primary" style="width:100%;margin-top:4px;"
            hx-post="/action/generate_dataset"
            hx-include="#gen-select, [name^=param_]"
            hx-target="#generate-panel"
            hx-swap="innerHTML">Generate</button>
    </div>
    """)


async def partial_checkpoint_indicator(request: Request):
    """Header badge showing current checkpoint tag + short ID."""
    tree = await _api("/checkpoints/tree")
    if not tree or not isinstance(tree, dict):
        return HTMLResponse('<span style="color:#484f58;">[no checkpoint]</span>')
    current_id = tree.get("current_id")
    if not current_id:
        return HTMLResponse('<span style="color:#484f58;">[no checkpoint]</span>')
    nodes = tree.get("nodes", [])
    tag = "unknown"
    for n in nodes:
        if n.get("id") == current_id:
            tag = n.get("tag", "unknown")
            break
    short_id = current_id[:8]
    return HTMLResponse(
        f'<span style="color:#d2a8ff;background:#1c1430;padding:3px 8px;border-radius:4px;border:1px solid #6e40aa;">CP: {tag} ({short_id})</span>'
    )


async def partial_training(request: Request):
    """Training panel — renders server-side loss summary for the most recent job.

    No 500ms race. No client-side-only data. The server always renders
    the latest loss summary directly in HTML. JS handles the chart and SSE.
    """
    job = await _api("/jobs/current")
    running = job and isinstance(job, dict) and job.get("state") == "running"

    # Find the most recent job (running or completed) for loss summary
    recent_job_id = None
    summary_html = '<div class="empty">No training data yet</div>'
    health_banner_html = ""

    if running and job:
        recent_job_id = job.get("id")
    else:
        # Check job history for the most recent completed job
        history = await _api("/jobs/history?limit=1")
        if history and isinstance(history, list) and len(history) > 0:
            recent_job_id = history[0].get("id")

    # Render loss summary table + health banner server-side
    summary_data = None
    if recent_job_id:
        summary_data = await _api(f"/jobs/{recent_job_id}/loss_summary")
        if not (summary_data and isinstance(summary_data, dict) and "error" not in summary_data):
            summary_data = None

    summary_html = C.loss_summary_table(summary_data)
    health_banner_html = C.health_banner(summary_data)

    # JS: initialize chart and load data — wrapped in requestAnimationFrame
    # to guarantee the canvas is in the DOM and laid out before Chart.js touches it
    if running and recent_job_id:
        init_js = f"requestAnimationFrame(function() {{ initChart(); loadLossHistory('{recent_job_id}'); setTimeout(function() {{ startSSE('{recent_job_id}'); }}, 300); }});"
    elif recent_job_id:
        init_js = f"requestAnimationFrame(function() {{ initChart(); loadLossHistory('{recent_job_id}'); }});"
    else:
        init_js = "requestAnimationFrame(function() { initChart(); });"

    running_indicator = '<span style="color:#f0883e;font-size:11px;"> (training...)</span>' if running else ''

    # When NOT running, poll every 3s to detect new jobs (JS-based polling).
    # When running, SSE handles live updates; no polling needed.
    # SSE 'done' handler refreshes the panel, which restarts the poll cycle.
    if not running:
        poll_js = """
        (function() {
            var pollTimer = setInterval(function() {
                fetch('/api/jobs/current').then(function(r) { return r.json(); }).then(function(job) {
                    if (job && job.state === 'running') {
                        clearInterval(pollTimer);
                        htmx.ajax('GET', '/partial/training', {target: '#training-panel', swap: 'innerHTML'});
                    }
                }).catch(function() {});
            }, 2000);
        })();
        """
    else:
        poll_js = ""

    return HTMLResponse(f"""
    <div class="panel">
        <h3>Loss Curves{running_indicator}</h3>
        {health_banner_html}
        <div class="chart-container">
            <canvas id="loss-chart"></canvas>
        </div>
        <div id="loss-log"></div>
    </div>
    <div class="panel" id="loss-summary-panel">
        <h3>Loss Summary</h3>
        <div id="loss-summary-content">{summary_html}</div>
    </div>
    <script>{init_js}{poll_js}</script>
    """)


async def partial_reconstructions(request: Request):
    """Side-by-side original vs reconstruction comparison."""
    health = await _api("/health")
    if not health or not health.get("has_model"):
        return HTMLResponse(C.no_model_guard("Reconstructions"))

    data = await _api("/eval/reconstructions", method="POST", json_data={"n": 8})
    if not data or "error" in (data if isinstance(data, dict) else {}):
        # Fallback: show samples only
        datasets = await _api("/datasets")
        if datasets and isinstance(datasets, list) and len(datasets) > 0:
            ds_name = datasets[0]["name"]
            samples = await _api(f"/datasets/{ds_name}/sample?n=8")
            if samples and "images" in samples:
                imgs = "".join(
                    f'<img src="data:image/png;base64,{b64}">' for b64 in samples["images"]
                )
                return HTMLResponse(f"""
                <div class="panel">
                    <h3>Reconstructions</h3>
                    <div class="recon-label">Samples from {ds_name} (no reconstructions yet):</div>
                    <div class="recon-grid">{imgs}</div>
                </div>
                """)
        return HTMLResponse(
            '<div class="panel"><h3>Reconstructions</h3><div class="empty">No data</div></div>'
        )

    # Side-by-side: original on top, reconstruction on bottom
    pairs = ""
    originals = data.get("originals", [])
    reconstructions = data.get("reconstructions", [])
    for i in range(len(originals)):
        orig_b64 = originals[i]
        recon_b64 = reconstructions[i] if i < len(reconstructions) else originals[i]
        pairs += f"""
        <div class="recon-pair">
            <img src="data:image/png;base64,{orig_b64}" title="original">
            <img src="data:image/png;base64,{recon_b64}" title="reconstruction" style="border-color:#58a6ff;">
        </div>"""

    return HTMLResponse(f"""
    <div class="panel">
        <h3>Reconstructions (top=input, bottom=output)</h3>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
            {pairs}
        </div>
        <button class="btn" style="margin-top:8px;"
            hx-get="/partial/reconstructions"
            hx-target="#recon-panel"
            hx-swap="innerHTML">Refresh</button>
    </div>
    """)


async def partial_eval(request: Request):
    """Eval metrics with optional checkpoint comparison.

    If comparison data exists in query params (from action_eval_compare),
    shows a side-by-side table. Otherwise shows single-model eval.
    """
    health = await _api("/health")
    if not health or not health.get("has_model"):
        return HTMLResponse(C.no_model_guard("Eval Metrics"))

    # Run eval on current model (the "base")
    results = await _api("/eval/run", method="POST")
    if not results or "error" in (results if isinstance(results, dict) else {}):
        # Build checkpoint selector even when no eval yet
        cp_selector = await _checkpoint_selector_html()
        return HTMLResponse(f"""
        <div class="panel">
            <h3>Eval Metrics</h3>
            <div class="empty">No eval results yet</div>
            <div style="display:flex;gap:8px;margin-top:8px;align-items:center;">
                <button class="btn"
                    hx-post="/action/eval"
                    hx-target="#eval-panel"
                    hx-swap="innerHTML">Run Eval</button>
            </div>
            {cp_selector}
        </div>
        """)

    # Single-model table (no comparison)
    rows = ""
    for task_name, metrics in results.items():
        for metric_name, value in metrics.items():
            css_class = C.metric_color(metric_name, value)
            rows += f"""
            <tr>
                <td style="color:#f0f6fc;">{task_name}</td>
                <td>{metric_name}</td>
                <td class="{css_class}">{value:.4f}</td>
            </tr>"""

    cp_selector = await _checkpoint_selector_html()

    return HTMLResponse(f"""
    <div class="panel">
        <h3>Eval Metrics (Current Model)</h3>
        <table class="eval-table">
            <thead><tr><th>Task</th><th>Metric</th><th>Value</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        <div style="display:flex;gap:8px;margin-top:8px;align-items:center;">
            <button class="btn"
                hx-post="/action/eval"
                hx-target="#eval-panel"
                hx-swap="innerHTML">Run Eval</button>
        </div>
        {cp_selector}
    </div>
    """)


async def _checkpoint_selector_html() -> str:
    """Build a checkpoint dropdown for comparison eval."""
    tree = await _api("/checkpoints/tree")
    if not tree or not tree.get("nodes"):
        return ""

    current_id = tree.get("current_id")
    options = '<option value="">-- compare with checkpoint --</option>'
    for node in tree["nodes"]:
        nid = node["id"]
        tag = node["tag"]
        short = nid[:8]
        is_current = " (current)" if nid == current_id else ""
        options += f'<option value="{nid}">{tag} ({short}){is_current}</option>'

    return f"""
    <div style="margin-top:10px;border-top:1px solid #30363d;padding-top:8px;">
        <div style="color:#8b949e;font-size:11px;margin-bottom:4px;">Compare with checkpoint:</div>
        <div style="display:flex;gap:6px;align-items:center;">
            <select id="eval-compare-cp" name="eval-compare-cp" style="flex:1;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:4px 8px;border-radius:4px;font-family:inherit;font-size:12px;">
                {options}
            </select>
            <button class="btn btn-primary btn-sm"
                hx-post="/action/eval_compare"
                hx-include="#eval-compare-cp"
                hx-target="#eval-panel"
                hx-swap="innerHTML">Compare</button>
        </div>
    </div>
    """


async def partial_eval_compare(request: Request):
    """Run eval on current model AND a comparison checkpoint, show side-by-side table."""
    form = await request.form()
    compare_cp_id = str(form.get("eval-compare-cp", ""))

    if not compare_cp_id:
        return await partial_eval(request)

    health = await _api("/health")
    if not health or not health.get("has_model"):
        return HTMLResponse(C.no_model_guard("Eval Metrics"))

    # Run eval on current model (base)
    base_results = await _api("/eval/run", method="POST")
    if not base_results or "error" in (base_results if isinstance(base_results, dict) else {}):
        return HTMLResponse(
            '<div class="panel"><h3>Eval Metrics</h3><div class="error">Base eval failed</div></div>'
        )

    # Run eval on comparison checkpoint
    compare_data = await _api("/eval/checkpoint", method="POST", json_data={"checkpoint_id": compare_cp_id})
    if not compare_data or "error" in (compare_data if isinstance(compare_data, dict) else {}):
        error_msg = compare_data.get("error", "Unknown error") if isinstance(compare_data, dict) else "Failed"
        return HTMLResponse(
            f'<div class="panel"><h3>Eval Metrics</h3><div class="error">Comparison eval failed: {error_msg}</div></div>'
        )

    compare_tag = compare_data.get("tag", compare_cp_id[:8])
    compare_metrics = compare_data.get("metrics", {})

    # Build comparison table
    # Collect all (task, metric) pairs from both
    all_rows = []
    all_tasks = set(base_results.keys()) | set(compare_metrics.keys())
    for task_name in sorted(all_tasks):
        base_task_metrics = base_results.get(task_name, {})
        comp_task_metrics = compare_metrics.get(task_name, {})
        all_metric_names = set(base_task_metrics.keys()) | set(comp_task_metrics.keys())
        for metric_name in sorted(all_metric_names):
            base_val = base_task_metrics.get(metric_name)
            comp_val = comp_task_metrics.get(metric_name)
            all_rows.append((task_name, metric_name, base_val, comp_val))

    rows_html = ""
    for task_name, metric_name, base_val, comp_val in all_rows:
        higher_better = C.metric_higher_is_better(metric_name)

        base_str = f"{base_val:.4f}" if base_val is not None else "-"
        comp_str = f"{comp_val:.4f}" if comp_val is not None else "-"

        # Determine which is best
        base_bold = ""
        comp_bold = ""
        if base_val is not None and comp_val is not None:
            if higher_better:
                if base_val >= comp_val:
                    base_bold = "font-weight:700;color:#7ee787;"
                else:
                    comp_bold = "font-weight:700;color:#7ee787;"
            else:
                if base_val <= comp_val:
                    base_bold = "font-weight:700;color:#7ee787;"
                else:
                    comp_bold = "font-weight:700;color:#7ee787;"

        rows_html += f"""
        <tr>
            <td style="color:#f0f6fc;">{task_name}</td>
            <td>{metric_name}</td>
            <td style="{base_bold}">{base_str}</td>
            <td style="{comp_bold}">{comp_str}</td>
        </tr>"""

    cp_selector = await _checkpoint_selector_html()

    return HTMLResponse(f"""
    <div class="panel">
        <h3>Eval Comparison: Current vs {compare_tag}</h3>
        <table class="eval-table">
            <thead>
                <tr>
                    <th>Task</th>
                    <th>Metric</th>
                    <th style="color:#58a6ff;">Current</th>
                    <th style="color:#d2a8ff;">{compare_tag}</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        <div style="display:flex;gap:8px;margin-top:8px;align-items:center;">
            <button class="btn"
                hx-post="/action/eval"
                hx-target="#eval-panel"
                hx-swap="innerHTML">Run Eval (single)</button>
        </div>
        {cp_selector}
    </div>
    """)


# _resolve_metric, _METRIC_THRESHOLDS, _metric_color, _metric_higher_is_better
# REMOVED: now in acc.ui.components (C.metric_color, C.metric_higher_is_better)


async def partial_jobs_history(request: Request):
    """Job history panel — recent jobs with summary, click to load loss curves."""
    history = await _api("/jobs/history?limit=10")
    if not history or isinstance(history, dict):
        return HTMLResponse(
            '<div class="panel"><h3>Job History</h3><div class="empty">No jobs yet</div></div>'
        )

    items = ""
    for j in history:
        state = j.get("state", "?")
        state_class = f"status-{state}"
        steps = j.get("current_step", 0)
        total = j.get("total_steps", 0)
        job_id = j["id"]

        # Final losses summary with health coloring
        final_losses = j.get("final_losses", {})
        overall_health = j.get("overall_health", "unknown")
        loss_parts = []
        for k, v in final_losses.items():
            if isinstance(v, dict):
                loss_val = v.get("loss", 0)
                health = v.get("health", "unknown")
                color = C.HEALTH_COLORS.get(health, "#8b949e")
                loss_parts.append(f'<span style="color:{color};">{k}: {loss_val:.3f}</span>')
            else:
                # Backward compat: old format was just a float
                loss_parts.append(f'<span style="color:#8b949e;">{k}: {v:.3f}</span>')
        loss_summary = "  ".join(loss_parts) if loss_parts else "no data"

        # Overall health indicator
        oh_color = C.HEALTH_COLORS.get(overall_health, "#8b949e")
        health_dot = f'<span style="color:{oh_color};">&#9679;</span>'

        items += f"""
        <div class="job-item" onclick="initChart(); loadLossHistory('{job_id}');">
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#f0f6fc;">{health_dot} {job_id[:8]}</span>
                <span class="{state_class}" style="font-size:11px;">{state}</span>
            </div>
            <div style="font-size:11px;">{steps}/{total} steps | {loss_summary}</div>
        </div>"""

    return HTMLResponse(f"""
    <div class="panel">
        <h3>Job History</h3>
        {items if items else '<div class="empty">No jobs yet</div>'}
    </div>
    """)


async def partial_checkpoints(request: Request):
    """Delegates to checkpoint tree view."""
    return await partial_checkpoints_tree(request)


async def partial_datasets(request: Request):
    """Dataset browser with sample thumbnails."""
    datasets = await _api("/datasets")
    if not datasets or (isinstance(datasets, dict) and "error" in datasets):
        return HTMLResponse(
            '<div class="panel"><h3>Datasets</h3><div class="empty">No datasets</div></div>'
        )

    items = ""
    for ds in datasets:
        name = ds["name"]
        size = ds["size"]
        shape = "x".join(str(d) for d in ds["image_shape"])
        target_type = ds.get("target_type", "?")
        items += f"""
        <div style="margin-bottom:10px;border-bottom:1px solid #21262d;padding-bottom:8px;">
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#f0f6fc;font-weight:600;">{name}</span>
                <span style="color:#8b949e;font-size:11px;">{size} images</span>
            </div>
            <div style="color:#8b949e;font-size:11px;">{shape} | target: {target_type}</div>
            <div id="ds-samples-{name}" style="margin-top:4px;"
                hx-get="/partial/dataset_samples/{name}"
                hx-trigger="load"
                hx-swap="innerHTML">
            </div>
        </div>"""

    return HTMLResponse(
        f'<div class="panel"><h3>Datasets</h3>{items if items else "<div class=empty>No datasets</div>"}</div>'
    )


async def partial_dataset_samples(request: Request):
    """Sample thumbnails for a dataset."""
    name = request.path_params["name"]
    samples = await _api(f"/datasets/{name}/sample?n=8")
    if samples and "images" in samples:
        imgs = "".join(
            f'<img src="data:image/png;base64,{b64}" style="width:32px;height:32px;image-rendering:pixelated;border:1px solid #30363d;">'
            for b64 in samples["images"]
        )
        return HTMLResponse(f'<div style="display:flex;gap:2px;flex-wrap:wrap;">{imgs}</div>')
    return HTMLResponse('<span style="color:#484f58;font-size:10px;">No samples</span>')


async def partial_step(request: Request):
    job = await _api("/jobs/current")
    if job and isinstance(job, dict) and "current_step" in job:
        return HTMLResponse(f"[step: {job['current_step']}]")
    jobs = await _api("/jobs")
    if jobs and isinstance(jobs, list) and len(jobs) > 0:
        return HTMLResponse(f"[step: {jobs[0].get('current_step', '-')}]")
    return HTMLResponse("[step: -]")


async def partial_health(request: Request):
    """Connection health indicator with device selector."""
    health = await _api("/health")
    connected = health and isinstance(health, dict) and health.get("status") == "ok"

    if connected:
        device = health.get("device", "?")
        n_tasks = health.get("num_tasks", 0)
        n_datasets = health.get("num_datasets", 0)

        # Fetch available devices for selector
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


# ─── Recipe + Tree + Viz Partials ───


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


async def partial_checkpoints_tree(request: Request):
    """Checkpoint tree visualization."""
    tree = await _api("/checkpoints/tree")
    if not tree or isinstance(tree, dict) and "error" in tree:
        return HTMLResponse(
            '<div class="panel"><h3>Checkpoint Tree</h3><div class="empty">No checkpoints</div></div>'
        )

    nodes = tree.get("nodes", [])
    current_id = tree.get("current_id")

    if not nodes:
        return HTMLResponse(
            '<div class="panel"><h3>Checkpoint Tree</h3><div class="empty">No checkpoints saved</div></div>'
        )

    children = {}
    roots = []
    for n in nodes:
        pid = n.get("parent_id")
        if pid is None:
            roots.append(n)
        else:
            children.setdefault(pid, []).append(n)

    def render_node(node, depth=0):
        nid = node["id"]
        tag = node["tag"]
        short_id = nid[:8]
        is_current = nid == current_id
        metrics = node.get("metrics", {})

        # Determine health indicator from loss_summary in metrics
        loss_summary = metrics.get("loss_summary", {})
        health_dot = ""
        if loss_summary:
            healths = [s.get("health", "unknown") for s in loss_summary.values() if isinstance(s, dict)]
            if "critical" in healths:
                health_dot = f'<span style="color:{C.HEALTH_COLORS["critical"]};">&#9679;</span>'
            elif "warning" in healths:
                health_dot = f'<span style="color:{C.HEALTH_COLORS["warning"]};">&#9679;</span>'
            elif healths:
                health_dot = f'<span style="color:{C.HEALTH_COLORS["healthy"]};">&#9679;</span>'

        indent = '<span class="tree-indent">&#9474;</span>' * depth
        if depth > 0:
            indent = (
                '<span class="tree-indent">&#9474;</span>' * (depth - 1)
                + '<span class="tree-branch">&#9500;&#9472;</span>'
            )

        current_cls = ' tree-current' if is_current else ''
        marker = ' &#9679;' if is_current else ''

        html = f"""
        <div class="tree-node{current_cls}">
            {indent}
            {health_dot}
            <span class="tree-tag">{tag}</span>
            <span class="tree-meta">({short_id}){marker}</span>
            <span class="tree-actions">
                <button class="btn btn-sm"
                    hx-post="/action/load_checkpoint/{nid}"
                    hx-target="#checkpoints-panel"
                    hx-swap="innerHTML">Load</button>
                <button class="btn btn-sm"
                    hx-post="/action/fork_checkpoint/{nid}"
                    hx-target="#checkpoints-panel"
                    hx-swap="innerHTML">Fork</button>
            </span>
        </div>
        """
        for child in children.get(nid, []):
            html += render_node(child, depth + 1)
        return html

    tree_html = ""
    for root in roots:
        tree_html += render_node(root, 0)

    return HTMLResponse(f"""
    <div class="panel">
        <h3>Checkpoint Tree</h3>
        {tree_html}
        <div style="margin-top:8px;">
            <button class="btn" style="width:100%;"
                hx-post="/action/save_checkpoint"
                hx-target="#checkpoints-panel"
                hx-swap="innerHTML">Save Checkpoint</button>
        </div>
    </div>
    """)


async def partial_traversals(request: Request):
    """Latent traversal grids for each factor group."""
    health = await _api("/health")
    if not health or not health.get("has_model"):
        return HTMLResponse(C.no_model_guard("Latent Traversals"))

    data = await _api("/eval/traversals?n_seeds=5&n_steps=9")
    if not data or (isinstance(data, dict) and "error" in data):
        error = data.get("error", "") if isinstance(data, dict) else ""
        return HTMLResponse(f"""
        <div class="panel">
            <h3>Latent Traversals</h3>
            <div class="empty">No traversal data. {error}</div>
            <button class="btn" style="margin-top:8px;"
                hx-get="/partial/traversals"
                hx-target="#traversal-panel"
                hx-swap="innerHTML">Generate Traversals</button>
        </div>
        """)

    groups_html = ""
    for factor_name, rows in data.items():
        n_cols = len(rows[0]) if rows else 0
        grid_items = ""
        for row in rows:
            for img_b64 in row:
                grid_items += f'<img src="data:image/png;base64,{img_b64}">'

        groups_html += f"""
        <div class="traversal-group">
            <h4>{factor_name} (dims -3 to +3)</h4>
            <div class="traversal-grid" style="grid-template-columns: repeat({n_cols}, 36px);">
                {grid_items}
            </div>
        </div>
        """

    return HTMLResponse(f"""
    <div class="panel">
        <h3>Latent Traversals</h3>
        {groups_html}
        <button class="btn" style="margin-top:8px;"
            hx-get="/partial/traversals"
            hx-target="#traversal-panel"
            hx-swap="innerHTML">Refresh Traversals</button>
    </div>
    """)


async def partial_sort_by_factor(request: Request):
    """Sort images by factor activation."""
    health = await _api("/health")
    if not health or not health.get("has_model"):
        return HTMLResponse(C.no_model_guard("Sort by Factor"))

    data = await _api("/eval/sort_by_factor?n_show=16")
    if not data or (isinstance(data, dict) and "error" in data):
        error = data.get("error", "") if isinstance(data, dict) else ""
        return HTMLResponse(f"""
        <div class="panel">
            <h3>Sort by Factor</h3>
            <div class="empty">No sort data. {error}</div>
            <button class="btn" style="margin-top:8px;"
                hx-get="/partial/sort_by_factor"
                hx-target="#sort-panel"
                hx-swap="innerHTML">Generate Sort</button>
        </div>
        """)

    groups_html = ""
    for factor_name, directions in data.items():
        lowest_imgs = "".join(
            f'<img src="data:image/png;base64,{b64}">'
            for b64 in directions.get("lowest", [])
        )
        highest_imgs = "".join(
            f'<img src="data:image/png;base64,{b64}">'
            for b64 in directions.get("highest", [])
        )
        groups_html += f"""
        <div class="sort-group">
            <h4>{factor_name}</h4>
            <div class="sort-label">Lowest activation:</div>
            <div class="sort-row">{lowest_imgs}</div>
            <div class="sort-label">Highest activation:</div>
            <div class="sort-row">{highest_imgs}</div>
        </div>
        """

    return HTMLResponse(f"""
    <div class="panel">
        <h3>Sort by Factor</h3>
        {groups_html}
        <button class="btn" style="margin-top:8px;"
            hx-get="/partial/sort_by_factor"
            hx-target="#sort-panel"
            hx-swap="innerHTML">Refresh Sort</button>
    </div>
    """)


async def partial_attention_maps(request: Request):
    """Per-factor attention heatmaps overlaid on input images."""
    health = await _api("/health")
    if not health or not health.get("has_model"):
        return HTMLResponse(C.no_model_guard("Attention Maps"))

    data = await _api("/eval/attention_maps?n_images=4")
    if not data or (isinstance(data, dict) and "error" in data):
        error = data.get("error", "") if isinstance(data, dict) else ""
        return HTMLResponse(f"""
        <div class="panel">
            <h3>Attention Maps</h3>
            <div class="empty">No attention data. {error}</div>
            <button class="btn" style="margin-top:8px;"
                hx-get="/partial/attention_maps"
                hx-target="#attention-panel"
                hx-swap="innerHTML">Generate Attention Maps</button>
        </div>
        """)

    originals = data.get("originals", [])
    factor_names = [k for k in data if k != "originals"]

    # Build a row per factor: originals on top, then per-factor heatmaps
    groups_html = ""

    # Original images row
    orig_imgs = "".join(
        f'<div class="attn-pair"><img src="data:image/png;base64,{b64}"><div class="attn-label">input</div></div>'
        for b64 in originals
    )
    groups_html += f"""
    <div class="attn-group">
        <h4>Original Images</h4>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">{orig_imgs}</div>
    </div>
    """

    # Per-factor rows
    for factor_name in factor_names:
        overlays = data[factor_name]
        factor_imgs = "".join(
            f'<div class="attn-pair"><img src="data:image/png;base64,{b64}"><div class="attn-label">{factor_name}</div></div>'
            for b64 in overlays
        )
        groups_html += f"""
        <div class="attn-group">
            <h4>{factor_name}</h4>
            <div style="display:flex;gap:8px;flex-wrap:wrap;">{factor_imgs}</div>
        </div>
        """

    return HTMLResponse(f"""
    <div class="panel">
        <h3>Attention Maps</h3>
        {groups_html}
        <button class="btn" style="margin-top:8px;"
            hx-get="/partial/attention_maps"
            hx-target="#attention-panel"
            hx-swap="innerHTML">Refresh Attention Maps</button>
    </div>
    """)


# ─── Action Endpoints ───


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


async def action_set_device(request: Request):
    """Change the active device via the trainer API."""
    form = await request.form()
    device = str(form.get("device", ""))
    if device:
        await _api("/device/set", method="POST", json_data={"device": device})
    return await partial_health(request)


async def action_eval(request: Request):
    """Run eval and return the metrics table."""
    return await partial_eval(request)


async def action_eval_compare(request: Request):
    """Run eval comparison between current model and a checkpoint."""
    return await partial_eval_compare(request)


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
    # HX-Trigger fires checkpoint-changed → all panels with that hx-trigger auto-refresh
    return HTMLResponse(
        '<div class="panel"><h3>Checkpoint Tree</h3><div class="empty">Loading checkpoint...</div></div>',
        headers={"HX-Trigger": E.CHECKPOINT_CHANGED},
    )


async def action_toggle_task(request: Request):
    task_name = request.path_params["name"]
    await _api(f"/tasks/{task_name}/toggle", method="POST")
    resp = await partial_tasks(request)
    resp.headers["HX-Trigger"] = E.TASKS_CHANGED
    return resp


async def action_remove_task(request: Request):
    task_name = request.path_params["name"]
    await _api(f"/tasks/{task_name}/remove", method="POST")
    resp = await partial_tasks(request)
    resp.headers["HX-Trigger"] = E.TASKS_CHANGED
    return resp


async def action_set_weight(request: Request):
    task_name = request.path_params["name"]
    form = await request.form()
    weight = float(form.get("weight", 1.0))
    await _api(f"/tasks/{task_name}/set_weight", method="POST", json_data={"weight": weight})
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

    if result and isinstance(result, dict) and result.get("error"):
        error = result["error"]
        return HTMLResponse(
            f'<div class="panel"><h3>+ Task</h3><div class="error">{error}</div></div>'
        )

    # Success — HX-Trigger fires tasks-changed → tasks panel + add-task panel auto-refresh
    return HTMLResponse(
        '<div class="panel"><h3>+ Task</h3><div style="color:#7ee787;">Task added!</div></div>',
        headers={"HX-Trigger": E.TASKS_CHANGED},
    )


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
    # HX-Trigger fires datasets-changed → datasets panel + add-task panel auto-refresh
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


# ─── SSE Proxy ───


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


# ─── API Proxies for JS ───


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


# ─── App ───

routes = [
    Route("/", index),
    # Partials
    Route("/partial/model", partial_model),
    Route("/partial/tasks", partial_tasks),
    Route("/partial/add_task", partial_add_task),
    Route("/partial/generate", partial_generate),
    Route("/partial/training", partial_training),
    Route("/partial/reconstructions", partial_reconstructions),
    Route("/partial/eval", partial_eval),
    Route("/partial/checkpoints", partial_checkpoints),
    Route("/partial/datasets", partial_datasets),
    Route("/partial/dataset_samples/{name}", partial_dataset_samples),
    Route("/partial/step", partial_step),
    Route("/partial/health", partial_health),
    Route("/partial/checkpoint_indicator", partial_checkpoint_indicator),
    Route("/partial/recipe", partial_recipe),
    Route("/partial/traversals", partial_traversals),
    Route("/partial/sort_by_factor", partial_sort_by_factor),
    Route("/partial/attention_maps", partial_attention_maps),
    Route("/partial/jobs_history", partial_jobs_history),
    # Actions
    Route("/action/train", action_train, methods=["POST"]),
    Route("/action/stop", action_stop, methods=["POST"]),
    Route("/action/eval", action_eval, methods=["POST"]),
    Route("/action/eval_compare", action_eval_compare, methods=["POST"]),
    Route("/action/set_device", action_set_device, methods=["POST"]),
    Route("/action/save_checkpoint", action_save_checkpoint, methods=["POST"]),
    Route("/action/load_checkpoint/{cp_id}", action_load_checkpoint, methods=["POST"]),
    Route("/action/fork_checkpoint/{cp_id}", action_fork_checkpoint, methods=["POST"]),
    Route("/action/toggle_task/{name}", action_toggle_task, methods=["POST"]),
    Route("/action/remove_task/{name}", action_remove_task, methods=["POST"]),
    Route("/action/set_weight/{name}", action_set_weight, methods=["POST"]),
    Route("/action/add_task", action_add_task, methods=["POST"]),
    Route("/action/generate_dataset", action_generate_dataset, methods=["POST"]),
    Route("/action/recipe_run", action_recipe_run, methods=["POST"]),
    Route("/action/recipe_stop", action_recipe_stop, methods=["POST"]),
    # SSE + API proxy
    Route("/sse/job/{job_id}", sse_job),
    Route("/api/jobs/current", api_jobs_current),
    Route("/api/jobs/{job_id}/loss_history", api_jobs_loss_history),
    Route("/api/jobs/{job_id}/loss_summary", api_jobs_loss_summary),
]

app = Starlette(
    routes=[
        Mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static"),
        *routes,
    ]
)
