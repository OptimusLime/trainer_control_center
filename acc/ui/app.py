"""UI Dashboard — Starlette + HTMX + SSE on localhost:8081.

Stateless: all state comes from the trainer API (localhost:6060).
HTMX for partial page updates. SSE for live training loss streaming.
"""

import json
import os
from typing import Optional

import httpx
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, StreamingResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.requests import Request

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
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'SF Mono', 'Menlo', 'Consolas', monospace; background: #0d1117; color: #c9d1d9; font-size: 13px; }}
        .layout {{ display: grid; grid-template-columns: 280px 1fr; min-height: 100vh; }}
        .sidebar {{ background: #161b22; border-right: 1px solid #30363d; padding: 16px; overflow-y: auto; }}
        .main {{ padding: 16px; overflow-y: auto; }}
        .header {{ background: #161b22; border-bottom: 1px solid #30363d; padding: 12px 16px; display: flex; justify-content: space-between; align-items: center; grid-column: 1 / -1; }}
        .header h1 {{ font-size: 16px; color: #58a6ff; font-weight: 600; }}
        .header .step {{ color: #8b949e; }}
        .panel {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; margin-bottom: 12px; }}
        .panel h3 {{ color: #58a6ff; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }}
        .task-card {{ background: #0d1117; border: 1px solid #30363d; border-radius: 4px; padding: 8px; margin-bottom: 6px; }}
        .task-card .name {{ color: #f0f6fc; font-weight: 600; }}
        .task-card .type {{ color: #8b949e; font-size: 11px; }}
        .task-card .metrics {{ color: #7ee787; font-size: 12px; margin-top: 4px; }}
        .btn {{ background: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-family: inherit; font-size: 12px; }}
        .btn:hover {{ background: #30363d; }}
        .btn-primary {{ background: #238636; border-color: #238636; color: #fff; }}
        .btn-primary:hover {{ background: #2ea043; }}
        .btn-danger {{ background: #da3633; border-color: #da3633; color: #fff; }}
        .btn-danger:hover {{ background: #f85149; }}
        .training-controls {{ display: flex; gap: 8px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }}
        .training-controls input {{ background: #0d1117; border: 1px solid #30363d; color: #c9d1d9; padding: 4px 8px; border-radius: 4px; width: 80px; font-family: inherit; }}
        .training-controls label {{ color: #8b949e; font-size: 11px; }}
        .chart-container {{ position: relative; height: 250px; margin: 8px 0; }}
        .model-desc {{ white-space: pre-wrap; font-size: 11px; color: #8b949e; background: #0d1117; padding: 8px; border-radius: 4px; overflow-x: auto; }}
        .checkpoint {{ padding: 4px 0; font-size: 12px; }}
        .checkpoint .tag {{ color: #f0f6fc; }}
        .checkpoint .meta {{ color: #8b949e; font-size: 11px; }}
        .recon-grid {{ display: flex; gap: 4px; flex-wrap: wrap; }}
        .recon-grid img {{ width: 48px; height: 48px; image-rendering: pixelated; border: 1px solid #30363d; }}
        .recon-row {{ margin-bottom: 4px; }}
        .recon-label {{ color: #8b949e; font-size: 11px; margin-bottom: 2px; }}
        .status-running {{ color: #f0883e; }}
        .status-completed {{ color: #7ee787; }}
        .status-stopped {{ color: #8b949e; }}
        .status-failed {{ color: #f85149; }}
        .job-item {{ padding: 4px 0; border-bottom: 1px solid #21262d; font-size: 12px; }}
        .empty {{ color: #484f58; font-style: italic; padding: 8px 0; }}
        .error {{ color: #f85149; padding: 8px; background: #1c0b0b; border: 1px solid #f85149; border-radius: 4px; margin: 8px 0; }}
        #loss-log {{ max-height: 120px; overflow-y: auto; font-size: 11px; color: #8b949e; background: #0d1117; padding: 4px; border-radius: 4px; margin-top: 8px; }}
        .recipe-select {{ background: #0d1117; border: 1px solid #30363d; color: #c9d1d9; padding: 4px 8px; border-radius: 4px; width: 100%; font-family: inherit; font-size: 12px; margin-bottom: 8px; }}
        .recipe-desc {{ color: #8b949e; font-size: 11px; margin-bottom: 8px; }}
        .phase-indicator {{ background: #0d1117; border: 1px solid #30363d; border-radius: 4px; padding: 8px; margin-top: 8px; }}
        .phase-current {{ color: #f0883e; font-weight: 600; font-size: 12px; }}
        .phase-done {{ color: #7ee787; font-size: 11px; }}
        .phase-item {{ padding: 2px 0; }}
        .tree-node {{ padding: 4px 0; font-size: 12px; display: flex; align-items: center; gap: 6px; }}
        .tree-indent {{ display: inline-block; width: 16px; text-align: center; color: #30363d; }}
        .tree-branch {{ color: #30363d; }}
        .tree-tag {{ color: #f0f6fc; font-weight: 600; }}
        .tree-meta {{ color: #8b949e; font-size: 11px; }}
        .tree-current {{ background: #1c2333; border-radius: 3px; padding: 2px 4px; }}
        .tree-actions {{ display: flex; gap: 4px; }}
        .traversal-group {{ margin-bottom: 12px; }}
        .traversal-group h4 {{ color: #d2a8ff; font-size: 12px; margin-bottom: 4px; }}
        .traversal-grid {{ display: grid; gap: 2px; }}
        .traversal-grid img {{ width: 36px; height: 36px; image-rendering: pixelated; border: 1px solid #21262d; }}
        .sort-group {{ margin-bottom: 12px; }}
        .sort-group h4 {{ color: #d2a8ff; font-size: 12px; margin-bottom: 4px; }}
        .sort-row {{ display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 4px; }}
        .sort-label {{ color: #8b949e; font-size: 11px; margin-bottom: 2px; }}
        .sort-row img {{ width: 32px; height: 32px; image-rendering: pixelated; border: 1px solid #21262d; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ACC -- Autoencoder Control Center</h1>
        <div style="display:flex;align-items:center;gap:12px;">
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
    <script>
        {_chart_js()}
    </script>
</body>
</html>"""


def _sidebar_placeholder() -> str:
    return """
        <div id="recipe-panel" hx-get="/partial/recipe" hx-trigger="load, every 3s">
            <div class="panel"><h3>Recipes</h3><div class="empty">Loading...</div></div>
        </div>
        <div id="model-panel" hx-get="/partial/model" hx-trigger="load, every 5s">
            <div class="panel"><h3>Model</h3><div class="empty">Loading...</div></div>
        </div>
        <div id="tasks-panel" hx-get="/partial/tasks" hx-trigger="load, every 3s">
            <div class="panel"><h3>Tasks</h3><div class="empty">Loading...</div></div>
        </div>
        <div id="checkpoints-panel" hx-get="/partial/checkpoints" hx-trigger="load, every 5s">
            <div class="panel"><h3>Checkpoint Tree</h3><div class="empty">Loading...</div></div>
        </div>
    """


def _main_placeholder() -> str:
    return """
        <div id="training-panel" hx-get="/partial/training" hx-trigger="load">
            <div class="panel"><h3>Training</h3><div class="empty">Loading...</div></div>
        </div>
        <div id="recon-panel" hx-get="/partial/reconstructions" hx-trigger="load, every 5s">
            <div class="panel"><h3>Reconstructions</h3><div class="empty">Loading...</div></div>
        </div>
        <div id="eval-panel" hx-get="/partial/eval" hx-trigger="load, every 5s">
            <div class="panel"><h3>Eval Metrics</h3><div class="empty">Loading...</div></div>
        </div>
        <div id="traversal-panel" hx-get="/partial/traversals" hx-trigger="load">
            <div class="panel"><h3>Latent Traversals</h3><div class="empty">Run eval to generate</div></div>
        </div>
        <div id="sort-panel" hx-get="/partial/sort_by_factor" hx-trigger="load">
            <div class="panel"><h3>Sort by Factor</h3><div class="empty">Run eval to generate</div></div>
        </div>
        <div id="datasets-panel" hx-get="/partial/datasets" hx-trigger="load, every 10s">
            <div class="panel"><h3>Dataset Browser</h3><div class="empty">Loading...</div></div>
        </div>
    """


def _chart_js() -> str:
    return """
    let lossChart = null;
    let lossData = {};
    let eventSource = null;
    let lastStep = 0;

    function initChart() {
        const ctx = document.getElementById('loss-chart');
        if (!ctx) return;
        if (lossChart) { lossChart.destroy(); }
        lossChart = new Chart(ctx, {
            type: 'line',
            data: { datasets: [] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: { type: 'linear', title: { display: true, text: 'Step', color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
                    y: { title: { display: true, text: 'Loss', color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: '#21262d' } }
                },
                plugins: { legend: { labels: { color: '#c9d1d9' } } }
            }
        });
    }

    function addLossPoint(step, taskName, loss) {
        if (!lossData[taskName]) {
            const colors = ['#58a6ff', '#7ee787', '#f0883e', '#f778ba', '#d2a8ff'];
            const idx = Object.keys(lossData).length;
            lossData[taskName] = { label: taskName, data: [], borderColor: colors[idx % colors.length], borderWidth: 1.5, pointRadius: 0, fill: false };
        }
        lossData[taskName].data.push({ x: step, y: loss });

        if (lossChart) {
            lossChart.data.datasets = Object.values(lossData);
            lossChart.update('none');
        }

        // Update step counter
        const counter = document.getElementById('step-counter');
        if (counter) counter.textContent = '[step: ' + step + ']';

        // Update loss log
        const log = document.getElementById('loss-log');
        if (log) {
            log.innerHTML += '<div>step ' + step + ' | ' + taskName + ': ' + loss.toFixed(4) + '</div>';
            log.scrollTop = log.scrollHeight;
        }
    }

    function startSSE(jobId) {
        if (eventSource) { eventSource.close(); }
        lossData = {};
        if (lossChart) { lossChart.data.datasets = []; lossChart.update(); }

        eventSource = new EventSource('/sse/job/' + jobId + '?from_step=' + lastStep);
        eventSource.onmessage = function(e) {
            const data = JSON.parse(e.data);
            if (data.done) {
                eventSource.close();
                htmx.trigger('#training-panel', 'refresh');
                htmx.trigger('#eval-panel', 'refresh');
                htmx.trigger('#recon-panel', 'refresh');
                htmx.trigger('#tasks-panel', 'refresh');
                return;
            }
            addLossPoint(data.step, data.task_name, data.task_loss);
            lastStep = data.step;
        };
    }

    // Auto-connect to running job on page load
    document.addEventListener('DOMContentLoaded', function() {
        fetch('/api/jobs/current').then(r => r.json()).then(data => {
            if (data && data.id && data.state === 'running') {
                setTimeout(function() { initChart(); startSSE(data.id); }, 500);
            }
        }).catch(() => {});
    });
    """


# ─── HTMX Partial Endpoints ───


async def index(request: Request):
    return HTMLResponse(_page("ACC", ""))


async def partial_model(request: Request):
    data = await _api("/model/describe")
    if data is None or "error" in data:
        return HTMLResponse(
            '<div class="panel"><h3>Model</h3><div class="empty">No model loaded</div></div>'
        )

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
    tasks = await _api("/tasks")
    if not tasks or "error" in (tasks if isinstance(tasks, dict) else {}):
        return HTMLResponse(
            '<div class="panel"><h3>Tasks</h3><div class="empty">No tasks</div></div>'
        )

    cards = ""
    for t in tasks:
        enabled = "on" if t.get("enabled") else "off"
        check = "checked" if t.get("enabled") else ""
        cards += f"""
        <div class="task-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span class="name">{t["name"]}</span>
                <input type="checkbox" {check}
                    hx-post="/action/toggle_task/{t["name"]}"
                    hx-target="#tasks-panel"
                    hx-swap="innerHTML">
            </div>
            <div class="type">{t["type"]} | w={t.get("weight", 1.0)} | {enabled}</div>
        </div>"""

    return HTMLResponse(f"""
    <div class="panel">
        <h3>Tasks</h3>
        {cards if cards else '<div class="empty">No tasks attached</div>'}
    </div>
    """)


async def partial_training(request: Request):
    job = await _api("/jobs/current")
    running = job and isinstance(job, dict) and job.get("state") == "running"

    return HTMLResponse(f"""
    <div class="panel">
        <h3>Training</h3>
        <div class="training-controls">
            <label>Steps:</label>
            <input type="number" id="train-steps" value="500" min="1">
            <label>LR:</label>
            <input type="text" id="train-lr" value="1e-3">
            <button class="btn btn-primary" id="train-btn"
                hx-post="/action/train"
                hx-include="#train-steps, #train-lr"
                hx-target="#training-panel"
                hx-swap="innerHTML"
                {"disabled" if running else ""}>Train</button>
            <button class="btn btn-danger"
                hx-post="/action/stop"
                hx-target="#training-panel"
                hx-swap="innerHTML"
                {"" if running else "disabled"}>Stop</button>
            <button class="btn"
                hx-post="/action/eval"
                hx-target="#eval-panel"
                hx-swap="innerHTML">Eval</button>
            <button class="btn"
                hx-post="/action/save_checkpoint"
                hx-target="#checkpoints-panel"
                hx-swap="innerHTML">Save Checkpoint</button>
        </div>
        <div class="chart-container">
            <canvas id="loss-chart"></canvas>
        </div>
        <div id="loss-log"></div>
    </div>
    <script>initChart();</script>
    """)


async def partial_reconstructions(request: Request):
    health = await _api("/health")
    if not health or not health.get("has_model"):
        return HTMLResponse(
            '<div class="panel"><h3>Reconstructions</h3><div class="empty">No model</div></div>'
        )

    # Get sample images from first dataset
    datasets = await _api("/datasets")
    if not datasets or (isinstance(datasets, dict) and "error" in datasets):
        return HTMLResponse(
            '<div class="panel"><h3>Reconstructions</h3><div class="empty">No datasets</div></div>'
        )

    if isinstance(datasets, list) and len(datasets) > 0:
        ds_name = datasets[0]["name"]
        samples = await _api(f"/datasets/{ds_name}/sample?n=8")
        if samples and "images" in samples:
            imgs = "".join(
                f'<img src="data:image/png;base64,{b64}">' for b64 in samples["images"]
            )
            return HTMLResponse(f"""
            <div class="panel">
                <h3>Reconstructions</h3>
                <div class="recon-row">
                    <div class="recon-label">Samples from {ds_name}:</div>
                    <div class="recon-grid">{imgs}</div>
                </div>
            </div>
            """)

    return HTMLResponse(
        '<div class="panel"><h3>Reconstructions</h3><div class="empty">No data</div></div>'
    )


async def partial_eval(request: Request):
    health = await _api("/health")
    if not health or not health.get("has_model"):
        return HTMLResponse(
            '<div class="panel"><h3>Eval Metrics</h3><div class="empty">No model</div></div>'
        )

    # Try to get eval results
    results = await _api("/eval/run", method="POST")
    if not results or "error" in (results if isinstance(results, dict) else {}):
        return HTMLResponse(
            '<div class="panel"><h3>Eval Metrics</h3><div class="empty">Run eval to see metrics</div></div>'
        )

    rows = ""
    for task_name, metrics in results.items():
        metric_str = "  ".join(f"{k} = {v:.4f}" for k, v in metrics.items())
        rows += f'<div style="margin-bottom:4px;"><span style="color:#f0f6fc;">{task_name}:</span> <span class="metrics">{metric_str}</span></div>'

    return HTMLResponse(f"""
    <div class="panel">
        <h3>Eval Metrics</h3>
        {rows if rows else '<div class="empty">No results</div>'}
    </div>
    """)


async def partial_checkpoints(request: Request):
    """Delegates to checkpoint tree view."""
    return await partial_checkpoints_tree(request)


async def partial_datasets(request: Request):
    datasets = await _api("/datasets")
    if not datasets or (isinstance(datasets, dict) and "error" in datasets):
        return HTMLResponse(
            '<div class="panel"><h3>Dataset Browser</h3><div class="empty">No datasets</div></div>'
        )

    items = ""
    for ds in datasets:
        items += f"""
        <div style="margin-bottom:8px;">
            <span style="color:#f0f6fc;font-weight:600;">[{ds["name"]}]</span>
            <span style="color:#8b949e;">{ds["size"]} images, {"x".join(str(d) for d in ds["image_shape"])}</span>
        </div>"""

    return HTMLResponse(
        f'<div class="panel"><h3>Dataset Browser</h3>{items if items else "<div class=empty>No datasets</div>"}</div>'
    )


async def partial_step(request: Request):
    job = await _api("/jobs/current")
    if job and isinstance(job, dict) and "current_step" in job:
        return HTMLResponse(f"[step: {job['current_step']}]")
    # Check last completed job
    jobs = await _api("/jobs")
    if jobs and isinstance(jobs, list) and len(jobs) > 0:
        return HTMLResponse(f"[step: {jobs[0].get('current_step', '-')}]")
    return HTMLResponse("[step: -]")


async def partial_health(request: Request):
    """Connection health indicator — pings trainer /health endpoint."""
    health = await _api("/health")
    connected = health and isinstance(health, dict) and health.get("status") == "ok"

    if connected:
        device = health.get("device", "?")
        return HTMLResponse(
            f'<span style="color:#7ee787;font-size:11px;">'
            f"&#9679; {TRAINER_URL} ({device})</span>"
        )
    else:
        return HTMLResponse(
            f'<span style="color:#f85149;font-size:11px;">'
            f"&#9679; Disconnected — {TRAINER_URL}</span>"
        )


# ─── Recipe + Tree + Viz Partials ───


async def partial_recipe(request: Request):
    """Recipe picker, run/stop button, and phase progress."""
    recipes = await _api("/recipes")
    current = await _api("/recipes/current")

    # Is a recipe running?
    running = (
        current
        and isinstance(current, dict)
        and current.get("state") == "running"
    )

    # Recipe options
    if not recipes or isinstance(recipes, dict):
        options = '<option value="">No recipes available</option>'
    else:
        options = '<option value="">-- select recipe --</option>'
        for r in recipes:
            options += f'<option value="{r["name"]}">{r["name"]}</option>'

    # Phase progress display
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

    # Run/stop buttons
    if running:
        buttons = """
        <button class="btn btn-danger" style="width:100%;"
            hx-post="/action/recipe_stop"
            hx-target="#recipe-panel"
            hx-swap="innerHTML">Stop Recipe</button>
        """
    else:
        buttons = f"""
        <select id="recipe-select" class="recipe-select">{options}</select>
        <button class="btn btn-primary" style="width:100%;"
            hx-post="/action/recipe_run"
            hx-include="#recipe-select"
            hx-target="#recipe-panel"
            hx-swap="innerHTML">Run Recipe</button>
        """

    return HTMLResponse(f"""
    <div class="panel">
        <h3>Recipes</h3>
        {buttons}
        {progress_html}
    </div>
    """)


async def partial_checkpoints_tree(request: Request):
    """Checkpoint tree visualization — replaces flat list."""
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

    # Build tree structure: group children by parent_id
    children = {}  # parent_id -> [node, ...]
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
            <span class="tree-tag">{tag}</span>
            <span class="tree-meta">({short_id}){marker}</span>
            <span class="tree-actions">
                <button class="btn" style="padding:1px 4px;font-size:10px;"
                    hx-post="/action/load_checkpoint/{nid}"
                    hx-target="#checkpoints-panel"
                    hx-swap="innerHTML">Load</button>
                <button class="btn" style="padding:1px 4px;font-size:10px;"
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
        return HTMLResponse(
            '<div class="panel"><h3>Latent Traversals</h3><div class="empty">No model loaded</div></div>'
        )

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
        # Each row is a list of base64 PNGs
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
    """Sort images by factor activation — lowest and highest."""
    health = await _api("/health")
    if not health or not health.get("has_model"):
        return HTMLResponse(
            '<div class="panel"><h3>Sort by Factor</h3><div class="empty">No model loaded</div></div>'
        )

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
        # Return training panel with SSE connection script
        return HTMLResponse(f"""
        <div class="panel">
            <h3>Training</h3>
            <div class="training-controls">
                <label>Steps:</label>
                <input type="number" id="train-steps" value="{steps}" min="1">
                <label>LR:</label>
                <input type="text" id="train-lr" value="{lr}">
                <button class="btn btn-primary" disabled>Training...</button>
                <button class="btn btn-danger"
                    hx-post="/action/stop"
                    hx-target="#training-panel"
                    hx-swap="innerHTML">Stop</button>
                <button class="btn"
                    hx-post="/action/save_checkpoint"
                    hx-target="#checkpoints-panel"
                    hx-swap="innerHTML">Save Checkpoint</button>
            </div>
            <div class="chart-container">
                <canvas id="loss-chart"></canvas>
            </div>
            <div id="loss-log"></div>
        </div>
        <script>initChart(); startSSE('{result["id"]}');</script>
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


async def action_eval(request: Request):
    results = await _api("/eval/run", method="POST")
    if not results or "error" in (results if isinstance(results, dict) else {}):
        error = results.get("error", "Eval failed") if results else "Failed to connect"
        return HTMLResponse(
            f'<div class="panel"><h3>Eval Metrics</h3><div class="error">{error}</div></div>'
        )

    rows = ""
    for task_name, metrics in results.items():
        metric_str = "  ".join(f"{k} = {v:.4f}" for k, v in metrics.items())
        rows += f'<div style="margin-bottom:4px;"><span style="color:#f0f6fc;">{task_name}:</span> <span class="metrics">{metric_str}</span></div>'

    return HTMLResponse(f'<div class="panel"><h3>Eval Metrics</h3>{rows}</div>')


async def action_save_checkpoint(request: Request):
    result = await _api("/checkpoints/save", method="POST", json_data={"tag": "checkpoint"})
    # Re-render checkpoints panel
    return HTMLResponse(
        '<div hx-get="/partial/checkpoints" hx-trigger="load" hx-swap="innerHTML"></div>'
    )


async def action_load_checkpoint(request: Request):
    cp_id = request.path_params["cp_id"]
    result = await _api("/checkpoints/load", method="POST", json_data={"id": cp_id})
    return HTMLResponse(
        '<div hx-get="/partial/checkpoints" hx-trigger="load" hx-swap="innerHTML"></div>'
    )


async def action_toggle_task(request: Request):
    task_name = request.path_params["name"]
    await _api(f"/tasks/{task_name}/toggle", method="POST")
    # Re-render tasks panel
    resp = await partial_tasks(request)
    return resp


async def action_recipe_run(request: Request):
    """Start a recipe by name."""
    form = await request.form()
    recipe_name = form.get("recipe-select", "")
    if not recipe_name:
        return HTMLResponse(
            '<div class="panel"><h3>Recipes</h3><div class="error">No recipe selected</div></div>'
        )
    result = await _api(f"/recipes/{recipe_name}/run", method="POST")
    if result and isinstance(result, dict) and "error" in result:
        return HTMLResponse(
            f'<div class="panel"><h3>Recipes</h3><div class="error">{result["error"]}</div></div>'
        )
    # Return recipe panel that auto-refreshes to show progress
    return HTMLResponse(
        '<div hx-get="/partial/recipe" hx-trigger="load" hx-swap="innerHTML"></div>'
    )


async def action_recipe_stop(request: Request):
    """Stop the running recipe."""
    await _api("/recipes/stop", method="POST")
    return HTMLResponse(
        '<div hx-get="/partial/recipe" hx-trigger="load" hx-swap="innerHTML"></div>'
    )


async def action_fork_checkpoint(request: Request):
    """Fork a checkpoint, creating a new branch in the tree."""
    cp_id = request.path_params["cp_id"]
    result = await _api(
        "/checkpoints/fork", method="POST",
        json_data={"id": cp_id, "new_tag": f"fork_{cp_id[:6]}"}
    )
    return HTMLResponse(
        '<div hx-get="/partial/checkpoints" hx-trigger="load" hx-swap="innerHTML"></div>'
    )


# ─── SSE Proxy ───


async def sse_job(request: Request):
    """Proxy SSE stream from trainer to the browser."""
    job_id = request.path_params["job_id"]
    from_step = int(request.query_params.get("from_step", 0))

    async def event_generator():
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "GET",
                    f"{TRAINER_URL}/jobs/{job_id}/stream?from_step={from_step}",
                    timeout=None,
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            yield f"{line}\n\n"
        except Exception:
            yield 'data: {"done": true}\n\n'

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ─── API Proxy for JS ───


async def api_jobs_current(request: Request):
    """Proxy for JS to check current job."""
    job = await _api("/jobs/current")
    return HTMLResponse(
        json.dumps(job),
        media_type="application/json",
    )


# ─── App ───

routes = [
    Route("/", index),
    # Partials
    Route("/partial/model", partial_model),
    Route("/partial/tasks", partial_tasks),
    Route("/partial/training", partial_training),
    Route("/partial/reconstructions", partial_reconstructions),
    Route("/partial/eval", partial_eval),
    Route("/partial/checkpoints", partial_checkpoints),
    Route("/partial/datasets", partial_datasets),
    Route("/partial/step", partial_step),
    Route("/partial/health", partial_health),
    Route("/partial/recipe", partial_recipe),
    Route("/partial/traversals", partial_traversals),
    Route("/partial/sort_by_factor", partial_sort_by_factor),
    # Actions
    Route("/action/train", action_train, methods=["POST"]),
    Route("/action/stop", action_stop, methods=["POST"]),
    Route("/action/eval", action_eval, methods=["POST"]),
    Route("/action/save_checkpoint", action_save_checkpoint, methods=["POST"]),
    Route("/action/load_checkpoint/{cp_id}", action_load_checkpoint, methods=["POST"]),
    Route("/action/fork_checkpoint/{cp_id}", action_fork_checkpoint, methods=["POST"]),
    Route("/action/toggle_task/{name}", action_toggle_task, methods=["POST"]),
    Route("/action/recipe_run", action_recipe_run, methods=["POST"]),
    Route("/action/recipe_stop", action_recipe_stop, methods=["POST"]),
    # SSE + API proxy
    Route("/sse/job/{job_id}", sse_job),
    Route("/api/jobs/current", api_jobs_current),
]

app = Starlette(routes=routes)
