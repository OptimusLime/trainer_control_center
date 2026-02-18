"""UI Dashboard — Starlette + HTMX + SSE on localhost:8081.

Thin router: imports partials/actions from submodules, registers routes,
serves the main page shell. All rendering logic lives in:
  - acc/ui/partials/ — panel rendering
  - acc/ui/actions/  — form/button handlers
  - acc/ui/api.py    — trainer API client
  - acc/ui/components.py — shared HTML builders
  - acc/ui/events.py — event name constants
  - acc/ui/static/   — JS and CSS
"""

from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.requests import Request

# Partials
from acc.ui.partials.model import partial_model, partial_tasks, partial_add_task
from acc.ui.partials.training import partial_training, partial_step, partial_jobs_history
from acc.ui.partials.eval import (
    partial_eval, partial_eval_compare, partial_reconstructions,
    partial_traversals, partial_sort_by_factor, partial_attention_maps,
)
from acc.ui.partials.checkpoints import (
    partial_checkpoints, partial_checkpoints_tree, partial_checkpoint_indicator,
)
from acc.ui.partials.datasets import partial_datasets, partial_dataset_samples, partial_generate
from acc.ui.partials.recipe import partial_recipe
from acc.ui.partials.health import partial_health

# Actions
from acc.ui.actions.training import action_train, action_stop
from acc.ui.actions.checkpoints import (
    action_save_checkpoint, action_load_checkpoint, action_fork_checkpoint,
)
from acc.ui.actions.tasks import (
    action_add_task, action_toggle_task, action_remove_task, action_set_weight,
)
from acc.ui.actions.datasets import action_generate_dataset, action_recipe_run, action_recipe_stop
from acc.ui.actions.eval import action_eval, action_eval_compare, action_set_device
from acc.ui.actions.api_proxies import sse_job, api_jobs_current, api_jobs_loss_history, api_jobs_loss_summary

_STATIC_DIR = Path(__file__).parent / "static"


# ─── Page Shell ───


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


async def index(request: Request):
    return HTMLResponse(_page("ACC", ""))


# ─── Routes ───

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
