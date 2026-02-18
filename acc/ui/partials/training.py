"""Training panel, step counter, and job history partials."""

from starlette.requests import Request
from starlette.responses import HTMLResponse

from acc.ui.api import call as _api
from acc.ui import components as C


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


async def partial_step(request: Request):
    job = await _api("/jobs/current")
    if job and isinstance(job, dict) and "current_step" in job:
        return HTMLResponse(f"[step: {job['current_step']}]")
    jobs = await _api("/jobs")
    if jobs and isinstance(jobs, list) and len(jobs) > 0:
        return HTMLResponse(f"[step: {jobs[0].get('current_step', '-')}]")
    return HTMLResponse("[step: -]")


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
