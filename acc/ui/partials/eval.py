"""Eval metrics, reconstructions, traversals, sort-by-factor, and attention maps partials."""

from starlette.requests import Request
from starlette.responses import HTMLResponse

from acc.ui.api import call as _api
from acc.ui import components as C


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
    """Eval metrics with optional checkpoint comparison."""
    health = await _api("/health")
    if not health or not health.get("has_model"):
        return HTMLResponse(C.no_model_guard("Eval Metrics"))

    results = await _api("/eval/run", method="POST")
    if not results or "error" in (results if isinstance(results, dict) else {}):
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


async def partial_eval_compare(request: Request):
    """Run eval on current model AND a comparison checkpoint, show side-by-side table."""
    form = await request.form()
    compare_cp_id = str(form.get("eval-compare-cp", ""))

    if not compare_cp_id:
        return await partial_eval(request)

    health = await _api("/health")
    if not health or not health.get("has_model"):
        return HTMLResponse(C.no_model_guard("Eval Metrics"))

    base_results = await _api("/eval/run", method="POST")
    if not base_results or "error" in (base_results if isinstance(base_results, dict) else {}):
        return HTMLResponse(
            '<div class="panel"><h3>Eval Metrics</h3><div class="error">Base eval failed</div></div>'
        )

    compare_data = await _api("/eval/checkpoint", method="POST", json_data={"checkpoint_id": compare_cp_id})
    if not compare_data or "error" in (compare_data if isinstance(compare_data, dict) else {}):
        error_msg = compare_data.get("error", "Unknown error") if isinstance(compare_data, dict) else "Failed"
        return HTMLResponse(
            f'<div class="panel"><h3>Eval Metrics</h3><div class="error">Comparison eval failed: {error_msg}</div></div>'
        )

    compare_tag = compare_data.get("tag", compare_cp_id[:8])
    compare_metrics = compare_data.get("metrics", {})

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

    groups_html = ""

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
