"""Checkpoint tree and indicator partials."""

from starlette.requests import Request
from starlette.responses import HTMLResponse

from acc.ui.api import call as _api, is_error
from acc.ui import components as C


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
    desc = ""
    for n in nodes:
        if n.get("id") == current_id:
            tag = n.get("tag", "unknown")
            desc = n.get("description") or ""
            break
    short_id = current_id[:8]
    tooltip = f' title="{desc}"' if desc else ""
    return HTMLResponse(
        f'<span style="color:#d2a8ff;background:#1c1430;padding:3px 8px;border-radius:4px;border:1px solid #6e40aa;"{tooltip}>CP: {tag} ({short_id})</span>'
    )


async def partial_checkpoints(request: Request):
    """Delegates to checkpoint tree view."""
    return await partial_checkpoints_tree(request)


def _config_summary(node: dict) -> str:
    """One-line model config summary from checkpoint metadata."""
    config = node.get("model_config", {})
    if not config:
        return ""

    parts = []
    latent = config.get("latent_channels")
    if latent:
        parts.append(f"{latent}ch")

    detach = config.get("detach_factor_grad")
    if detach is True:
        parts.append("stop-grad")
    elif detach is False:
        parts.append("no stop-grad")

    groups = config.get("factor_groups", [])
    if groups:
        group_strs = [f"{g['name']}:{g['dim']}" for g in groups]
        parts.append(" ".join(group_strs))

    return " | ".join(parts)


def _loss_summary_line(node: dict) -> str:
    """Compact loss summary: key metrics with health colors."""
    metrics = node.get("metrics", {})
    loss_summary = metrics.get("loss_summary", {})
    if not loss_summary:
        return ""

    parts = []
    for task_name, s in loss_summary.items():
        if not isinstance(s, dict):
            continue
        health = s.get("health", "unknown")
        color = C.HEALTH_COLORS.get(health, "#8b949e")
        final = s.get("final", 0)
        n_steps = s.get("n_steps", 0)
        parts.append(f'<span style="color:{color};">{task_name}:{final:.4f}</span>')

    if not parts:
        return ""

    # Show total steps from any task
    any_task = next(iter(loss_summary.values()), {})
    n_steps = any_task.get("n_steps", 0) if isinstance(any_task, dict) else 0
    steps_str = f" ({n_steps} steps)" if n_steps else ""

    return f'<div style="font-size:10px;color:#8b949e;margin-top:2px;">{" ".join(parts)}{steps_str}</div>'


def _health_dot(node: dict) -> str:
    """Health indicator dot based on loss summary."""
    metrics = node.get("metrics", {})
    loss_summary = metrics.get("loss_summary", {})
    if not loss_summary:
        return ""
    healths = [s.get("health", "unknown") for s in loss_summary.values() if isinstance(s, dict)]
    if "critical" in healths:
        return f'<span style="color:{C.HEALTH_COLORS["critical"]};">&#9679;</span> '
    elif "warning" in healths:
        return f'<span style="color:{C.HEALTH_COLORS["warning"]};">&#9679;</span> '
    elif healths:
        return f'<span style="color:{C.HEALTH_COLORS["healthy"]};">&#9679;</span> '
    return ""


async def partial_checkpoints_tree(request: Request):
    """Checkpoint tree visualization with rich metadata.

    Shows recipe name, model config, description, and loss summary
    for each checkpoint. Groups branches under their root nodes.
    """
    tree = await _api("/checkpoints/tree")
    if not tree or is_error(tree):
        return HTMLResponse(
            '<div class="panel"><h3>Checkpoint Tree</h3><div class="empty">No checkpoints</div></div>'
        )

    nodes = tree.get("nodes", [])
    current_id = tree.get("current_id")

    if not nodes:
        return HTMLResponse(
            '<div class="panel"><h3>Checkpoint Tree</h3><div class="empty">No checkpoints saved</div></div>'
        )

    # Build parent->children map
    children = {}
    roots = []
    node_map = {}
    for n in nodes:
        node_map[n["id"]] = n
        pid = n.get("parent_id")
        if pid is None:
            roots.append(n)
        else:
            children.setdefault(pid, []).append(n)

    # Group roots by recipe_name (if available)
    recipe_groups = {}
    for root in roots:
        recipe = root.get("recipe_name") or "Manual checkpoints"
        recipe_groups.setdefault(recipe, []).append(root)

    def render_node(node, depth=0):
        nid = node["id"]
        tag = node["tag"]
        short_id = nid[:8]
        is_current = nid == current_id
        desc = node.get("description") or ""

        health = _health_dot(node)
        config_str = _config_summary(node)
        loss_line = _loss_summary_line(node)

        current_cls = ' tree-current' if is_current else ''
        current_marker = ' <span style="color:#d2a8ff;font-size:10px;">&#9668; current</span>' if is_current else ''

        # Indent
        indent = ""
        if depth > 0:
            indent = (
                '<span class="tree-indent">&#9474;</span>' * (depth - 1)
                + '<span class="tree-branch">&#9500;&#9472;</span>'
            )

        # Description line (if present and different from tag)
        desc_html = ""
        if desc and desc != tag:
            desc_html = f'<div style="font-size:10px;color:#8b949e;margin-left:{20 + depth*16}px;">{desc}</div>'

        # Config line (only show on root or when config differs from parent)
        config_html = ""
        if config_str:
            config_html = f'<div style="font-size:10px;color:#6e7681;margin-left:{20 + depth*16}px;">{config_str}</div>'

        html = f"""
        <div class="tree-node{current_cls}" style="margin-bottom:2px;">
            <div style="display:flex;align-items:center;gap:4px;">
                {indent}
                {health}
                <span class="tree-tag">{tag}</span>
                <span class="tree-meta">({short_id}){current_marker}</span>
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
            {desc_html}
            {config_html}
            {loss_line}
        </div>
        """
        for child in children.get(nid, []):
            html += render_node(child, depth + 1)
        return html

    tree_html = ""
    for recipe_name, recipe_roots in recipe_groups.items():
        # Recipe header
        if recipe_name != "Manual checkpoints":
            tree_html += f"""
            <div style="margin-bottom:8px;padding:6px 8px;background:#161b22;border:1px solid #30363d;border-radius:4px;">
                <div style="color:#d2a8ff;font-weight:600;font-size:12px;">&#9881; {recipe_name}</div>
            </div>
            """
        for root in recipe_roots:
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
