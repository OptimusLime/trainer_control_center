"""Checkpoint tree and indicator partials."""

from starlette.requests import Request
from starlette.responses import HTMLResponse

from acc.ui.api import call as _api
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
    for n in nodes:
        if n.get("id") == current_id:
            tag = n.get("tag", "unknown")
            break
    short_id = current_id[:8]
    return HTMLResponse(
        f'<span style="color:#d2a8ff;background:#1c1430;padding:3px 8px;border-radius:4px;border:1px solid #6e40aa;">CP: {tag} ({short_id})</span>'
    )


async def partial_checkpoints(request: Request):
    """Delegates to checkpoint tree view."""
    return await partial_checkpoints_tree(request)


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
