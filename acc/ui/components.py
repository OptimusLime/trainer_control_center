"""Shared HTML component builders for the ACC dashboard.

Every piece of repeated HTML — panels, guards, tables, badges — lives here
as a function. Partials call these instead of duplicating rendering logic.

This is the single source of truth for:
- HEALTH_COLORS dict (was duplicated 4 times)
- No-model guard HTML (was copy-pasted 6 times)
- Error guard pattern (was copy-pasted 6 times)
- Panel wrapper HTML (was repeated ~18 times)
- Loss summary table rendering (was duplicated in Python and JS)
- Eval metric color coding (used by eval table)
- Training panel skeleton (was duplicated twice)
"""

from acc.eval_metric import EvalMetric


# ─── Constants ───

HEALTH_COLORS: dict[str, str] = {
    "healthy": "#7ee787",
    "warning": "#f0883e",
    "critical": "#f85149",
}

CHART_COLORS: list[str] = [
    "#58a6ff", "#7ee787", "#f0883e", "#f778ba",
    "#d2a8ff", "#ff7b72", "#79c0ff", "#a5d6ff",
]

# Thresholds for eval metric color coding: (good_threshold, mid_threshold)
# For higher-is-better: good if value > good_thresh, mid if > mid_thresh
# For lower-is-better: good if value < good_thresh, mid if < mid_thresh
_METRIC_THRESHOLDS: dict[EvalMetric, tuple[float, float]] = {
    EvalMetric.ACCURACY: (0.9, 0.5),
    EvalMetric.PSNR: (25.0, 15.0),
    EvalMetric.L1: (0.05, 0.2),
    EvalMetric.MAE: (0.05, 0.2),
    EvalMetric.MSE: (0.05, 0.2),
    EvalMetric.KL: (5.0, 20.0),
}


# ─── Component Builders ───


def panel(title: str, body: str, **attrs: str) -> str:
    """Wrap content in a standard panel div with title.

    Extra HTML attributes can be passed as keyword args (e.g. id="foo").
    """
    attr_str = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    if attr_str:
        attr_str = " " + attr_str
    return f'<div class="panel"{attr_str}><h3>{title}</h3>{body}</div>'


def empty(text: str) -> str:
    """Render placeholder text in the empty style."""
    return f'<div class="empty">{text}</div>'


def error_div(text: str) -> str:
    """Render an error message."""
    return f'<div class="error">{text}</div>'


def no_model_guard(title: str) -> str:
    """Return a panel indicating no model is loaded.

    Replaces the 6 copy-pasted guard blocks throughout partials.
    """
    return panel(title, empty("No model loaded"))


def error_guard(title: str, error_msg: str) -> str:
    """Return a panel showing an error message."""
    return panel(title, error_div(error_msg))


# ─── Eval Metric Helpers ───


def resolve_metric(metric_name: str) -> EvalMetric | None:
    """Resolve a metric name string to an EvalMetric enum, or None if unknown."""
    try:
        return EvalMetric(metric_name)
    except ValueError:
        return None


def metric_color(metric_name: str, value: float) -> str:
    """Return CSS class for metric value color coding."""
    metric = resolve_metric(metric_name)
    if metric is None:
        return ""

    thresholds = _METRIC_THRESHOLDS.get(metric)
    if thresholds is None:
        return ""

    good_thresh, mid_thresh = thresholds

    if metric.higher_is_better:
        if value > good_thresh:
            return "metric-good"
        elif value > mid_thresh:
            return "metric-mid"
        return "metric-bad"
    else:
        if value < good_thresh:
            return "metric-good"
        elif value < mid_thresh:
            return "metric-mid"
        return "metric-bad"


def metric_higher_is_better(metric_name: str) -> bool:
    """Determine if higher values are better for this metric."""
    metric = resolve_metric(metric_name)
    if metric is None:
        return False
    return metric.higher_is_better


# ─── Loss Summary Table ───


def loss_summary_table(summary_data: dict | None) -> str:
    """Render the per-task loss summary table from summary API data.

    This replaces the duplicated table rendering in both Python (partial_training)
    and JavaScript (loadLossSummary). The Python version is the server-rendered
    source of truth; the JS version will call this endpoint for live updates.

    Args:
        summary_data: dict of {task_name: {final, mean, min, max, trend, health}}

    Returns:
        HTML string of the summary table.
    """
    if not summary_data:
        return empty("No training data yet")

    rows = ""
    for task_name, s in summary_data.items():
        if not isinstance(s, dict):
            continue
        color = HEALTH_COLORS.get(s.get("health", ""), "#8b949e")
        trend = s.get("trend", "flat")
        trend_icon = "&#9660;" if trend == "improving" else "&#9650;" if trend == "worsening" else "&#9644;"
        trend_color = "#7ee787" if trend == "improving" else "#f85149" if trend == "worsening" else "#8b949e"
        h = s.get("health", "unknown")
        rows += f'''<tr>
            <td style="color:#f0f6fc;">{task_name}</td>
            <td style="color:{color};font-weight:700;">{s.get("final", 0):.4f}</td>
            <td>{s.get("mean", 0):.4f}</td>
            <td>{s.get("min", 0):.4f}</td>
            <td>{s.get("max", 0):.4f}</td>
            <td style="color:{trend_color};">{trend_icon} {trend}</td>
            <td style="color:{color};font-weight:700;">{h.upper()}</td>
        </tr>'''

    if not rows:
        return empty("No training data yet")

    return f'''<table class="eval-table">
        <thead><tr><th>Task</th><th>Final</th><th>Mean</th><th>Min</th><th>Max</th><th>Trend</th><th>Health</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>'''


def health_banner(summary_data: dict | None) -> str:
    """Render the health banner from loss summary data.

    Returns a visible banner (colored background) if any task is warning/critical,
    or a hidden placeholder banner div for JS to populate during live training.

    Args:
        summary_data: dict of {task_name: {final, health, ...}} or None/empty

    Returns:
        HTML string of the health banner div.
    """
    hidden_banner = '<div id="health-banner" style="display:none;padding:6px 10px;border-radius:4px;border:1px solid #30363d;margin-bottom:8px;font-size:12px;font-weight:600;"></div>'

    if not summary_data:
        return hidden_banner

    worst_health = "healthy"
    banner_parts = []
    for task_name, s in summary_data.items():
        if not isinstance(s, dict):
            continue
        color = HEALTH_COLORS.get(s.get("health", ""), "#8b949e")
        h = s.get("health", "unknown")
        if h == "critical":
            worst_health = "critical"
        elif h == "warning" and worst_health != "critical":
            worst_health = "warning"
        banner_parts.append(
            f'<span style="color:{color};">{task_name}: {s.get("final", 0):.4f}</span>'
        )

    if worst_health == "healthy":
        return hidden_banner

    bg = "#3d1114" if worst_health == "critical" else "#3d2e14"
    border = HEALTH_COLORS.get(worst_health, "#30363d")
    banner_content = " &nbsp;|&nbsp; ".join(banner_parts)
    return f'<div id="health-banner" style="padding:6px 10px;border-radius:4px;border:1px solid {border};margin-bottom:8px;font-size:12px;font-weight:600;background:{bg};">{banner_content}</div>'


def training_panel_skeleton() -> str:
    """Render the training panel structure with chart canvas and log div.

    This is the shared skeleton used by both partial_training (initial load)
    and action_train (when training starts). Caller wraps with panel() and
    adds appropriate <script> for chart init/SSE.
    """
    return '''<div id="health-banner" style="display:none;padding:6px 10px;border-radius:4px;border:1px solid #30363d;margin-bottom:8px;font-size:12px;font-weight:600;"></div>
        <div class="chart-container">
            <canvas id="loss-chart"></canvas>
        </div>
        <div id="loss-log"></div>'''
