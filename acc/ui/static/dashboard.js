/* ACC Dashboard JavaScript — extracted from app.py inline _chart_js().
   Single source of truth for chart, SSE, loss rendering, and health banner logic.

   HEALTH_COLORS is the JS-side canonical source. The Python-side canonical
   source is acc.ui.components.HEALTH_COLORS. Both must match. */

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
            plugins: { legend: { labels: { color: '#c9d1d9', font: { size: 11 } } } }
        }
    });
}

const CHART_COLORS = ['#58a6ff', '#7ee787', '#f0883e', '#f778ba', '#d2a8ff', '#ff7b72', '#79c0ff', '#a5d6ff'];

const HEALTH_COLORS = { healthy: '#7ee787', warning: '#f0883e', critical: '#f85149' };

// --- Buffered loss rendering ---
// Incoming SSE points are pushed to a buffer. A 200ms timer flushes
// the buffer: updates chart once, appends log entries in one DOM write,
// and updates the health banner. This prevents the browser from
// drowning in per-step reflows during fast training.
let lossBuf = [];
let flushTimer = null;
const FLUSH_INTERVAL = 200; // ms

function addLossPoint(step, taskName, loss, health) {
    if (!lossData[taskName]) {
        const idx = Object.keys(lossData).length;
        lossData[taskName] = {
            label: taskName,
            data: [],
            borderColor: CHART_COLORS[idx % CHART_COLORS.length],
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false
        };
    }
    lossData[taskName].data.push({ x: step, y: loss });
    lossBuf.push({ step, taskName, loss, health });

    if (!flushTimer) {
        flushTimer = setTimeout(flushLossBuf, FLUSH_INTERVAL);
    }
}

function flushLossBuf() {
    flushTimer = null;
    if (lossBuf.length === 0) return;

    // Chart: single update
    if (lossChart) {
        lossChart.data.datasets = Object.values(lossData);
        lossChart.update('none');
    }

    // Step counter: latest step
    const lastEntry = lossBuf[lossBuf.length - 1];
    const counter = document.getElementById('step-counter');
    if (counter) counter.textContent = '[step: ' + lastEntry.step + ']';

    // Loss log: batch append (only last 20 entries to avoid DOM bloat)
    const log = document.getElementById('loss-log');
    if (log) {
        const tail = lossBuf.slice(-20);
        let html = '';
        for (const e of tail) {
            const color = HEALTH_COLORS[e.health] || '#8b949e';
            html += '<div style="color:' + color + ';">step ' + e.step + ' | ' + e.taskName + ': ' + e.loss.toFixed(4) + (e.health === 'critical' ? ' !!!' : e.health === 'warning' ? ' !' : '') + '</div>';
        }
        log.innerHTML += html;
        // Cap total log entries to prevent memory bloat
        while (log.childElementCount > 500) { log.removeChild(log.firstChild); }
        log.scrollTop = log.scrollHeight;
    }

    // Health banner: only update with the latest per-task values
    for (const e of lossBuf) {
        if (e.health) {
            updateHealthBanner(e.taskName, e.loss, e.health);
        }
    }

    lossBuf = [];
}

// Track worst health across all tasks for the banner
let taskHealthState = {};
function updateHealthBanner(taskName, loss, health) {
    taskHealthState[taskName] = { loss: loss, health: health };
    const banner = document.getElementById('health-banner');
    if (!banner) return;
    let worst = 'healthy';
    let parts = [];
    for (const [tn, st] of Object.entries(taskHealthState)) {
        if (st.health === 'critical') worst = 'critical';
        else if (st.health === 'warning' && worst !== 'critical') worst = 'warning';
        const c = HEALTH_COLORS[st.health] || '#8b949e';
        parts.push('<span style="color:' + c + ';">' + tn + ': ' + st.loss.toFixed(4) + '</span>');
    }
    const bgColor = worst === 'critical' ? '#3d1114' : worst === 'warning' ? '#3d2e14' : '#14261a';
    const borderColor = HEALTH_COLORS[worst];
    banner.style.background = bgColor;
    banner.style.borderColor = borderColor;
    banner.style.display = 'block';
    banner.innerHTML = parts.join(' &nbsp;|&nbsp; ');
}

function loadLossHistory(jobId) {
    // Fetch full loss history for a job and populate chart
    fetch('/api/jobs/' + jobId + '/loss_history').then(r => r.json()).then(data => {
        if (!Array.isArray(data)) return;
        lossData = {};
        taskHealthState = {};
        data.forEach(function(entry) {
            addLossPoint(entry.step, entry.task_name, entry.task_loss, entry.health || null);
        });
        lastStep = data.length > 0 ? data[data.length - 1].step : 0;
        // Also load and display the loss summary
        loadLossSummary(jobId);
    }).catch(() => {});
}

function loadLossSummary(jobId) {
    fetch('/api/jobs/' + jobId + '/loss_summary').then(r => r.json()).then(data => {
        const panel = document.getElementById('loss-summary-content');
        if (!panel || !data || data.error) return;
        let html = '<table class="eval-table"><thead><tr><th>Task</th><th>Final</th><th>Mean</th><th>Min</th><th>Max</th><th>Trend</th><th>Health</th></tr></thead><tbody>';
        for (const [taskName, s] of Object.entries(data)) {
            const c = HEALTH_COLORS[s.health] || '#8b949e';
            const trendIcon = s.trend === 'improving' ? '&#9660;' : s.trend === 'worsening' ? '&#9650;' : '&#9644;';
            const trendColor = s.trend === 'improving' ? '#7ee787' : s.trend === 'worsening' ? '#f85149' : '#8b949e';
            html += '<tr><td style="color:#f0f6fc;">' + taskName + '</td>';
            html += '<td style="color:' + c + ';font-weight:700;">' + s.final.toFixed(4) + '</td>';
            html += '<td>' + s.mean.toFixed(4) + '</td>';
            html += '<td>' + s.min.toFixed(4) + '</td>';
            html += '<td>' + s.max.toFixed(4) + '</td>';
            html += '<td style="color:' + trendColor + ';">' + trendIcon + ' ' + s.trend + '</td>';
            html += '<td style="color:' + c + ';font-weight:700;">' + s.health.toUpperCase() + '</td>';
            html += '</tr>';
        }
        html += '</tbody></table>';
        panel.innerHTML = html;
    }).catch(() => {});
}

function startSSE(jobId) {
    if (eventSource) { eventSource.close(); }
    // Do NOT wipe lossData here — loadLossHistory may have already populated it.
    // SSE picks up from lastStep, so existing data is kept and new points are appended.

    eventSource = new EventSource('/sse/job/' + jobId + '?from_step=' + lastStep);
    eventSource.onmessage = function(e) {
        const data = JSON.parse(e.data);
        if (data.done) {
            eventSource.close();
            // Load final loss summary for the completed job
            loadLossSummary(jobId);
            // Fire training-done event — all panels with hx-trigger="training-done from:body" auto-refresh
            htmx.trigger(document.body, 'training-done');
            return;
        }
        addLossPoint(data.step, data.task_name, data.task_loss, data.health || null);
        lastStep = data.step;
    };
}

// No DOMContentLoaded race — partial_training() renders server-side
// and emits the correct JS to initialize the chart and load data.
