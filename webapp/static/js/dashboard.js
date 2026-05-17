/**
 * ShrimpVision — Dashboard & Charts (Minimal)
 */

const Dashboard = (() => {
    let donutChart = null;
    let activeTimelineChart = null;
    let inactiveTimelineChart = null;
    let _lastResult = null;

    const COLORS = {
        active: '#4ade80',
        inactive: '#f87171',
        grid: 'rgba(255,255,255,0.05)',
    };

    const HEALTH = {
        healthy: { icon: '🟢', title: 'High Activity', desc: 'Most shrimp are actively moving. This is a behavioral indicator, not a disease diagnosis.' },
        warning: { icon: '🟡', title: 'Reduced Activity', desc: 'Movement is lower than expected. Check water quality and culture conditions.' },
        critical: { icon: '🔴', title: 'Low Activity', desc: 'Most shrimp are inactive. Immediate observation and water-quality checking are recommended.' },
        unknown: { icon: '🔵', title: 'Detected', desc: 'Behavior analysis requires video input.' },
    };

    // ── Charts ─────────────────────────────────────────────

    function renderDonut(active, inactive) {
        const canvas = document.getElementById('chart-donut');
        if (!canvas || !window.Chart) return;
        if (donutChart) donutChart.destroy();

        donutChart = new Chart(canvas, {
            type: 'doughnut',
            data: {
                labels: ['Active', 'Inactive'],
                datasets: [{ data: [active, inactive], backgroundColor: [COLORS.active, COLORS.inactive], borderWidth: 0, spacing: 2 }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                animation: false,
                cutout: '70%',
                plugins: {
                    legend: { position: 'bottom', labels: { color: '#6b7280', font: { size: 11 } } },
                    tooltip: { callbacks: { label: (c) => ` ${c.label}: ${c.raw}` } },
                },
            },
        });
    }

    function renderTimelineLayer(canvasId, chartRef, label, color, labels, values) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || !window.Chart) return;
        if (chartRef) chartRef.destroy();

        return new Chart(canvas, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    {
                        label,
                        data: values,
                        borderColor: color,
                        backgroundColor: color + '26',
                        fill: true,
                        tension: 0.35,
                        pointRadius: 0,
                        borderWidth: 2
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: { grid: { display: false }, ticks: { color: '#6b7280', maxTicksLimit: 20, font: { size: 9 } } },
                    y: {
                        grid: { color: COLORS.grid },
                        ticks: { color: '#6b7280', precision: 0 },
                        beginAtZero: true,
                        min: 0,
                    },
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(15,17,23,0.95)',
                        padding: 10,
                        cornerRadius: 6,
                        callbacks: { label: (c) => ` ${label}: ${c.raw}` },
                    },
                },
            },
        });
    }

    function renderTimeline(frameStats) {
        const stats = Array.isArray(frameStats) ? frameStats : [];
        const labels = stats.map((f) => `F${f.frame}`);
        const activeValues = stats.map((f) => f.active ?? 0);
        const inactiveValues = stats.map((f) => f.inactive ?? 0);

        activeTimelineChart = renderTimelineLayer(
            'chart-active-timeline',
            activeTimelineChart,
            'Active',
            COLORS.active,
            labels,
            activeValues,
        );
        inactiveTimelineChart = renderTimelineLayer(
            'chart-inactive-timeline',
            inactiveTimelineChart,
            'Inactive',
            COLORS.inactive,
            labels,
            inactiveValues,
        );
    }

    // ── Stat Cards ─────────────────────────────────────────

    function setVal(id, val) {
        const el = document.getElementById(id);
        if (el) el.textContent = val ?? '—';
    }

    function updateStatCards(total, active, inactive, activityScore, inferenceMs) {
        setVal('stat-total', total);
        setVal('stat-active', active);
        setVal('stat-inactive', inactive);
        setVal('stat-health-score', activityScore != null ? activityScore : '—');
        if (inferenceMs != null && inferenceMs > 0) {
            setVal('stat-inf-speed', `${(1000 / inferenceMs).toFixed(1)} fps`);
        } else {
            setVal('stat-inf-speed', '—');
        }
    }

    // ── Health Banner ──────────────────────────────────────

    function showHealthBanner(status) {
        const banner = document.getElementById('health-banner');
        if (!banner) return;
        const cfg = HEALTH[status] || HEALTH.unknown;
        banner.hidden = false;
        banner.className = `health-banner health-banner--${status}`;
        setVal('health-banner-icon', cfg.icon);
        setVal('health-banner-title', cfg.title);
        setVal('health-banner-desc', cfg.desc);
    }

    // ── Details Table ──────────────────────────────────────

    function populateDetails(rows) {
        const tbody = document.getElementById('details-tbody');
        const card = document.getElementById('details-card');
        if (!tbody || !card) return;
        card.hidden = false;
        tbody.innerHTML = '';
        rows.forEach(([label, value]) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `<td>${label}</td><td>${value}</td>`;
            tbody.appendChild(tr);
        });
    }

    // ── Render Image ───────────────────────────────────────

    function renderImageResults(apiData) {
        const det = apiData.detection;
        const beh = apiData.behavior;

        updateStatCards(beh.total_shrimp, '—', '—', null, det.inference_ms || null);
        showHealthBanner(beh.health_status);
        renderDonut(0, 0);
        renderTimeline([{ frame: 0, count: beh.total_shrimp, active: 0, inactive: 0 }]);
        populateDetails([
            ['Analysis Type', 'Image'],
            ['Model', det.model_name || det.model_key || '—'],
            ['Total Shrimp', det.count],
            ['Density', beh.density],
            ['Behavior Status', 'Not available for still image'],
            ['Resolution', `${det.image_width} × ${det.image_height}`],
            ['Inference Time', det.inference_ms != null ? `${det.inference_ms} ms` : '—'],
            ['Inference Speed', det.inference_ms != null ? `${(1000 / det.inference_ms).toFixed(1)} fps` : '—'],
        ]);

        _lastResult = { type: 'image', detection: det, behavior: beh, timestamp: new Date().toISOString() };
        const btn = document.getElementById('btn-export-report');
        if (btn) btn.disabled = false;

        App.showDashboard();
    }

    // ── Render Video ───────────────────────────────────────

    function renderVideoSummary(summary) {
        const avg = Math.round(summary.avg_shrimp_count);
        const active = Math.round(avg * summary.avg_active_ratio);
        const inactive = avg - active;
        let infMs = null;
        if (summary.avg_inference_fps > 0) infMs = 1000 / summary.avg_inference_fps;

        updateStatCards(avg, active, inactive, summary.overall_health_score, infMs);
        showHealthBanner(summary.overall_health_status);
        renderDonut(active, inactive);
        if (summary.frame_stats?.length) renderTimeline(summary.frame_stats);
        populateDetails([
            ['Analysis Type', 'Video'],
            ['Model', summary.model_name || '—'],
            ['Frames Analyzed', summary.total_frames_analyzed],
            ['Avg Shrimp Count', summary.avg_shrimp_count],
            ['Peak Count', summary.peak_count],
            ['Min Count', summary.min_count],
            ['Avg Active Ratio', `${Math.round(summary.avg_active_ratio * 100)}%`],
            ['Activity Score', `${summary.overall_health_score}/100`],
            ['Behavior Status', summary.overall_health_status],
        ]);

        _lastResult = { type: 'video', summary, timestamp: new Date().toISOString() };
        const btn = document.getElementById('btn-export-report');
        if (btn) btn.disabled = false;

        App.showDashboard();
    }

    // ── Live Update (per-frame) ────────────────────────────

    function liveUpdate(frameData) {
        const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
        set('live-frame', `${frameData.frame_number}/${frameData.total_frames}`);
        set('live-count', frameData.count);
        set('live-active', frameData.active_count ?? '—');
        set('live-inactive', frameData.inactive_count ?? '—');
        set('live-health', frameData.health_score ?? '—');
        if (frameData.avg_fps_inference != null)
            set('live-fps', `${frameData.avg_fps_inference.toFixed(1)} fps`);
    }

    // ── Export ─────────────────────────────────────────────

    function exportReport() {
        if (!_lastResult) { alert('Run a detection first.'); return; }
        const blob = new Blob([JSON.stringify({ generated_at: new Date().toISOString(), result: _lastResult }, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = Object.assign(document.createElement('a'), { href: url, download: `shrimp_report_${Date.now()}.json` });
        document.body.appendChild(a); a.click(); URL.revokeObjectURL(url); a.remove();
    }

    // ── Init ───────────────────────────────────────────────

    function init() {
        if (window.Chart) {
            Chart.defaults.color = '#6b7280';
            Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
        }
        const btn = document.getElementById('btn-export-report');
        if (btn) btn.addEventListener('click', exportReport);
    }

    document.addEventListener('DOMContentLoaded', init);

    return { renderImageResults, renderVideoSummary, liveUpdate, updateStatCards, showHealthBanner };
})();
