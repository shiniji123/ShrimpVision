/**
 * ShrimpVision — Video Stream Handler (Minimal)
 */

const VideoStream = (() => {
    let eventSource = null;
    let canvas = null;
    let ctx = null;
    let completed = false;

    function cacheDom() {
        canvas = document.getElementById('video-canvas');
        ctx = canvas?.getContext('2d');
    }

    function hideOverlay() {
        const el = document.getElementById('video-overlay');
        if (el) el.classList.add('hidden');
    }

    function showOverlay(msg) {
        const el = document.getElementById('video-overlay');
        const label = document.getElementById('video-progress-label');
        if (el) el.classList.remove('hidden');
        if (label && msg) label.textContent = msg;
    }

    function drawFrame(b64) {
        if (!ctx || !canvas) return;
        const img = new Image();
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
        };
        img.src = 'data:image/jpeg;base64,' + b64;
    }

    function start(jobId, modelName) {
        cacheDom();
        completed = false;
        showOverlay(`Processing with ${modelName}…`);
        App.setStatus('Processing video…');

        const badge = document.getElementById('result-model-badge');
        if (badge && modelName) { badge.textContent = modelName; badge.hidden = false; }

        if (eventSource) eventSource.close();
        eventSource = new EventSource(`/api/stream/${jobId}`);

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'frame') {
                    if (data.annotated_frame_b64) { hideOverlay(); drawFrame(data.annotated_frame_b64); }
                    // Update progress label
                    const label = document.getElementById('video-progress-label');
                    if (label) label.textContent = `Frame ${data.frame_number} / ${data.total_frames}`;
                    Dashboard.liveUpdate(data);
                } else if (data.type === 'complete') {
                    completed = true;
                    eventSource.close(); eventSource = null;
                    hideOverlay();
                    App.setStatus('Ready');
                    if (data.summary) {
                        Dashboard.renderVideoSummary(data.summary);
                        App.recordAnalysis();
                    }
                } else if (data.type === 'error') {
                    completed = true;
                    eventSource.close(); eventSource = null;
                    App.setStatus('Error');
                    showOverlay(`Video error: ${data.message}`);
                    alert(`Video error: ${data.message}`);
                }
            } catch (err) {
                console.error('SSE parse error:', err);
            }
        };

        eventSource.onerror = () => {
            if (completed || eventSource?.readyState === EventSource.CLOSED) return;
            eventSource?.close();
            eventSource = null;
            App.setStatus('Connection lost');
            showOverlay('Connection lost while processing. Please run the video again.');
        };
    }

    function stop() {
        completed = true;
        if (eventSource) { eventSource.close(); eventSource = null; }
    }

    document.addEventListener('DOMContentLoaded', cacheDom);

    return { start, stop };
})();
