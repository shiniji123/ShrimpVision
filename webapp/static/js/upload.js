/**
 * ShrimpVision — Upload Handler (Minimal)
 */

const Upload = (() => {
    const ALLOWED_IMAGE = ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff'];
    const ALLOWED_VIDEO = ['mp4', 'mov', 'avi', 'mkv', 'webm'];
    const MAX_UPLOAD_MB = 75;

    let zone, inner, fileInput, browseBtn;
    let progressWrap, progressFill, progressStatus, progressFilename;

    function cacheDom() {
        zone = document.getElementById('upload-zone');
        inner = document.getElementById('upload-inner');
        fileInput = document.getElementById('file-input');
        browseBtn = document.getElementById('upload-browse');
        progressWrap = document.getElementById('upload-progress');
        progressFill = document.getElementById('progress-fill');
        progressStatus = document.getElementById('progress-status');
        progressFilename = document.getElementById('progress-filename');
    }

    function getFileType(filename) {
        const ext = filename.split('.').pop().toLowerCase();
        if (ALLOWED_IMAGE.includes(ext)) return 'image';
        if (ALLOWED_VIDEO.includes(ext)) return 'video';
        return null;
    }

    function showProgress(filename, msg = 'Uploading…') {
        if (inner) inner.hidden = true;
        if (progressWrap) progressWrap.hidden = false;
        if (progressStatus) progressStatus.textContent = msg;
        if (progressFilename) progressFilename.textContent = filename;
        if (progressFill) progressFill.style.width = '30%';
    }

    function hideProgress() {
        if (inner) inner.hidden = false;
        if (progressWrap) progressWrap.hidden = true;
        if (progressFill) progressFill.style.width = '0%';
    }

    async function handleFile(file) {
        const type = getFileType(file.name);
        if (!type) {
            alert(`Unsupported file type: .${file.name.split('.').pop()}`);
            return;
        }
        if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
            alert(`File too large (max ${MAX_UPLOAD_MB} MB)`);
            return;
        }

        showProgress(file.name);
        App.setStatus('Processing…');

        const formData = new FormData();
        formData.append('file', file);
        formData.append('model_key', App.getSelectedModel());
        // Confidence is fixed at backend (25%) and follows the report experiment.

        try {
            if (type === 'image') {
                await uploadImage(formData);
            } else {
                await uploadVideo(formData);
            }
        } catch (err) {
            console.error('Upload failed:', err);
            alert(`Error: ${err.message}`);
            App.setStatus('Error');
            hideProgress();
        }
    }

    async function uploadImage(formData) {
        if (progressFill) progressFill.style.width = '60%';
        if (progressStatus) progressStatus.textContent = 'Running inference…';

        const resp = await fetch('/api/detect/image', { method: 'POST', body: formData });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.error || 'Detection failed');
        }
        const data = await resp.json();
        handleImageResult(data);
    }

    async function uploadVideo(formData) {
        if (progressStatus) progressStatus.textContent = 'Uploading video…';
        if (progressFill) progressFill.style.width = '80%';

        const resp = await fetch('/api/detect/video', { method: 'POST', body: formData });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.error || 'Video upload failed');
        }
        const data = await resp.json();
        App.state.jobId = data.job_id;

        hideProgress();
        App.showResults('video');
        VideoStream.start(data.job_id, data.model_name || App.getSelectedModel());
    }

    function handleImageResult(data) {
        hideProgress();
        App.setStatus('Ready');

        const img = document.getElementById('result-annotated-image');
        if (img) img.src = data.detection.annotated_image_url;

        const badge = document.getElementById('result-model-badge');
        if (badge) {
            badge.textContent = data.detection.model_name || '';
            badge.hidden = false;
        }

        App.showResults('image');
        Dashboard.renderImageResults(data);
        App.recordAnalysis();

        setTimeout(() => App.showDashboard(), 600);
    }

    function reset() {
        hideProgress();
        if (fileInput) fileInput.value = '';
        App.setStatus('Ready');
    }

    function init() {
        cacheDom();
        if (!zone) return;

        browseBtn?.addEventListener('click', (e) => { e.stopPropagation(); fileInput?.click(); });
        zone.addEventListener('click', () => fileInput?.click());
        fileInput?.addEventListener('change', (e) => { const f = e.target.files?.[0]; if (f) handleFile(f); });

        zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
        zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('dragover');
            const f = e.dataTransfer.files?.[0];
            if (f) handleFile(f);
        });
    }

    document.addEventListener('DOMContentLoaded', init);

    return { reset, handleFile };
})();
