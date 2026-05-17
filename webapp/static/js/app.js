/**
 * ShrimpVision — App Controller (Minimal)
 */

const App = (() => {
    const state = {
        currentSection: 'upload',
        analysisType: null,
        jobId: null,
        selectedModel: 'yolov10s',
        sessionStats: { totalAnalyses: 0 },
    };

    const $ = (id) => document.getElementById(id);
    const $$ = (sel) => document.querySelectorAll(sel);

    // ── Navigation ──────────────────────────────────────────

    function showSection(name) {
        ['upload', 'results', 'dashboard'].forEach((s) => {
            const el = $(`${s}-section`);
            if (el) el.hidden = s !== name;
        });
        $$('.nav-link').forEach((l) => l.classList.toggle('active', l.dataset.section === name));
        state.currentSection = name;
    }

    function showResults(type) {
        state.analysisType = type;
        const imgView = $('result-image-view');
        const vidView = $('result-video-view');
        if (imgView) imgView.hidden = type !== 'image';
        if (vidView) vidView.hidden = type !== 'video';
        showSection('results');
    }

    function showDashboard() { showSection('dashboard'); }

    // ── Model ────────────────────────────────────────────────

    function setSelectedModel(key) { state.selectedModel = key; }
    function getSelectedModel() { return state.selectedModel; }

    // Confidence is fixed at backend (0.25) and follows the report experiment.
    function getConfidence() { return 0.25; }

    // ── Status ───────────────────────────────────────────────

    function setStatus(text) {
        const el = $('nav-status');
        if (el) el.textContent = text;
    }

    // ── Toast (simple) ───────────────────────────────────────

    function toast(message, type = 'info') {
        console.log(`[${type.toUpperCase()}] ${message}`);
    }

    // ── Session ──────────────────────────────────────────────

    function recordAnalysis() { state.sessionStats.totalAnalyses++; }

    // ── Init ──────────────────────────────────────────────────

    function init() {
        $$('.nav-link').forEach((link) => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const s = link.dataset.section;
                if (s) showSection(s);
            });
        });

        const btnNew = $('btn-new-upload');
        if (btnNew) {
            btnNew.addEventListener('click', () => {
                showSection('upload');
                if (typeof Upload !== 'undefined') Upload.reset();
            });
        }

        // Load models
        fetch('/api/models')
            .then((r) => r.json())
            .then((d) => {
                ModelSelector.render(d.models, d.default);
                setSelectedModel(d.default);
            })
            .catch((e) => console.warn('Could not load models:', e));

        showSection('upload');
    }

    document.addEventListener('DOMContentLoaded', init);

    return {
        state,
        showSection,
        showResults,
        showDashboard,
        setStatus,
        toast,
        setSelectedModel,
        getSelectedModel,
        getConfidence,
        recordAnalysis,
    };
})();
