/**
 * ShrimpVision — Model Selector (Minimal Pills)
 */

const ModelSelector = (() => {
    const BADGE_COLORS = {
        'Recommended': 'green',
        'Highest mAP': 'purple',
        'NMS-Free': 'blue',
        'Stable': 'gray',
        'Latest Model': 'amber',
        'Baseline': 'gray',
        'Alternative': 'amber',
        'Selected': 'blue',
    };

    function render(models, defaultKey) {
        const container = document.getElementById('model-selector-grid');
        if (!container) return;

        container.innerHTML = '';

        function selectModel(key) {
            container.querySelectorAll('.model-pill').forEach((p) => {
                const selected = p.dataset.key === key;
                p.classList.toggle('selected', selected);
                p.setAttribute('aria-pressed', String(selected));
            });
            App.setSelectedModel(key);
            const selectedModel = models.find((m) => m.key === key);
            App.setStatus(selectedModel ? `Model: ${selectedModel.name}` : 'Ready');
        }

        models.forEach((m) => {
            const isDefault = m.key === defaultKey;
            const badgeColor = BADGE_COLORS[m.badge] || 'gray';

            const pill = document.createElement('button');
            pill.type = 'button';
            pill.className = `model-pill ${isDefault ? 'selected' : ''}`;
            pill.dataset.key = m.key;
            pill.title = m.description;
            pill.setAttribute('aria-pressed', String(isDefault));

            pill.innerHTML = `
                ${m.name}
                <span class="pill-badge" style="color:var(--${badgeColor})">${m.badge}</span>
            `;

            if (!m.available) {
                pill.disabled = true;
                pill.style.opacity = '0.4';
            }

            pill.addEventListener('click', () => {
                if (pill.disabled) return;
                selectModel(m.key);
            });

            container.appendChild(pill);
        });
    }

    return { render };
})();
