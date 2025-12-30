/**
 * biocomp-tuner Frontend Application
 */

const API_BASE = '';

// State
let currentParams = {};
let paramsMetadata = [];

// DOM Elements
const initBtn = document.getElementById('init-btn');
const computeBtn = document.getElementById('compute-btn');
const resetBtn = document.getElementById('reset-btn');
const zeroBtn = document.getElementById('zero-btn');
const exportBtn = document.getElementById('export-btn');
const importBtn = document.getElementById('import-btn');
const importFile = document.getElementById('import-file');

const initSection = document.getElementById('init-section');
const vizSection = document.getElementById('visualization-section');
const paramsSection = document.getElementById('params-section');
const paramsContainer = document.getElementById('params-container');

const initStatus = document.getElementById('init-status');
const networkName = document.getElementById('network-name');
const gridInfo = document.getElementById('grid-info');

// Initialize handlers
document.addEventListener('DOMContentLoaded', () => {
    initBtn.addEventListener('click', handleInit);
    computeBtn.addEventListener('click', handleCompute);
    resetBtn.addEventListener('click', handleReset);
    zeroBtn.addEventListener('click', handleZeroAll);
    exportBtn.addEventListener('click', handleExport);
    importBtn.addEventListener('click', () => importFile.click());
    importFile.addEventListener('change', handleImport);

    checkStatus();
});

async function checkStatus() {
    try {
        const resp = await fetch(`${API_BASE}/status`);
        const data = await resp.json();
        if (data.initialized) {
            showInitialized(data);
            await loadParams();
            await handleCompute();
        }
    } catch (e) {
        console.log('Session not initialized');
    }
}

async function handleInit() {
    const modelPath = document.getElementById('model-path').value;
    const scaffoldPath = document.getElementById('scaffold-path').value;
    const targetPath = document.getElementById('target-path').value;
    const gridRes = parseInt(document.getElementById('grid-res').value) || 32;

    if (!modelPath || !scaffoldPath || !targetPath) {
        showStatus('Please fill in all paths', 'error');
        return;
    }

    initBtn.disabled = true;
    showStatus('Initializing...', 'info');

    try {
        const resp = await fetch(`${API_BASE}/init`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_path: modelPath,
                scaffold_path: scaffoldPath,
                target_path: targetPath,
                grid_resolution: [gridRes, gridRes]
            })
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Init failed');
        }

        const data = await resp.json();
        showStatus('Session initialized!', 'success');
        showInitialized(data);

        await loadParams();
        await handleCompute();

    } catch (e) {
        showStatus(`Error: ${e.message}`, 'error');
    } finally {
        initBtn.disabled = false;
    }
}

function showInitialized(data) {
    networkName.textContent = data.network_name || 'Unknown';
    gridInfo.textContent = `Grid: ${data.grid_resolution?.[0] || 32}x${data.grid_resolution?.[1] || 32}`;

    // Populate the form fields with loaded config values
    if (data.model_name) {
        document.getElementById('model-path').value = data.model_name;
    }
    if (data.scaffold_name) {
        document.getElementById('scaffold-path').value = data.scaffold_name;
    }
    if (data.target_name) {
        document.getElementById('target-path').value = data.target_name;
    }
    if (data.grid_resolution) {
        document.getElementById('grid-res').value = data.grid_resolution[0] || 32;
    }

    // Hide the init button if already initialized
    if (data.initialized) {
        initBtn.style.display = 'none';
        initStatus.textContent = 'Session initialized from config';
        initStatus.className = 'status success';
    }

    vizSection.classList.remove('hidden');
    paramsSection.classList.remove('hidden');
}

async function loadParams() {
    try {
        const resp = await fetch(`${API_BASE}/params`);
        const data = await resp.json();

        paramsMetadata = data.params;
        currentParams = {};
        for (const p of paramsMetadata) {
            currentParams[p.path] = p.current_value;
        }

        renderParams(data.by_category);

    } catch (e) {
        console.error('Failed to load params:', e);
    }
}

function renderParams(byCategory) {
    paramsContainer.innerHTML = '';

    const categoryOrder = ['ratios', 'embeddings', 'bias', 'other'];

    for (const cat of categoryOrder) {
        const params = byCategory[cat];
        if (!params || params.length === 0) continue;

        const section = document.createElement('div');
        section.className = 'param-category';

        const header = document.createElement('div');
        header.className = 'category-header';
        header.innerHTML = `
            <span class="collapse-icon">▼</span>
            <span class="category-name">${cat.toUpperCase()}</span>
            <span class="category-count">(${params.length} parameters)</span>
        `;
        header.addEventListener('click', () => {
            section.classList.toggle('collapsed');
            header.querySelector('.collapse-icon').textContent =
                section.classList.contains('collapsed') ? '▶' : '▼';
        });

        const content = document.createElement('div');
        content.className = 'category-content';

        for (const param of params) {
            const paramEl = createParamEditor(param, cat);
            content.appendChild(paramEl);
        }

        section.appendChild(header);
        section.appendChild(content);
        paramsContainer.appendChild(section);
    }
}

function createParamEditor(param, category) {
    const container = document.createElement('div');
    container.className = 'param-editor';
    container.dataset.path = param.path;

    const label = document.createElement('div');
    label.className = 'param-label';
    label.textContent = param.display_name;
    label.title = param.path;

    const inputContainer = document.createElement('div');
    inputContainer.className = 'param-inputs';

    const value = param.current_value;

    if (Array.isArray(value)) {
        if (Array.isArray(value[0])) {
            // 2D array
            for (let i = 0; i < value.length; i++) {
                const row = document.createElement('div');
                row.className = 'param-row';
                for (let j = 0; j < value[i].length; j++) {
                    const input = createInput(param, [i, j], value[i][j]);
                    row.appendChild(input);
                }
                inputContainer.appendChild(row);
            }
        } else {
            // 1D array
            const row = document.createElement('div');
            row.className = 'param-row';
            for (let i = 0; i < value.length; i++) {
                const input = createInput(param, [i], value[i]);
                row.appendChild(input);
            }
            inputContainer.appendChild(row);
        }
    } else {
        // Scalar
        const input = createInput(param, [], value);
        inputContainer.appendChild(input);
    }

    container.appendChild(label);
    container.appendChild(inputContainer);

    return container;
}

function createInput(param, indices, value) {
    const input = document.createElement('input');
    input.type = 'number';
    input.step = param.category === 'ratios' ? '0.1' : '0.01';
    input.value = typeof value === 'number' ? value.toFixed(4) : value;
    input.className = 'param-input';
    input.dataset.path = param.path;
    input.dataset.indices = JSON.stringify(indices);

    if (param.min_value !== null) input.min = param.min_value;
    if (param.max_value !== null) input.max = param.max_value;

    input.addEventListener('change', (e) => {
        const newValue = parseFloat(e.target.value);
        updateParamValue(param.path, indices, newValue);
    });

    return input;
}

function updateParamValue(path, indices, value) {
    if (!currentParams[path]) return;

    let arr = currentParams[path];
    if (indices.length === 0) {
        currentParams[path] = value;
    } else if (indices.length === 1) {
        arr[indices[0]] = value;
    } else if (indices.length === 2) {
        arr[indices[0]][indices[1]] = value;
    }
}

async function handleCompute() {
    computeBtn.disabled = true;
    computeBtn.textContent = 'Computing...';

    try {
        // First update params
        const updatesResp = await fetch(`${API_BASE}/params`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ updates: currentParams })
        });

        if (!updatesResp.ok) {
            throw new Error('Failed to update params');
        }

        // Then compute
        const resp = await fetch(`${API_BASE}/compute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Compute failed');
        }

        const data = await resp.json();
        updateVisualization(data);
        updateLosses(data.losses, data.penalties);

    } catch (e) {
        console.error('Compute failed:', e);
        alert(`Compute failed: ${e.message}`);
    } finally {
        computeBtn.disabled = false;
        computeBtn.textContent = 'Compute';
    }
}

function updateVisualization(data) {
    const layout = {
        margin: { t: 10, b: 30, l: 30, r: 10 },
        paper_bgcolor: '#1e1e1e',
        plot_bgcolor: '#1e1e1e',
        font: { color: '#e0e0e0' },
        coloraxis: { colorscale: 'Viridis' },
        yaxis: { scaleanchor: 'x', scaleratio: 1 },
        xaxis: { constrain: 'domain' }
    };

    const targetTrace = {
        z: data.Y_target,
        type: 'heatmap',
        colorscale: 'Viridis',
        showscale: false
    };

    const predTrace = {
        z: data.Y_pred,
        type: 'heatmap',
        colorscale: 'Viridis',
        showscale: true
    };

    Plotly.newPlot('target-heatmap', [targetTrace], layout, { responsive: true });
    Plotly.newPlot('prediction-heatmap', [predTrace], layout, { responsive: true });
}

function updateLosses(losses, penalties) {
    const format = (v) => v !== undefined && v !== null ? v.toFixed(4) : '-';

    document.getElementById('loss-sinkhorn').textContent = format(losses.sinkhorn);
    document.getElementById('loss-lncc').textContent = format(losses.lncc);
    document.getElementById('loss-mse').textContent = format(losses.mse);
    document.getElementById('loss-total').textContent = format(losses.total);

    document.getElementById('penalty-tucount').textContent = format(penalties.tucount);
    document.getElementById('penalty-spread').textContent = format(penalties.spread);

    // Color coding
    const totalEl = document.getElementById('loss-total');
    totalEl.classList.remove('loss-good', 'loss-medium', 'loss-bad');
    const total = losses.total;
    if (total < 0.1) {
        totalEl.classList.add('loss-good');
    } else if (total < 0.3) {
        totalEl.classList.add('loss-medium');
    } else {
        totalEl.classList.add('loss-bad');
    }
}

async function handleReset() {
    if (!confirm('Reset all parameters to random values?')) return;

    try {
        const resp = await fetch(`${API_BASE}/reset`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (!resp.ok) throw new Error('Reset failed');

        await loadParams();
        await handleCompute();

    } catch (e) {
        alert(`Reset failed: ${e.message}`);
    }
}

async function handleZeroAll() {
    // Zero out all ratio parameters
    for (const param of paramsMetadata) {
        if (param.category === 'ratios') {
            const value = param.current_value;
            if (Array.isArray(value)) {
                if (Array.isArray(value[0])) {
                    currentParams[param.path] = value.map(row => row.map(() => 0));
                } else {
                    currentParams[param.path] = value.map(() => 0);
                }
            } else {
                currentParams[param.path] = 0;
            }
        }
    }

    await loadParams();  // Refresh UI
    await handleCompute();
}

async function handleExport() {
    try {
        const resp = await fetch(`${API_BASE}/export`);
        const data = await resp.json();

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'tuner_params.json';
        a.click();
        URL.revokeObjectURL(url);

    } catch (e) {
        alert(`Export failed: ${e.message}`);
    }
}

async function handleImport(event) {
    const file = event.target.files[0];
    if (!file) return;

    try {
        const text = await file.text();
        const params = JSON.parse(text);

        const resp = await fetch(`${API_BASE}/import`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });

        if (!resp.ok) throw new Error('Import failed');

        await loadParams();
        await handleCompute();

        alert('Parameters imported successfully!');

    } catch (e) {
        alert(`Import failed: ${e.message}`);
    }

    event.target.value = '';
}

function showStatus(message, type) {
    initStatus.textContent = message;
    initStatus.className = `status ${type}`;
}

function toggleCollapsible(element) {
    element.classList.toggle('collapsed');
    const icon = element.querySelector('.collapse-icon');
    if (icon) {
        icon.textContent = element.classList.contains('collapsed') ? '▶' : '▼';
    }
}
