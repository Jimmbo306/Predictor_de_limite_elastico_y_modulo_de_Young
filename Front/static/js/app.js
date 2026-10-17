/**
 * MPEA Predictor - Lógica Frontend
 * Control de navegación por pestañas, llamadas asíncronas a la API y renderizado dinámico.
 */

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initPresets();
  initForms();
});

/**
 * Control de Pestañas
 */
function initTabs() {
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');

      // Actualizar botones
      tabButtons.forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
      });
      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');

      // Actualizar paneles
      tabPanes.forEach(pane => {
        pane.classList.remove('active');
        if (pane.id === targetId) {
          pane.classList.add('active');
        }
      });
    });
  });
}

/**
 * Autocompletado de presets rápidos
 */
function initPresets() {
  const chips = document.querySelectorAll('.chip-btn');

  chips.forEach(chip => {
    chip.addEventListener('click', () => {
      const target = chip.getAttribute('data-target');

      if (target === 'young') {
        const formulaInput = document.getElementById('young-formula');
        const densityInput = document.getElementById('young-density');
        if (formulaInput) formulaInput.value = chip.getAttribute('data-formula') || '';
        if (densityInput) densityInput.value = chip.getAttribute('data-density') || '';
      } else if (target === 'ys') {
        const formulaInput = document.getElementById('ys-formula');
        const tempInput = document.getElementById('ys-temp');
        const testSelect = document.getElementById('ys-test-type');
        const procSelect = document.getElementById('ys-proc');
        const crystSelect = document.getElementById('ys-cryst');
        const microSelect = document.getElementById('ys-micro');

        if (formulaInput) formulaInput.value = chip.getAttribute('data-formula') || '';
        if (tempInput) tempInput.value = chip.getAttribute('data-temp') || '25';
        if (testSelect) testSelect.value = chip.getAttribute('data-test') || 'C';
        if (procSelect) procSelect.value = chip.getAttribute('data-proc') || 'CAST';
        if (crystSelect) crystSelect.value = chip.getAttribute('data-cryst') || 'FCC';
        if (microSelect) microSelect.value = chip.getAttribute('data-micro') || 'FCC';
      }
    });
  });
}

/**
 * Manejo de Formularios y Peticiones API
 */
function initForms() {
  // Formulario Módulo de Young
  const formYoung = document.getElementById('form-young');
  if (formYoung) {
    formYoung.addEventListener('submit', async (e) => {
      e.preventDefault();
      await handlePredictYoung();
    });
  }

  // Formulario Límite Elástico
  const formYS = document.getElementById('form-ys');
  if (formYS) {
    formYS.addEventListener('submit', async (e) => {
      e.preventDefault();
      await handlePredictYS();
    });
  }
}

/**
 * Predicción de Módulo de Young
 */
async function handlePredictYoung() {
  const formula = document.getElementById('young-formula').value.trim();
  const density = parseFloat(document.getElementById('young-density').value);
  const btn = document.getElementById('btn-predict-young');
  const spinner = btn.querySelector('.btn-spinner');
  const btnText = btn.querySelector('.btn-text');

  if (!formula || isNaN(density)) {
    showToast('Por favor ingresa una fórmula válida y una densidad calculada.', 'error');
    return;
  }

  setLoading(btn, spinner, btnText, true, 'Featurizando y prediciendo...');

  try {
    const res = await fetch('/api/predict/young', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ formula, density }),
    });

    const data = await res.json();

    if (!res.ok || !data.success) {
      throw new Error(data.error || 'Error al procesar la predicción.');
    }

    renderYoungResults(data);
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    setLoading(btn, spinner, btnText, false, 'Calcular Descriptores y Predecir Módulo de Young');
  }
}

/**
 * Predicción de Límite Elástico
 */
async function handlePredictYS() {
  const formula = document.getElementById('ys-formula').value.trim();
  const temp = parseFloat(document.getElementById('ys-temp').value);
  const tipo_ensayo = document.getElementById('ys-test-type').value;
  const metodo_proc = document.getElementById('ys-proc').value;
  const est_cristalina = document.getElementById('ys-cryst').value;
  const microestructura = document.getElementById('ys-micro').value;

  const btn = document.getElementById('btn-predict-ys');
  const spinner = btn.querySelector('.btn-spinner');
  const btnText = btn.querySelector('.btn-text');

  if (!formula || isNaN(temp)) {
    showToast('Por favor completa la fórmula y la temperatura de prueba.', 'error');
    return;
  }

  setLoading(btn, spinner, btnText, true, 'Featurizando y prediciendo...');

  try {
    const res = await fetch('/api/predict/limite_elastico', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        formula,
        temp,
        tipo_ensayo,
        metodo_proc,
        est_cristalina,
        microestructura,
      }),
    });

    const data = await res.json();

    if (!res.ok || !data.success) {
      throw new Error(data.error || 'Error al procesar la predicción.');
    }

    renderYSResults(data);
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    setLoading(btn, spinner, btnText, false, 'Calcular Descriptores y Predecir Límite Elástico');
  }
}

/**
 * Renderiza resultados para Módulo de Young
 */
function renderYoungResults(data) {
  const placeholder = document.getElementById('young-placeholder');
  const resultsContainer = document.getElementById('young-results');

  if (placeholder) placeholder.classList.add('hidden');
  if (resultsContainer) resultsContainer.classList.remove('hidden');

  // Métrica principal
  document.getElementById('young-val').textContent = data.prediccion.toFixed(1);
  document.getElementById('young-formula-display').textContent = data.formula_normalizada;

  // Desglose elemental
  const elementsContainer = document.getElementById('young-elements-bars');
  elementsContainer.innerHTML = '';
  if (data.fracciones_elementales) {
    Object.entries(data.fracciones_elementales).forEach(([el, pct]) => {
      const pill = document.createElement('div');
      pill.className = 'element-pill';
      pill.innerHTML = `
        <span class="element-symbol">${el}</span>
        <span class="element-percent">${pct}%</span>
      `;
      elementsContainer.appendChild(pill);
    });
  }

  // Descriptores clave
  const descriptorsContainer = document.getElementById('young-descriptors-grid');
  descriptorsContainer.innerHTML = '';
  if (data.descriptores_clave) {
    Object.entries(data.descriptores_clave).forEach(([name, val]) => {
      const item = document.createElement('div');
      item.className = 'descriptor-item';
      item.innerHTML = `
        <span class="descriptor-name">${name}</span>
        <span class="descriptor-value">${val}</span>
      `;
      descriptorsContainer.appendChild(item);
    });
  }
}

/**
 * Renderiza resultados para Límite Elástico
 */
function renderYSResults(data) {
  const placeholder = document.getElementById('ys-placeholder');
  const resultsContainer = document.getElementById('ys-results');

  if (placeholder) placeholder.classList.add('hidden');
  if (resultsContainer) resultsContainer.classList.remove('hidden');

  // Métrica principal
  document.getElementById('ys-val').textContent = data.prediccion.toFixed(1);
  document.getElementById('ys-formula-display').textContent = data.formula_normalizada;

  // Desglose elemental
  const elementsContainer = document.getElementById('ys-elements-bars');
  elementsContainer.innerHTML = '';
  if (data.fracciones_elementales) {
    Object.entries(data.fracciones_elementales).forEach(([el, pct]) => {
      const pill = document.createElement('div');
      pill.className = 'element-pill';
      pill.innerHTML = `
        <span class="element-symbol">${el}</span>
        <span class="element-percent">${pct}%</span>
      `;
      elementsContainer.appendChild(pill);
    });
  }

  // Descriptores clave
  const descriptorsContainer = document.getElementById('ys-descriptors-grid');
  descriptorsContainer.innerHTML = '';
  if (data.descriptores_clave) {
    Object.entries(data.descriptores_clave).forEach(([name, val]) => {
      const item = document.createElement('div');
      item.className = 'descriptor-item';
      item.innerHTML = `
        <span class="descriptor-name">${name}</span>
        <span class="descriptor-value">${val}</span>
      `;
      descriptorsContainer.appendChild(item);
    });
  }
}

/**
 * Helper de estado de carga
 */
function setLoading(btn, spinner, btnText, loading, text) {
  if (loading) {
    btn.disabled = true;
    if (spinner) spinner.classList.remove('hidden');
    if (btnText) btnText.innerHTML = text;
  } else {
    btn.disabled = false;
    if (spinner) spinner.classList.add('hidden');
    if (btnText) {
      btnText.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M5 12h14"></path>
          <path d="m12 5 7 7-7 7"></path>
        </svg>
        ${text}
      `;
    }
  }
}

/**
 * Sistema de notificación Toast
 */
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <circle cx="12" cy="12" r="10"></circle>
      <line x1="12" y1="8" x2="12" y2="12"></line>
      <line x1="12" y1="16" x2="12.01" y2="16"></line>
    </svg>
    <span>${message}</span>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}
