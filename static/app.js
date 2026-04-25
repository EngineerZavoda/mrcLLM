const form = document.getElementById("check-form");
const submitBtn = document.getElementById("submit-btn");
const compareBtn = document.getElementById("compare-btn");
const autodetectBtn = document.getElementById("autodetect-btn");
const errorEl = document.getElementById("error");
const resultPanel = document.getElementById("result-panel");
const recommendationEl = document.getElementById("recommendation");
const platformEl = document.getElementById("platform");
const languageEl = document.getElementById("language");
const ramLabelEl = document.getElementById("ram_label");
const vramLabelEl = document.getElementById("vram_label");
const gpuLabelEl = document.getElementById("gpu_label");
const platformHintEl = document.getElementById("platform_hint");
const hasGpuFieldEl = document.getElementById("has_gpu_field");
const hasGpuEl = document.getElementById("has_gpu");
const weightsEl = document.getElementById("weights");
const kvCacheEl = document.getElementById("kv_cache");
const totalEl = document.getElementById("total");
const quantizationsEl = document.getElementById("quantizations");
const statusChip = document.getElementById("status-chip");
const comparePanel = document.getElementById("compare-panel");
const compareListEl = document.getElementById("compare-list");
const clearCompareBtn = document.getElementById("clear-compare");
const historyPanel = document.getElementById("history-panel");
const historyListEl = document.getElementById("history-list");
const clearHistoryBtn = document.getElementById("clear-history");

const STORAGE_HISTORY = "llm_calc_history_v1";
const STORAGE_HISTORY_SCHEMA = "llm_calc_history_schema";
const HISTORY_SCHEMA_VERSION = 2;

let lastResult = null;
let compareItems = [];

const i18n = {
  ru: {
    title: "Проверка совместимости модели с вашим железом",
    subtitle: "Введите название модели, выберите режим расчета и сравните требуемую память с вашей системой.",
    model: "Название модели",
    lang: "Язык",
    platform: "Платформа",
    mode: "Режим",
    context: "Контекст, токенов",
    os: "ОС",
    cpu: "CPU ядер",
    hasGpu: "Есть GPU",
    detect: "Определить систему",
    check: "Проверить модель",
    compareAdd: "Добавить в сравнение",
    compareTitle: "Сравнение моделей",
    historyTitle: "Последние проверки",
    clear: "Очистить",
    result: "Результат",
    weights: "Веса",
    kv: "KV cache",
    total: "Итого VRAM",
    quants: "Варианты квантования",
    ready: "Готово",
    loading: "Считаем...",
    busy: "Расчет",
    noData: "Нет данных",
    noModel: "Введите название модели.",
    invalidNumbers: "Проверьте числовые поля: значения должны быть положительными, контекст 256..131072.",
    netErr: "Ошибка сети. Проверьте сервер.",
    hintMac: "Для Mac (особенно Apple Silicon) обычно используется объединенная память. Часто имеет смысл ставить GPU память близкой к Unified Memory.",
    hintPc: "Для PC указывайте отдельные RAM и VRAM. Если дискретной видеокарты нет, снимите галочку \"Есть GPU\".",
    historyEmpty: "История пока пуста.",
    compareEmpty: "Добавьте результаты в сравнение.",
  },
  en: {
    title: "Check model compatibility with your hardware",
    subtitle: "Enter model name, select memory mode, and compare requirements with your system.",
    model: "Model name",
    lang: "Language",
    platform: "Platform",
    mode: "Mode",
    context: "Context tokens",
    os: "OS",
    cpu: "CPU cores",
    hasGpu: "Has GPU",
    detect: "Auto-detect system",
    check: "Check model",
    compareAdd: "Add to compare",
    compareTitle: "Model comparison",
    historyTitle: "Recent checks",
    clear: "Clear",
    result: "Result",
    weights: "Weights",
    kv: "KV cache",
    total: "Total VRAM",
    quants: "Quantization options",
    ready: "Ready",
    loading: "Calculating...",
    busy: "Running",
    noData: "No data",
    noModel: "Please enter model name.",
    invalidNumbers: "Please check numeric fields: values must be positive, context must be 256..131072.",
    netErr: "Network error. Check server.",
    hintMac: "On Mac (especially Apple Silicon), unified memory is common. It's usually reasonable to set GPU memory near unified memory.",
    hintPc: "On PC, provide separate RAM and VRAM values. If no discrete GPU, uncheck 'Has GPU'.",
    historyEmpty: "No history yet.",
    compareEmpty: "Add results to compare.",
  },
};

function tr(key) {
  const lang = languageEl.value || "ru";
  return i18n[lang][key] || key;
}

function asNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function toGB(value) {
  return `${asNumber(value).toFixed(1)} GB`;
}

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  submitBtn.textContent = isLoading ? tr("loading") : tr("check");
  statusChip.textContent = isLoading ? tr("busy") : tr("ready");
}

function applyLanguage() {
  document.getElementById("title").textContent = tr("title");
  document.getElementById("subtitle").textContent = tr("subtitle");
  document.getElementById("label_model").textContent = tr("model");
  document.getElementById("label_lang").textContent = tr("lang");
  document.getElementById("label_platform").textContent = tr("platform");
  document.getElementById("label_mode").textContent = tr("mode");
  document.getElementById("label_context").textContent = tr("context");
  document.getElementById("label_os").textContent = tr("os");
  document.getElementById("label_cpu").textContent = tr("cpu");
  document.getElementById("has_gpu_text").textContent = tr("hasGpu");
  autodetectBtn.textContent = tr("detect");
  submitBtn.textContent = tr("check");
  compareBtn.textContent = tr("compareAdd");
  document.getElementById("compare_title").textContent = tr("compareTitle");
  document.getElementById("history_title").textContent = tr("historyTitle");
  document.getElementById("clear-compare").textContent = tr("clear");
  document.getElementById("clear-history").textContent = tr("clear");
  document.getElementById("result_title").textContent = tr("result");
  document.getElementById("metric_weights").textContent = tr("weights");
  document.getElementById("metric_kv").textContent = tr("kv");
  document.getElementById("metric_total").textContent = tr("total");
  document.getElementById("quant_title").textContent = tr("quants");
  statusChip.textContent = tr("ready");

  applyPlatformPreset();
  renderCompare();
  renderHistory();
}

function applyPlatformPreset() {
  const platform = platformEl.value;
  const ramInput = document.getElementById("ram_gb");
  const vramInput = document.getElementById("vram_gb");
  const gpuInput = document.getElementById("gpu_name");
  const osInput = document.getElementById("os");

  if (platform === "mac") {
    ramLabelEl.textContent = "Unified Memory, GB";
    vramLabelEl.textContent = languageEl.value === "ru" ? "Память под GPU, GB" : "GPU memory share, GB";
    gpuLabelEl.textContent = "Apple GPU/Chip";
    hasGpuFieldEl.classList.add("hidden");
    hasGpuEl.checked = true;

    if (!ramInput.value || asNumber(ramInput.value) === 32) ramInput.value = "36";
    if (!vramInput.value || asNumber(vramInput.value) === 12) vramInput.value = ramInput.value;
    if (!gpuInput.value || gpuInput.value === "NVIDIA RTX 3060") gpuInput.value = "Apple M3 Pro";
    if (!osInput.value || osInput.value === "Linux") osInput.value = "macOS";

    platformHintEl.textContent = tr("hintMac");
    return;
  }

  ramLabelEl.textContent = "RAM, GB";
  vramLabelEl.textContent = "VRAM, GB";
  gpuLabelEl.textContent = "GPU";
  hasGpuFieldEl.classList.remove("hidden");

  if (!gpuInput.value || gpuInput.value === "Apple M3 Pro") gpuInput.value = "NVIDIA RTX 3060";
  if (!osInput.value || osInput.value === "macOS") osInput.value = "Linux";
  if (!vramInput.value || asNumber(vramInput.value) === asNumber(ramInput.value)) vramInput.value = "12";

  platformHintEl.textContent = tr("hintPc");
}

function validatePayload(payload) {
  if (!payload.model_name) return tr("noModel");
  if (payload.context_len < 256 || payload.context_len > 131072) return tr("invalidNumbers");
  if (payload.system_info.ram_gb <= 0 || payload.system_info.vram_gb < 0 || payload.system_info.cpu_cores < 1) return tr("invalidNumbers");
  return null;
}

function collectPayload() {
  const modelName = document.getElementById("model_name").value.trim();
  const mode = document.getElementById("mode").value;
  const platform = platformEl.value;
  const hasGpu = platform === "mac" ? true : hasGpuEl.checked;

  return {
    model_name: modelName,
    mode,
    context_len: asNumber(document.getElementById("context_len").value),
    language: languageEl.value,
    system_info: {
      ram_gb: asNumber(document.getElementById("ram_gb").value),
      vram_gb: asNumber(document.getElementById("vram_gb").value),
      has_gpu: hasGpu,
      gpu_name: document.getElementById("gpu_name").value.trim() || "Unknown",
      os: document.getElementById("os").value.trim() || "Unknown",
      cpu_cores: asNumber(document.getElementById("cpu_cores").value) || 1,
    },
  };
}

function renderQuantizations(quantizations) {
  quantizationsEl.innerHTML = "";
  const entries = Object.entries(quantizations || {}).sort((a, b) => a[1] - b[1]);
  if (!entries.length) {
    quantizationsEl.innerHTML = `<p class="muted">${tr("noData")}</p>`;
    return;
  }

  for (const [name, value] of entries) {
    const row = document.createElement("div");
    row.className = "quant-item";
    row.innerHTML = `<span>${name}</span><strong>${toGB(value)}</strong>`;
    quantizationsEl.appendChild(row);
  }
}

function ensureHistorySchema() {
  const current = Number(localStorage.getItem(STORAGE_HISTORY_SCHEMA) || "0");
  if (current !== HISTORY_SCHEMA_VERSION) {
    localStorage.removeItem(STORAGE_HISTORY);
    localStorage.setItem(STORAGE_HISTORY_SCHEMA, String(HISTORY_SCHEMA_VERSION));
  }
}

function normalizeHistoryItem(item) {
  if (!item || typeof item !== "object") return null;
  if (typeof item.model_name !== "string" || !item.model_name.trim()) return null;
  if (typeof item.mode !== "string" || !item.mode.trim()) return null;

  const context_len = asNumber(item.context_len);
  if (context_len < 256 || context_len > 131072) return null;

  return {
    model_name: item.model_name.trim(),
    mode: item.mode.trim(),
    context_len,
    total_vram_gb: asNumber(item.total_vram_gb),
    ram_gb: asNumber(item.ram_gb),
    vram_gb: asNumber(item.vram_gb),
    has_gpu: Boolean(item.has_gpu),
    gpu_name: typeof item.gpu_name === "string" ? item.gpu_name : "Unknown",
    os: typeof item.os === "string" ? item.os : "Unknown",
    cpu_cores: Math.max(1, asNumber(item.cpu_cores) || 1),
    language: typeof item.language === "string" ? item.language : "ru",
    platform: typeof item.platform === "string" ? item.platform : "pc",
    ts: asNumber(item.ts) || Date.now(),
  };
}

function pushHistory(entry) {
  ensureHistorySchema();
  const current = getHistory();
  const normalized = normalizeHistoryItem(entry);
  if (!normalized) return;

  const next = [
    normalized,
    ...current.filter((x) => !(x.model_name === normalized.model_name && x.mode === normalized.mode && x.context_len === normalized.context_len)),
  ].slice(0, 8);
  localStorage.setItem(STORAGE_HISTORY, JSON.stringify(next));
}

function getHistory() {
  ensureHistorySchema();
  const raw = JSON.parse(localStorage.getItem(STORAGE_HISTORY) || "[]");
  if (!Array.isArray(raw)) return [];
  const normalized = raw.map(normalizeHistoryItem).filter(Boolean);
  if (normalized.length !== raw.length) {
    localStorage.setItem(STORAGE_HISTORY, JSON.stringify(normalized));
  }
  return normalized;
}

function fillFormFromHistory(item) {
  document.getElementById("model_name").value = item.model_name;
  document.getElementById("mode").value = item.mode;
  document.getElementById("context_len").value = item.context_len;
  document.getElementById("ram_gb").value = item.ram_gb;
  document.getElementById("vram_gb").value = item.vram_gb;
  document.getElementById("gpu_name").value = item.gpu_name;
  document.getElementById("os").value = item.os;
  document.getElementById("cpu_cores").value = item.cpu_cores;
  hasGpuEl.checked = Boolean(item.has_gpu);
  platformEl.value = item.platform === "mac" ? "mac" : "pc";
  applyPlatformPreset();
}

function renderHistory() {
  const list = getHistory();
  historyListEl.innerHTML = "";
  if (!list.length) {
    historyListEl.innerHTML = `<p class="muted">${tr("historyEmpty")}</p>`;
    historyPanel.classList.remove("hidden");
    return;
  }
  for (const item of list) {
    const row = document.createElement("div");
    row.className = "history-item";
    row.innerHTML = `<span>${item.model_name} | ${item.mode} | контекст ${item.context_len} токенов | RAM ${item.ram_gb.toFixed(1)} | VRAM ${item.vram_gb.toFixed(1)}</span><strong>${toGB(item.total_vram_gb)}</strong>`;
    row.addEventListener("click", () => fillFormFromHistory(item));
    historyListEl.appendChild(row);
  }
  historyPanel.classList.remove("hidden");
}

function renderCompare() {
  compareListEl.innerHTML = "";
  if (!compareItems.length) {
    compareListEl.innerHTML = `<p class="muted">${tr("compareEmpty")}</p>`;
    comparePanel.classList.remove("hidden");
    return;
  }

  compareItems.forEach((item) => {
    const row = document.createElement("div");
    row.className = "compare-item";
    row.innerHTML = `<span>${item.model_name} | ${item.mode} | контекст ${item.context_len} токенов</span><strong>${toGB(item.total_vram_gb)}</strong>`;
    compareListEl.appendChild(row);
  });
  comparePanel.classList.remove("hidden");
}

function renderResult(data, payload) {
  recommendationEl.textContent = data.recommendation || tr("noData");
  weightsEl.textContent = toGB(data.memory_requirements?.weights_gb);
  kvCacheEl.textContent = toGB(data.memory_requirements?.kv_cache_gb);
  totalEl.textContent = toGB(data.memory_requirements?.total_vram_gb);
  renderQuantizations(data.possible_quantizations);
  resultPanel.classList.remove("hidden");
  compareBtn.disabled = false;

  lastResult = {
    model_name: payload.model_name,
    mode: payload.mode,
    context_len: payload.context_len,
    total_vram_gb: data.memory_requirements?.total_vram_gb || 0,
    ram_gb: payload.system_info?.ram_gb || 0,
    vram_gb: payload.system_info?.vram_gb || 0,
    has_gpu: payload.system_info?.has_gpu || false,
    gpu_name: payload.system_info?.gpu_name || "Unknown",
    os: payload.system_info?.os || "Unknown",
    cpu_cores: payload.system_info?.cpu_cores || 1,
    language: payload.language || "ru",
    platform: platformEl.value || "pc",
  };

  pushHistory({ ...lastResult, ts: Date.now() });
  renderHistory();
}

async function autoDetectSystem() {
  errorEl.textContent = "";
  autodetectBtn.disabled = true;
  try {
    const response = await fetch("/api/system_info");
    const data = await response.json();
    if (!response.ok) throw new Error(data?.detail || "Auto-detect failed");

    document.getElementById("ram_gb").value = asNumber(data.ram_gb).toFixed(1);
    document.getElementById("vram_gb").value = asNumber(data.vram_gb).toFixed(1);
    document.getElementById("gpu_name").value = data.gpu_name || "Unknown";
    document.getElementById("os").value = data.os || "Unknown";
    document.getElementById("cpu_cores").value = data.cpu_cores || 1;
    hasGpuEl.checked = Boolean(data.has_gpu);

    const osNorm = (data.os || "").toLowerCase();
    if (osNorm.includes("mac") || osNorm.includes("darwin") || osNorm.includes("os x")) {
      platformEl.value = "mac";
    } else {
      platformEl.value = "pc";
    }
    applyPlatformPreset();
  } catch (error) {
    errorEl.textContent = error.message || tr("netErr");
  } finally {
    autodetectBtn.disabled = false;
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorEl.textContent = "";

  const payload = collectPayload();
  const validationErr = validatePayload(payload);
  if (validationErr) {
    errorEl.textContent = validationErr;
    return;
  }

  setLoading(true);
  try {
    const response = await fetch("/api/check_model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(String(data?.detail || "Request failed"));
    }

    renderResult(data, payload);
  } catch (error) {
    errorEl.textContent = error.message || tr("netErr");
  } finally {
    setLoading(false);
  }
});

compareBtn.addEventListener("click", () => {
  if (!lastResult) return;
  const already = compareItems.some((x) => x.model_name === lastResult.model_name && x.mode === lastResult.mode && x.context_len === lastResult.context_len);
  if (!already) {
    compareItems = [...compareItems, lastResult].slice(-3);
    renderCompare();
  }
});

clearCompareBtn.addEventListener("click", () => {
  compareItems = [];
  renderCompare();
});

clearHistoryBtn.addEventListener("click", () => {
  localStorage.removeItem(STORAGE_HISTORY);
  localStorage.setItem(STORAGE_HISTORY_SCHEMA, String(HISTORY_SCHEMA_VERSION));
  renderHistory();
});

autodetectBtn.addEventListener("click", autoDetectSystem);
platformEl.addEventListener("change", applyPlatformPreset);
languageEl.addEventListener("change", applyLanguage);

applyLanguage();
ensureHistorySchema();
renderHistory();
renderCompare();
