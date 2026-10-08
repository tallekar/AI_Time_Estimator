/* ================================================================
   TimeForge AI — Frontend Logic
   ================================================================ */

// --- Feature importances (injected from server via template, fallback below) ---
const FEATURE_LABELS = {
  num_employees:       "Team Size",
  complexity:          "Complexity",
  team_experience:     "Team Experience",
  num_modules:         "No. of Modules",
  working_hours_day:   "Working Hrs/Day",
  deadline_pressure:   "Deadline Pressure",
  requirement_clarity: "Req. Clarity",
};

// ---- Slider value display ----
document.querySelectorAll(".slider").forEach(slider => {
  const pill = document.getElementById("val_" + slider.id);
  if (!pill) return;
  const update = () => {
    pill.textContent = slider.value;
    // Dynamic accent colour based on value vs midpoint
    const ratio = (slider.value - slider.min) / (slider.max - slider.min);
    pill.style.color = ratio > 0.7 ? "var(--amber)" : "var(--accent)";
  };
  slider.addEventListener("input", update);
  update();
});

// ---- Tab switching ----
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById("tab-" + tab.dataset.tab).classList.add("active");
    clearError();
  });
});

// ---- NLP char counter ----
const nlpText = document.getElementById("nlp-text");
const nlpChars = document.getElementById("nlp-chars");
nlpText.addEventListener("input", () => { nlpChars.textContent = nlpText.value.length; });

// ---- Predict (manual) ----
document.getElementById("btn-predict").addEventListener("click", async () => {
  clearError();
  const features = getManualFeatures();
  await runPrediction("/predict", features, false);
});

// ---- Predict (NLP) ----
document.getElementById("btn-nlp").addEventListener("click", async () => {
  clearError();
  const text = nlpText.value.trim();
  if (!text) { showError("Please describe your project first."); return; }
  await runPrediction("/predict_nlp", { text }, true);
});

// ---- Core predict function ----
async function runPrediction(endpoint, payload, isNlp) {
  const btn = isNlp
    ? document.getElementById("btn-nlp")
    : document.getElementById("btn-predict");

  btn.classList.add("loading");
  btn.querySelector(".btn-text").textContent = "Estimating…";

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok || data.error) {
      showError(data.error || "Prediction failed. Please try again.");
      return;
    }

    renderResults(data, isNlp);
  } catch (e) {
    showError("Network error: " + e.message);
  } finally {
    btn.classList.remove("loading");
    btn.querySelector(".btn-text").textContent = isNlp ? "Analyse & Estimate" : "Estimate Duration";
  }
}

// ---- Read manual form values ----
function getManualFeatures() {
  const ids = [
    "num_employees", "complexity", "team_experience",
    "num_modules", "working_hours_day", "deadline_pressure", "requirement_clarity",
  ];
  const f = {};
  ids.forEach(id => { f[id] = parseFloat(document.getElementById(id).value); });
  return f;
}

// ---- Render results ----
function renderResults(data, isNlp) {
  document.getElementById("empty-state").style.display = "none";
  const content = document.getElementById("results-content");
  content.style.display = "block";

  // Re-trigger animation
  content.style.animation = "none";
  content.offsetHeight; // reflow
  content.style.animation = "";

  // Main values
  animateNumber(document.getElementById("res-days"), data.days);
  document.getElementById("res-weeks").textContent  = data.weeks;
  document.getElementById("res-months").textContent = data.months;

  // Confidence
  document.getElementById("conf-lo").textContent = data.low + "d";
  document.getElementById("conf-hi").textContent = data.high + "d";
  document.getElementById("conf-range-text").textContent =
    `Prediction range: ${data.low} – ${data.high} days`;

  // Confidence bar fill
  const range  = data.high - data.low;
  const filled = range > 0 ? 100 : 80;
  const markerPct = range > 0 ? ((data.days - data.low) / range) * 100 : 50;
  document.getElementById("conf-fill").style.width  = filled + "%";
  document.getElementById("conf-fill").style.left   = "0%";
  document.getElementById("conf-marker").style.left = Math.max(5, Math.min(95, markerPct)) + "%";

  // Feature importance (from server-side JSON injected into page or fetched)
  renderImportances();

  // NLP parsed features
  const parsedCard = document.getElementById("nlp-parsed-card");
  if (isNlp && data.parsed_features) {
    parsedCard.style.display = "block";
    renderParsedFeatures(data.parsed_features);
  } else {
    parsedCard.style.display = "none";
  }

  // Suggestions
  renderSuggestions(data.suggestions || []);
}

// ---- Animate number count-up ----
function animateNumber(el, target) {
  const duration = 800;
  const start    = Date.now();
  const from     = parseFloat(el.textContent) || 0;

  function tick() {
    const elapsed = Date.now() - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = (from + (target - from) * eased).toFixed(1);
    if (progress < 1) requestAnimationFrame(tick);
    else el.textContent = target;
  }
  requestAnimationFrame(tick);
}

// ---- Feature importances ----
let _importances = null;
async function renderImportances() {
  if (!_importances) {
    // Grab from embedded script tag or fetch
    const script = document.getElementById("importances-data");
    if (script) {
      _importances = JSON.parse(script.textContent);
    } else {
      // fallback: already loaded via template variable
      _importances = window.__importances__ || {};
    }
  }

  const container = document.getElementById("importance-bars");
  container.innerHTML = "";

  const entries = Object.entries(_importances).sort((a, b) => b[1] - a[1]);
  const max     = entries[0]?.[1] || 1;

  entries.forEach(([key, val]) => {
    const pct = (val / max) * 100;
    const row = document.createElement("div");
    row.className = "imp-row";
    row.innerHTML = `
      <span class="imp-name">${FEATURE_LABELS[key] || key}</span>
      <div class="imp-bar-outer">
        <div class="imp-bar-inner" style="width:0%" data-target="${pct}"></div>
      </div>
      <span class="imp-val">${(val * 100).toFixed(1)}%</span>
    `;
    container.appendChild(row);
  });

  // Animate bars
  setTimeout(() => {
    container.querySelectorAll(".imp-bar-inner").forEach(bar => {
      bar.style.width = bar.dataset.target + "%";
    });
  }, 50);
}

// ---- Parsed features ----
function renderParsedFeatures(features) {
  const grid = document.getElementById("parsed-grid");
  grid.innerHTML = "";
  Object.entries(features).forEach(([k, v]) => {
    const item = document.createElement("div");
    item.className = "parsed-item";
    item.innerHTML = `<span class="parsed-key">${FEATURE_LABELS[k] || k}</span><span class="parsed-val">${v}</span>`;
    grid.appendChild(item);
  });
}

// ---- Suggestions ----
function renderSuggestions(suggestions) {
  const ul = document.getElementById("suggestions-list");
  ul.innerHTML = "";
  suggestions.forEach(s => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="sug-icon">${s.icon}</span><span>${s.text}</span>`;
    ul.appendChild(li);
  });
}

// ---- Error helpers ----
function showError(msg) {
  const box = document.getElementById("error-box");
  box.textContent = "⚠ " + msg;
  box.style.display = "block";
}
function clearError() {
  const box = document.getElementById("error-box");
  box.style.display = "none";
  box.textContent   = "";
}

// ---- Init: pre-load importances from template ----
(function initImportances() {
  // Attempt to read from a data attribute set by Flask template
  const el = document.getElementById("__importances__");
  if (el) {
    try { _importances = JSON.parse(el.textContent); } catch(e) {}
  }
})();
