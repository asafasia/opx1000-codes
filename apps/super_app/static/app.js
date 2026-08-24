const grid = document.querySelector("#appGrid");
const systemDot = document.querySelector("#systemDot");
const systemText = document.querySelector("#systemText");
const lastChecked = document.querySelector("#lastChecked");
const refreshButton = document.querySelector("#refreshButton");
const serviceCount = document.querySelector("#serviceCount");
const isDesktop = new URLSearchParams(window.location.search).has("desktop");

const icons = {
  "data-review": `<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M5 24V11m7 13V6m7 18v-9m7 9V9"/><path d="M3 27h26"/></svg>`,
  "lab-monitor": `<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M3 17h5l3-8 5 16 4-11 3 3h6"/><circle cx="16" cy="16" r="13"/></svg>`,
  "fridge-monitor": `<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M14 5a3 3 0 0 1 6 0v13.4a6 6 0 1 1-6 0z"/><path d="M17 9v12"/></svg>`,
  "oscilloscope": `<svg viewBox="0 0 32 32" aria-hidden="true"><rect x="3" y="6" width="26" height="19" rx="2"/><path d="M6 17h4l3-6 5 12 4-8 4 2"/><path d="M12 28h8"/></svg>`,
  "profile-studio": `<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M7 4h18v24H7z"/><path d="M11 10h10M11 16h10M11 22h6"/><circle cx="23" cy="22" r="4"/></svg>`,
  "parameter-sweep": `<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M4 25c6 0 6-18 12-18s6 18 12 18"/><path d="M4 7v18h24"/></svg>`,
  "wiki": `<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M6 5h8c2 0 3 1 3 3v19c0-2-1-3-3-3H6z"/><path d="M26 5h-8c-1 0-1 .5-1 1.5V27c0-2 1-3 3-3h6z"/></svg>`,
};

const submodules = {
  "data-review": ["Runs", "Figures", "AI review", "Trends"],
  "lab-monitor": ["QOP jobs", "Temperature", "History"],
  "fridge-monitor": ["Mixing chamber", "Still", "Pressure", "History"],
  "oscilloscope": ["Live waveform", "CH1–CH4", "Acquisition", "Timing"],
  "profile-studio": ["Qubits", "Pulses", "Connectivity", "Validation"],
  "parameter-sweep": ["Configure", "Live metrics", "Variance", "Drift"],
  "wiki": ["Protocols", "Hardware", "Calibration docs"],
};

function appUrl(port) {
  const host = window.location.hostname || "127.0.0.1";
  return `${window.location.protocol}//${host}:${port}/${isDesktop ? "?desktop=1" : ""}`;
}

function moduleUrl(app) {
  const wrappers = {
    "fridge-monitor": "/fridge.html",
    "oscilloscope": "/oscilloscope.html",
  };
  if (wrappers[app.id]) {
    return `${wrappers[app.id]}${isDesktop ? "?desktop=1" : ""}`;
  }
  return app.href || appUrl(app.port);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function card(app, index) {
  const hasError = !app.running && app.error && app.error !== "Starting…";
  const status = app.running ? "Ready" : (hasError ? "Issue" : "Starting");
  const disabled = app.running ? "" : `aria-disabled="true"`;
  const error = hasError
    ? `<p class="app-error" title="${escapeHtml(app.error)}">${escapeHtml(app.error)}</p>`
    : "";
  const innerModules = (submodules[app.id] || [])
    .map(module => `<span>${escapeHtml(module)}</span>`)
    .join("");
  return `
    <article class="app-card card-${index + 1} ${app.running ? "is-ready" : "is-waiting"}">
      <div class="module-rail">
        <span class="module-index">${String(index + 1).padStart(2, "0")}</span>
        <span class="app-icon">${icons[app.id] || ""}</span>
      </div>
      <div class="module-copy">
        <div class="module-heading">
          <p class="eyebrow">${app.eyebrow}</p>
          <span class="status"><i></i>${status}</span>
        </div>
        <h3>${app.name}</h3>
        <p class="description">${app.description}</p>
      </div>
      <div class="module-tools">
        <div class="card-meta"><span>${app.port ? "LOCAL SERVICE" : "KNOWLEDGE BASE"}</span><span>${app.port ? `TCP / ${app.port}` : "INDEX / MD"}</span></div>
        <p class="submodule-label">Submodules</p>
        <div class="submodule-list">${innerModules}</div>
        ${error}
      </div>
      <a class="module-action" href="${moduleUrl(app)}" ${disabled}>
        <span>${app.running ? app.action : "Waiting for app"}</span>
        <span class="arrow">${isDesktop ? "→" : "↗"}</span>
      </a>
    </article>`;
}

async function refresh() {
  refreshButton.disabled = true;
  try {
    const response = await fetch("/api/apps", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    const wiki = {
      id: "wiki", name: "Repository Wiki", eyebrow: "Learn",
      description: "Search and read every Markdown document across the repository.",
      action: "Browse documentation", href: `/wiki.html${isDesktop ? "?desktop=1" : ""}`, running: true, error: null,
    };
    grid.innerHTML = [...payload.apps, wiki].map(card).join("");
    const ready = payload.apps.filter(app => app.running).length;
    serviceCount.textContent = `${ready} / ${payload.apps.length}`;
    systemDot.classList.toggle("ready", ready === payload.apps.length);
    systemText.textContent = ready === payload.apps.length
      ? "All systems ready"
      : `${ready} of ${payload.apps.length} apps ready`;
    lastChecked.textContent = `Checked ${new Date(payload.checked_at * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"})}`;
  } catch (error) {
    serviceCount.textContent = "-- / 6";
    systemDot.classList.remove("ready");
    systemText.textContent = "Hub status unavailable";
    lastChecked.textContent = error.message;
  } finally {
    refreshButton.disabled = false;
  }
}

refreshButton.addEventListener("click", refresh);
refresh();
window.setInterval(refresh, 4000);
