const state = {
  timer: null,
  inFlight: false,
  windowSeconds: 15 * 60,
};
document.querySelectorAll("[data-hub-link]").forEach(link => { link.href = `${location.protocol}//${location.hostname}:8890/`; });

const WINDOW_LABELS = new Map([
  [15 * 60, "15 min"],
  [60 * 60, "1 hour"],
  [24 * 60 * 60, "24 hours"],
  [7 * 24 * 60 * 60, "1 week"],
]);

const els = {
  liveDot: document.querySelector("#liveDot"),
  liveState: document.querySelector("#liveState"),
  connectionText: document.querySelector("#connectionText"),
  jobTitle: document.querySelector("#jobTitle"),
  jobGlyph: document.querySelector("#jobGlyph"),
  jobStatus: document.querySelector("#jobStatus"),
  jobType: document.querySelector("#jobType"),
  jobDescription: document.querySelector("#jobDescription"),
  temperatureTitle: document.querySelector("#temperatureTitle"),
  temperatureWindow: document.querySelector("#temperatureWindow"),
  temperatureGrid: document.querySelector("#temperatureGrid"),
  lastPoll: document.querySelector("#lastPoll"),
  openQms: document.querySelector("#openQms"),
};

function setLiveState(kind, heading, text) {
  els.liveDot.className = `live-dot ${kind || ""}`.trim();
  els.liveState.textContent = heading;
  els.connectionText.textContent = text;
}

function formatTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function render(payload) {
  const jobPayload = payload.jobs || {};
  const temperature = payload.temperature || {};
  const jobs = jobPayload.jobs || [];
  const openQms = jobPayload.open_qms || [];
  const job = jobs[0];

  if (payload.qpu_live) {
    setLiveState("live", "Live", "Temperature acquisition can reach the QPU.");
  } else {
    setLiveState("offline", "Not live", "Temperature acquisition cannot reach the QPU.");
  }

  if (jobPayload.has_active_jobs && job) {
    els.jobGlyph.className = "job-orbit running";
    els.jobTitle.textContent = job.id || "Active job";
    els.jobStatus.textContent = job.status || "Running";
    els.jobType.textContent = job.is_simulation ? "Simulation" : "Hardware";
    els.jobDescription.textContent = job.description || "No description reported.";
  } else {
    els.jobGlyph.className = "job-orbit waiting";
    els.jobTitle.textContent = "No active job";
    els.jobStatus.textContent = "-";
    els.jobType.textContent = "-";
    els.jobDescription.textContent = payload.jobs_error || "No active QOP job reported.";
  }

  renderTemperature(temperature);
  els.lastPoll.textContent = `Last checked: ${formatTime(payload.polled_at)}`;
  els.openQms.textContent = `Open QMs: ${openQms.length}`;
}

function renderTemperature(temperature) {
  const latest = temperature.latest || {};
  const entries = Object.entries(latest);
  const history = temperature.history || [];
  updateWindowButtons();
  if (temperature.connected) {
    els.temperatureTitle.textContent = "Scanning";
  } else {
    els.temperatureTitle.textContent = "Disconnected";
  }
  if (!entries.length) {
    els.temperatureGrid.innerHTML = `<p class="muted">${escapeHtml(
      temperature.error || "Waiting for the first temperature sample."
    )}</p>`;
    return;
  }
  els.temperatureGrid.innerHTML = entries.map(([name, value]) => `
    <article class="temperature-sensor">
      <div class="sensor-top">
        <div class="sensor-name">${escapeHtml(name)}</div>
        <div class="sensor-value">${Number(value).toFixed(1)} C</div>
      </div>
      ${temperatureSparkline(name, history, state.windowSeconds)}
    </article>
  `).join("");
}

function updateWindowButtons() {
  els.temperatureWindow.querySelectorAll("button").forEach((button) => {
    const value = Number(button.dataset.windowSeconds);
    button.classList.toggle("active", value === state.windowSeconds);
  });
}

function sampleX(sample, fallback) {
  const time = Date.parse(sample.time || "");
  if (Number.isFinite(time)) return time / 1000;
  const elapsed = Number(sample.elapsed_seconds);
  return Number.isFinite(elapsed) ? elapsed : fallback;
}

function temperatureSparkline(name, history, windowSeconds) {
  const points = history
    .map((sample, index) => ({
      x: sampleX(sample, index),
      y: Number((sample.temperatures || {})[name]),
    }))
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));

  if (points.length < 2) {
    return '<div class="sparkline empty-line">Collecting trend...</div>';
  }

  const width = 260;
  const height = 92;
  const leftPad = 42;
  const rightPad = 8;
  const topPad = 10;
  const bottomPad = 16;
  const minX = Math.min(...points.map((point) => point.x));
  const maxX = Math.max(...points.map((point) => point.x));
  const minYRaw = Math.min(...points.map((point) => point.y));
  const maxYRaw = Math.max(...points.map((point) => point.y));
  const yPadding = Math.max(0.2, (maxYRaw - minYRaw) * 0.18);
  const minY = minYRaw - yPadding;
  const maxY = maxYRaw + yPadding;
  const xSpan = Math.max(1, maxX - minX);
  const ySpan = Math.max(0.1, maxY - minY);
  const path = points.map((point, index) => {
    const x = leftPad + ((point.x - minX) / xSpan) * (width - leftPad - rightPad);
    const y = height - bottomPad - ((point.y - minY) / ySpan) * (height - topPad - bottomPad);
    return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(" ");
  const ticks = [maxYRaw, (minYRaw + maxYRaw) / 2, minYRaw].map((value) => {
    const y = height - bottomPad - ((value - minY) / ySpan) * (height - topPad - bottomPad);
    return { value, y };
  });
  const tickMarkup = ticks.map((tick) => `
    <path class="grid-line" d="M ${leftPad} ${tick.y.toFixed(1)} H ${width - rightPad}"></path>
    <text class="axis-label" x="${leftPad - 6}" y="${(tick.y + 4).toFixed(1)}" text-anchor="end">${tick.value.toFixed(1)}</text>
  `).join("");
  const label = WINDOW_LABELS.get(windowSeconds) || "selected range";

  return `
    <div class="sparkline" aria-label="${escapeHtml(name)} temperature trend for ${escapeHtml(label)}">
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-hidden="true">
        ${tickMarkup}
        <path class="axis-line" d="M ${leftPad} ${topPad} V ${height - bottomPad}"></path>
        <path class="trend-line" d="${path}"></path>
      </svg>
    </div>
  `;
}

async function refresh() {
  if (state.inFlight) return;
  state.inFlight = true;
  setLiveState("", "Checking", "Polling QOP...");

  try {
    const response = await fetch(`/api/status?window_seconds=${state.windowSeconds}`, { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || `Request failed with ${response.status}`);
    }
    render(payload);
  } catch (error) {
    setLiveState("offline", "Not live", "Cannot reach QOP from this dashboard.");
    els.jobTitle.textContent = "No job data";
    els.jobStatus.textContent = "Disconnected";
    els.jobType.textContent = "-";
    els.jobDescription.textContent = error.message;
    els.lastPoll.textContent = `Last checked: ${formatTime(new Date().toISOString())}`;
    els.openQms.textContent = "Open QMs: -";
  } finally {
    state.inFlight = false;
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function schedule() {
  if (state.timer) window.clearInterval(state.timer);
  state.timer = window.setInterval(refresh, 3000);
}

els.temperatureWindow.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-window-seconds]");
  if (!button) return;
  const nextWindow = Number(button.dataset.windowSeconds);
  if (!Number.isFinite(nextWindow) || nextWindow === state.windowSeconds) return;
  state.windowSeconds = nextWindow;
  updateWindowButtons();
  refresh();
});

schedule();
refresh();
