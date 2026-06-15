const state = { experiments: [], selectedSeries: 0, lastStatus: null };
const $ = id => document.getElementById(id);
const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, c => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#039;" }[c]));
const fmt = value => value == null || !Number.isFinite(Number(value)) ? "--" : Number(value).toPrecision(5);

function applyTheme(theme) {
  const selected = theme || localStorage.getItem("parameterScanTheme") || "light";
  localStorage.setItem("parameterScanTheme", selected);
  $("themeSelect").value = selected;
  const resolved = selected === "system" && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : selected;
  document.body.classList.toggle("theme-dark", resolved === "dark");
  renderPlot(state.lastStatus?.series || []);
}

async function api(url, options) {
  const response = await fetch(url, options);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `Request failed (${response.status})`);
  return value;
}

async function loadExperiments() {
  const data = await api("/api/experiments");
  state.experiments = data.experiments;
  $("qubitSelect").innerHTML = (data.qubits || []).map(qubit => `<option value="${escapeHtml(qubit)}">${escapeHtml(qubit)}</option>`).join("");
  if ([...$("qubitSelect").options].some(option => option.value === "q10")) $("qubitSelect").value = "q10";
  $("experimentCount").textContent = data.experiments.length;
  $("experimentList").innerHTML = data.experiments.map(item => `
    <label class="experiment-option">
      <input type="checkbox" value="${escapeHtml(item.script)}" ${item.preferred ? "checked" : ""}>
      <span><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(item.script)}</small></span>
    </label>
  `).join("");
}

function selectedExperiments() {
  return [...document.querySelectorAll("#experimentList input:checked")].map(input => input.value);
}

async function startRun() {
  const payload = {
    name: $("scanName").value.trim() || "live_scan",
    repetitions: Number($("repetitions").value || 1),
    interval_seconds: Number($("intervalSeconds").value || 0),
    continue_on_error: $("continueOnError").checked,
    save_full_results: $("saveFullResults").checked,
    qubit: $("qubitSelect").value,
    experiments: selectedExperiments(),
  };
  await api("/api/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  state.selectedSeries = 0;
  await refreshStatus();
}

async function stopRun() {
  await api("/api/stop", { method: "POST" });
  await refreshStatus();
}

async function refreshStatus() {
  try {
    const data = await api("/api/status");
    state.lastStatus = data;
    renderStatus(data);
  } catch (error) {
    showErrors([error.message]);
  }
}

function renderStatus(data) {
  const status = data.status?.status || (data.running ? "running" : "idle");
  $("statusText").textContent = status;
  $("statusDot").className = `dot ${data.running ? "running" : status === "failed" ? "failed" : ""}`;
  $("runButton").disabled = data.running;
  $("stopButton").disabled = !data.running;
  $("runPath").textContent = data.run_path || "No active run";
  $("recordCount").textContent = data.records?.length || 0;
  $("seriesCount").textContent = data.series?.length || 0;
  $("lastUpdate").textContent = data.status?.updated_at ? new Date(data.status.updated_at).toLocaleTimeString() : "--";
  renderSeriesSelect(data.series || []);
  renderPlot(data.series || []);
  renderLatest(data.series || []);
  renderAnalysis(data.series || []);
  renderTerminal(data);
  showErrors((data.errors || []).map(error => `${error.experiment_name}: ${error.error}`));
}

function renderSeriesSelect(series) {
  const select = $("seriesSelect");
  const previous = select.value;
  select.innerHTML = series.length ? series.map((item, index) => {
    const label = `${item.experiment_name} | ${item.qubit || "run"} | ${item.parameter}`;
    return `<option value="${index}">${escapeHtml(label)}</option>`;
  }).join("") : `<option>No parameters yet</option>`;
  if (previous && Number(previous) < series.length) select.value = previous;
  state.selectedSeries = Number(select.value || 0);
}

function renderLatest(series) {
  const latest = series.map(item => ({ item, point: item.points[item.points.length - 1] })).filter(entry => entry.point);
  $("latestCount").textContent = latest.length;
  $("latestGrid").innerHTML = latest.slice(-12).reverse().map(({ item, point }) => `
    <article class="latest-card">
      <span>${escapeHtml(item.experiment_name)} / ${escapeHtml(item.qubit || "run")}</span>
      <strong>${fmt(point.value)} ${escapeHtml(item.unit || "")}</strong>
      <small>${escapeHtml(item.parameter)} | cycle ${point.cycle}</small>
    </article>
  `).join("") || `<p class="mono">Values will appear after the first experiment finishes.</p>`;
}

function renderAnalysis(series) {
  if (!series.length) {
    $("bestVerdict").textContent = "Idle";
    $("analysisPanel").innerHTML = `<p class="mono">Waiting for fitted parameters.</p>`;
    return;
  }
  const interesting = [...series].filter(item => item.analysis?.count >= 2).sort((a, b) => {
    const av = Math.abs(a.analysis.drift_per_hour || 0) / (Math.abs(a.analysis.mean || 1));
    const bv = Math.abs(b.analysis.drift_per_hour || 0) / (Math.abs(b.analysis.mean || 1));
    return bv - av;
  });
  const top = interesting[0] || series[0];
  $("bestVerdict").textContent = top.analysis?.verdict || "collecting";
  const cards = interesting.slice(0, 6).map(item => {
    const a = item.analysis;
    const cls = a.verdict === "jump" || a.verdict === "drifting" ? "bad" : a.verdict === "watch" ? "warn" : "";
    return `<article class="analysis-item ${cls}">
      <strong>${escapeHtml(item.parameter)} on ${escapeHtml(item.qubit || "run")}: ${escapeHtml(a.verdict)}</strong>
      <p>std ${fmt(a.std)} ${escapeHtml(item.unit || "")}, drift ${fmt(a.drift_per_hour)} ${escapeHtml(item.unit || "")}/hour, relative std ${(100 * (a.relative_std || 0)).toPrecision(3)}%.</p>
    </article>`;
  }).join("");
  $("analysisPanel").innerHTML = cards || `<p class="mono">Need at least two points for drift analysis.</p>`;
}

function showErrors(errors) {
  $("errorPanel").classList.toggle("hidden", !errors.length);
  $("errorPanel").textContent = errors.join("\n");
}

function renderTerminal(data) {
  const output = $("terminalOutput");
  const wasAtBottom = output.scrollTop + output.clientHeight >= output.scrollHeight - 24;
  const terminalText = data.terminal_output || "Press Run to start a scan.";
  output.textContent = terminalText;
  $("terminalState").textContent = data.running ? "running" : (data.status?.status || "idle");
  if (wasAtBottom || data.running) output.scrollTop = output.scrollHeight;
}

function renderPlot(series) {
  const canvas = $("trendPlot");
  const context = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const ratio = devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * ratio));
  canvas.height = Math.max(1, Math.round(rect.height * ratio));
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);
  if (!series.length) {
    context.fillStyle = getComputedStyle(document.body).getPropertyValue("--muted").trim();
    context.font = "13px system-ui";
    context.fillText("Waiting for first fitted parameter...", 22, 32);
    $("plotMeta").innerHTML = "";
    return;
  }
  const selected = series[Math.min(state.selectedSeries, series.length - 1)];
  const points = selected.points || [];
  $("plotMeta").innerHTML = `<span>${points.length} point${points.length === 1 ? "" : "s"}</span><span>${escapeHtml(selected.unit || "unitless")}</span><span>${escapeHtml(selected.analysis?.verdict || "collecting")}</span>`;
  if (points.length < 1) return;
  const values = points.map(point => Number(point.value)).filter(Number.isFinite);
  const xs = points.map((point, index) => point.timestamp_epoch ? (point.timestamp_epoch - points[0].timestamp_epoch) / 3600 : index);
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...values), y1 = Math.max(...values);
  if (x0 === x1) x1 = x0 + 1;
  if (y0 === y1) { y0 -= 1; y1 += 1; }
  const pad = { l: 74, r: 20, t: 18, b: 42 };
  const sx = x => pad.l + (x - x0) / (x1 - x0) * (rect.width - pad.l - pad.r);
  const sy = y => rect.height - pad.b - (y - y0) / (y1 - y0) * (rect.height - pad.t - pad.b);
  const styles = getComputedStyle(document.body);
  const lineColor = styles.getPropertyValue("--line").trim();
  const mutedColor = styles.getPropertyValue("--muted").trim();
  const accentColor = styles.getPropertyValue("--accent").trim();
  const badColor = styles.getPropertyValue("--bad").trim();
  context.strokeStyle = lineColor;
  context.fillStyle = mutedColor;
  context.font = "10px system-ui";
  for (let i = 0; i <= 5; i++) {
    const x = pad.l + i * (rect.width - pad.l - pad.r) / 5;
    const y = pad.t + i * (rect.height - pad.t - pad.b) / 5;
    context.beginPath();
    context.moveTo(x, pad.t); context.lineTo(x, rect.height - pad.b);
    context.moveTo(pad.l, y); context.lineTo(rect.width - pad.r, y);
    context.stroke();
    context.fillText((x0 + i * (x1 - x0) / 5).toPrecision(3), x - 10, rect.height - 20);
    context.fillText((y1 - i * (y1 - y0) / 5).toPrecision(4), 8, y + 3);
  }
  context.strokeStyle = accentColor;
  context.lineWidth = 1.8;
  context.beginPath();
  points.forEach((point, index) => {
    const x = sx(xs[index]), y = sy(Number(point.value));
    index ? context.lineTo(x, y) : context.moveTo(x, y);
  });
  context.stroke();
  points.forEach((point, index) => {
    context.fillStyle = String(point.success) === "False" || String(point.success) === "false" ? badColor : accentColor;
    context.beginPath();
    context.arc(sx(xs[index]), sy(Number(point.value)), 3, 0, Math.PI * 2);
    context.fill();
  });
  context.fillStyle = mutedColor;
  context.fillText("hours since scan start", rect.width / 2 - 45, rect.height - 7);
  context.save();
  context.translate(12, rect.height / 2 + 40);
  context.rotate(-Math.PI / 2);
  context.fillText(`${selected.parameter}${selected.unit ? " [" + selected.unit + "]" : ""}`, 0, 0);
  context.restore();
}

$("runButton").onclick = () => startRun().catch(error => showErrors([error.message]));
$("stopButton").onclick = () => stopRun().catch(error => showErrors([error.message]));
$("seriesSelect").onchange = event => { state.selectedSeries = Number(event.target.value || 0); renderPlot(state.lastStatus?.series || []); };
$("themeSelect").onchange = event => applyTheme(event.target.value);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (($("themeSelect").value || localStorage.getItem("parameterScanTheme")) === "system") applyTheme("system");
});
window.addEventListener("resize", () => renderPlot(state.lastStatus?.series || []));

applyTheme(localStorage.getItem("parameterScanTheme") || "light");
loadExperiments().catch(error => showErrors([error.message]));
refreshStatus();
setInterval(refreshStatus, 2000);
