const COLORS = { 1: "#f4d34d", 2: "#54d5ff", 3: "#df67ff", 4: "#70ee88" };
const canvas = document.querySelector("#scope");
const ctx = canvas.getContext("2d");
const emptyState = document.querySelector("#emptyState");
const connection = document.querySelector("#connection");
const statusText = document.querySelector("#statusText");
const runButton = document.querySelector("#runButton");
const singleButton = document.querySelector("#singleButton");
const intervalSelect = document.querySelector("#interval");
const maxPointsSelect = document.querySelector("#maxPoints");
let running = true;
let fetching = false;
let timer = null;
let traces = [];
let lastCapture = null;

function selectedChannels() {
  return [...document.querySelectorAll(".channel input:checked")].map((el) => Number(el.value));
}

function setStatus(kind, message) {
  connection.className = `connection ${kind}`;
  statusText.textContent = message;
}

function engineering(value, unit) {
  if (value == null || !Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  const scales = [[1e9, "G"], [1e6, "M"], [1e3, "k"], [1, ""], [1e-3, "m"], [1e-6, "µ"], [1e-9, "n"]];
  const scale = scales.find(([factor]) => abs >= factor * .8) || scales.at(-1);
  const rendered = value / scale[0];
  return `${rendered.toFixed(Math.abs(rendered) >= 100 ? 1 : Math.abs(rendered) >= 10 ? 2 : 3)} ${scale[1]}${unit}`;
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  draw();
}

function draw() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  ctx.clearRect(0, 0, width, height);
  if (!traces.length) return;
  const allValues = traces.flatMap((trace) => trace.values).filter(Number.isFinite);
  let yMin = Math.min(...allValues);
  let yMax = Math.max(...allValues);
  const padding = Math.max((yMax - yMin) * .12, 1e-9);
  yMin -= padding;
  yMax += padding;
  const left = 54, right = 18, top = 22, bottom = 34;
  const plotW = width - left - right, plotH = height - top - bottom;

  ctx.font = "10px ui-monospace, Consolas, monospace";
  ctx.fillStyle = "#5d6b69";
  ctx.strokeStyle = "#253233";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 8; i++) {
    const x = left + (i / 8) * plotW;
    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + plotH); ctx.stroke();
  }
  for (let i = 0; i <= 6; i++) {
    const y = top + (i / 6) * plotH;
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + plotW, y); ctx.stroke();
    const label = engineering(yMax - (i / 6) * (yMax - yMin), "V");
    ctx.fillText(label, 7, y + 3);
  }

  for (const trace of traces) {
    const values = trace.values;
    const step = Math.max(1, Math.floor(values.length / Math.max(plotW * 2, 1)));
    ctx.beginPath();
    ctx.strokeStyle = COLORS[trace.channel];
    ctx.lineWidth = 1.4;
    ctx.shadowColor = COLORS[trace.channel];
    ctx.shadowBlur = 5;
    for (let i = 0; i < values.length; i += step) {
      const x = left + (i / Math.max(values.length - 1, 1)) * plotW;
      const y = top + (1 - (values[i] - yMin) / (yMax - yMin)) * plotH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }
}

function renderCapture(data) {
  traces = data.traces;
  lastCapture = data.captured_at * 1000;
  emptyState.classList.add("hidden");
  setStatus("connected", "Live");
  document.querySelector("#identity").textContent = data.identity;
  const longest = traces.reduce((best, trace) => trace.values.length > best.values.length ? trace : best, traces[0]);
  const start = longest.x_zero;
  const end = start + (longest.values.length - 1) * longest.x_increment;
  document.querySelector("#windowLabel").textContent = `${engineering(end - start, "s")} window`;
  document.querySelector("#pointsLabel").textContent = `${longest.values.length.toLocaleString()} points`;
  document.querySelector("#xStart").textContent = engineering(start, "s");
  document.querySelector("#xEnd").textContent = engineering(end, "s");
  document.querySelector("#legend").innerHTML = traces.map((trace) =>
    `<span class="legend-row"><b style="color:${COLORS[trace.channel]}">CH${trace.channel}</b>${engineering(trace.metrics.pk_pk, "Vpp")}</span>`
  ).join("");
  document.querySelector("#measurements").innerHTML = traces.map((trace) => `
    <div class="measurement"><small style="color:${COLORS[trace.channel]}">CH${trace.channel} PK–PK</small><strong>${engineering(trace.metrics.pk_pk, "V")}</strong></div>
    <div class="measurement"><small>RMS</small><strong>${engineering(trace.metrics.rms, "V")}</strong></div>
    <div class="measurement"><small>MIN / MAX</small><strong>${engineering(trace.metrics.min, "V")} <em>/</em> ${engineering(trace.metrics.max, "V")}</strong></div>
    <div class="measurement"><small>SAMPLE RATE</small><strong>${engineering(trace.metrics.sample_rate, "Sa/s")}</strong></div>
  `).join("");
  draw();
}

async function acquire() {
  if (fetching) return;
  const channels = selectedChannels();
  if (!channels.length) {
    setStatus("error", "Select a channel");
    return;
  }
  fetching = true;
  const controller = new AbortController();
  const abortTimer = setTimeout(() => controller.abort(), 10000);
  try {
    const response = await fetch(`/api/waveforms?channels=${channels.join(",")}&max_points=${maxPointsSelect.value}`, { cache: "no-store", signal: controller.signal });
    const data = await response.json();
    if (response.status === 503 && data.error === "Waiting for the first instrument capture") {
      connection.className = "connection";
      statusText.textContent = "Connecting";
      emptyState.classList.remove("hidden");
      emptyState.querySelector("h2").textContent = "Connecting to oscilloscope";
      emptyState.querySelector("p").textContent = "Waiting for the first VISA waveform capture…";
      return;
    }
    if (!response.ok) throw new Error(data.error || `Instrument error ${response.status}`);
    renderCapture(data);
  } catch (error) {
    setStatus("error", "Scope unavailable");
    emptyState.classList.remove("hidden");
    emptyState.querySelector("h2").textContent = "Instrument unavailable";
    emptyState.querySelector("p").textContent = `${error.message}. Retrying automatically…`;
  } finally {
    clearTimeout(abortTimer);
    fetching = false;
  }
}

function schedule() {
  clearTimeout(timer);
  if (!running) return;
  timer = setTimeout(async () => {
    await acquire();
    schedule();
  }, Number(intervalSelect.value));
}

function toggleRun() {
  running = !running;
  runButton.classList.toggle("paused", !running);
  runButton.querySelector("span:last-child").textContent = running ? "Pause" : "Run";
  if (running) { acquire(); schedule(); } else clearTimeout(timer);
}

runButton.addEventListener("click", toggleRun);
singleButton.addEventListener("click", acquire);
intervalSelect.addEventListener("change", schedule);
maxPointsSelect.addEventListener("change", acquire);
document.querySelectorAll(".channel input").forEach((input) => input.addEventListener("change", acquire));
window.addEventListener("resize", resizeCanvas);
setInterval(() => {
  document.querySelector("#captureAge").textContent = lastCapture ? `Captured ${Math.max(0, (Date.now() - lastCapture) / 1000).toFixed(1)} s ago` : "No capture";
}, 250);
resizeCanvas();
acquire();
schedule();
