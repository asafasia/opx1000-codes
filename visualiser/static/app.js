const state = { experiments: [], filtered: [], selected: null, detail: null, calibrationType: null, qubit: "all", tab: "figures", figureIndex: 0, plotData: null, plotResult: 0 };
const $ = (id) => document.getElementById(id);
const escapeHtml = (value = "") => String(value).replace(/[&<>"']/g, c => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#039;" }[c]));
const formatBytes = n => n == null ? "unavailable" : n < 1024 ? `${n} B` : n < 1048576 ? `${(n/1024).toFixed(1)} KB` : `${(n/1048576).toFixed(1)} MB`;
let refreshInProgress = false;

async function api(url) {
  const response = await fetch(url);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `Request failed (${response.status})`);
  return value;
}

async function loadDates() {
  setStatus("Indexing archive");
  try {
    const { dates } = await api("/api/dates");
    $("dateSelect").innerHTML = dates.map(d => `<option value="${d}">${d}</option>`).join("");
    if (dates.length) await loadExperiments(dates[0]);
    else showGlobalError("No dated experiments were found under data/.");
    setStatus("Live archive");
  } catch (error) { showGlobalError(error.message); setStatus("Connection error"); }
}

async function refreshLiveData() {
  if (refreshInProgress) return;
  refreshInProgress = true;
  try {
    const currentDate = $("dateSelect").value;
    const selectedId = state.selected?.id;
    const [{ dates }, data] = await Promise.all([
      api("/api/dates"),
      api(`/api/experiments?date=${encodeURIComponent(currentDate)}`)
    ]);
    $("dateSelect").innerHTML = dates.map(date => `<option value="${date}" ${date === currentDate ? "selected" : ""}>${date}</option>`).join("");
    state.experiments = data.experiments;
    if (!state.experiments.some(experiment => experiment.type === state.calibrationType)) {
      state.calibrationType = state.experiments.find(experiment => experiment.kind === "calibration")?.type || state.experiments[0]?.type || null;
      state.qubit = "all";
    }
    applyFilters();
    if (selectedId && state.experiments.some(experiment => experiment.id === selectedId)) {
      state.selected = state.experiments.find(experiment => experiment.id === selectedId);
      applyFilters();
    } else if (state.filtered.length) {
      await selectLatestRun();
    }
    setStatus("Live archive");
  } catch (error) {
    setStatus("Refresh failed");
    console.warn("Automatic refresh failed", error);
  } finally {
    refreshInProgress = false;
  }
}

async function loadExperiments(date) {
  $("experimentList").innerHTML = `<div class="empty-state"><p>Scanning ${escapeHtml(date)}...</p></div>`;
  state.selected = null; state.detail = null;
  $("detailView").classList.add("hidden"); $("emptyState").classList.remove("hidden");
  try {
    const data = await api(`/api/experiments?date=${encodeURIComponent(date)}`);
    state.experiments = data.experiments;
    const firstCalibration = state.experiments.find(e => e.kind === "calibration");
    state.calibrationType = firstCalibration?.type || (state.experiments[0]?.type ?? null);
    state.qubit = "all";
    $("dateBadge").textContent = date;
    applyFilters();
    await selectLatestRun();
    if (data.errors.length) console.warn("Archive scan warnings", data.errors);
  } catch (error) { showGlobalError(error.message); }
}

async function selectLatestRun() {
  const latest = state.filtered[0];
  if (latest) await selectExperiment(latest.id);
}

function applyFilters() {
  const q = $("searchInput").value.trim().toLowerCase();
  renderCalibrationNav();
  renderQubitNav();
  state.filtered = state.experiments.filter(e =>
    e.type === state.calibrationType
    && (state.qubit === "all" || (e.qubits || []).includes(state.qubit))
    && `${e.name} ${e.type} ${(e.qubits || []).join(" ")} ${e.path}`.toLowerCase().includes(q)
  );
  $("resultCount").textContent = `${state.filtered.length} run${state.filtered.length === 1 ? "" : "s"}`;
  $("experimentList").innerHTML = state.filtered.length ? state.filtered.map(e => `
    <article class="experiment-card ${state.selected?.id === e.id ? "active" : ""}" data-id="${escapeHtml(e.id)}">
      <div class="experiment-time">${escapeHtml(e.time.slice(0,5))}</div>
      <div><h3>${escapeHtml((e.qubits || []).join(", ") || e.name)}</h3><p>${escapeHtml(e.type)} | ${escapeHtml(e.path)}</p></div>
    </article>`).join("") : `<div class="empty-state"><p>No experiments match this filter.</p></div>`;
  document.querySelectorAll(".experiment-card").forEach(card => card.onclick = () => selectExperiment(card.dataset.id));
}

function renderCalibrationNav() {
  const counts = new Map();
  state.experiments.forEach(e => counts.set(e.type, (counts.get(e.type) || 0) + 1));
  $("calibrationNav").innerHTML = [...counts.entries()].map(([type, count]) =>
    `<button class="side-tab ${state.calibrationType === type ? "active" : ""}" data-type="${escapeHtml(type)}"><span>${escapeHtml(type)}</span><small>${count}</small></button>`
  ).join("");
  document.querySelectorAll(".side-tab[data-type]").forEach(button => button.onclick = async () => {
    state.calibrationType = button.dataset.type;
    state.qubit = "all";
    state.selected = null;
    state.detail = null;
    $("detailView").classList.add("hidden");
    $("emptyState").classList.remove("hidden");
    applyFilters();
    await selectLatestRun();
  });
}

function renderQubitNav() {
  const qubits = [...new Set(state.experiments.filter(e => e.type === state.calibrationType).flatMap(e => e.qubits || []))].sort();
  $("qubitSection").classList.toggle("hidden", !qubits.length);
  $("qubitNav").innerHTML = qubits.length ? [
    `<button class="side-tab ${state.qubit === "all" ? "active" : ""}" data-qubit="all"><span>All qubits</span></button>`,
    ...qubits.map(qubit => `<button class="side-tab ${state.qubit === qubit ? "active" : ""}" data-qubit="${escapeHtml(qubit)}"><span>${escapeHtml(qubit)}</span></button>`)
  ].join("") : "";
  document.querySelectorAll(".side-tab[data-qubit]").forEach(button => button.onclick = async () => {
    state.qubit = button.dataset.qubit;
    applyFilters();
    await selectLatestRun();
  });
}

async function selectExperiment(id) {
  state.selected = state.experiments.find(e => e.id === id);
  state.tab = "figures";
  state.figureIndex = 0;
  state.plotData = null;
  state.plotResult = 0;
  applyFilters();
  $("emptyState").classList.add("hidden"); $("detailView").classList.remove("hidden");
  $("tabContent").innerHTML = `<div class="empty-state"><p>Loading experiment files...</p></div>`;
  setStatus("Loading experiment");
  try {
    state.detail = await api(`/api/experiment?path=${encodeURIComponent(id)}`);
    populateHero();
    renderTab();
    setStatus("Live archive");
  } catch (error) {
    $("tabContent").innerHTML = ""; showDetailErrors([error.message]); setStatus("Read error");
  }
}

function populateHero() {
  const s = state.detail.summary;
  $("kindTag").textContent = s.kind;
  $("dateTag").textContent = s.date;
  $("experimentName").textContent = s.name;
  $("experimentPath").textContent = s.path;
  $("runTime").textContent = s.time;
  $("figureCount").textContent = state.detail.figures.length;
  showDetailErrors(state.detail.errors);
}

async function renderTab() {
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.tab === state.tab));
  if (state.tab === "interactive" && !state.plotData) {
    $("tabContent").innerHTML = `<div class="empty-state"><p>Loading NPZ plotting data...</p></div>`;
    try {
      state.plotData = await api(`/api/npz-plot?path=${encodeURIComponent(state.selected.id)}`);
    } catch (error) {
      $("tabContent").innerHTML = `<div class="error-panel">${escapeHtml(error.message)}</div>`;
      return;
    }
  }
  const renderers = { overview: renderOverview, figures: renderFigures, interactive: renderInteractivePlot, data: renderData, calibrations: renderCalibrations };
  $("tabContent").innerHTML = renderers[state.tab]();
  bindFigures();
  if (state.tab === "interactive") bindInteractivePlot();
}

const section = (title, count, body) => `<div class="section-title"><h3>${title}</h3><span>${count}</span></div>${body}`;
const card = (item, body) => `<article class="data-card"><div class="card-head"><strong>${escapeHtml(item.relative || item.name)}</strong><a href="${item.url}" target="_blank">open</a></div>${body}</article>`;
const jsonCard = item => card(item, item.error ? `<pre class="text-view">${escapeHtml(item.error)}</pre>` : `<pre class="json-view">${escapeHtml(JSON.stringify(item.value, null, 2))}</pre>`);
const textCard = item => card(item, `<pre class="text-view">${escapeHtml(item.error || item.value || "Empty file")}</pre>`);
const artifactRows = items => `<div class="artifact-list">${items.map(i => `<div class="artifact"><a href="${i.url}" target="_blank">${escapeHtml(i.relative || i.name)}</a><span>${formatBytes(i.size)} | ${escapeHtml(i.extension || "file")}</span></div>`).join("") || `<p class="mono">No files in this section.</p>`}</div>`;

function renderOverview() {
  const d = state.detail;
  return section("Metadata & Parameters", `${d.metadata.length} JSON files`, `<div class="card-grid">${d.metadata.map(jsonCard).join("") || `<p class="mono">No JSON metadata found.</p>`}</div>`)
    + section("Notes & Reports", `${d.text.length} text files`, `<div class="card-grid">${d.text.map(textCard).join("") || `<p class="mono">No reports found.</p>`}</div>`)
    + section("Quick Figures", `${d.figures.length} figures`, figureMarkup(d.figures.slice(0, 4)));
}

function activeQubits() {
  const profile = state.detail?.metadata.find(item => item.relative === "profile/profile.json")?.value;
  return Array.isArray(profile?.active_qubits) ? profile.active_qubits.map(String) : (state.selected?.qubits || []);
}
function figureLabel(item, index) {
  const qubits = activeQubits();
  if (qubits.length === state.detail.figures.length) return qubits[index];
  if (qubits.length === 1) return qubits[0];
  return item.name.replace(/\.[^.]+$/, "");
}
function figureMarkup(items) {
  return `<div class="figure-grid">${items.map(i => `<article class="figure-card"><div class="card-head"><strong>${escapeHtml(i.relative)}</strong><a href="${i.url}" target="_blank">open original</a></div><div class="figure-stage" data-src="${i.url}" data-name="${escapeHtml(i.relative)}"><img loading="lazy" src="${i.url}" alt="${escapeHtml(i.name)}"></div></article>`).join("") || `<p class="mono">No generated figures found.</p>`}</div>`;
}
function renderFigures() {
  const figures = state.detail.figures;
  if (!figures.length) return section("Generated Figures", "0 images", `<p class="mono">No generated figures found.</p>`);
  const index = Math.min(state.figureIndex, figures.length - 1);
  const figure = figures[index];
  const selectors = figures.map((item, itemIndex) => `<button class="qubit-figure-tab ${itemIndex === index ? "active" : ""}" data-figure-index="${itemIndex}">${escapeHtml(figureLabel(item, itemIndex))}</button>`).join("");
  return `<div class="fit-tab-panel"><div class="figure-toolbar"><div><span class="section-label">Qubit / Figure</span><div class="qubit-figure-tabs">${selectors}</div></div><a href="${figure.url}" target="_blank">Open original</a></div>
    <article class="focused-figure"><div class="focused-figure-stage figure-stage" data-src="${figure.url}" data-name="${escapeHtml(figure.relative)}"><img src="${figure.url}" alt="${escapeHtml(figure.name)}"></div><div class="focused-figure-caption"><strong>${escapeHtml(figureLabel(figure, index))}</strong><span>${escapeHtml(figure.relative)}</span></div></article></div>`;
}
function renderInteractivePlot() {
  const results = state.plotData?.results || [];
  if (!results.length) return `<div class="error-panel">No numeric NPZ result arrays are available for plotting.</div>`;
  const selected = results[Math.min(state.plotResult, results.length - 1)];
  return `<div class="fit-tab-panel"><div class="plot-toolbar">
    <label><span class="section-label">Result array</span><select id="plotResultSelect">${results.map((result, index) => `<option value="${index}" ${index === state.plotResult ? "selected" : ""}>${escapeHtml(result.name)} [${result.shape.join(" x ")}]</option>`).join("")}</select></label>
    <div class="plot-meta"><span>x: ${escapeHtml(selected.x_name)}</span><span>${selected.traces.length} trace${selected.traces.length === 1 ? "" : "s"}</span>${selected.downsampled ? "<span>downsampled</span>" : ""}</div>
    <button id="resetPlot" class="plot-button">Reset view</button>
  </div>
  <div class="interactive-plot-wrap"><canvas id="interactivePlot"></canvas><div id="plotTooltip" class="plot-tooltip hidden"></div></div>
  <div id="plotLegend" class="plot-legend"></div></div>`;
}
function renderData() {
  const d = state.detail;
  const tables = d.tables.map(item => card(item, item.error ? `<pre class="text-view">${escapeHtml(item.error)}</pre>` : `<div class="table-wrap"><table>${item.value.rows.map(row => `<tr>${row.map(cell => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`).join("")}</table></div>`)).join("");
  return section("Table Previews", `${d.tables.length} CSV/TSV files`, `<div class="card-grid">${tables || `<p class="mono">No tabular data found.</p>`}</div>`)
    + section("Binary & Other Artifacts", `${d.artifacts.length} files`, artifactRows(d.artifacts));
}
function renderCalibrations() {
  const c = state.detail.calibrations;
  const updates = c.updates.map(u => `<article class="data-card"><div class="card-head"><strong>Update ${escapeHtml(u.time)}</strong><span class="mono">${escapeHtml(u.path)}</span></div><div style="padding:10px">${artifactRows(u.files)}</div></article>`).join("");
  return section("Calibration Source", `${c.scripts.length} matching scripts`, artifactRows(c.scripts))
    + section("Associated Calibration Updates", `${c.updates.length} nearby updates`, `<div class="card-grid">${updates || `<p class="mono">No matching calibration proposals found.</p>`}</div>`);
}

function bindFigures() {
  document.querySelectorAll(".figure-stage").forEach(el => el.onclick = () => openLightbox(el.dataset.src, el.dataset.name));
  document.querySelectorAll("[data-figure-index]").forEach(button => button.onclick = () => {
    state.figureIndex = Number(button.dataset.figureIndex);
    renderTab();
  });
}

function bindInteractivePlot() {
  const canvas = $("interactivePlot");
  if (!canvas) return;
  const result = state.plotData.results[state.plotResult];
  const plot = createCanvasPlot(canvas, result, $("plotTooltip"), $("plotLegend"));
  $("plotResultSelect").onchange = event => { state.plotResult = Number(event.target.value); renderTab(); };
  $("resetPlot").onclick = () => plot.reset();
}

function createCanvasPlot(canvas, result, tooltip, legend) {
  const colors = ["#176b87", "#b45b2a", "#6f55a5", "#25805a", "#b33e65", "#707b24"];
  const validX = result.x.filter(value => value != null);
  const validY = result.traces.flatMap(trace => trace.y.filter(value => value != null));
  const full = { x0: Math.min(...validX), x1: Math.max(...validX), y0: Math.min(...validY), y1: Math.max(...validY) };
  if (full.x0 === full.x1) full.x1 += 1;
  if (full.y0 === full.y1) full.y1 += 1;
  let view = {...full}, dragging = false, start = null;
  const context = canvas.getContext("2d");
  legend.innerHTML = result.traces.map((trace, index) => `<span><i style="background:${colors[index % colors.length]}"></i>${escapeHtml(trace.label)}</span>`).join("");
  function resize() {
    const rect = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * ratio); canvas.height = Math.round(rect.height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0); draw();
  }
  const pad = {l:68,r:18,t:18,b:48};
  const sx = (x,w) => pad.l + (x-view.x0)/(view.x1-view.x0)*(w-pad.l-pad.r);
  const sy = (y,h) => h-pad.b - (y-view.y0)/(view.y1-view.y0)*(h-pad.t-pad.b);
  function draw() {
    const w=canvas.clientWidth,h=canvas.clientHeight; context.clearRect(0,0,w,h); context.strokeStyle="#dfe3e6"; context.fillStyle="#68747d"; context.font="10px system-ui";
    for(let i=0;i<=5;i++){const x=pad.l+i*(w-pad.l-pad.r)/5,y=pad.t+i*(h-pad.t-pad.b)/5;context.beginPath();context.moveTo(x,pad.t);context.lineTo(x,h-pad.b);context.moveTo(pad.l,y);context.lineTo(w-pad.r,y);context.stroke();context.fillText((view.x0+i*(view.x1-view.x0)/5).toPrecision(4),x-16,h-24);context.fillText((view.y1-i*(view.y1-view.y0)/5).toPrecision(4),8,y+3);}
    result.traces.forEach((trace,ti)=>{context.strokeStyle=colors[ti%colors.length];context.lineWidth=1.5;context.beginPath();let started=false;trace.y.forEach((y,i)=>{const x=result.x[i];if(x==null||y==null)return;const px=sx(x,w),py=sy(y,h);started?context.lineTo(px,py):context.moveTo(px,py);started=true;});context.stroke();});
    context.fillStyle="#68747d"; context.fillText(result.x_name,w/2,h-7); context.save();context.translate(12,h/2);context.rotate(-Math.PI/2);context.fillText(result.name,0,0);context.restore();
  }
  canvas.onwheel=e=>{e.preventDefault();const rect=canvas.getBoundingClientRect(),mx=view.x0+(e.clientX-rect.left-pad.l)/(rect.width-pad.l-pad.r)*(view.x1-view.x0),my=view.y1-(e.clientY-rect.top-pad.t)/(rect.height-pad.t-pad.b)*(view.y1-view.y0),factor=e.deltaY>0?1.18:.84;view={x0:mx+(view.x0-mx)*factor,x1:mx+(view.x1-mx)*factor,y0:my+(view.y0-my)*factor,y1:my+(view.y1-my)*factor};draw();};
  canvas.onmousedown=e=>{dragging=true;start={x:e.clientX,y:e.clientY,view:{...view}};};
  window.addEventListener("mouseup",()=>dragging=false,{once:false});
  canvas.onmousemove=e=>{if(!dragging)return;const dx=(e.clientX-start.x)/canvas.clientWidth*(start.view.x1-start.view.x0),dy=(e.clientY-start.y)/canvas.clientHeight*(start.view.y1-start.view.y0);view={x0:start.view.x0-dx,x1:start.view.x1-dx,y0:start.view.y0+dy,y1:start.view.y1+dy};draw();};
  const observer=new ResizeObserver(resize);observer.observe(canvas);resize();
  return {reset(){view={...full};draw();}};
}
let zoom = 1, panX = 0, panY = 0, dragging = false, startX = 0, startY = 0;
function openLightbox(src, name) { $("lightboxImage").src = src; $("lightboxName").textContent = name; $("lightbox").classList.remove("hidden"); zoom=1; panX=panY=0; transformImage(); }
function transformImage() { $("lightboxImage").style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`; }
$("lightboxStage").onwheel = e => { e.preventDefault(); zoom = Math.min(8, Math.max(.2, zoom * (e.deltaY > 0 ? .88 : 1.14))); transformImage(); };
$("lightboxStage").onmousedown = e => { dragging=true; startX=e.clientX-panX; startY=e.clientY-panY; };
window.onmousemove = e => { if (dragging) { panX=e.clientX-startX; panY=e.clientY-startY; transformImage(); } };
window.onmouseup = () => dragging=false;
$("closeLightbox").onclick = () => $("lightbox").classList.add("hidden");
window.onkeydown = e => { if (e.key === "Escape") $("lightbox").classList.add("hidden"); };

function showDetailErrors(errors) { $("errorPanel").textContent = errors.join("\n"); $("errorPanel").classList.toggle("hidden", !errors.length); }
function showGlobalError(message) { $("experimentList").innerHTML = `<div class="error-panel">${escapeHtml(message)}</div>`; }
function setStatus(value) { $("statusText").textContent = value; }
$("dateSelect").onchange = e => loadExperiments(e.target.value);
$("searchInput").oninput = applyFilters;
$("refreshButton").onclick = () => loadDates();
document.querySelectorAll(".tab").forEach(button => button.onclick = () => { state.tab=button.dataset.tab; renderTab(); });
loadDates();
setInterval(refreshLiveData, 10_000);
