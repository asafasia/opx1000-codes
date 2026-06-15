const state = { experiments: [], filtered: [], selected: null, detail: null, calibrationType: null, qubit: "all", tab: "figures", figureIndex: 0, plotData: null, plotResult: 0, plotHeatmap: 0, plotSlice: null, trendData: null, trendSeries: 0 };
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
  state.plotHeatmap = 0;
  state.plotSlice = null;
  state.trendData = null;
  state.trendSeries = 0;
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
  if (state.tab === "trends" && !state.trendData) {
    $("tabContent").innerHTML = `<div class="empty-state"><p>Loading parameter trends...</p></div>`;
    try {
      state.trendData = await api(`/api/parameter-scan?path=${encodeURIComponent(state.selected.id)}`);
    } catch (error) {
      $("tabContent").innerHTML = `<div class="error-panel">${escapeHtml(error.message)}</div>`;
      return;
    }
  }
  const renderers = { overview: renderOverview, figures: renderFigures, interactive: renderInteractivePlot, trends: renderParameterTrends, data: renderData, calibrations: renderCalibrations };
  $("tabContent").innerHTML = renderers[state.tab]();
  bindFigures();
  if (state.tab === "interactive") bindInteractivePlot();
  if (state.tab === "trends") bindParameterTrends();
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
  const heatmaps = selected.heatmaps || [];
  const heatmap = heatmaps[Math.min(state.plotHeatmap, heatmaps.length - 1)];
  const heatmapControl = heatmaps.length > 1 ? `<label><span class="section-label">Qubit / slice</span><select id="plotHeatmapSelect">${heatmaps.map((item, index) => `<option value="${index}" ${index === state.plotHeatmap ? "selected" : ""}>${escapeHtml(item.label)}</option>`).join("")}</select></label>` : "";
  const plotArea = heatmap ? `<div class="heatmap-layout"><div class="heatmap-panel"><canvas id="heatmapPlot"></canvas></div><div class="slice-panel"><canvas id="slicePlot"></canvas></div></div><div class="slice-control"><span>${escapeHtml(heatmap.y_name)}</span><input id="sliceSlider" type="range" min="0" max="${heatmap.y.length - 1}" value="${state.plotSlice ?? Math.floor(heatmap.y.length / 2)}"><strong id="sliceValue"></strong></div>`
    : `<div class="interactive-plot-wrap"><canvas id="interactivePlot"></canvas><div id="plotTooltip" class="plot-tooltip hidden"></div></div>`;
  return `<div class="fit-tab-panel"><div class="plot-toolbar">
    <label><span class="section-label">Result array</span><select id="plotResultSelect">${results.map((result, index) => `<option value="${index}" ${index === state.plotResult ? "selected" : ""}>${escapeHtml(result.name)} [${result.shape.join(" x ")}]</option>`).join("")}</select></label>
    ${heatmapControl}
    <div class="plot-meta"><span>x: ${escapeHtml(selected.x_name)}</span><span>${selected.traces.length} trace${selected.traces.length === 1 ? "" : "s"}</span>${selected.downsampled ? "<span>downsampled</span>" : ""}</div>
    <button id="resetPlot" class="plot-button">Reset view</button>
  </div>
  ${plotArea}
  <div id="plotLegend" class="plot-legend"></div></div>`;
}
function renderParameterTrends() {
  const series = state.trendData?.series || [];
  if (!series.length) return `<div class="error-panel">No successful parameter values were found in this scan summary.</div>`;
  const index = Math.min(state.trendSeries, series.length - 1);
  const selected = series[index];
  const options = series.map((item, itemIndex) => {
    const label = `${item.experiment_name} | ${item.qubit || "run"} | ${item.parameter}`;
    return `<option value="${itemIndex}" ${itemIndex === index ? "selected" : ""}>${escapeHtml(label)}</option>`;
  }).join("");
  return `<div class="fit-tab-panel"><div class="plot-toolbar">
    <label><span class="section-label">Parameter</span><select id="trendSeriesSelect">${options}</select></label>
    <div class="plot-meta"><span>${selected.points.length} point${selected.points.length === 1 ? "" : "s"}</span><span>${escapeHtml(selected.unit || "unitless")}</span></div>
    <button id="resetTrend" class="plot-button">Reset view</button>
  </div>
  <div class="interactive-plot-wrap"><canvas id="trendPlot"></canvas></div>
  <div id="trendLegend" class="plot-legend"></div></div>`;
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
  const result = state.plotData.results[state.plotResult];
  let plot;
  if (result.heatmaps?.length) {
    plot = createHeatmapPlot($("heatmapPlot"), $("slicePlot"), result.heatmaps[state.plotHeatmap]);
    $("plotHeatmapSelect") && ($("plotHeatmapSelect").onchange = event => { state.plotHeatmap = Number(event.target.value); state.plotSlice = null; renderTab(); });
    $("sliceSlider").oninput = event => { state.plotSlice = Number(event.target.value); plot.setSlice(state.plotSlice); };
  } else if (canvas) {
    plot = createCanvasPlot(canvas, result, $("plotTooltip"), $("plotLegend"));
  } else return;
  $("plotResultSelect").onchange = event => { state.plotResult = Number(event.target.value); renderTab(); };
  $("resetPlot").onclick = () => plot.reset();
}

function bindParameterTrends() {
  const series = state.trendData.series[state.trendSeries];
  const plot = createTrendPlot($("trendPlot"), series, $("trendLegend"));
  $("trendSeriesSelect").onchange = event => { state.trendSeries = Number(event.target.value); renderTab(); };
  $("resetTrend").onclick = () => plot.reset();
}

function createTrendPlot(canvas, series, legend) {
  const context = canvas.getContext("2d");
  const points = series.points.map((point, index) => ({ x: index + 1, y: point.value, label: point.timestamp, success: point.success }));
  const validY = points.map(point => point.y).filter(value => Number.isFinite(value));
  const full = { x0: 1, x1: Math.max(points.length, 2), y0: Math.min(...validY), y1: Math.max(...validY) };
  if (full.y0 === full.y1) { full.y0 -= 1; full.y1 += 1; }
  let view = {...full};
  const pad = {l:72,r:20,t:18,b:48};
  legend.innerHTML = `<span><i style="background:#176b87"></i>${escapeHtml(series.experiment_name)} / ${escapeHtml(series.qubit || "run")} / ${escapeHtml(series.parameter)}</span>`;
  function resize() {
    const rect = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * ratio); canvas.height = Math.round(rect.height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0); draw();
  }
  const sx = (x,w) => pad.l + (x-view.x0)/(view.x1-view.x0)*(w-pad.l-pad.r);
  const sy = (y,h) => h-pad.b - (y-view.y0)/(view.y1-view.y0)*(h-pad.t-pad.b);
  function draw() {
    const w=canvas.clientWidth,h=canvas.clientHeight; context.clearRect(0,0,w,h); context.strokeStyle="#dfe3e6"; context.fillStyle="#68747d"; context.font="10px system-ui";
    for(let i=0;i<=5;i++){const x=pad.l+i*(w-pad.l-pad.r)/5,y=pad.t+i*(h-pad.t-pad.b)/5;context.beginPath();context.moveTo(x,pad.t);context.lineTo(x,h-pad.b);context.moveTo(pad.l,y);context.lineTo(w-pad.r,y);context.stroke();context.fillText((view.x0+i*(view.x1-view.x0)/5).toFixed(1),x-12,h-24);context.fillText((view.y1-i*(view.y1-view.y0)/5).toPrecision(4),8,y+3);}
    context.strokeStyle="#176b87"; context.lineWidth=1.6; context.beginPath(); points.forEach((point,index)=>{const x=sx(point.x,w),y=sy(point.y,h);index?context.lineTo(x,y):context.moveTo(x,y);}); context.stroke();
    points.forEach(point=>{context.fillStyle=String(point.success)==="false"?"#a43a42":"#176b87";context.beginPath();context.arc(sx(point.x,w),sy(point.y,h),3,0,Math.PI*2);context.fill();});
    context.fillStyle="#68747d"; context.fillText("cycle point",w/2,h-7); context.save();context.translate(12,h/2);context.rotate(-Math.PI/2);context.fillText(`${series.parameter}${series.unit ? " [" + series.unit + "]" : ""}`,0,0);context.restore();
  }
  canvas.onwheel=e=>{e.preventDefault();const rect=canvas.getBoundingClientRect(),mx=view.x0+(e.clientX-rect.left-pad.l)/(rect.width-pad.l-pad.r)*(view.x1-view.x0),my=view.y1-(e.clientY-rect.top-pad.t)/(rect.height-pad.t-pad.b)*(view.y1-view.y0),factor=e.deltaY>0?1.18:.84;view={x0:mx+(view.x0-mx)*factor,x1:mx+(view.x1-mx)*factor,y0:my+(view.y0-my)*factor,y1:my+(view.y1-my)*factor};draw();};
  const observer=new ResizeObserver(resize);observer.observe(canvas);resize();
  return {reset(){view={...full};draw();}};
}

function createHeatmapPlot(heatCanvas, sliceCanvas, heatmap) {
  const heatContext = heatCanvas.getContext("2d"), sliceContext = sliceCanvas.getContext("2d");
  const slider = $("sliceSlider"), valueLabel = $("sliceValue");
  let sliceIndex = Math.min(Number(slider.value), heatmap.y.length - 1);
  const finite = heatmap.z.flat().filter(value => value != null), zMin = Math.min(...finite), zMax = Math.max(...finite);
  const color = value => {
    const t = Math.max(0, Math.min(1, (value - zMin) / (zMax - zMin || 1)));
    return `hsl(${250 - t * 250} 75% ${32 + t * 28}%)`;
  };
  function sizeCanvas(canvas, context) { const r=canvas.getBoundingClientRect(),d=devicePixelRatio||1;canvas.width=r.width*d;canvas.height=r.height*d;context.setTransform(d,0,0,d,0,0);return r; }
  function drawHeatmap() {
    const r=sizeCanvas(heatCanvas,heatContext), p={l:68,r:18,t:15,b:38}, pw=r.width-p.l-p.r,ph=r.height-p.t-p.b,rows=heatmap.y.length,cols=heatmap.x.length,cw=pw/cols,ch=ph/rows;
    heatContext.clearRect(0,0,r.width,r.height);
    heatmap.z.forEach((row,yi)=>row.forEach((v,xi)=>{heatContext.fillStyle=v==null?"#eee":color(v);heatContext.fillRect(p.l+xi*cw,p.t+(rows-1-yi)*ch,cw+1,ch+1);}));
    const lineY=p.t+(rows-1-sliceIndex+.5)*ch;heatContext.strokeStyle="#fff";heatContext.lineWidth=2;heatContext.beginPath();heatContext.moveTo(p.l,lineY);heatContext.lineTo(r.width-p.r,lineY);heatContext.stroke();heatContext.strokeStyle="#111";heatContext.lineWidth=.7;heatContext.stroke();
    heatContext.fillStyle="#68747d";heatContext.font="10px system-ui";heatContext.fillText(heatmap.x_name,r.width/2,r.height-7);heatContext.save();heatContext.translate(12,r.height/2);heatContext.rotate(-Math.PI/2);heatContext.fillText(heatmap.y_name,0,0);heatContext.restore();
  }
  function drawSlice() {
    const r=sizeCanvas(sliceCanvas,sliceContext), p={l:68,r:18,t:12,b:35}, values=heatmap.z[sliceIndex], valid=values.filter(v=>v!=null), y0=Math.min(...valid),y1=Math.max(...valid),x0=heatmap.x[0],x1=heatmap.x[heatmap.x.length-1];
    sliceContext.clearRect(0,0,r.width,r.height);sliceContext.strokeStyle="#dfe3e6";for(let i=0;i<=5;i++){let x=p.l+i*(r.width-p.l-p.r)/5;sliceContext.beginPath();sliceContext.moveTo(x,p.t);sliceContext.lineTo(x,r.height-p.b);sliceContext.stroke();}
    sliceContext.strokeStyle="#176b87";sliceContext.lineWidth=1.5;sliceContext.beginPath();values.forEach((v,i)=>{if(v==null)return;const x=p.l+(heatmap.x[i]-x0)/(x1-x0||1)*(r.width-p.l-p.r),y=r.height-p.b-(v-y0)/(y1-y0||1)*(r.height-p.t-p.b);i?sliceContext.lineTo(x,y):sliceContext.moveTo(x,y);});sliceContext.stroke();
    sliceContext.fillStyle="#68747d";sliceContext.font="10px system-ui";sliceContext.fillText(`${heatmap.x_name} spectrum at ${heatmap.y_name} = ${Number(heatmap.y[sliceIndex]).toPrecision(5)}`,p.l,r.height-8);
    valueLabel.textContent=Number(heatmap.y[sliceIndex]).toPrecision(6);
  }
  function setSlice(index){sliceIndex=Math.max(0,Math.min(heatmap.y.length-1,index));slider.value=sliceIndex;drawHeatmap();drawSlice();}
  heatCanvas.onclick=e=>{const r=heatCanvas.getBoundingClientRect(),pTop=15,pBottom=38,ratio=1-(e.clientY-r.top-pTop)/(r.height-pTop-pBottom);setSlice(Math.round(ratio*(heatmap.y.length-1)));state.plotSlice=sliceIndex;};
  heatCanvas.onmousemove=e=>{if(e.buttons===1)heatCanvas.onclick(e);};
  const observer=new ResizeObserver(()=>{drawHeatmap();drawSlice();});observer.observe(heatCanvas);observer.observe(sliceCanvas);setSlice(sliceIndex);
  return {setSlice,reset(){setSlice(Math.floor(heatmap.y.length/2));}};
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
