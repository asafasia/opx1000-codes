const sections = {
  profile: { label: "Profile", description: "Manifest, file references, and active qubits" },
  qubits: { label: "Qubits", description: "Frequencies, coherence, readout, and operations" },
  pulses: { label: "Pulses", description: "Reusable control and readout pulse definitions" },
  connectivity: { label: "Connectivity", description: "Network, controllers, ports, lines, and LOs" },
};
const state = { profile: "", section: "profile", view: "fields", documents: {}, selectedQubits: {}, selectedPulse: null };
const $ = id => document.getElementById(id);
const escapeHtml = (value = "") => String(value).replace(/[&<>"']/g, c => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#039;" }[c]));

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `Request failed (${response.status})`);
  return value;
}

function documentKey(section = state.section) { return `${state.profile}/${section}`; }
function currentDocument() { return state.documents[documentKey()]; }
function setStatus(text, working = false) {
  $("statusText").textContent = text;
  document.querySelector(".status-dot").classList.toggle("working", working);
}
function showMessage(text = "", kind = "error") {
  $("message").textContent = text;
  $("message").className = text ? `message ${kind}` : "message hidden";
}

async function initialize() {
  try {
    const result = await api("/api/profiles");
    $("profileSelect").innerHTML = result.profiles.map(name => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("");
    if (!result.profiles.length) throw new Error("No complete profiles were found.");
    state.profile = result.profiles.includes("main") ? "main" : result.profiles[0];
    $("profileSelect").value = state.profile;
    renderTabs();
    await loadSection();
  } catch (error) {
    showMessage(error.message);
    setStatus("Connection error");
  }
}

async function loadSection(force = false) {
  const key = documentKey();
  if (state.documents[key]?.dirty && force && !confirm("Discard unsaved changes in this tab?")) return;
  showMessage();
  setStatus("Loading profile", true);
  try {
    const result = await api(`/api/section?profile=${encodeURIComponent(state.profile)}&section=${encodeURIComponent(state.section)}`);
    state.documents[key] = { data: result.data, digest: result.digest, dirty: false };
    render();
    setStatus("Ready");
  } catch (error) {
    showMessage(error.message);
    setStatus("Read error");
  }
}

function renderTabs() {
  $("tabs").innerHTML = Object.entries(sections).map(([key, item]) => {
    const dirty = state.documents[documentKey(key)]?.dirty;
    return `<button class="side-tab ${key === state.section ? "active" : ""}" data-section="${key}">
      <span><strong>${item.label}</strong><small>${item.description}</small></span><i class="${dirty ? "dirty" : ""}"></i>
    </button>`;
  }).join("");
  document.querySelectorAll("[data-section]").forEach(button => button.onclick = async () => {
    state.section = button.dataset.section;
    renderTabs();
    if (currentDocument()) render(); else await loadSection();
  });
}

function render() {
  const doc = currentDocument();
  const item = sections[state.section];
  $("sectionTitle").textContent = item.label;
  $("filePath").textContent = `profiles/${state.profile}/${state.section}.json`;
  $("dirtyBadge").classList.toggle("hidden", !doc?.dirty);
  $("saveButton").disabled = !doc?.dirty;
  document.querySelectorAll("[data-view]").forEach(button => button.classList.toggle("active", button.dataset.view === state.view));
  if (!doc) return;
  if (state.view === "raw") {
    $("editor").innerHTML = `<textarea id="rawEditor" class="raw-editor" spellcheck="false">${escapeHtml(JSON.stringify(doc.data, null, 2))}</textarea>`;
    $("rawEditor").oninput = event => {
      try {
        doc.data = JSON.parse(event.target.value);
        markDirty();
        showMessage();
      } catch (error) {
        showMessage(`Raw JSON is not valid yet: ${error.message}`, "warning");
      }
    };
  } else {
    $("editor").innerHTML = renderFields(doc.data);
    bindFields();
  }
  renderTabs();
}

function renderFields(data) {
  const collection = data?.[state.section];
  if (!["qubits", "pulses"].includes(state.section) || !collection || typeof collection !== "object" || Array.isArray(collection)) {
    return `<div class="field-tree">${renderNode(data, [])}</div>`;
  }
  const qubits = Object.keys(collection);
  if (!qubits.length) return `<div class="empty-state">No qubits are defined in this file.</div>`;
  const selectionKey = `${state.profile}/${state.section}`;
  if (!qubits.includes(state.selectedQubits[selectionKey])) state.selectedQubits[selectionKey] = qubits[0];
  const selected = state.selectedQubits[selectionKey];
  const tabs = qubits.map(qubit =>
    `<button class="qubit-tab ${qubit === selected ? "active" : ""}" data-qubit-tab="${escapeHtml(qubit)}">${escapeHtml(qubit)}</button>`
  ).join("");
  const pulses = state.section === "pulses" ? Object.keys(collection[selected] || {}) : [];
  if (pulses.length && !pulses.includes(state.selectedPulse)) state.selectedPulse = pulses[0];
  const selectedPulse = state.selectedPulse;
  const pulseTabs = pulses.map(pulse =>
    `<button class="pulse-tab ${pulse === selectedPulse ? "active" : ""}" data-pulse-tab="${escapeHtml(pulse)}">${escapeHtml(pulse)}</button>`
  ).join("");
  const selectedData = state.section === "pulses" ? collection[selected][selectedPulse] : collection[selected];
  const selectedPath = state.section === "pulses" ? [state.section, selected, selectedPulse] : [state.section, selected];
  const selectedName = state.section === "pulses" ? selectedPulse : selected;
  return `<div class="qubit-editor">
    <div class="qubit-sticky">
      <div class="qubit-tabs" role="tablist" aria-label="${escapeHtml(sections[state.section].label)} qubit selection">${tabs}</div>
      <div class="qubit-panel-head"><span>Editing</span><strong>${escapeHtml(selected)}</strong><small>${escapeHtml(sections[state.section].label)}</small></div>
      ${pulses.length ? `<div class="pulse-tabs" role="tablist" aria-label="${escapeHtml(selected)} pulse selection">${pulseTabs}</div>` : ""}
    </div>
    <div class="qubit-panel">
      <div class="field-tree">${renderNode(selectedData, selectedPath, selectedName)}</div>
    </div>
  </div>`;
}

function renderNode(value, path, name = null) {
  if (Array.isArray(value)) {
    return `<details class="group" open><summary>${nodeTitle(name, "array", value.length)}</summary><div class="children">${value.map((item, index) => renderNode(item, [...path, index], `[${index}]`)).join("")}</div></details>`;
  }
  if (value !== null && typeof value === "object") {
    const entries = Object.entries(value);
    return `<details class="group" open><summary>${nodeTitle(name, "object", entries.length)}</summary><div class="children">${entries.map(([key, item]) => renderNode(item, [...path, key], key)).join("")}</div></details>`;
  }
  const type = value === null ? "null" : typeof value;
  const encodedPath = escapeHtml(JSON.stringify(path));
  if (type === "boolean") {
    return `<label class="field"><span>${escapeHtml(name)}</span><span class="type">${type}</span><input data-path="${encodedPath}" data-type="${type}" type="checkbox" ${value ? "checked" : ""}></label>`;
  }
  if (type === "null") {
    return `<label class="field"><span>${escapeHtml(name)}</span><span class="type">${type}</span><input data-path="${encodedPath}" data-type="${type}" value="null" disabled></label>`;
  }
  const inputType = type === "number" ? "number" : "text";
  const step = type === "number" ? ' step="any"' : "";
  return `<label class="field"><span>${escapeHtml(name)}</span><span class="type">${type}</span><input data-path="${encodedPath}" data-type="${type}" type="${inputType}"${step} value="${escapeHtml(value)}"></label>`;
}

function nodeTitle(name, type, count) {
  return `<span>${escapeHtml(name ?? sections[state.section].label)}</span><small>${type} · ${count} item${count === 1 ? "" : "s"}</small>`;
}

function bindFields() {
  document.querySelectorAll("[data-qubit-tab]").forEach(button => button.onclick = () => {
    state.selectedQubits[`${state.profile}/${state.section}`] = button.dataset.qubitTab;
    render();
  });
  document.querySelectorAll("[data-pulse-tab]").forEach(button => button.onclick = () => {
    state.selectedPulse = button.dataset.pulseTab;
    render();
  });
  document.querySelectorAll("[data-path]").forEach(input => input.onchange = () => {
    const path = JSON.parse(input.dataset.path);
    let value;
    if (input.dataset.type === "boolean") value = input.checked;
    else if (input.dataset.type === "number") {
      value = Number(input.value);
      if (!Number.isFinite(value)) {
        showMessage(`"${path.at(-1)}" must be a valid number.`);
        render();
        return;
      }
    } else value = input.value;
    setAtPath(currentDocument().data, path, value);
    markDirty();
  });
}

function setAtPath(root, path, value) {
  let parent = root;
  for (const part of path.slice(0, -1)) parent = parent[part];
  parent[path.at(-1)] = value;
}

function markDirty() {
  currentDocument().dirty = true;
  $("dirtyBadge").classList.remove("hidden");
  $("saveButton").disabled = false;
  renderTabs();
  setStatus("Unsaved changes");
}

async function save() {
  const doc = currentDocument();
  if (!doc?.dirty) return;
  if (state.view === "raw") {
    try { doc.data = JSON.parse($("rawEditor").value); }
    catch (error) { showMessage(`Cannot save invalid JSON: ${error.message}`); return; }
  }
  setStatus("Saving profile", true);
  $("saveButton").disabled = true;
  try {
    const result = await api("/api/section", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile: state.profile, section: state.section, data: doc.data, digest: doc.digest }),
    });
    doc.digest = result.digest;
    doc.dirty = false;
    showMessage(`${sections[state.section].label} saved successfully.`, "success");
    render();
    setStatus("Saved");
  } catch (error) {
    showMessage(error.message);
    $("saveButton").disabled = false;
    setStatus("Save failed");
  }
}

$("profileSelect").onchange = async event => {
  if (Object.values(state.documents).some(doc => doc.dirty) && !confirm("Switch profile and leave unsaved changes open?")) {
    event.target.value = state.profile;
    return;
  }
  state.profile = event.target.value;
  renderTabs();
  if (currentDocument()) render(); else await loadSection();
};
$("reloadButton").onclick = () => loadSection(true);
$("saveButton").onclick = save;
document.querySelectorAll("[data-view]").forEach(button => button.onclick = () => {
  if (state.view === "raw") {
    try { currentDocument().data = JSON.parse($("rawEditor").value); }
    catch (error) { showMessage(`Fix the raw JSON before switching views: ${error.message}`); return; }
  }
  state.view = button.dataset.view;
  render();
});
window.addEventListener("beforeunload", event => {
  if (Object.values(state.documents).some(doc => doc.dirty)) event.preventDefault();
});
initialize();
