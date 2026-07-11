const state = {
  timer: null,
  inFlight: false,
};

const els = {
  liveDot: document.querySelector("#liveDot"),
  liveState: document.querySelector("#liveState"),
  connectionText: document.querySelector("#connectionText"),
  jobTitle: document.querySelector("#jobTitle"),
  jobStatus: document.querySelector("#jobStatus"),
  jobType: document.querySelector("#jobType"),
  jobDescription: document.querySelector("#jobDescription"),
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
  const jobs = payload.jobs || [];
  const openQms = payload.open_qms || [];
  const job = jobs[0];

  if (payload.has_active_jobs && job) {
    setLiveState("live", "Live", "A QOP job is active right now.");
    els.jobTitle.textContent = job.id || "Active job";
    els.jobStatus.textContent = job.status || "Running";
    els.jobType.textContent = job.is_simulation ? "Simulation" : "Hardware";
    els.jobDescription.textContent = job.description || "No description reported.";
  } else {
    setLiveState("", "Not live", "No active QOP jobs were reported.");
    els.jobTitle.textContent = "No active job";
    els.jobStatus.textContent = "-";
    els.jobType.textContent = "-";
    els.jobDescription.textContent = "The monitor is connected and waiting.";
  }

  els.lastPoll.textContent = `Last checked: ${formatTime(payload.polled_at)}`;
  els.openQms.textContent = `Open QMs: ${openQms.length}`;
}

async function refresh() {
  if (state.inFlight) return;
  state.inFlight = true;
  setLiveState("", "Checking", "Polling QOP...");

  try {
    const response = await fetch("/api/jobs", { cache: "no-store" });
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

function schedule() {
  if (state.timer) window.clearInterval(state.timer);
  state.timer = window.setInterval(refresh, 3000);
}

schedule();
refresh();
