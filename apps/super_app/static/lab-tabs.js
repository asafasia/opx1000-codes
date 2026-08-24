(function () {
  "use strict";

  const currentScript = document.currentScript;
  const scriptUrl = new URL(currentScript.src, window.location.href);
  const activeTab = document.body.dataset.labTab || "overview";
  const desktopSuffix = new URLSearchParams(window.location.search).has("desktop") ? "?desktop=1" : "";
  const host = scriptUrl.hostname || "127.0.0.1";
  const protocol = scriptUrl.protocol === "https:" ? "https:" : "http:";
  const serviceUrl = (port, path = "/") => `${protocol}//${host}:${port}${path}${desktopSuffix}`;

  const tabs = [
    { id: "overview", label: "Overview", href: serviceUrl(8890) },
    { id: "data-review", label: "Data Review", href: serviceUrl(8892) },
    { id: "lab-monitor", label: "Lab Monitor", href: serviceUrl(8895) },
    { id: "fridge-monitor", label: "Fridge Monitor", href: serviceUrl(8890, "/fridge.html") },
    { id: "oscilloscope", label: "Oscilloscope", href: serviceUrl(8890, "/oscilloscope.html") },
    { id: "profile-studio", label: "Profile Studio", href: serviceUrl(8893) },
    { id: "parameter-sweep", label: "Parameter Sweep", href: serviceUrl(8770) },
    { id: "wiki", label: "Wiki", href: serviceUrl(8890, "/wiki.html") },
  ];

  const nav = document.createElement("header");
  nav.className = "lab-tabs-shell";
  nav.setAttribute("aria-label", "Quantum coherence lab navigation");

  const brand = document.createElement("a");
  brand.className = "lab-tabs-brand";
  brand.href = serviceUrl(8890);
  brand.setAttribute("aria-label", "Quantum coherence lab overview");
  brand.innerHTML = `
    <span class="lab-tabs-mark" aria-hidden="true">Q</span>
    <span class="lab-tabs-brand-copy">
      <strong>OPX1000</strong>
      <small>Quantum coherence lab</small>
    </span>`;

  const tabList = document.createElement("nav");
  tabList.className = "lab-tabs-list";
  tabList.setAttribute("aria-label", "Laboratory applications");

  tabs.forEach((tab) => {
    const link = document.createElement("a");
    const isActive = tab.id === activeTab;
    link.className = `lab-tabs-link${isActive ? " is-active" : ""}`;
    link.href = tab.href;
    link.textContent = tab.label;
    link.dataset.labTarget = tab.id;
    if (isActive) link.setAttribute("aria-current", "page");
    tabList.appendChild(link);
  });

  nav.append(brand, tabList);

  if (activeTab === "overview") {
    const telemetry = document.createElement("div");
    telemetry.className = "lab-tabs-telemetry";
    telemetry.innerHTML = `
      <span class="lab-tabs-service"><small>Services</small><strong id="serviceCount">-- / 6</strong></span>
      <span class="lab-tabs-state" aria-live="polite">
        <i class="lab-tabs-pulse" id="systemDot" aria-hidden="true"></i>
        <span id="systemText">Initializing</span>
      </span>`;
    nav.appendChild(telemetry);
  }

  document.body.prepend(nav);
})();
