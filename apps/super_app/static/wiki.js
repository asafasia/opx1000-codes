const nav = document.querySelector("#documentNav");
const documentView = document.querySelector("#document");
const documentCount = document.querySelector("#documentCount");
const searchInput = document.querySelector("#searchInput");
const breadcrumb = document.querySelector("#breadcrumb");
const refreshWiki = document.querySelector("#refreshWiki");
const menuButton = document.querySelector("#menuButton");
const scrim = document.querySelector("#scrim");
let documents = [];
let currentPath = "";
let searchTimer;

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function resolvePath(target) {
  if (/^(https?:|mailto:|#)/i.test(target)) return target;
  const base = currentPath.includes("/") ? currentPath.slice(0, currentPath.lastIndexOf("/") + 1) : "";
  const parts = `${base}${target}`.split("/");
  const resolved = [];
  for (const part of parts) {
    if (!part || part === ".") continue;
    if (part === "..") resolved.pop(); else resolved.push(part);
  }
  return resolved.join("/");
}

function inline(text) {
  let rendered = escapeHtml(text);
  rendered = rendered.replace(/!\[([^\]]*)\]\(([^)\s]+)(?:\s+&quot;.*?&quot;)?\)/g, (_m, alt, target) => {
    const resolved = resolvePath(target);
    if (/^https?:/i.test(resolved)) return `<img src="${escapeHtml(resolved)}" alt="${alt}" loading="lazy">`;
    return `<img src="/api/wiki/asset?path=${encodeURIComponent(resolved)}" alt="${alt}" loading="lazy">`;
  });
  rendered = rendered.replace(/\[([^\]]+)\]\(([^)\s]+)(?:\s+&quot;.*?&quot;)?\)/g, (_m, label, target) => {
    const resolved = resolvePath(target);
    if (resolved.startsWith("#")) return `<a href="${escapeHtml(resolved)}">${label}</a>`;
    if (/^(https?:|mailto:)/i.test(resolved)) return `<a href="${escapeHtml(resolved)}" target="_blank" rel="noreferrer">${label}</a>`;
    if (resolved.toLowerCase().endsWith(".md")) return `<a href="/wiki.html?path=${encodeURIComponent(resolved)}" data-wiki-path="${escapeHtml(resolved)}">${label}</a>`;
    return `<span title="${escapeHtml(resolved)}">${label}</span>`;
  });
  rendered = rendered.replace(/`([^`]+)`/g, "<code>$1</code>");
  rendered = rendered.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  rendered = rendered.replace(/__([^_]+)__/g, "<strong>$1</strong>");
  rendered = rendered.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, "<em>$1</em>");
  return rendered;
}

function slug(text) {
  return text.toLowerCase().replace(/<[^>]+>/g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function renderMarkdown(source) {
  const lines = source.replaceAll("\r\n", "\n").split("\n");
  const output = [];
  let paragraph = [];
  let listType = null;
  let inCode = false;
  let code = [];

  const flushParagraph = () => {
    if (paragraph.length) output.push(`<p>${inline(paragraph.join(" "))}</p>`);
    paragraph = [];
  };
  const closeList = () => {
    if (listType) output.push(`</${listType}>`);
    listType = null;
  };

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (line.startsWith("```")) {
      flushParagraph(); closeList();
      if (inCode) {
        output.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
        code = [];
      }
      inCode = !inCode;
      continue;
    }
    if (inCode) { code.push(line); continue; }
    if (!line.trim()) { flushParagraph(); closeList(); continue; }

    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      flushParagraph(); closeList();
      const level = heading[1].length;
      const content = inline(heading[2]);
      output.push(`<h${level} id="${slug(content)}">${content}</h${level}>`);
      continue;
    }
    if (/^\s*([-*_])(?:\s*\1){2,}\s*$/.test(line)) {
      flushParagraph(); closeList(); output.push("<hr>"); continue;
    }
    const list = line.match(/^\s*(?:([-+*])|(\d+)\.)\s+(.+)$/);
    if (list) {
      flushParagraph();
      const type = list[2] ? "ol" : "ul";
      if (listType !== type) { closeList(); output.push(`<${type}>`); listType = type; }
      output.push(`<li>${inline(list[3])}</li>`);
      continue;
    }
    if (line.startsWith("> ")) {
      flushParagraph(); closeList(); output.push(`<blockquote>${inline(line.slice(2))}</blockquote>`); continue;
    }

    const next = lines[index + 1] || "";
    if (line.includes("|") && /^\s*\|?\s*:?-{3,}/.test(next)) {
      flushParagraph(); closeList();
      const headers = line.replace(/^\||\|$/g, "").split("|");
      output.push(`<table><thead><tr>${headers.map(cell => `<th>${inline(cell.trim())}</th>`).join("")}</tr></thead><tbody>`);
      index += 1;
      while (index + 1 < lines.length && lines[index + 1].includes("|") && lines[index + 1].trim()) {
        index += 1;
        const cells = lines[index].replace(/^\||\|$/g, "").split("|");
        output.push(`<tr>${cells.map(cell => `<td>${inline(cell.trim())}</td>`).join("")}</tr>`);
      }
      output.push("</tbody></table>");
      continue;
    }
    paragraph.push(line.trim());
  }
  if (inCode) output.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
  flushParagraph(); closeList();
  return output.join("\n");
}

function renderIndex(items) {
  const groups = new Map();
  for (const item of items) {
    if (!groups.has(item.section)) groups.set(item.section, []);
    groups.get(item.section).push(item);
  }
  nav.innerHTML = [...groups.entries()].map(([section, entries]) => `
    <section class="nav-group">
      <h2>${escapeHtml(section)}</h2>
      ${entries.map(item => `
        <button class="doc-link ${item.path === currentPath ? "active" : ""}" data-path="${escapeHtml(item.path)}" type="button">
          <strong>${escapeHtml(item.title)}</strong>
          <span>${escapeHtml(item.path)}</span>
        </button>`).join("")}
    </section>`).join("");
  documentCount.textContent = `${items.length} Markdown document${items.length === 1 ? "" : "s"}`;
}

async function loadIndex(query = "") {
  try {
    const response = await fetch(`/api/wiki?q=${encodeURIComponent(query)}`, {cache: "no-store"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    documents = payload.documents;
    renderIndex(documents);
  } catch (error) {
    nav.innerHTML = `<p class="error-state">Could not load the wiki index: ${escapeHtml(error.message)}</p>`;
  }
}

async function openDocument(path, {pushHistory = true} = {}) {
  try {
    documentView.innerHTML = `<div class="empty-state"><p>Loading</p><h1>Opening document…</h1></div>`;
    const response = await fetch(`/api/wiki/file?path=${encodeURIComponent(path)}`, {cache: "no-store"});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    currentPath = payload.path;
    breadcrumb.textContent = payload.path;
    document.title = `${payload.title} · OPX1000 Wiki`;
    documentView.innerHTML = `
      <header class="document-meta">
        <p>${escapeHtml(payload.section)}</p>
        <h1>${escapeHtml(payload.title)}</h1>
        <span>${escapeHtml(payload.path)} · ${Math.max(1, Math.round(payload.size / 1024))} KB</span>
      </header>
      <div class="markdown-body">${renderMarkdown(payload.content)}</div>`;
    renderIndex(documents);
    if (pushHistory) history.pushState({path}, "", `/wiki.html?path=${encodeURIComponent(path)}`);
    window.scrollTo({top: 0, behavior: "instant"});
    document.body.classList.remove("menu-open");
  } catch (error) {
    documentView.innerHTML = `<p class="error-state">Could not open this document: ${escapeHtml(error.message)}</p>`;
  }
}

nav.addEventListener("click", event => {
  const button = event.target.closest("[data-path]");
  if (button) openDocument(button.dataset.path);
});
documentView.addEventListener("click", event => {
  const link = event.target.closest("[data-wiki-path]");
  if (link) { event.preventDefault(); openDocument(link.dataset.wikiPath); }
});
searchInput.addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => loadIndex(searchInput.value), 180);
});
document.addEventListener("keydown", event => {
  if (event.key === "/" && document.activeElement !== searchInput) { event.preventDefault(); searchInput.focus(); }
});
refreshWiki.addEventListener("click", () => loadIndex(searchInput.value));
menuButton.addEventListener("click", () => document.body.classList.toggle("menu-open"));
scrim.addEventListener("click", () => document.body.classList.remove("menu-open"));
window.addEventListener("popstate", event => {
  const path = event.state?.path || new URLSearchParams(location.search).get("path");
  if (path) openDocument(path, {pushHistory: false});
});

(async () => {
  await loadIndex();
  const requested = new URLSearchParams(location.search).get("path");
  if (requested) openDocument(requested, {pushHistory: false});
  else if (documents.length) openDocument(documents.find(item => item.path === "README.md")?.path || documents[0].path, {pushHistory: false});
})();
