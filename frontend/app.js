/* SignalForge frontend: vanilla JS, hash routing, Chart.js. */
"use strict";

const S = { meta: null, projects: [], pid: null, charts: [], es: null, poll: null, datasets: [] };
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const view = () => $("#view");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 1) => (x == null || Number.isNaN(x) ? "–" : `${(x * 100).toFixed(d)}%`);
const num = (x, d = 2) => (x == null || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
const sgn = (x) => (x == null ? "" : x > 0 ? "pos" : x < 0 ? "neg" : "");
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const ago = (iso) => {
  if (!iso) return "";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
};

async function api(path, opts = {}) {
  const init = { ...opts, headers: { ...(opts.headers || {}) } };
  if (opts.json !== undefined) {
    init.body = JSON.stringify(opts.json);
    init.headers["Content-Type"] = "application/json";
    init.method = init.method || "POST";
  }
  const res = await fetch(path, init);
  const isJson = (res.headers.get("content-type") || "").includes("application/json");
  const body = isJson ? await res.json() : await res.text();
  if (!res.ok) {
    const err = new Error(isJson ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : `Request failed (${res.status})`);
    err.report = isJson ? body.report : null;
    err.body = isJson ? body : null;
    err.status = res.status;
    throw err;
  }
  return body;
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.remove("show"), 2600);
}

function cleanup() {
  S.charts.forEach((c) => c.destroy());
  S.charts = [];
  if (S.es) { S.es.close(); S.es = null; }
  if (S.poll) { clearInterval(S.poll); S.poll = null; }
}

/* ------------------------------------------------------------------ charts */

const dividerPlugin = {
  id: "divider",
  afterDatasetsDraw(chart, _a, opts) {
    if (opts == null || opts.index == null) return;
    const { ctx, chartArea: a, scales: { x } } = chart;
    const px = x.getPixelForValue(opts.index);
    ctx.save();
    ctx.fillStyle = cssVar("--grid") || "rgba(0,0,0,0.04)";
    ctx.globalAlpha = 1;
    ctx.fillStyle = "rgba(124,140,255,0.07)";
    ctx.fillRect(a.left, a.top, px - a.left, a.bottom - a.top);
    ctx.strokeStyle = cssVar("--ink");
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(px, a.top); ctx.lineTo(px, a.bottom); ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = "600 12px 'IBM Plex Sans', sans-serif";
    const tag = (text, x, align, color) => {
      const w = ctx.measureText(text).width;
      const bx = align === "right" ? x - w - 8 : x;
      ctx.fillStyle = cssVar("--panel");
      ctx.fillRect(bx, a.bottom - 26, w + 8, 20);
      ctx.fillStyle = color;
      ctx.textAlign = "left";
      ctx.fillText(text, bx + 4, a.bottom - 12);
    };
    if (px - a.left > 120) tag(opts.left || "", px - 6, "right", cssVar("--muted"));
    tag(opts.right || "", px + 6, "left", cssVar("--ink"));
    ctx.restore();
  },
};

function baseOptions({ yPct = false, log = false, yTitle = "" } = {}) {
  const ink = cssVar("--muted"), rule = cssVar("--rule");
  return {
    responsive: true, maintainAspectRatio: false, animation: false, normalized: true,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.y == null ? "–" : yPct ? pct(c.parsed.y) : num(c.parsed.y, 3)}` },
      },
    },
    scales: {
      x: {
        ticks: {
          color: ink, maxTicksLimit: window.innerWidth < 640 ? 3 : 6, maxRotation: 0, autoSkip: true,
          callback(v) { const l = String(this.getLabelForValue(v) ?? ""); return /^\d{4}-\d{2}-\d{2}$/.test(l) ? l.slice(0, 7) : l; },
        },
        grid: { display: false }, border: { color: rule },
      },
      y: {
        type: log ? "logarithmic" : "linear",
        ticks: { color: ink, maxTicksLimit: 6, callback: (v) => (yPct ? `${Math.round(v * 100)}%` : log ? `${Number(v).toPrecision(2)}×` : Math.abs(v) >= 10000 ? new Intl.NumberFormat('en', { notation: 'compact' }).format(v) : v) },
        grid: { color: rule }, border: { display: false },
        title: { display: !!yTitle, text: yTitle, color: ink },
      },
    },
    elements: { point: { radius: 0, hoverRadius: 3 }, line: { borderWidth: 1.8, tension: 0 } },
  };
}

function makeChart(canvas, config) {
  const c = new Chart(canvas, config);
  S.charts.push(c);
  return c;
}

function line(label, data, color, dashed = false, width = 1.8) {
  return { label, data, borderColor: color, backgroundColor: color, borderDash: dashed ? [5, 4] : [], borderWidth: width, spanGaps: true };
}

/* ------------------------------------------------------------------ router */

async function ensureMeta() {
  if (!S.meta) S.meta = await api("/api/meta");
}

async function loadProjects() {
  S.projects = await api("/api/projects");
  const sel = $("#projectSelect");
  sel.innerHTML = S.projects.length
    ? S.projects.map((p) => `<option value="${p.id}">${esc(p.name)}</option>`).join("")
    : `<option value="">No projects yet</option>`;
  if (S.pid) sel.value = S.pid;
}

function renderNav(active) {
  const p = S.projects.find((x) => x.id === S.pid);
  if (!p) { $("#nav").innerHTML = ""; return; }
  const items = [
    ["home", `#/p/${p.id}`, "Experiments", p.n_experiments],
    ["new", `#/p/${p.id}/new`, "New experiment", ""],
    ["datasets", `#/p/${p.id}/datasets`, "Datasets", p.n_datasets],
    ["compare", `#/p/${p.id}/compare`, "Compare", ""],
  ];
  $("#nav").innerHTML = items
    .map(([k, href, label, c]) => `<a href="${href}" class="${k === active ? "active" : ""}" ${k === active ? 'aria-current="page"' : ""}>${label}<span class="count">${c ?? ""}</span></a>`)
    .join("");
}

async function route() {
  cleanup();
  const hash = location.hash.replace(/^#/, "") || "/";
  const [path, qs] = hash.split("?");
  const q = new URLSearchParams(qs || "");
  const parts = path.split("/").filter(Boolean);
  try {
    await ensureMeta();
    if (!S.projects.length) await loadProjects();
    if (parts[0] === "p" && parts[1]) S.pid = Number(parts[1]);
    if (parts[0] === "e" && parts[1]) {
      const exp = await api(`/api/experiments/${parts[1]}`);
      S.pid = exp.project_id;
      $("#projectSelect").value = S.pid;
      renderNav("home");
      return renderExperiment(exp);
    }
    if (parts.length === 0) {
      if (S.projects.length) { location.replace(`#/p/${S.projects[0].id}`); return; }
      location.replace("#/projects"); return;
    }
    if (parts[0] === "projects") { renderNav(""); return renderProjects(); }
    $("#projectSelect").value = S.pid;
    const sub = parts[2] || "home";
    renderNav(sub);
    if (sub === "home") return renderHome();
    if (sub === "datasets") return renderDatasets(parts[3] ? Number(parts[3]) : null);
    if (sub === "new") return renderNew(q);
    if (sub === "compare") return renderCompare(q);
    view().innerHTML = `<div class="empty"><h2>Page not found</h2><p><a href="#/">Go to your experiments</a></p></div>`;
  } catch (e) {
    view().innerHTML = `<div class="empty"><h2>Could not load this page</h2><p>${esc(e.message)}</p><p><a class="btn" href="#/">Back to start</a></p></div>`;
  }
  view().focus({ preventScroll: true });
}

/* ------------------------------------------------------------------ projects */

async function renderProjects() {
  await loadProjects();
  view().innerHTML = `
    <div class="head"><div><h1>Projects</h1><p class="lede">Each project keeps its own datasets, experiments and leaderboard.</p></div></div>
    <div class="grid2">
      <div class="panel">
        <h2>Your projects</h2>
        ${S.projects.length ? `<div class="tbl-wrap"><table><thead><tr><th>Name</th><th class="num">Datasets</th><th class="num">Experiments</th><th></th></tr></thead><tbody>
          ${S.projects.map((p) => `<tr class="click" data-href="#/p/${p.id}"><td><b>${esc(p.name)}</b><div class="muted small">${esc(p.description)}</div></td>
            <td class="num">${p.n_datasets}</td><td class="num">${p.n_experiments}</td>
            <td><button class="btn small danger" data-del="${p.id}" aria-label="Delete ${esc(p.name)}">Delete</button></td></tr>`).join("")}
        </tbody></table></div>` : `<p class="muted">No projects yet. Create one or load the demo workspace.</p>`}
      </div>
      <div class="stack">
        <form class="panel" id="newProject">
          <h2>Create a project</h2>
          <div class="stack">
            <label class="field">Name<input type="text" name="name" required maxlength="120" placeholder="e.g. NIFTY momentum signals"></label>
            <label class="field">Description <span class="hint">Optional</span><textarea name="description" placeholder="What are you trying to find out?"></textarea></label>
            <div><button class="btn primary" type="submit">Create project</button></div>
          </div>
        </form>
        <div class="panel">
          <h2>Demo workspace</h2>
          <p class="muted small">Adds a project with four synthetic assets, a messy file, and six preset experiments (including the overfit trap and the leakage demo).</p>
          <button class="btn" id="seed">Load demo workspace</button>
        </div>
      </div>
    </div>`;
  $$("tr.click").forEach((tr) => tr.addEventListener("click", (e) => { if (!e.target.closest("button")) location.hash = tr.dataset.href; }));
  $$("[data-del]").forEach((b) => b.addEventListener("click", async () => {
    if (!confirm("Delete this project with all its datasets and experiments? This cannot be undone.")) return;
    await api(`/api/projects/${b.dataset.del}`, { method: "DELETE" });
    toast("Project deleted");
    S.projects = []; S.pid = null; renderProjects();
  }));
  $("#newProject").addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    try {
      const p = await api("/api/projects", { json: { name: fd.get("name"), description: fd.get("description") } });
      await loadProjects();
      toast("Project created");
      location.hash = `#/p/${p.id}/datasets`;
    } catch (err) { toast(err.message); }
  });
  $("#seed").addEventListener("click", async (e) => {
    e.target.disabled = true;
    const r = await api("/api/demo/seed", { method: "POST" });
    await loadProjects();
    location.hash = `#/p/${r.project_id}`;
  });
}

/* ------------------------------------------------------------------ home: leaderboard + experiments */

async function renderHome() {
  const [project, lb, exps, datasets] = await Promise.all([
    api(`/api/projects/${S.pid}`), api(`/api/projects/${S.pid}/leaderboard`),
    api(`/api/projects/${S.pid}/experiments`), api(`/api/projects/${S.pid}/datasets`),
  ]);
  S.datasets = datasets;
  if (!datasets.length) {
    view().innerHTML = `<div class="head"><div><h1>${esc(project.name)}</h1></div></div>
      <div class="panel empty"><h2>Start by adding market data</h2><p>Upload daily prices as CSV, Parquet or a ZIP of many stocks (date, open, high, low, close, volume). Every file is validated before it can be used.</p>
      <a class="btn primary" href="#/p/${S.pid}/datasets">Upload a dataset</a></div>`;
    return;
  }
  const naive = lb.naive_rank_by_train_accuracy;
  const naiveRank = (id) => naive.indexOf(id) + 1;
  const ranked = lb.ranked;
  const running = exps.some((e) => e.status === "queued" || e.status === "running");

  view().innerHTML = `
    <div class="head">
      <div><h1>${esc(project.name)}</h1><p class="lede">${esc(project.description || "Experiments, ranked by how well they hold up on data the model never saw.")}</p></div>
      <div class="actions"><a class="btn" href="#/p/${S.pid}/datasets">Upload data</a><a class="btn primary" href="#/p/${S.pid}/new">New experiment</a></div>
    </div>

    <section class="panel">
      <div class="panel-head"><h2>Leaderboard</h2><span class="muted small">Ranked by robustness, not training accuracy</span></div>
      ${ranked.length ? `<div class="tbl-wrap"><table>
        <thead><tr><th>#</th><th>Experiment</th><th class="num">Train acc</th><th class="num">Test acc</th><th class="num">Test Sharpe</th>
        <th class="num">Walk-forward Sharpe</th><th class="num">Buy &amp; hold</th><th class="num">Score</th><th class="num" title="Where this run would rank if you sorted by training accuracy">Naive rank</th></tr></thead>
        <tbody>${ranked.map((e, i) => {
          const h = e.headline;
          return `<tr class="click ${i === 0 ? "top" : ""}" data-href="#/e/${e.id}">
            <td class="rank">${i + 1}</td>
            <td><b>${esc(e.name)}</b> <span class="pill ver">v${e.version}</span><div class="muted small">${esc(e.model)} · ${esc(h.dataset)}</div></td>
            <td class="num">${pct(h.train_accuracy)}</td><td class="num">${pct(h.test_accuracy)}</td>
            <td class="num ${sgn(h.test_sharpe)}">${num(h.test_sharpe)}</td><td class="num ${sgn(h.wf_mean_sharpe)}">${num(h.wf_mean_sharpe)}</td>
            <td class="num">${num(h.buy_hold_sharpe)}</td><td class="num score">${num(e.score)}</td>
            <td class="num muted">${naiveRank(e.id) || "–"}</td></tr>`;
        }).join("")}</tbody></table></div>` : `<p class="muted">No finished experiments yet.</p>`}
      ${lb.disqualified.length ? `<h3 style="margin-top:16px">Disqualified</h3><div class="tbl-wrap"><table><tbody>
        ${lb.disqualified.map((e) => `<tr class="click dq" data-href="#/e/${e.id}"><td><b>${esc(e.name)}</b></td>
          <td>${(e.headline.disqualified_reasons || []).map((r) => `<span class="pill dq">${esc(r)}</span>`).join(" ")}</td>
          <td class="num">Test Sharpe ${num(e.headline.test_sharpe)} <span class="muted">(would score ${num(e.headline.raw_score)})</span></td></tr>`).join("")}
      </tbody></table></div>` : ""}
      <details style="margin-top:12px"><summary class="small">How the score works</summary><p class="small muted">${esc(lb.formula)}</p></details>
    </section>

    <section class="panel">
      <div class="panel-head"><h2>All experiments</h2><button class="btn small" id="cmpBtn" disabled>Compare selected</button></div>
      ${exps.length ? `<div class="tbl-wrap"><table>
        <thead><tr><th><span class="sr-only">Select</span></th><th>Experiment</th><th>Status</th><th class="num">Test acc</th><th class="num">Test return</th><th class="num">Test Sharpe</th><th>Created</th></tr></thead>
        <tbody>${exps.map((e) => `<tr class="click" data-href="#/e/${e.id}">
          <td><input type="checkbox" class="pick" value="${e.id}" ${e.status !== "done" ? "disabled" : ""} aria-label="Select ${esc(e.name)}"></td>
          <td><b>${esc(e.name)}</b> <span class="pill ver">#${e.id} v${e.version}</span>${e.reproduce_of ? ` <span class="muted small">reproduces #${e.reproduce_of}</span>` : ""}<div class="muted small">${esc(e.model)} · ${e.n_features} features</div></td>
          <td>${statusCell(e)}</td>
          <td class="num">${pct(e.headline?.test_accuracy)}</td>
          <td class="num ${sgn(e.headline?.test_return)}">${pct(e.headline?.test_return)}</td>
          <td class="num ${sgn(e.headline?.test_sharpe)}">${num(e.headline?.test_sharpe)}</td>
          <td class="muted small">${ago(e.created_at)}</td></tr>`).join("")}</tbody></table></div>`
        : `<div class="empty"><p>No experiments yet.</p><a class="btn primary" href="#/p/${S.pid}/new">Configure your first experiment</a></div>`}
    </section>`;

  $$("tr.click").forEach((tr) => tr.addEventListener("click", (e) => { if (!e.target.closest("input,button,a")) location.hash = tr.dataset.href; }));
  const updateCmp = () => { $("#cmpBtn").disabled = $$(".pick:checked").length < 2; };
  $$(".pick").forEach((c) => c.addEventListener("change", updateCmp));
  $("#cmpBtn")?.addEventListener("click", () => {
    location.hash = `#/p/${S.pid}/compare?ids=${$$(".pick:checked").map((c) => c.value).join(",")}`;
  });
  if (running) {
    S.poll = setInterval(async () => {
      const ex = await api(`/api/projects/${S.pid}/experiments`);
      if (!ex.some((e) => e.status === "queued" || e.status === "running")) { await loadProjects(); route(); return; }
      ex.forEach((e) => {
        const cell = $(`tr[data-href="#/e/${e.id}"] td:nth-child(3)`);
        if (cell) cell.innerHTML = statusCell(e);
      });
    }, 1500);
  }
}

function statusCell(e) {
  if (e.status === "running" || e.status === "queued")
    return `<span class="pill running">${e.status === "queued" ? "Queued" : "Running"}</span> <span class="mini-bar"><div style="width:${Math.round((e.progress || 0) * 100)}%"></div></span>`;
  if (e.status === "failed") return `<span class="pill failed" title="${esc(e.error)}">Failed</span>`;
  if (e.headline?.disqualified) return `<span class="pill dq">Disqualified</span>`;
  if (e.headline?.reproduced === true) return `<span class="pill done">Reproduced ✓</span>`;
  if (e.headline?.reproduced === false) return `<span class="pill failed">Mismatch</span>`;
  return `<span class="pill done">Done</span>`;
}

/* ------------------------------------------------------------------ datasets */

function validationBlock(rep) {
  if (!rep) return "";
  const list = (items, cls) => (items && items.length ? `<ul class="issue-list ${cls}">${items.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "");
  return `
    ${rep.errors?.length ? `<div class="banner bad"><b>This file cannot be used.</b>${list(rep.errors)}</div>` : ""}
    ${rep.warnings?.length ? `<div class="banner warn"><b>${rep.warnings.length} issue(s) found and handled</b>${list(rep.warnings)}</div>` : ""}
    ${rep.fixes?.length ? `<div class="banner good"><b>Automatic fixes</b>${list(rep.fixes)}</div>` : ""}
    ${!rep.errors?.length && !rep.warnings?.length && !rep.fixes?.length ? `<div class="banner good"><b>Clean file.</b> No problems found.</div>` : ""}`;
}

async function renderDatasets(did) {
  const [datasets, samples] = await Promise.all([api(`/api/projects/${S.pid}/datasets`), api("/api/samples")]);
  S.datasets = datasets;
  view().innerHTML = `
    <div class="head"><div><h1>Datasets</h1><p class="lede">Daily price history. Upload a CSV or Parquet file for one stock, a file with many symbols, or a whole ZIP (like Kaggle's NIFTY-50 archive). Required columns: date and close; open, high, low and volume are used when present.</p></div></div>
    <div class="grid2">
      <form class="panel" id="upload">
        <h2>Upload market data</h2>
        <div class="dropzone" id="drop"><p>Drop a .csv, .parquet or .zip file here, or</p><input type="file" name="file" accept=".csv,.parquet,.pq,.zip,.txt,text/csv,application/zip" required aria-label="Data file"></div>
        <p class="muted small" style="margin:8px 0 0">Splits and bonus issues are detected and adjusted automatically. Renamed tickers are merged.</p>
        <div class="form-grid" style="margin-top:12px"><label class="field">Name <span class="hint">Single-stock files only; defaults to the symbol</span><input type="text" name="name" maxlength="80" placeholder="e.g. RELIANCE"></label></div>
        <div style="margin-top:12px"><button class="btn primary" type="submit">Validate and upload</button></div>
        <div id="uploadResult" style="margin-top:14px"></div>
      </form>
      <div class="panel">
        <h2>Sample files</h2>
        <p class="muted small">Synthetic data for trying things out. The files marked as broken show what validation catches.</p>
        <ul class="issue-list">${samples.map((s) => `<li><a href="/sample/${encodeURIComponent(s.name)}" download>${esc(s.name)}</a> ${s.kind === "bad" ? '<span class="muted small">(broken on purpose)</span>' : ""}</li>`).join("")}</ul>
        <p class="muted small">Real NSE data: upload Kaggle's NIFTY-50 <code>archive.zip</code> as-is, Parquet files from the Hugging Face <code>indian-market-historical-ohlcv</code> dataset, or run <code>python scripts/fetch_yfinance.py RELIANCE.NS TCS.NS</code>.</p>
      </div>
    </div>
    <section class="panel">
      <h2>In this project</h2>
      ${datasets.length ? `<div class="tbl-wrap"><table><thead><tr><th>Name</th><th>Period</th><th class="num">Rows</th><th>Validation</th><th></th></tr></thead><tbody>
        ${datasets.map((d) => `<tr class="click" data-href="#/p/${S.pid}/datasets/${d.id}"><td><b>${esc(d.name)}</b><div class="muted small">${esc(d.filename)}</div></td>
          <td>${d.start_date} → ${d.end_date}</td><td class="num">${d.rows.toLocaleString()}</td>
          <td>${d.validation.warnings.length ? `<span class="pill queued">${d.validation.warnings.length} warning(s)</span>` : `<span class="pill done">Clean</span>`}
            ${(d.validation.fixes || []).some((f) => f.includes("split/bonus")) ? ' <span class="pill ver">Split-adjusted</span>' : ""}</td>
          <td><button class="btn small danger" data-del="${d.id}">Delete</button></td></tr>`).join("")}
      </tbody></table></div>` : `<p class="muted">No datasets yet.</p>`}
    </section>
    <div id="preview"></div>`;

  $$("tr.click").forEach((tr) => tr.addEventListener("click", (e) => { if (!e.target.closest("button")) location.hash = tr.dataset.href; }));
  $$("[data-del]").forEach((b) => b.addEventListener("click", async () => {
    if (!confirm("Delete this dataset?")) return;
    try { await api(`/api/datasets/${b.dataset.del}`, { method: "DELETE" }); toast("Dataset deleted"); await loadProjects(); route(); }
    catch (err) { toast(err.message); }
  }));
  const drop = $("#drop"), input = $('input[type=file]', drop);
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { if (e.dataTransfer.files.length) input.files = e.dataTransfer.files; });
  $("#upload").addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    const out = $("#uploadResult");
    out.innerHTML = `<p class="muted">Validating and importing… large ZIP files can take a little while.</p>`;
    const btn = $('button[type=submit]', e.target);
    btn.disabled = true;
    try {
      const r = await api(`/api/projects/${S.pid}/datasets`, { method: "POST", body: fd });
      await loadProjects();
      renderNav("datasets");
      if (r.created.length === 1 && !r.skipped.some((x) => !x.ignored)) {
        toast(`Imported ${r.created[0].name}`);
        location.hash = `#/p/${S.pid}/datasets/${r.created[0].id}`;
        return;
      }
      toast(`Imported ${r.created.length} dataset(s)`);
      S.lastImport = importSummary(r);
      if (location.hash === `#/p/${S.pid}/datasets`) route(); else location.hash = `#/p/${S.pid}/datasets`;
    } catch (err) {
      btn.disabled = false;
      const skipped = err.body?.skipped || [];
      out.innerHTML = (err.report ? validationBlock(err.report) : `<div class="banner bad">${esc(err.message)}</div>`)
        + (skipped.length > 1 ? importSummary({ created: [], skipped }) : "");
    }
  });
  if (S.lastImport) { $("#uploadResult").innerHTML = S.lastImport; S.lastImport = null; }
  if (did) renderPreview(did);
}

function importSummary(r) {
  const fixes = (d) => (d.validation?.fixes || []).filter((f) => /split|Symbol changed/.test(f));
  const problems = r.skipped.filter((x) => !x.ignored), ignored = r.skipped.filter((x) => x.ignored);
  return `
    ${r.created.length ? `<div class="banner good"><b>Imported ${r.created.length} dataset(s)</b>
      <ul class="issue-list">${r.created.map((d) => `<li><a href="#/p/${S.pid}/datasets/${d.id}">${esc(d.name)}</a> · ${d.rows.toLocaleString()} rows, ${d.start_date} → ${d.end_date}${fixes(d).length ? `<br><span class="small">${esc(fixes(d).map((f) => f.split(". Without")[0]).join(" "))}</span>` : ""}</li>`).join("")}</ul></div>` : ""}
    ${problems.length ? `<div class="banner bad"><b>${problems.length} could not be imported</b><ul class="issue-list">${problems.map((x) => `<li><b>${esc(x.name)}</b>: ${esc(x.reason)}</li>`).join("")}</ul></div>` : ""}
    ${ignored.length ? `<div class="banner warn"><b>${ignored.length} skipped</b><ul class="issue-list">${ignored.map((x) => `<li>${esc(x.reason)}</li>`).join("")}</ul></div>` : ""}`;
}

async function renderPreview(did) {
  const d = await api(`/api/datasets/${did}`);
  const s = d.summary;
  const miss = d.validation.missing_by_column || {};
  $("#preview").innerHTML = `
    <section class="panel">
      <div class="panel-head"><h2>${esc(d.name)}</h2><a class="btn small primary" href="#/p/${S.pid}/new?dataset=${d.id}">Use in an experiment</a></div>
      ${validationBlock(d.validation)}
      <div class="grid3">
        <dl class="kv"><dt>Period</dt><dd>${s.start} → ${s.end}</dd><dt>Rows</dt><dd>${s.rows.toLocaleString()} (raw ${d.validation.rows_raw?.toLocaleString()})</dd>
          <dt>SHA-256</dt><dd><code>${d.sha256.slice(0, 16)}…</code></dd></dl>
        <dl class="kv"><dt>First / last close</dt><dd>${num(s.first_close)} / ${num(s.last_close)}</dd><dt>Annual volatility</dt><dd>${pct(s.annualized_volatility)}</dd>
          <dt>Up days</dt><dd>${pct(s.pct_up_days)}</dd></dl>
        <dl class="kv"><dt>Missing values (raw)</dt><dd>${Object.keys(miss).length ? Object.entries(miss).map(([k, v]) => `${k}: ${v}`).join(", ") : "none"}</dd>
          <dt>Volume</dt><dd>${s.has_volume ? "available" : "not provided"}</dd></dl>
      </div>
      <h3 style="margin-top:18px">Close price</h3><div class="chart-box"><canvas id="pxChart" aria-label="Close price chart"></canvas></div>
      ${s.has_volume ? `<h3 style="margin-top:14px">Volume</h3><div class="chart-box short"><canvas id="volChart" aria-label="Volume chart"></canvas></div>` : ""}
      <h3 style="margin-top:14px">First rows after cleaning</h3>
      <div class="tbl-wrap"><table><thead><tr>${Object.keys(d.head[0]).map((k) => `<th class="${k === "date" ? "" : "num"}">${k}</th>`).join("")}</tr></thead>
      <tbody>${d.head.map((r) => `<tr>${Object.entries(r).map(([k, v]) => `<td class="${k === "date" ? "" : "num"}">${v ?? "–"}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
    </section>`;
  makeChart($("#pxChart"), { type: "line", data: { labels: d.series.dates, datasets: [line("Close", d.series.close, cssVar("--cobalt"))] }, options: baseOptions() });
  if (s.has_volume) {
    const o = baseOptions();
    makeChart($("#volChart"), { type: "bar", data: { labels: d.series.dates, datasets: [{ label: "Volume", data: d.series.volume, backgroundColor: cssVar("--bh"), barPercentage: 1, categoryPercentage: 1 }] }, options: o });
  }
  $("#preview").scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ------------------------------------------------------------------ new experiment */

function defaultConfig() {
  const preferred = S.datasets.find((d) => d.name === "DEMO_TECH") || S.datasets[0];
  return {
    dataset_id: preferred?.id ?? null, features: [...S.meta.default_features],
    model: { type: "logistic_regression", params: {} }, target: { horizon: 1, threshold: 0 },
    split: { mode: "fraction", train_fraction: 0.7, train_start: "", train_end: "", test_start: "", test_end: "" },
    embargo_days: 0, signal: { mode: "long_only", prob_threshold: 0.5 }, costs: { commission_bps: 10, slippage_bps: 5 },
    walk_forward: { enabled: true, n_folds: 5, window: "expanding", window_days: 756 },
    generalization_dataset_ids: S.datasets.filter((d) => d.id !== preferred?.id && d.name.startsWith("DEMO_")).map((d) => d.id),
    allow_lookahead: false, seed: 42,
  };
}

function deepMerge(a, b) {
  const out = JSON.parse(JSON.stringify(a));
  for (const [k, v] of Object.entries(b || {})) {
    if (v && typeof v === "object" && !Array.isArray(v) && out[k] && typeof out[k] === "object") out[k] = deepMerge(out[k], v);
    else out[k] = v;
  }
  return out;
}

async function renderNew(q) {
  S.datasets = await api(`/api/projects/${S.pid}/datasets`);
  if (!S.datasets.length) {
    view().innerHTML = `<div class="panel empty"><h2>Upload a dataset first</h2><p>Experiments need price history to train on.</p><a class="btn primary" href="#/p/${S.pid}/datasets">Upload a dataset</a></div>`;
    return;
  }
  let cfg = defaultConfig(), parent = null, name = "", notes = "", activePreset = null;
  if (q.get("from")) {
    parent = await api(`/api/experiments/${q.get("from")}`);
    cfg = deepMerge(cfg, parent.config);
    cfg.split = { ...defaultConfig().split, ...parent.config.split };
    name = parent.name.replace(/ \(v\d+\)$/, "");
    notes = "";
  }
  if (q.get("dataset")) cfg.dataset_id = Number(q.get("dataset"));
  if (q.get("preset")) {
    const p = S.meta.presets.find((x) => x.key === q.get("preset"));
    if (p) { cfg = deepMerge(defaultConfig(), p.config); name = p.name; activePreset = p.key; }
  }

  const draw = (err = "") => {
    const m = S.meta.models[cfg.model.type];
    const groups = {};
    S.meta.features.forEach((f) => { (groups[f.group] = groups[f.group] || []).push(f); });
    const ds = S.datasets.find((d) => d.id === cfg.dataset_id);
    view().innerHTML = `
      <div class="head"><div>
        <h1>${parent ? `New version of “${esc(parent.name)}”` : "New experiment"}</h1>
        <p class="lede">${parent ? "Change what you want to test. Differences from the original are recorded with the new version." : "Pick a starting point, adjust, then run. Training and backtesting happen in the background."}</p>
      </div></div>
      ${err ? `<div class="form-error" role="alert">${esc(err)}</div>` : ""}
      ${parent ? "" : `<section class="panel"><h2>Start from a preset</h2><div class="presets">
        ${S.meta.presets.map((p) => `<button type="button" class="preset ${activePreset === p.key ? "on" : ""}" data-preset="${p.key}"><b>${esc(p.name)}</b><span>${esc(p.description)}</span></button>`).join("")}
      </div></section>`}
      <form id="expForm" novalidate>
        <section class="panel">
          <h2>Data and target</h2>
          <div class="form-grid">
            <label class="field">Experiment name<input type="text" name="name" value="${esc(name)}" maxlength="120" placeholder="Auto-named if left empty"></label>
            <label class="field">Train on dataset<select name="dataset_id">${S.datasets.map((d) => `<option value="${d.id}" ${d.id === cfg.dataset_id ? "selected" : ""}>${esc(d.name)} (${d.start_date.slice(0, 4)}–${d.end_date.slice(0, 4)})</option>`).join("")}</select></label>
            <label class="field">Predict direction over<span class="hint">Trading days ahead (1–20)</span><input type="number" name="horizon" min="1" max="20" step="1" value="${cfg.target.horizon}"></label>
            <label class="field">“Up” means return above<span class="hint">0 = any rise; 0.002 = rise above 0.2%</span><input type="number" name="threshold" step="0.001" min="-0.2" max="0.2" value="${cfg.target.threshold}"></label>
          </div>
        </section>

        <section class="panel">
          <fieldset><legend>Features <span class="muted small" id="featCount"></span></legend>
          <div class="feat-groups">${Object.entries(groups).map(([g, fs]) => `
            <div class="feat-group ${fs[0].leaky ? "leaky" : ""}"><h4>${esc(g)}</h4>
              ${fs.map((f) => `<label class="check"><input type="checkbox" name="feat" value="${f.name}" ${cfg.features.includes(f.name) ? "checked" : ""} ${f.leaky && !cfg.allow_lookahead ? "disabled" : ""}>
                <span>${esc(f.name)}<small>${esc(f.description)}</small></span></label>`).join("")}
            </div>`).join("")}
          </div></fieldset>
          <label class="check" style="margin-top:14px"><input type="checkbox" name="allow_lookahead" ${cfg.allow_lookahead ? "checked" : ""}>
            <span>Allow look-ahead features (demo only)<small>Unlocks the leaky features so you can see what leakage does. Results are flagged and disqualified.</small></span></label>
          ${ds && !ds.has_volume ? `<p class="small muted">This dataset has no volume, so volume features will be rejected.</p>` : ""}
        </section>

        <section class="panel">
          <h2>Model</h2>
          <div class="form-grid">
            <label class="field">Model type<select name="model_type">${Object.entries(S.meta.models).map(([k, v]) => `<option value="${k}" ${k === cfg.model.type ? "selected" : ""}>${esc(v.label)}</option>`).join("")}</select></label>
            ${Object.entries(m.params).map(([p, spec]) => {
              const v = cfg.model.params[p] !== undefined ? cfg.model.params[p] : spec.default;
              return `<label class="field">${esc(p.replace(/_/g, " "))}<span class="hint">${esc(spec.help)}</span>
                <input type="number" name="param_${p}" value="${v ?? ""}" step="${spec.type === "int" ? 1 : "any"}" min="${spec.min}" max="${spec.max}" ${spec.nullable || spec.default === null ? 'placeholder="unlimited"' : ""}></label>`;
            }).join("")}
            <label class="field">Random seed<span class="hint">Same seed + config + data = same result</span><input type="number" name="seed" value="${cfg.seed}" step="1"></label>
          </div>
        </section>

        <section class="panel">
          <h2>Evaluation</h2>
          <div class="form-grid">
            <label class="field">Split<select name="split_mode"><option value="fraction" ${cfg.split.mode === "fraction" ? "selected" : ""}>By fraction of history</option><option value="date" ${cfg.split.mode === "date" ? "selected" : ""}>By dates</option></select></label>
            ${cfg.split.mode === "fraction"
              ? `<label class="field">Training share<span class="hint">Earliest part of history used to train</span><input type="number" name="train_fraction" min="0.3" max="0.9" step="0.05" value="${cfg.split.train_fraction}"></label>`
              : `<label class="field">Train start <span class="hint">Optional</span><input type="date" name="train_start" value="${cfg.split.train_start || ""}"></label>
                 <label class="field">Train end<input type="date" name="train_end" value="${cfg.split.train_end || ""}" required></label>
                 <label class="field">Test start <span class="hint">Default: day after train end</span><input type="date" name="test_start" value="${cfg.split.test_start || ""}"></label>
                 <label class="field">Test end <span class="hint">Optional</span><input type="date" name="test_end" value="${cfg.split.test_end || ""}"></label>`}
            <label class="field">Embargo<span class="hint">Extra days skipped after training</span><input type="number" name="embargo_days" min="0" max="60" step="1" value="${cfg.embargo_days}"></label>
          </div>
          <div class="form-grid" style="margin-top:14px">
            <label class="check"><input type="checkbox" name="wf_enabled" ${cfg.walk_forward.enabled ? "checked" : ""}><span>Walk-forward evaluation<small>Retrain and test on several later periods</small></span></label>
            <label class="field">Folds<input type="number" name="wf_folds" min="2" max="12" step="1" value="${cfg.walk_forward.n_folds}"></label>
            <label class="field">Training window<select name="wf_window"><option value="expanding" ${cfg.walk_forward.window === "expanding" ? "selected" : ""}>Expanding (all past data)</option><option value="rolling" ${cfg.walk_forward.window === "rolling" ? "selected" : ""}>Rolling (fixed length)</option></select></label>
            <label class="field">Rolling window length<span class="hint">Trading days</span><input type="number" name="wf_days" min="120" step="1" value="${cfg.walk_forward.window_days}"></label>
          </div>
          <h3 style="margin-top:16px">Test on unseen assets</h3>
          <div class="form-grid">${S.datasets.filter((d) => d.id !== cfg.dataset_id).map((d) => `<label class="check"><input type="checkbox" name="gen" value="${d.id}" ${cfg.generalization_dataset_ids.includes(d.id) ? "checked" : ""}><span>${esc(d.name)}</span></label>`).join("") || '<p class="muted small">Upload more datasets to test generalization.</p>'}</div>
        </section>

        <section class="panel">
          <h2>Trading rule and costs</h2>
          <div class="form-grid">
            <label class="field">Positions<select name="signal_mode"><option value="long_only" ${cfg.signal.mode === "long_only" ? "selected" : ""}>Long or cash</option><option value="long_short" ${cfg.signal.mode === "long_short" ? "selected" : ""}>Long or short</option></select></label>
            <label class="field">Go long when P(up) ≥<input type="number" name="prob_threshold" min="0.3" max="0.8" step="0.01" value="${cfg.signal.prob_threshold}"></label>
            <label class="field">Fees and taxes<span class="hint">bps per trade; NSE delivery STT alone is 10</span><input type="number" name="commission_bps" min="0" max="200" step="0.5" value="${cfg.costs.commission_bps}"></label>
            <label class="field">Slippage<span class="hint">Basis points per trade</span><input type="number" name="slippage_bps" min="0" max="200" step="0.5" value="${cfg.costs.slippage_bps}"></label>
          </div>
          <p class="muted small" style="margin-top:10px">The position for each day is decided at the previous day's close, so a prediction can never trade on the price it was made from.</p>
          <label class="field" style="margin-top:12px">Notes <span class="hint">Your hypothesis, saved with the run</span><textarea name="notes">${esc(notes)}</textarea></label>
        </section>
        <div class="actions"><button class="btn primary" type="submit">Run experiment</button><a class="btn" href="#/p/${S.pid}">Cancel</a></div>
      </form>`;

    const updateCount = () => { $("#featCount").textContent = `(${$$('input[name=feat]:checked').length} selected)`; };
    updateCount();
    $$('input[name=feat]').forEach((c) => c.addEventListener("change", updateCount));
    $$("[data-preset]").forEach((b) => b.addEventListener("click", () => {
      const p = S.meta.presets.find((x) => x.key === b.dataset.preset);
      const keepDs = collect().dataset_id;
      cfg = deepMerge(defaultConfig(), p.config);
      cfg.dataset_id = keepDs;
      cfg.generalization_dataset_ids = defaultConfig().generalization_dataset_ids.filter((x) => x !== keepDs);
      name = p.name; notes = p.description; activePreset = p.key;
      draw();
    }));
    const redrawOn = (sel) => $(sel).addEventListener("change", () => { cfg = collect(); name = $('input[name=name]').value; notes = $('textarea[name=notes]').value; draw(); });
    redrawOn('select[name=model_type]');
    redrawOn('select[name=split_mode]');
    redrawOn('select[name=dataset_id]');
    redrawOn('input[name=allow_lookahead]');
    $("#expForm").addEventListener("submit", async (e) => {
      e.preventDefault();
      const c = collect();
      const btn = $('button[type=submit]', e.target);
      btn.disabled = true; btn.textContent = "Starting…";
      try {
        const exp = await api(`/api/projects/${S.pid}/experiments`, {
          json: { name: $('input[name=name]').value, notes: $('textarea[name=notes]').value, parent_id: parent?.id, config: c },
        });
        await loadProjects();
        location.hash = `#/e/${exp.id}`;
      } catch (err) {
        cfg = c; name = $('input[name=name]').value; notes = $('textarea[name=notes]').value;
        draw(err.message);
        window.scrollTo({ top: 0, behavior: "smooth" });
      }
    });
  };

  const collect = () => {
    const f = $("#expForm");
    const val = (n) => f.elements[n]?.value;
    const numv = (n) => (val(n) === "" || val(n) == null ? null : Number(val(n)));
    const mtype = val("model_type");
    const params = {};
    Object.keys(S.meta.models[mtype].params).forEach((p) => {
      if (mtype !== cfg.model.type) return;  // switching model: fall back to defaults
      params[p] = numv(`param_${p}`);
    });
    const mode = val("split_mode");
    const split = mode === cfg.split.mode
      ? (mode === "fraction" ? { mode, train_fraction: numv("train_fraction") }
        : { mode, train_start: val("train_start") || null, train_end: val("train_end") || null, test_start: val("test_start") || null, test_end: val("test_end") || null })
      : { mode, train_fraction: 0.7 };
    const allow = f.elements.allow_lookahead.checked;
    const feats = $$('input[name=feat]:checked', f).map((c) => c.value).filter((n) => allow || !S.meta.features.find((x) => x.name === n).leaky);
    const ds = Number(val("dataset_id"));
    return {
      dataset_id: ds, features: feats, model: { type: mtype, params },
      target: { horizon: numv("horizon"), threshold: numv("threshold") ?? 0 },
      split, embargo_days: numv("embargo_days") ?? 0,
      signal: { mode: val("signal_mode"), prob_threshold: numv("prob_threshold") },
      costs: { commission_bps: numv("commission_bps") ?? 0, slippage_bps: numv("slippage_bps") ?? 0 },
      walk_forward: { enabled: f.elements.wf_enabled.checked, n_folds: numv("wf_folds"), window: val("wf_window"), window_days: numv("wf_days") },
      generalization_dataset_ids: $$('input[name=gen]:checked', f).map((c) => Number(c.value)).filter((x) => x !== ds),
      allow_lookahead: allow, seed: numv("seed") ?? 42,
    };
  };
  draw();
}

/* ------------------------------------------------------------------ experiment detail */

function renderExperiment(exp) {
  if (exp.status === "queued" || exp.status === "running") return renderRunning(exp);
  const head = `
    <div class="head">
      <div>
        <h1>${esc(exp.name)}</h1>
        <p class="lede" style="margin-bottom:0">#${exp.id} <span class="pill ver">v${exp.version}</span> · ${esc(exp.model)} · ${exp.n_features} features
          ${exp.reproduce_of ? ` · reproduction of <a href="#/e/${exp.reproduce_of}">#${exp.reproduce_of}</a>` : ""}
          ${exp.parent_id ? ` · new version of <a href="#/e/${exp.parent_id}">#${exp.parent_id}</a>` : ""}</p>
        ${exp.notes ? `<p class="muted small" style="max-width:72ch">${esc(exp.notes)}</p>` : ""}
      </div>
      <div class="actions">
        ${exp.status === "done" ? `<a class="btn" href="/api/experiments/${exp.id}/report" target="_blank" rel="noopener">View report</a>
          <a class="btn" href="/api/experiments/${exp.id}/report?download=1">Download report</a>
          <button class="btn" id="reproBtn">Reproduce</button>` : ""}
        <a class="btn primary" href="#/p/${exp.project_id}/new?from=${exp.id}">Edit and run new version</a>
        <button class="btn" id="renameBtn">Rename</button>
        <button class="btn danger" id="delBtn">Delete</button>
      </div>
    </div>`;
  if (exp.status === "failed") {
    view().innerHTML = `${head}<div class="banner bad"><b>The experiment could not run.</b><br>${esc(exp.error)}</div>
      <div class="panel"><h2>Configuration</h2><pre class="json">${esc(JSON.stringify(exp.config, null, 2))}</pre></div>`;
    bindExpActions(exp);
    return;
  }
  const r = exp.results, pm = r.predictive, st = r.strategy, bh = r.benchmark, rob = r.robustness, sp = r.split;
  const leaky = Object.entries(r.leakage.features).filter(([, v]) => v.leaky).map(([k]) => k);
  const rc = r.reproduction_check;
  const ci = st.test.sharpe_ci95 || [null, null];

  view().innerHTML = `${head}
    ${rob.disqualified ? `<div class="banner bad"><b>Disqualified from the leaderboard.</b> ${esc(rob.disqualified_reasons.join("; "))}. ${leaky.length ? "Every number below is inflated by information from the future." : ""}</div>` : ""}
    ${rc ? `<div class="banner ${rc.reproduced ? "good" : "bad"}"><b>${rc.reproduced ? "Reproduced exactly." : "Reproduction mismatch."}</b>
      Re-ran experiment <a href="#/e/${rc.original_id}">#${rc.original_id}</a> from its saved configuration:
      config ${rc.config_hash_match ? "✓" : "✗"} · dataset checksum ${rc.dataset_match ? "✓" : "✗"} · predictions ${rc.predictions_match ? "✓" : "✗"} · metrics ${rc.metrics_match ? "✓" : "✗"}.
      Test Sharpe ${num(rc.original_test_sharpe, 4)} → ${num(rc.new_test_sharpe, 4)}.</div>` : ""}

    <section class="panel">
      <div class="panel-head"><h2>Before and after the model saw the future</h2>
        <span class="muted small">Training ${sp.train_start} → ${sp.train_end} · Test ${sp.test_start} → ${sp.test_end}</span></div>
      <div class="hero-chart"><canvas id="heroChart" aria-label="Equity curve across training and test periods"></canvas></div>
      <div class="legend"><span><i style="background:${cssVar("--cobalt")}"></i>Strategy, net of costs</span><span><i class="dash"></i>Buy and hold</span>
        <span class="muted">Log scale. Shaded area is the training period, where the model has already seen the answers.</span></div>
    </section>

    <div class="grid3" style="margin-bottom:18px">
      ${questionPanel("Does it predict direction?", pct(pm.test.accuracy), `test accuracy vs ${pct(pm.test.majority_baseline_accuracy)} for always guessing the majority`,
        [["Edge over naive guess", pct(pm.test.edge_vs_majority), pct(pm.train.edge_vs_majority)], ["ROC AUC", num(pm.test.roc_auc, 3), num(pm.train.roc_auc, 3)],
         ["Precision", pct(pm.test.precision), pct(pm.train.precision)], ["Accuracy", pct(pm.test.accuracy), pct(pm.train.accuracy)]], "Test", "Train", sgn(pm.test.edge_vs_majority))}
      ${questionPanel("Does trading it make money?", pct(st.test.total_return), `test return after costs vs ${pct(bh.test.total_return)} buy and hold`,
        [["CAGR", pct(st.test.cagr), pct(bh.test.cagr)], ["Before costs", pct(st.test.gross_total_return), "–"], ["Costs paid", pct(st.test.total_costs), pct(bh.test.total_costs)],
         ["Position changes", st.test.n_trades, bh.test.n_trades], ["Time in market", pct(st.test.exposure, 0), pct(bh.test.exposure, 0)]], "Strategy", "Buy & hold", sgn(st.test.total_return - bh.test.total_return))}
      ${questionPanel("Is it worth the risk?", num(st.test.sharpe), `test Sharpe, 95% CI [${num(ci[0])}, ${num(ci[1])}]; buy and hold ${num(bh.test.sharpe)}`,
        [["Sortino", num(st.test.sortino), num(bh.test.sortino)], ["Max drawdown", pct(st.test.max_drawdown), pct(bh.test.max_drawdown)], ["Calmar", num(st.test.calmar), num(bh.test.calmar)],
         ["Volatility", pct(st.test.annual_volatility), pct(bh.test.annual_volatility)], ["Sharpe before costs", num(st.test.gross_sharpe), "–"]], "Strategy", "Buy & hold", sgn(st.test.sharpe - bh.test.sharpe))}
    </div>

    ${r.latest_signal ? `<div class="grid2" style="margin-bottom:18px">
      <section class="panel q"><p class="ask">Latest signal, after the last day of data</p>
        <div class="big ${r.latest_signal.direction === "UP" ? "pos" : r.latest_signal.direction === "DOWN" ? "neg" : ""}">${esc(r.latest_signal.direction)}</div>
        <div class="vs">P(up over the next ${r.latest_signal.horizon_days} day${r.latest_signal.horizon_days > 1 ? "s" : ""}) = ${pct(r.latest_signal.prob_up)} · as of ${r.latest_signal.as_of}</div>
        <p class="muted small" style="margin:0">${esc(r.latest_signal.note)} ${rob.disqualified ? "<b>This run is disqualified, so ignore this signal.</b>" : ""}</p>
      </section>
      <section class="panel"><h2>Predicted vs what actually happened</h2>
        <p class="muted small">Last ${r.recent_predictions.length} days of the test period. The model never saw these outcomes.</p>
        <div class="tbl-wrap"><table><thead><tr><th>Date</th><th class="num">P(up)</th><th>Predicted</th><th class="num">Actual return</th><th></th></tr></thead><tbody>
        ${r.recent_predictions.map((p) => `<tr><td class="small">${p.date}</td><td class="num">${pct(p.prob_up)}</td><td>${p.predicted}</td>
          <td class="num ${sgn(p.actual_return)}">${pct(p.actual_return, 2)}</td><td class="${p.correct ? "pos" : "neg"}">${p.correct ? "✓" : "✗"}</td></tr>`).join("")}
        </tbody></table></div>
      </section>
    </div>` : ""}

    <div class="grid2" style="margin-bottom:18px">
      <section class="panel"><h2>What the evidence says</h2><ul class="findings">${r.diagnostics.map((d) => `<li class="${d.level}"><b>${esc(d.title)}</b><span>${esc(d.detail)}</span></li>`).join("")}</ul></section>
      <section class="panel">
        <div class="panel-head"><h2>Robustness score</h2><span class="score">${rob.disqualified ? "Disqualified" : num(rob.score, 3)}</span></div>
        <table><tbody>
          <tr><td>Out-of-sample Sharpe ${r.walk_forward?.summary ? "(walk-forward mean)" : "(test)"}</td><td class="num">${num(rob.components.oos_sharpe, 3)}</td></tr>
          <tr><td>Instability penalty</td><td class="num">${num(rob.components.instability_penalty, 3)}</td></tr>
          <tr><td>Overfitting penalty (train − test accuracy)</td><td class="num">${num(rob.components.overfit_penalty, 3)}</td></tr>
          <tr><td>Bonus for beating buy and hold</td><td class="num">${num(rob.components.benchmark_bonus, 3)}</td></tr>
        </tbody></table>
        <h3 style="margin-top:16px">Drawdown in the test period</h3>
        <div class="chart-box short"><canvas id="ddChart" aria-label="Drawdown chart"></canvas></div>
      </section>
    </div>

    <section class="panel">
      <div class="panel-head"><h2>Walk-forward evaluation</h2>${r.walk_forward?.summary ? `<span class="muted small">${esc(r.walk_forward.window)} window · positive in ${pct(r.walk_forward.summary.pct_folds_positive, 0)} of folds · beat buy and hold in ${pct(r.walk_forward.summary.pct_folds_beat_buy_hold, 0)}</span>` : ""}</div>
      ${r.walk_forward?.folds ? `<div class="grid2"><div class="chart-box"><canvas id="wfChart" aria-label="Sharpe by fold"></canvas></div>
        <div class="tbl-wrap"><table><thead><tr><th>Fold</th><th>Test period</th><th class="num">Train acc</th><th class="num">Test acc</th><th class="num">Sharpe</th><th class="num">B&amp;H</th></tr></thead><tbody>
        ${r.walk_forward.folds.map((f) => `<tr><td>${f.fold}</td><td class="small">${f.test_start} → ${f.test_end}</td><td class="num">${pct(f.train_accuracy)}</td><td class="num">${pct(f.accuracy)}</td>
          <td class="num ${sgn(f.strategy_sharpe)}">${num(f.strategy_sharpe)}</td><td class="num">${num(f.buy_hold_sharpe)}</td></tr>`).join("")}
        </tbody></table></div></div>`
        : `<p class="muted">${esc(r.walk_forward?.skipped || "Walk-forward evaluation was turned off for this run.")}</p>`}
    </section>

    <div class="grid2" style="margin-bottom:18px">
      <section class="panel"><h2>Across market regimes</h2>
        <p class="muted small">Regimes are labelled with information available at the time: price vs its 200-day average, and 20-day volatility vs the training-period median.</p>
        ${r.regimes.length ? `<div class="tbl-wrap"><table><thead><tr><th>Regime</th><th class="num">Days</th><th class="num">Accuracy</th><th class="num">Sharpe</th><th class="num">B&amp;H</th></tr></thead><tbody>
        ${r.regimes.map((g) => `<tr><td>${esc(g.regime)}</td><td class="num">${g.days}</td><td class="num">${pct(g.accuracy)}</td><td class="num ${sgn(g.strategy_sharpe)}">${num(g.strategy_sharpe)}</td><td class="num">${num(g.buy_hold_sharpe)}</td></tr>`).join("")}
        </tbody></table></div>` : `<p class="muted">Not enough test days in each regime.</p>`}
      </section>
      <section class="panel"><h2>On assets the model never saw</h2>
        <p class="muted small">The trained model is applied unchanged to other datasets.</p>
        ${r.generalization.length ? `<div class="tbl-wrap"><table><thead><tr><th>Asset</th><th class="num">Accuracy</th><th class="num">Sharpe</th><th class="num">B&amp;H</th></tr></thead><tbody>
        ${r.generalization.map((g) => `<tr><td>${esc(g.name)}<div class="muted small">${g.start} → ${g.end}</div></td><td class="num">${pct(g.accuracy)}</td>
          <td class="num ${sgn(g.strategy_sharpe)}">${num(g.strategy_sharpe)}</td><td class="num">${num(g.buy_hold_sharpe)}</td></tr>`).join("")}</tbody></table></div>`
        : `<p class="muted">No other assets were selected for this run.</p>`}
      </section>
    </div>

    <div class="grid2" style="margin-bottom:18px">
      <section class="panel"><h2>What the model relies on</h2>
        ${r.feature_importance.length ? `<p class="muted small">${esc(r.feature_importance[0].kind)}</p>` + r.feature_importance.slice(0, 12).map((f) => `<div class="hbar"><span>${esc(f.feature)}${f.feature.endsWith("LEAK") ? ' <span class="tag-leak">leak</span>' : ""}</span>
          <div class="track"><div class="fill" style="width:${(f.importance * 100 / r.feature_importance[0].importance).toFixed(1)}%;${f.feature.endsWith("LEAK") ? `background:${cssVar("--loss")}` : ""}"></div></div><span class="num">${pct(f.importance, 0)}</span></div>`).join("") : '<p class="muted">Not available for this model.</p>'}
      </section>
      <section class="panel"><h2>Look-ahead check</h2>
        <p class="muted small">${esc(r.leakage.method)}</p>
        ${leaky.length ? `<div class="banner bad"><b>${leaky.length} feature(s) use future data:</b> ${esc(leaky.join(", "))}</div>`
          : `<div class="banner good"><b>All ${Object.keys(r.leakage.features).length} features passed.</b> None of them changes value when future rows are added.</div>`}
        <details><summary class="small">Per-feature results</summary><div class="tbl-wrap"><table><tbody>${Object.entries(r.leakage.features).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="${v.leaky ? "neg" : "pos"}">${v.leaky ? "Uses future data" : "Causal"}</td><td class="muted small">checked at ${v.checked_at.length} dates</td></tr>`).join("")}</tbody></table></div></details>
        <p class="muted small">Split safeguards: ${sp.purged_rows} overlapping training row(s) purged, ${sp.embargo_rows} embargo row(s); scaler fitted on training data only.</p>
      </section>
    </div>

    <section class="panel"><h2>Reproducibility</h2>
      <dl class="kv">
        <dt>Config hash</dt><dd><code>${r.reproducibility.config_hash}</code></dd>
        <dt>Dataset SHA-256</dt><dd><code>${r.reproducibility.dataset_sha256}</code></dd>
        <dt>Result fingerprint</dt><dd><code>${r.reproducibility.result_fingerprint}</code></dd>
        <dt>Seed</dt><dd>${r.reproducibility.seed}</dd>
        <dt>Environment</dt><dd>${Object.entries(r.reproducibility.environment).map(([k, v]) => `${k} ${v}`).join(", ")}</dd>
        <dt>Run time</dt><dd>${r.duration_seconds}s</dd>
      </dl>
      ${exp.config_diff ? `<h3 style="margin-top:14px">Changes from <a href="#/e/${exp.parent_id}">#${exp.parent_id}</a></h3>${diffTable(exp.config_diff)}` : ""}
      ${exp.lineage.length > 1 ? `<h3 style="margin-top:14px">Version history</h3><div class="tbl-wrap"><table><tbody>${exp.lineage.map((l) => `<tr class="click" data-href="#/e/${l.id}"><td>v${l.version}</td><td>${esc(l.name)} ${l.id === exp.id ? '<span class="muted">(this)</span>' : ""}</td><td class="num">${num(l.headline?.test_sharpe)}</td><td class="num">${num(l.score)}</td></tr>`).join("")}</tbody></table></div>` : ""}
      <details style="margin-top:12px"><summary>Full configuration</summary><pre class="json">${esc(JSON.stringify(exp.config, null, 2))}</pre></details>
    </section>`;

  bindExpActions(exp);
  $$("tr.click").forEach((tr) => tr.addEventListener("click", () => { location.hash = tr.dataset.href; }));
  drawHero(r);
  const o = baseOptions({ yPct: true });
  const t = r.curves.test, tb = r.curves.test_buy_hold;
  makeChart($("#ddChart"), { type: "line", data: { labels: t.dates, datasets: [line("Strategy", t.drawdown, cssVar("--loss")), line("Buy and hold", tb.drawdown, cssVar("--bh"), true, 1.4)] }, options: o });
  if (r.walk_forward?.folds) {
    const wo = baseOptions(); wo.plugins.legend = { display: true, labels: { color: cssVar("--ink-2"), boxWidth: 12 } };
    wo.scales.y.ticks.callback = (v) => v;
    makeChart($("#wfChart"), {
      type: "bar",
      data: {
        labels: r.walk_forward.folds.map((f) => `Fold ${f.fold}`),
        datasets: [
          { label: "Strategy Sharpe", data: r.walk_forward.folds.map((f) => f.strategy_sharpe), backgroundColor: cssVar("--cobalt") },
          { label: "Buy & hold Sharpe", data: r.walk_forward.folds.map((f) => f.buy_hold_sharpe), backgroundColor: cssVar("--bh") },
        ],
      },
      options: wo,
    });
  }
}

function questionPanel(ask, big, vs, rows, colA, colB, cls) {
  return `<section class="panel q"><p class="ask">${esc(ask)}</p><div class="big ${cls}">${big}</div><div class="vs">${esc(vs)}</div>
    <dl><dt></dt><dd class="dlh">${colA}</dd><dd class="dlh">${colB}</dd>
    ${rows.map(([k, a, b]) => `<dt>${esc(k)}</dt><dd>${a ?? "–"}</dd><dd class="sub">${b ?? "–"}</dd>`).join("")}</dl></section>`;
}

function diffTable(diff) {
  if (!diff.length) return `<p class="muted small">Identical configuration.</p>`;
  const fmt = (v) => (v === null || v === undefined ? "–" : typeof v === "object" ? JSON.stringify(v) : String(v));
  const row = (d) => {
    if (Array.isArray(d.before) && Array.isArray(d.after)) {
      const add = d.after.filter((x) => !d.before.includes(x)), rem = d.before.filter((x) => !d.after.includes(x));
      return `<tr><td>${esc(d.field)}</td><td class="neg">${rem.length ? "− " + esc(rem.join(", ")) : "–"}</td><td class="pos">${add.length ? "+ " + esc(add.join(", ")) : "–"}</td></tr>`;
    }
    return `<tr><td>${esc(d.field)}</td><td class="muted">${esc(fmt(d.before))}</td><td>${esc(fmt(d.after))}</td></tr>`;
  };
  return `<div class="tbl-wrap"><table><thead><tr><th>Setting</th><th>Before</th><th>After</th></tr></thead><tbody>
    ${diff.map(row).join("")}</tbody></table></div>`;
}

function drawHero(r) {
  const tr = r.curves.train, te = r.curves.test, trb = r.curves.train_buy_hold, teb = r.curves.test_buy_hold;
  const lastS = tr.equity[tr.equity.length - 1], lastB = trb.equity[trb.equity.length - 1];
  const labels = [...tr.dates, ...te.dates];
  const strat = [...tr.equity, ...te.equity.map((v) => v * lastS)];
  const bhv = [...trb.equity, ...teb.equity.map((v) => v * lastB)];
  const o = baseOptions({ log: true });
  o.plugins.divider = { index: tr.dates.length, left: "Training (answers seen)", right: "Test (never seen)" };
  makeChart($("#heroChart"), {
    type: "line",
    data: { labels, datasets: [line("Strategy (net)", strat, cssVar("--cobalt"), false, 2.2), line("Buy and hold", bhv, cssVar("--bh"), true, 1.4)] },
    options: o, plugins: [dividerPlugin],
  });
}

function bindExpActions(exp) {
  $("#renameBtn")?.addEventListener("click", async () => {
    const name = prompt("New name for this experiment", exp.name);
    if (!name || !name.trim() || name.trim() === exp.name) return;
    try { await api(`/api/experiments/${exp.id}`, { method: "PATCH", json: { name: name.trim() } }); toast("Renamed"); route(); }
    catch (e) { toast(e.message); }
  });
  $("#delBtn")?.addEventListener("click", async () => {
    if (!confirm("Delete this experiment? Its saved results will be lost.")) return;
    try { await api(`/api/experiments/${exp.id}`, { method: "DELETE" }); await loadProjects(); toast("Experiment deleted"); location.hash = `#/p/${exp.project_id}`; }
    catch (e) { toast(e.message); }
  });
  $("#reproBtn")?.addEventListener("click", async (ev) => {
    ev.target.disabled = true;
    const r = await api(`/api/experiments/${exp.id}/reproduce`, { method: "POST" });
    await loadProjects();
    location.hash = `#/e/${r.id}`;
  });
}

function renderRunning(exp) {
  view().innerHTML = `
    <div class="head"><div><h1>${esc(exp.name)}</h1><p class="lede">#${exp.id} · ${esc(exp.model)} · ${exp.n_features} features</p></div></div>
    <div class="progress" aria-live="polite">
      <h2 id="stage">${esc(exp.stage || "Queued")}</h2>
      <div class="bar"><div id="barFill" style="width:${Math.round((exp.progress || 0) * 100)}%"></div></div>
      <p class="muted small"><span id="pctTxt">${Math.round((exp.progress || 0) * 100)}%</span> · runs in the background, so you can leave this page and come back.</p>
    </div>
    <div class="panel"><h2>Configuration</h2><pre class="json">${esc(JSON.stringify(exp.config, null, 2))}</pre></div>`;
  let finished = false;
  const show = (stage, progress) => {
    const s = $("#stage"), b = $("#barFill"), t = $("#pctTxt");
    if (!s || !b || !t) return;  // user already navigated away
    s.textContent = stage;
    b.style.width = `${Math.round((progress || 0) * 100)}%`;
    t.textContent = `${Math.round((progress || 0) * 100)}%`;
  };
  const finish = async () => {
    if (finished) return;
    finished = true;
    if (S.es) { S.es.close(); S.es = null; }
    if (S.poll) { clearInterval(S.poll); S.poll = null; }
    await loadProjects();
    if (location.hash === `#/e/${exp.id}`) route();
  };
  const es = new EventSource(`/api/experiments/${exp.id}/events`);
  S.es = es;
  es.onmessage = (m) => {
    const st = JSON.parse(m.data);
    show(st.stage || st.status, st.progress);
    if (st.status === "done" || st.status === "failed") finish();
  };
  es.onerror = () => {  // stream closed or blocked by a proxy: fall back to polling
    es.close();
    if (S.es === es) S.es = null;
    if (finished || S.poll || location.hash !== `#/e/${exp.id}`) return;
    S.poll = setInterval(async () => {
      const e = await api(`/api/experiments/${exp.id}`);
      if (e.status === "done" || e.status === "failed") finish();
      else show(e.stage, e.progress);
    }, 1000);
  };
}

/* ------------------------------------------------------------------ compare */

async function renderCompare(q) {
  const exps = (await api(`/api/projects/${S.pid}/experiments`)).filter((e) => e.status === "done");
  const ids = (q.get("ids") || "").split(",").filter(Boolean).map(Number);
  view().innerHTML = `
    <div class="head"><div><h1>Compare experiments</h1><p class="lede">Pick two to six finished experiments. Every run is scored by the same backtest rules and costs.</p></div></div>
    <form class="panel" id="cmpForm"><div class="form-grid">${exps.map((e) => `<label class="check"><input type="checkbox" name="id" value="${e.id}" ${ids.includes(e.id) ? "checked" : ""}><span>${esc(e.name)}<small>#${e.id} v${e.version} · ${esc(e.model)}</small></span></label>`).join("") || '<p class="muted">No finished experiments yet.</p>'}</div>
      <div style="margin-top:12px"><button class="btn primary" type="submit">Compare</button></div></form>
    <div id="cmpOut"></div>`;
  $("#cmpForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const sel = $$('input[name=id]:checked').map((c) => c.value);
    location.hash = `#/p/${S.pid}/compare?ids=${sel.join(",")}`;
  });
  if (ids.length < 2) return;
  let data;
  try { data = await api(`/api/compare?ids=${ids.join(",")}`); }
  catch (e) { $("#cmpOut").innerHTML = `<div class="banner bad">${esc(e.message)}</div>`; return; }
  const palette = [cssVar("--cobalt"), cssVar("--gain"), cssVar("--loss"), cssVar("--caution"), "#8B5CF6", "#0E7490"];
  const rows = [
    ["Train accuracy", (r) => pct(r.predictive.train.accuracy)], ["Test accuracy", (r) => pct(r.predictive.test.accuracy)],
    ["Test ROC AUC", (r) => num(r.predictive.test.roc_auc, 3)], ["Test return (net)", (r) => pct(r.strategy.test.total_return)],
    ["Buy & hold return", (r) => pct(r.benchmark.test.total_return)], ["Costs paid", (r) => pct(r.strategy.test.total_costs)],
    ["Test Sharpe", (r) => num(r.strategy.test.sharpe)], ["Sharpe 95% CI", (r) => `[${num(r.strategy.test.sharpe_ci95?.[0])}, ${num(r.strategy.test.sharpe_ci95?.[1])}]`],
    ["Max drawdown", (r) => pct(r.strategy.test.max_drawdown)], ["Walk-forward Sharpe", (r) => num(r.walk_forward?.summary?.mean_sharpe)],
    ["Robustness score", (r) => (r.robustness.disqualified ? "Disqualified" : num(r.robustness.score, 3))],
  ];
  $("#cmpOut").innerHTML = `
    <section class="panel"><h2>Test-period equity</h2><div class="hero-chart"><canvas id="cmpChart" aria-label="Equity comparison"></canvas></div>
      <div class="legend">${data.map((e, i) => `<span><i style="background:${palette[i]}"></i>${esc(e.name)}</span>`).join("")}<span><i class="dash"></i>Buy and hold (${esc(data[0].results.dataset.name)})</span></div></section>
    <section class="panel"><h2>Side by side</h2><div class="tbl-wrap"><table><thead><tr><th></th>${data.map((e) => `<th class="num"><a href="#/e/${e.id}">${esc(e.name)}</a></th>`).join("")}</tr></thead><tbody>
      ${rows.map(([k, f]) => `<tr><td>${k}</td>${data.map((e) => `<td class="num">${f(e.results)}</td>`).join("")}</tr>`).join("")}</tbody></table></div></section>
    ${data.slice(1).map((e) => `<section class="panel"><h3>Settings that differ: ${esc(e.name)} vs ${esc(data[0].name)}</h3>${diffTable(e.config_diff_vs_first)}</section>`).join("")}`;
  const allDates = [...new Set(data.flatMap((e) => e.results.curves.test.dates))].sort();
  const series = data.map((e, i) => {
    const m = new Map(e.results.curves.test.dates.map((d, j) => [d, e.results.curves.test.equity[j]]));
    return line(e.name, allDates.map((d) => m.get(d) ?? null), palette[i]);
  });
  const b = data[0].results.curves.test_buy_hold, bm = new Map(b.dates.map((d, j) => [d, b.equity[j]]));
  series.push(line("Buy and hold", allDates.map((d) => bm.get(d) ?? null), cssVar("--bh"), true, 1.4));
  makeChart($("#cmpChart"), { type: "line", data: { labels: allDates, datasets: series }, options: baseOptions({ log: true }) });
  renderPortfolioPanel(data);
}

function renderPortfolioPanel(data) {
  const host = document.createElement("section");
  host.className = "panel";
  host.innerHTML = `
    <div class="panel-head"><h2>Combine into a portfolio</h2><span class="muted small">One signal per stock; uses each experiment’s saved test-period positions</span></div>
    <p class="muted small" style="max-width:80ch">Each day, every stock whose signal is active gets an equal share of the allowed exposure, capped per stock; the rest stays in cash. Compared with equal-weight buy-and-hold of the same stocks.</p>
    <form id="pfForm" class="form-grid" style="margin-top:8px">
      <label class="field">Max weight per stock<span class="hint">e.g. 0.25 = 25%</span><input type="number" name="max_weight" min="0.05" max="1" step="0.05" value="0.25"></label>
      <label class="field">Max total exposure<span class="hint">1 = fully invested, no leverage</span><input type="number" name="max_gross" min="0.1" max="1" step="0.05" value="1"></label>
      <label class="field">Fees and taxes<span class="hint">bps per trade</span><input type="number" name="commission_bps" min="0" max="200" step="0.5" value="10"></label>
      <label class="field">Slippage<span class="hint">bps per trade</span><input type="number" name="slippage_bps" min="0" max="200" step="0.5" value="5"></label>
      <div style="align-self:end"><button class="btn primary" type="submit">Build portfolio</button></div>
    </form>
    <div id="pfOut" style="margin-top:16px"></div>`;
  $("#cmpOut").appendChild(host);
  $("#pfForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = new FormData(e.target);
    const out = $("#pfOut");
    out.innerHTML = `<p class="muted">Combining…</p>`;
    try {
      const r = await api(`/api/projects/${S.pid}/portfolio`, { json: {
        experiment_ids: data.map((x) => x.id), max_weight: Number(f.get("max_weight")), max_gross: Number(f.get("max_gross")),
        commission_bps: Number(f.get("commission_bps")), slippage_bps: Number(f.get("slippage_bps")) } });
      const p = r.portfolio, b = r.benchmark;
      out.innerHTML = `
        <ul class="findings">${r.findings.map((t) => `<li class="info"><span>${esc(t)}</span></li>`).join("")}</ul>
        <div class="chart-box" style="margin-top:14px"><canvas id="pfChart" aria-label="Portfolio equity"></canvas></div>
        <div class="legend"><span><i style="background:${cssVar("--cobalt")}"></i>Portfolio, net of costs</span><span><i class="dash"></i>Equal-weight buy and hold</span><span class="muted">${r.start} → ${r.end}, ${r.days} days</span></div>
        <div class="tbl-wrap" style="margin-top:12px"><table><thead><tr><th></th><th class="num">Return</th><th class="num">Sharpe</th><th class="num">95% CI</th><th class="num">Max drawdown</th><th class="num">Avg. invested</th></tr></thead><tbody>
          <tr><td><b>Portfolio</b></td><td class="num ${sgn(p.total_return)}">${pct(p.total_return)}</td><td class="num">${num(p.sharpe)}</td><td class="num">[${num(p.sharpe_ci95?.[0])}, ${num(p.sharpe_ci95?.[1])}]</td><td class="num">${pct(p.max_drawdown)}</td><td class="num">${pct(r.avg_exposure, 0)}</td></tr>
          <tr><td>Equal-weight buy and hold</td><td class="num ${sgn(b.total_return)}">${pct(b.total_return)}</td><td class="num">${num(b.sharpe)}</td><td class="num">[${num(b.sharpe_ci95?.[0])}, ${num(b.sharpe_ci95?.[1])}]</td><td class="num">${pct(b.max_drawdown)}</td><td class="num">100%</td></tr>
          ${r.legs.map((l) => `<tr><td class="muted">${esc(l.name)} alone</td><td class="num ${sgn(l.total_return)}">${pct(l.total_return)}</td><td class="num">${num(l.sharpe)}</td><td></td><td class="num">${pct(l.max_drawdown)}</td><td class="num">${pct(l.avg_weight, 0)} of portfolio</td></tr>`).join("")}
        </tbody></table></div>`;
      const c = r.curve, bc = r.benchmark_curve;
      makeChart($("#pfChart"), { type: "line", data: { labels: c.dates, datasets: [line("Portfolio", c.equity, cssVar("--cobalt"), false, 2.2), line("Equal-weight buy and hold", bc.equity, cssVar("--bh"), true, 1.4)] }, options: baseOptions({ log: true }) });
    } catch (err) {
      out.innerHTML = `<div class="banner bad">${esc(err.message)}</div>`;
    }
  });
}

/* ------------------------------------------------------------------ boot */

$("#projectSelect").addEventListener("change", (e) => { if (e.target.value) location.hash = `#/p/${e.target.value}`; });
window.addEventListener("hashchange", route);
route();
