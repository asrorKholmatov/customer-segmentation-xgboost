// SegmentIQ — evaluates the exported XGBoost trees in the browser.
const GITHUB_URL = "https://github.com/asrorKholmatov/customer-segmentation-xgboost";

const SEGMENTS = {
  A: {
    color: "var(--seg-a)", name: "Creative Individualists",
    blurb: "Mid-40s artists and entertainers with small households — mostly careful spenders.",
    advice: ["Lead with design, style and self-expression rather than price.",
      "Promote compact, individual-use products and flexible plans.",
      "Use creative channels — Instagram, events, artist collaborations."],
  },
  B: {
    color: "var(--seg-b)", name: "Established Professionals",
    blurb: "Married, educated late-40s customers with a balanced spending mix.",
    advice: ["Emphasise reliability, value for money and total cost of ownership.",
      "Offer upgrade paths and trade-in deals to move them up a tier.",
      "Email and LinkedIn campaigns work well for this group."],
  },
  C: {
    color: "var(--seg-c)", name: "Affluent Family Builders",
    blurb: "The most educated, mostly married, highest share of average/high spenders.",
    advice: ["Pitch premium and family-oriented options — space, safety, comfort.",
      "Bundle extras (warranty, servicing) — they are willing to pay for peace of mind.",
      "Loyalty programmes and referrals: they are high lifetime-value customers."],
  },
  D: {
    color: "var(--seg-d)", name: "Young Starters",
    blurb: "Early-30s, mostly single, often in healthcare — 87% are low spenders.",
    advice: ["Entry-level products, financing and first-time-buyer discounts.",
      "Mobile-first, social and short-video channels.",
      "Build the relationship early — they grow into higher segments over time."],
  },
};

const RAW_FIELDS = ["Gender", "Ever_Married", "Age", "Graduated", "Profession",
  "Work_Experience", "Spending_Score", "Family_Size", "Var_1"];

let M; // model payload

const $ = (s) => document.querySelector(s);
const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
const clean = (v) => (v == null ? "" : String(v).trim());

// ---------- model ----------
function num(v) {
  const s = clean(v);
  if (s === "") return NaN;
  const n = Number(s);
  return Number.isFinite(n) ? n : NaN;
}
function mapOr(v, table) {
  const s = clean(v);
  return s in table ? table[s] : NaN;
}

// Mirrors src/dataset.py::encode exactly.
function encode(row) {
  const yn = { No: 0, Yes: 1 };
  const prof = clean(row.Profession), cat = clean(row.Var_1);
  return [
    mapOr(row.Gender, { Female: 0, Male: 1 }),
    mapOr(row.Ever_Married, yn),
    num(row.Age),
    mapOr(row.Graduated, yn),
    num(row.Work_Experience),
    mapOr(row.Spending_Score, { Low: 0, Average: 1, High: 2 }),
    num(row.Family_Size),
    ...M.professions.map((p) => (prof ? +(prof === p) : NaN)),
    ...M.categories.map((c) => (cat ? +(cat === c) : NaN)),
  ];
}

function predict(row) {
  const x = encode(row);
  const k = M.classes.length;
  const margin = M.base_margin.slice();
  M.trees.forEach(([f, t, yes, no, miss], idx) => {
    let i = 0;
    while (f[i] >= 0) {
      const v = x[f[i]];
      i = Number.isNaN(v) ? miss[i] : v < t[i] ? yes[i] : no[i];
    }
    margin[idx % k] += t[i];
  });
  const mx = Math.max(...margin);
  const e = margin.map((m) => Math.exp(m - mx));
  const s = e.reduce((a, b) => a + b, 0);
  const probs = e.map((v) => v / s);
  const best = probs.indexOf(Math.max(...probs));
  return { segment: M.classes[best], confidence: probs[best], probs };
}

// ---------- single prediction ----------
function readForm() {
  const fd = new FormData($("#form"));
  return Object.fromEntries(RAW_FIELDS.map((f) => [f, fd.get(f)]));
}

function renderResult() {
  const r = predict(readForm());
  const seg = SEGMENTS[r.segment];
  $("#result").style.setProperty("--seg", seg.color);
  $("#r-badge").textContent = r.segment;
  $("#r-title").textContent = seg.name;
  const level = r.confidence >= 0.6 ? "high" : r.confidence >= 0.45 ? "medium" : "low";
  $("#r-conf").textContent = `${pct(r.confidence)} confidence · ${level}`;
  $("#r-probs").innerHTML = M.classes.map((c, i) => `
    <div class="prob" style="--seg:${SEGMENTS[c].color}">
      <span>Segment ${c}</span>
      <div class="track"><div class="fill" style="width:${pct(r.probs[i], 1)}"></div></div>
      <b>${pct(r.probs[i])}</b>
    </div>`).join("");
  $("#r-advice").innerHTML = seg.advice.map((a) => `<li>${a}</li>`).join("");
}

function randomize() {
  const pick = (a) => a[Math.floor(Math.random() * a.length)];
  const f = $("#form").elements;
  f.Gender.value = pick(["Male", "Female"]);
  f.Age.value = 18 + Math.floor(Math.random() * 60);
  f.Ever_Married.value = pick(["Yes", "No"]);
  f.Graduated.value = pick(["Yes", "No"]);
  f.Profession.value = pick(M.professions);
  f.Spending_Score.value = pick(["Low", "Average", "High"]);
  f.Work_Experience.value = pick([0, 0, 1, 1, 2, 4, 8, 12]);
  f.Family_Size.value = pick([1, 2, 2, 3, 4, 5, 6]);
  f.Var_1.value = pick(M.categories);
  $("#age-out").textContent = f.Age.value;
  renderResult();
}

// ---------- batch ----------
function parseCSV(text) {
  const rows = [];
  let row = [], field = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') q = false;
      else field += ch;
    } else if (ch === '"') q = true;
    else if (ch === ",") { row.push(field); field = ""; }
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.some((c) => c !== "")) rows.push(row);
      row = [];
    } else field += ch;
  }
  row.push(field);
  if (row.some((c) => c !== "")) rows.push(row);
  const [header, ...body] = rows;
  const cols = header.map((h) => h.trim().replace(/^﻿/, ""));
  return { cols, data: body.map((r) => Object.fromEntries(cols.map((c, i) => [c, r[i] ?? ""]))) };
}

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const csvCell = (s) => (/[",\n]/.test(s) ? `"${String(s).replace(/"/g, '""')}"` : s);

let lastCSV = "";

function runBatch(text) {
  const { cols, data } = parseCSV(text);
  const missing = RAW_FIELDS.filter((f) => !cols.includes(f));
  if (missing.length > 4 || !data.length) {
    alert(`This file doesn't look like customer data. Missing columns: ${missing.join(", ")}`);
    return;
  }
  const results = data.map((r) => ({ row: r, ...predict(r) }));
  const outCols = [...cols.filter((c) => c !== "Segmentation" && c !== "Predicted_Segment"),
    "Predicted_Segment", "Confidence", ...M.classes.map((c) => `P_${c}`)];
  lastCSV = [outCols.join(","), ...results.map((r) => outCols.map((c) => csvCell(
    c === "Predicted_Segment" ? r.segment
      : c === "Confidence" ? r.confidence.toFixed(4)
      : c.startsWith("P_") && M.classes.includes(c.slice(2)) ? r.probs[M.classes.indexOf(c.slice(2))].toFixed(4)
      : r.row[c] ?? "")).join(","))].join("\n");

  const counts = Object.fromEntries(M.classes.map((c) => [c, 0]));
  results.forEach((r) => counts[r.segment]++);
  $("#dist").innerHTML = M.classes.filter((c) => counts[c]).map((c) => {
    const share = counts[c] / results.length;
    return `<div style="--seg:${SEGMENTS[c].color};flex:${counts[c]}" title="${SEGMENTS[c].name}">${c} · ${pct(share)}</div>`;
  }).join("");

  const shown = cols.filter((c) => c !== "Predicted_Segment").slice(0, 11);
  const hasTruth = cols.includes("Segmentation");
  $("#table").innerHTML = `<thead><tr><th>Predicted</th><th>Confidence</th>${shown.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
    <tbody>${results.slice(0, 200).map((r) => `<tr>
      <td><span class="pill" style="--seg:${SEGMENTS[r.segment].color}">${r.segment}</span></td>
      <td>${pct(r.confidence)}</td>
      ${shown.map((c) => `<td>${esc(r.row[c])}</td>`).join("")}</tr>`).join("")}</tbody>`;

  let note = `${results.length.toLocaleString()} customers classified`;
  if (results.length > 200) note += " · showing the first 200 — download for all rows";
  if (hasTruth) {
    const labelled = results.filter((r) => M.classes.includes(clean(r.row.Segmentation)));
    const hit = labelled.filter((r) => clean(r.row.Segmentation) === r.segment).length;
    if (labelled.length) note += ` · ${pct(hit / labelled.length, 1)} match the file's Segmentation column`;
  }
  $("#batch-note").textContent = note;
  $("#batch-out").hidden = false;
  $("#download").disabled = false;
}

function downloadResults() {
  const blob = new Blob([lastCSV], { type: "text/csv" });
  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(blob), download: "segmentiq_predictions.csv",
  });
  a.click();
  URL.revokeObjectURL(a.href);
}

// ---------- static sections ----------
function renderStats() {
  const m = M.metrics;
  $("#hero-stats").innerHTML = [
    [pct(m.accuracy, 1), "test accuracy"],
    [`${(m.accuracy / m.baseline_accuracy).toFixed(1)}×`, "better than guessing"],
    [(m.n_train + m.n_test).toLocaleString(), "customers in training data"],
    [M.trees.length.toLocaleString(), "decision trees"],
  ].map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");

  $("#kpis").innerHTML = [
    [pct(m.accuracy, 1), "Test accuracy"],
    [m.macro_f1.toFixed(3), "Macro F1-score"],
    [pct(m.cv_accuracy, 1), "5-fold CV accuracy"],
    [pct(m.baseline_accuracy, 1), "Always-guess-D baseline"],
  ].map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");

  const cm = m.confusion;
  const max = Math.max(...cm.flat());
  let html = `<div class="hd"></div>${M.classes.map((c) => `<div class="hd">${c}</div>`).join("")}`;
  cm.forEach((row, i) => {
    html += `<div class="hd">${M.classes[i]}</div>`;
    row.forEach((v, j) => {
      const a = v / max;
      const color = i === j ? "79,70,229" : "148,153,190";
      html += `<div style="background:rgba(${color},${(0.08 + a * 0.85).toFixed(2)});color:${a > 0.5 ? "#fff" : "inherit"}">${v}</div>`;
    });
  });
  $("#confusion").innerHTML = html;

  const pretty = (f) => f.replace("Profession_", "Job: ").replace("Var_1_", "Category ").replace(/_/g, " ");
  const top = M.importance.slice(0, 10);
  const imax = top[0].gain;
  $("#importance").innerHTML = top.map((d) => `
    <div class="bar"><span>${pretty(d.feature)}</span>
      <div class="track"><div class="fill" style="width:${(d.gain / imax) * 100}%"></div></div>
      <b>${pct(d.gain, 1)}</b></div>`).join("");

  $("#seg-grid").innerHTML = M.classes.map((c) => {
    const p = M.profiles[c], s = SEGMENTS[c];
    const job = Object.entries(p.top_professions)[0];
    const spend = Object.entries(p.spending).sort((a, b) => b[1] - a[1])[0];
    const rec = m.per_class[c].recall;
    return `<div class="card seg" style="--seg:${s.color}">
      <div class="seg-letter">SEGMENT ${c}</div>
      <h3>${s.name}</h3>
      <p>${s.blurb}</p>
      <dl>
        <dt>Customers</dt><dd>${p.count.toLocaleString()}</dd>
        <dt>Average age</dt><dd>${p.avg_age}</dd>
        <dt>Married</dt><dd>${p.married_pct}%</dd>
        <dt>Graduated</dt><dd>${p.graduated_pct}%</dd>
        <dt>Avg family size</dt><dd>${p.avg_family}</dd>
        <dt>Top profession</dt><dd>${job[0]} (${pct(job[1])})</dd>
        <dt>Typical spending</dt><dd>${spend[0]} (${pct(spend[1])})</dd>
        <dt>Model recall</dt><dd>${pct(rec)}</dd>
      </dl></div>`;
  }).join("");
}

// ---------- boot ----------
async function init() {
  M = await (await fetch("model.json")).json();
  $("#gh-link").href = GITHUB_URL;
  $("#profession").innerHTML = M.professions.map((p) => `<option>${p}</option>`).join("") + `<option value="">Unknown</option>`;
  $("#profession").value = "Engineer";
  $("#category").innerHTML = M.categories.map((c) => `<option>${c}</option>`).join("") + `<option value="">Unknown</option>`;
  $("#category").value = "Cat_6";

  const form = $("#form");
  form.addEventListener("input", (e) => {
    if (e.target.name === "Age") $("#age-out").textContent = e.target.value;
    renderResult();
  });
  $("#randomize").addEventListener("click", randomize);

  const drop = $("#drop");
  const readFile = (file) => file && file.text().then(runBatch);
  $("#file").addEventListener("change", (e) => readFile(e.target.files[0]));
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => readFile(e.dataTransfer.files[0]));
  $("#use-sample").addEventListener("click", async () => runBatch(await (await fetch("sample_customers.csv")).text()));
  $("#download").addEventListener("click", downloadResults);

  renderStats();
  renderResult();
}

init();
