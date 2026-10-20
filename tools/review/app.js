"use strict";
// Tracking-Review: 3 synchronized clips + overlays, ID filter, A/B run
// comparison, and a run-path pitch for the selected IDs.

const CLIPS = ["match_center", "match_left", "match_right"];
const CLIP_LABEL = { match_center: "Mitte", match_left: "Links", match_right: "Rechts" };
const CLIP_COLOR = { match_center: "#ffd54a", match_left: "#37c0ff", match_right: "#ff7ad1" };
// Track-IDs are per-camera (local): id 5 in "Mitte" is a different person than id 5
// in "Links". Selection is therefore keyed by camera+id, shown as M5 / L12 / R3.
const CLIP_TOKEN = { match_center: "M", match_left: "L", match_right: "R" };
const TOKEN_CLIP = { M: "match_center", L: "match_left", R: "match_right" };
const keyOf = (clip, id) => `${clip}::${id}`;
const VW = 1920, VH = 1080;
const PW = 120, PH = 80;                 // StatsBomb pitch units
const TEAM_COLORS = { 0: "#37c0ff", 1: "#ff5a52" };
const TEAM_NONE = "#b0b0b0";

let FPS = 30, NFRAMES = 600;
let runA = "x_", runB = "", mode = "A"; // mode: A | B | AB
const panes = {};                        // clip -> {video, canvas, ctx, stage}
const data = {};                         // prefix -> clip -> {byFrame, byId, ids}
const runMeta = {};                      // prefix -> {label, global, ...}
let selection = new Set();               // global run: "id"; local run: "clip::id"
let scrubbing = false;
const isGlobal = () => !!runMeta[runA]?.global;   // active run A merges cameras?

const $ = (id) => document.getElementById(id);
const master = () => panes["match_center"].video;
const usesA = () => mode === "A" || mode === "AB";
const usesB = () => mode === "B" || mode === "AB";

// ---- setup ---------------------------------------------------------------
function buildPanes() {
  const grid = $("grid");
  for (const clip of CLIPS) {
    const pane = document.createElement("div");
    pane.className = "pane";
    pane.innerHTML =
      `<div class="cap"><b>${CLIP_LABEL[clip]}</b><span class="cnt" data-clip="${clip}">–</span></div>
       <div class="stage">
         <video muted playsinline preload="auto" src="/video/${clip}.mp4"></video>
         <canvas></canvas>
       </div>`;
    grid.appendChild(pane);
    const video = pane.querySelector("video");
    const canvas = pane.querySelector("canvas");
    const stage = pane.querySelector(".stage");
    panes[clip] = { video, canvas, ctx: canvas.getContext("2d"), stage, cnt: pane.querySelector(".cnt") };
    video.addEventListener("loadedmetadata", () => resizeCanvas(clip));
    canvas.addEventListener("click", (e) => onCanvasClick(clip, e));
  }
  window.addEventListener("resize", () => { CLIPS.forEach(resizeCanvas); sizePitch(); });
}

function resizeCanvas(clip) {
  const { canvas, stage } = panes[clip];
  const r = stage.getBoundingClientRect();
  canvas.width = Math.round(r.width);
  canvas.height = Math.round(r.height);
}

function sizePitch() {
  const c = $("pitch");
  const w = c.clientWidth || 720;
  c.width = Math.round(w);
  c.height = Math.round(w * PH / PW);
}

// ---- data ----------------------------------------------------------------
async function ensureRun(prefix) {
  if (!prefix && prefix !== "") return;
  if (data[prefix]) return;
  data[prefix] = {};
  await Promise.all(CLIPS.map(async (clip) => {
    const d = await fetch(`/api/tracks?run=${encodeURIComponent(prefix)}&clip=${clip}`).then((r) => r.json());
    const byFrame = new Map(), byId = new Map();
    for (const b of d.boxes) {
      if (!byFrame.has(b[0])) byFrame.set(b[0], []);
      byFrame.get(b[0]).push(b);
      if (!byId.has(b[1])) byId.set(b[1], []);
      byId.get(b[1]).push(b);
    }
    for (const arr of byId.values()) arr.sort((a, c) => a[0] - c[0]);
    data[prefix][clip] = { byFrame, byId, ids: d.ids || [] };
  }));
}

async function loadRuns() {
  const res = await fetch("/api/runs").then((r) => r.json());
  FPS = res.fps || 30;
  const runsSel = $("run"), runsSelB = $("runB");
  for (const sel of [runsSel, runsSelB]) sel.innerHTML = "";
  for (const run of res.runs) {
    runMeta[run.prefix] = run;
    for (const sel of [runsSel, runsSelB]) {
      const o = document.createElement("option");
      o.value = run.prefix;
      o.textContent = `${run.label} (${run.clips.length}/3)`;
      sel.appendChild(o);
    }
  }
  const has = (p) => res.runs.some((r) => r.prefix === p);
  runA = has("x_") ? "x_" : (res.runs[0]?.prefix ?? "x_");
  runB = res.runs.find((r) => r.prefix !== runA)?.prefix ?? runA;
  runsSel.value = runA; runsSelB.value = runB;
  runsSel.addEventListener("change", async () => {
    runA = runsSel.value;
    selection.clear(); syncIdInput(); updateFilterHint();  // keys differ global/local
    await refresh(); await loadFeedback();
  });
  runsSelB.addEventListener("change", async () => { runB = runsSelB.value; await refresh(); });
  updateFilterHint();
}

async function refresh() {
  setStatus("lade Trackingdaten …");
  await ensureRun(runA);
  if (usesB()) await ensureRun(runB);
  const n = CLIPS.reduce((a, c) => a + (data[runA]?.[c]?.ids.length || 0), 0);
  const bl = usesB() ? ` · B: ${$("runB").selectedOptions[0].textContent}` : "";
  setStatus(`A: ${$("run").selectedOptions[0].textContent} (${n} IDs)${bl}`);
}

// ---- playback ------------------------------------------------------------
function frameNow() { return Math.round(master().currentTime * FPS); }

function togglePlay() {
  const m = master();
  if (m.paused) { CLIPS.forEach((c) => panes[c].video.play()); $("play").textContent = "⏸ Pause"; }
  else { pauseAll(); }
}
function pauseAll() { CLIPS.forEach((c) => panes[c].video.pause()); $("play").textContent = "▶ Play"; }

function seekTo(frame) {
  frame = Math.max(0, Math.min(NFRAMES - 1, frame));
  const t = frame / FPS;
  CLIPS.forEach((c) => { panes[c].video.currentTime = t; });
}

function tick() {
  const m = master();
  for (const clip of CLIPS) {
    if (clip === "match_center") continue;
    const v = panes[clip].video;
    if (Math.abs(v.currentTime - m.currentTime) > 0.08) v.currentTime = m.currentTime;
  }
  const f = frameNow();
  if (!scrubbing) $("scrub").value = f;
  $("counter").textContent = `f${f} / ${NFRAMES - 1}`;
  $("fb-frame").textContent = `f${f}`;
  CLIPS.forEach((clip) => drawClip(clip, f));
  drawPitch(f);
  requestAnimationFrame(tick);
}

// ---- overlay drawing -----------------------------------------------------
function drawBoxes(clip, frame, rundata, dashed) {
  const { canvas, ctx } = panes[clip];
  const cd = rundata?.[clip];
  if (!cd) return 0;
  const boxes = cd.byFrame.get(frame) || [];
  const sx = canvas.width / VW, sy = canvas.height / VH;
  ctx.lineWidth = 2;
  ctx.setLineDash(dashed ? [5, 4] : []);
  ctx.font = "600 13px system-ui, sans-serif";
  ctx.textBaseline = "bottom";
  let shown = 0;
  const global = isGlobal();
  for (const b of boxes) {
    const id = b[1];
    const key = global ? String(id) : keyOf(clip, id);
    if (selection.size && !selection.has(key)) continue;
    shown++;
    const col = (b[6] === 0 || b[6] === 1) ? TEAM_COLORS[b[6]] : TEAM_NONE;
    const x = b[2] * sx, y = b[3] * sy, w = (b[4] - b[2]) * sx, h = (b[5] - b[3]) * sy;
    ctx.strokeStyle = col;
    ctx.strokeRect(x, y, w, h);
    const label = dashed ? `${id}'` : String(id);
    const tw = ctx.measureText(label).width + 6;
    const ly = dashed ? Math.min(canvas.height, y + h + 15) : Math.max(0, y - 15);
    ctx.setLineDash([]);
    ctx.fillStyle = col;
    ctx.fillRect(x, ly - (dashed ? 15 : 0), tw, 15);
    ctx.fillStyle = "#0d0f12";
    ctx.fillText(label, x + 3, ly - (dashed ? 2 : 2));
    ctx.setLineDash(dashed ? [5, 4] : []);
  }
  ctx.setLineDash([]);
  return { shown, total: boxes.length };
}

function drawClip(clip, frame) {
  const { canvas, ctx } = panes[clip];
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  let primary = null;
  if (usesA()) primary = drawBoxes(clip, frame, data[runA], false);
  if (usesB()) { const r = drawBoxes(clip, frame, data[runB], usesA()); if (!primary) primary = r; }
  if (primary) {
    panes[clip].cnt.textContent = selection.size ? `${primary.shown}/${primary.total}` : `${primary.total}`;
  }
}

// ---- run-path pitch ------------------------------------------------------
function drawPitchLines(ctx, W, H) {
  ctx.strokeStyle = "rgba(255,255,255,0.55)";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(2, 2, W - 4, H - 4);
  ctx.beginPath(); ctx.moveTo(W / 2, 2); ctx.lineTo(W / 2, H - 2); ctx.stroke();
  ctx.beginPath(); ctx.arc(W / 2, H / 2, H * 0.11, 0, 2 * Math.PI); ctx.stroke();
  const pen = (x0, dir) => {
    const bw = W * (16.5 / PW), bh = H * (40.3 / PH);
    const y = (H - bh) / 2;
    ctx.strokeRect(dir > 0 ? x0 : x0 - bw, y, bw, bh);
  };
  pen(2, 1); pen(W - 2, -1);
}

function px2cx(px, W) { return px / PW * W; }
function py2cy(py, H) { return py / PH * H; }

function drawPitch(frame) {
  const c = $("pitch"), ctx = c.getContext("2d");
  const W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  drawPitchLines(ctx, W, H);
  if (!selection.size || !data[runA]) return;
  // (clip, id) pairs to draw: local run = the selected key; global run = the
  // selected global id in EVERY camera that sees it (shows the hand-off).
  const pairs = [];
  if (isGlobal()) {
    for (const idStr of selection)
      for (const clip of CLIPS) pairs.push([clip, +idStr]);
  } else {
    for (const key of selection) { const [clip, idStr] = key.split("::"); pairs.push([clip, +idStr]); }
  }
  for (const [clip, id] of pairs) {
    const cd = data[runA][clip];
    if (!cd) continue;
    drawPath(ctx, W, H, cd.byId.get(id) || [], CLIP_COLOR[clip], frame);
  }
}

function drawPath(ctx, W, H, allRows, color, frame) {
  const rows = allRows.filter((b) => b[7] != null && b[8] != null);
  if (rows.length < 2) return;
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2;
  ctx.beginPath();
  let pen = false;
  for (let i = 0; i < rows.length; i++) {
    const b = rows[i];
    const x = px2cx(b[7], W), y = py2cy(b[8], H);
    const gap = i > 0 ? b[0] - rows[i - 1][0] : 0;
    if (!pen || gap > 10) { ctx.moveTo(x, y); pen = true; } else { ctx.lineTo(x, y); }
  }
  ctx.stroke();
  let cur = null;
  for (const b of rows) { if (b[0] <= frame) cur = b; else break; }
  if (cur && frame - cur[0] <= 10) {
    ctx.beginPath();
    ctx.arc(px2cx(cur[7], W), py2cy(cur[8], H), 4, 0, 2 * Math.PI);
    ctx.fill();
  }
}

// ---- ID selection --------------------------------------------------------
function onCanvasClick(clip, e) {
  const { canvas } = panes[clip];
  const r = canvas.getBoundingClientRect();
  const ix = (e.clientX - r.left) / r.width * VW;
  const iy = (e.clientY - r.top) / r.height * VH;
  const src = usesA() ? data[runA] : data[runB];
  const boxes = (src?.[clip]?.byFrame.get(frameNow()) || [])
    .filter((b) => ix >= b[2] && ix <= b[4] && iy >= b[3] && iy <= b[5])
    .sort((a, bb) => ((a[4] - a[2]) * (a[5] - a[3])) - ((bb[4] - bb[2]) * (bb[5] - bb[3])));
  if (!boxes.length) return;
  const id = boxes[0][1];
  const key = isGlobal() ? String(id) : keyOf(clip, id);
  if (selection.has(key)) selection.delete(key); else selection.add(key);
  syncIdInput();
  $("fb-clip").value = clip;   // prefill feedback with what you just clicked
  $("fb-id").value = id;
}
function syncIdInput() {
  let toks;
  if (isGlobal()) {
    toks = [...selection].map(Number).sort((a, b) => a - b);
  } else {
    toks = [...selection].map((k) => {
      const [clip, id] = k.split("::");
      return CLIP_TOKEN[clip] + id;
    }).sort();
  }
  $("id-input").value = toks.join(", ");
}
function parseIdInput() {
  const raw = $("id-input").value.trim();
  selection = new Set();
  if (!raw) return;
  if (isGlobal()) {
    for (const tok of raw.split(/[,\s]+/)) {
      const n = parseInt(tok.replace(/[^0-9]/g, ""), 10);
      if (Number.isFinite(n)) selection.add(String(n));
    }
    return;
  }
  for (const tok of raw.split(/[,\s]+/)) {
    const m = tok.match(/^([mMlLrR])?(\d+)$/);
    if (!m) continue;
    if (m[1]) selection.add(keyOf(TOKEN_CLIP[m[1].toUpperCase()], +m[2]));
    else CLIPS.forEach((clip) => selection.add(keyOf(clip, +m[2]))); // bare number = alle Kameras
  }
}
function updateFilterHint() {
  $("id-hint").textContent = isGlobal()
    ? "Globaler Lauf: eine ID = ein Spieler über alle Kameras. Klick auf eine Box wählt sie an/ab."
    : "IDs sind pro Kamera (M=Mitte, L=Links, R=Rechts). Klick auf eine Box wählt sie an/ab.";
}

// ---- wiring --------------------------------------------------------------
function wireControls() {
  $("play").addEventListener("click", togglePlay);
  $("step-back").addEventListener("click", () => { pauseAll(); seekTo(frameNow() - 1); });
  $("step-fwd").addEventListener("click", () => { pauseAll(); seekTo(frameNow() + 1); });
  const scrub = $("scrub");
  scrub.max = NFRAMES - 1;
  scrub.addEventListener("input", () => { scrubbing = true; seekTo(parseInt(scrub.value, 10)); });
  scrub.addEventListener("change", () => { scrubbing = false; });
  $("speed").addEventListener("change", (e) => {
    const rate = parseFloat(e.target.value);
    CLIPS.forEach((c) => { panes[c].video.playbackRate = rate; });
  });
  $("id-input").addEventListener("change", parseIdInput);
  $("id-input").addEventListener("keyup", (e) => { if (e.key === "Enter") parseIdInput(); });
  $("id-clear").addEventListener("click", () => { selection.clear(); syncIdInput(); });
  $("mode").addEventListener("click", async (e) => {
    const btn = e.target.closest("button[data-mode]");
    if (!btn) return;
    mode = btn.dataset.mode;
    [...$("mode").children].forEach((b) => b.classList.toggle("active", b === btn));
    await refresh();
  });
}

// ---- feedback ------------------------------------------------------------
const ISSUE_COLOR = { Abbruch: "#ffd54a", Verwechslung: "#ff5a52", Doppelung: "#ff7ad1", Geist: "#b0b0b0", Sonstiges: "#37c0ff" };

function initFeedbackForm() {
  const sel = $("fb-clip");
  for (const clip of CLIPS) {
    const o = document.createElement("option");
    o.value = clip; o.textContent = CLIP_LABEL[clip];
    sel.appendChild(o);
  }
  $("fb-save").addEventListener("click", saveFeedback);
}

async function saveFeedback() {
  const id = parseInt($("fb-id").value, 10);
  if (!Number.isFinite(id)) { setStatus("Feedback: bitte Track-ID angeben."); return; }
  const entry = {
    run: runA, clip: $("fb-clip").value, track_id: id,
    issue: $("fb-issue").value, frame: frameNow(), note: $("fb-note").value.trim(),
  };
  const res = await fetch("/api/feedback", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(entry),
  }).then((r) => r.json());
  $("fb-note").value = "";
  setStatus(`Feedback gespeichert (${res.count} für Lauf A).`);
  await loadFeedback();
}

async function loadFeedback() {
  const list = await fetch(`/api/feedback?run=${encodeURIComponent(runA)}`).then((r) => r.json());
  const ul = $("fb-list");
  ul.innerHTML = "";
  for (const e of (list.feedback || []).slice().reverse()) {
    const li = document.createElement("li");
    const col = ISSUE_COLOR[e.issue] || "#37c0ff";
    li.innerHTML =
      `<span class="tag" style="background:${col}">${e.issue}</span>` +
      `<span><b>${CLIP_LABEL[e.clip] || e.clip}</b> · ID ${e.track_id} · f${e.frame}</span>` +
      (e.note ? `<span class="muted">${e.note.replace(/</g, "&lt;")}</span>` : "");
    ul.appendChild(li);
  }
}

function setStatus(msg) { $("status").textContent = msg; }

async function init() {
  buildPanes();
  wireControls();
  initFeedbackForm();
  sizePitch();
  await loadRuns();
  await refresh();
  await loadFeedback();
  CLIPS.forEach(resizeCanvas);
  requestAnimationFrame(tick);
}
init();
