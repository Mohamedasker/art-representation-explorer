"""
Self-contained interactive HTML explorer for a set of embedded images.

One page shows a 2D projection of the samples on a zoomable canvas.  Hovering
a point shows the original image (a thumbnail stored next to the page), its
art style and cluster membership; clicking pins it.  Dropdowns switch between
projections, k-means cluster counts and cluster spaces, and the colouring.

The page is plain HTML + canvas + vanilla JS (no CDN), so it works offline.
Keep it next to the ``thumbs/`` and ``meta/`` folders it references.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg:#111; --panel:#1a1a1a; --line:#333; --fg:#e6e6e6; --muted:#8a8a8a; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { display:flex; flex-wrap:wrap; gap:10px 16px; align-items:center;
         padding:10px 16px; border-bottom:1px solid var(--line); }
header h1 { font-size:14px; font-weight:600; margin:0 8px 0 0; }
header .sub { color:var(--muted); font-size:12px; }
header a { color:var(--muted); text-decoration:none; margin-right:8px; }
header a.cur { color:var(--fg); border-bottom:1px solid var(--fg); }
label { color:var(--muted); }
select, input[type=range] { background:var(--panel); color:var(--fg); border:1px solid var(--line);
         border-radius:4px; padding:3px 6px; }
main { display:grid; grid-template-columns:minmax(0,1fr) 12px var(--side, 360px); gap:0;
       padding:12px 16px; height:calc(100vh - 92px); }
#split { cursor:col-resize; position:relative; touch-action:none; }
#split::after { content:""; position:absolute; top:50%; left:50%; width:4px; height:56px; transform:translate(-50%,-50%);
                border-radius:2px; background:var(--line); transition:background .15s, height .15s; }
#split:hover::after, #split.drag::after { background:#8a8a8a; height:96px; }
body.resizing { user-select:none; cursor:col-resize; }
.expl { color:#ffb74d; font-size:12px; cursor:pointer; white-space:nowrap; } .expl input { vertical-align:middle; }
.expl span { color:var(--muted); }
#mapwrap { position:relative; background:var(--panel); border:1px solid var(--line);
           border-radius:6px; min-height:300px; }
#map { width:100%; height:100%; display:block; cursor:crosshair; }
#legend { position:absolute; top:8px; left:8px; background:#111d; padding:6px 8px;
          border-radius:4px; font-size:11px; max-height:70%; overflow:auto; }
#legend div { display:flex; align-items:center; gap:6px; cursor:pointer; padding:1px 0; }
#legend div.off { opacity:.35; }
#legend div:hover { background:#fff1; }
#legend i { width:10px; height:10px; border-radius:50%; display:inline-block; flex:none; }
#hint { position:absolute; bottom:6px; right:10px; color:var(--muted); font-size:11px; }
aside { display:flex; flex-direction:column; gap:10px; overflow:auto; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:10px; }
.card h2 { font-size:12px; font-weight:600; color:var(--muted); margin:0 0 6px;
           text-transform:uppercase; letter-spacing:.04em; }
#thumb { width:100%; max-height:60vh; aspect-ratio:1; object-fit:contain; background:#000; display:block;
         border-radius:4px; }
#info { margin:0; font-size:12px; font-variant-numeric:tabular-nums; }
#info div { display:flex; gap:8px; }
#info dt { color:var(--muted); min-width:84px; }
#info dd { margin:0; overflow-wrap:anywhere; }
.bar { display:flex; align-items:center; gap:6px; font-size:11px; margin:2px 0; }
.bar span.n { width:110px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.bar span.b { flex:1; height:8px; background:#0003; border-radius:2px; overflow:hidden; }
.bar span.b i { display:block; height:100%; }
.bar span.p { width:38px; text-align:right; color:var(--muted); }
#palette { display:flex; height:22px; border-radius:4px; overflow:hidden; margin-top:6px; }
#palette i { display:block; height:100%; }
.chips { display:inline-flex; gap:2px; vertical-align:middle; }
.chips b { display:inline-block; width:12px; height:10px; border-radius:2px; }
#groupcard .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:6px 0; font-size:12px; color:var(--muted); }
#groupcard h3 { margin:0; font-size:13px; font-weight:600; color:var(--fg); }
#gallery, #selgallery { margin-top:6px; }
#selcard .row { display:flex; gap:6px; align-items:center; flex-wrap:wrap; margin:6px 0; font-size:12px; color:var(--muted); }
#selcard h3 { margin:0; font-size:13px; font-weight:600; color:var(--fg); }
#selcard button:disabled { opacity:.4; cursor:default; }
#tools { position:absolute; top:8px; right:8px; display:flex; gap:4px; background:#111d; padding:4px; border-radius:6px; z-index:4; }
#tools button { padding:3px 9px; font-size:12px; }
#tools button.on { background:#ffd54f; color:#000; border-color:#ffd54f; }
#tip { position:absolute; z-index:5; pointer-events:none; background:#000d; border:1px solid var(--line); border-radius:6px;
       padding:4px; font-size:11px; max-width:150px; }
#tip img { width:140px; height:140px; object-fit:contain; display:block; background:#000; }
#pin { color:#ffb74d; font-size:11px; margin-left:6px; }
.strip { display:grid; grid-template-columns:repeat(auto-fill, minmax(96px, 1fr)); gap:6px; margin-top:8px; }
.strip figure { margin:0; position:relative; cursor:pointer; border-radius:4px; overflow:hidden; background:#000;
                outline:2px solid transparent; }
.strip figure:hover { outline-color:#fff6; }
.strip img { width:100%; aspect-ratio:1; object-fit:cover; display:block; }
.strip figcaption { position:absolute; left:0; right:0; bottom:0; padding:1px 4px; font-size:10px; background:#000b;
                    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.strip .tag { position:absolute; top:0; left:0; padding:0 5px; font-size:11px; font-weight:600; color:#000; border-radius:0 0 4px 0; }
#modehelp { color:var(--muted); font-size:12px; }
.modectl { display:inline-flex; gap:6px; align-items:center; }
button { background:var(--panel); color:var(--fg); border:1px solid var(--line); border-radius:4px; padding:3px 8px; cursor:pointer; }
button:hover { border-color:#666; }
[hidden] { display:none !important; }
@media (max-width: 1000px) { main { grid-template-columns:1fr; height:auto; } #mapwrap { height:70vh; } #split { display:none; } }
</style></head>
<body>
<header>
  <h1>__TITLE__</h1><span class="sub" id="subtitle"></span>
  <span class="sub" id="nav"></span>
  <span id="varwrap" hidden><label>Augmentation set </label><select id="varsel"></select></span>
  <span id="setwrap" hidden><label>Images </label><select id="setsel"></select></span>
  <span><label>Projection </label><select id="projsel"></select></span>
  <span><label>Cluster space </label><select id="spacesel"></select></span>
  <span><label>k </label><select id="ksel"></select></span>
  <span><label>Colour by </label><select id="colorsel">
    <option value="cluster_style">k-means cluster (coloured by its dominant art style)</option>
    <option value="cluster">k-means cluster (distinct colours)</option><option value="style">art style</option>
    <option value="theme">colour theme (k-means on palettes)</option>
    <option value="dominant">dominant colour (actual)</option>
    <option value="dominant_ref">dominant colour (shared palette)</option>
    <option value="light">lightness</option><option value="chroma">colourfulness</option>
    <option value="year">painting date (from file name)</option></select></span>
  <span><label>Mode </label><select id="modesel">
    <option value="explore">explore</option>
    <option value="knn">nearest neighbours</option>
    <option value="interp">interpolate A &rarr; B</option></select></span>
  <span class="modectl" id="knnctl" hidden><label>neighbours </label>
    <input id="kk" type="range" min="3" max="30" value="12"><span id="kkv">12</span></span>
  <span class="modectl" id="interpctl" hidden><label>steps </label>
    <input id="steps" type="range" min="3" max="15" value="9"><span id="stepsv">9</span>
    <button id="swapab">swap A/B</button><button id="clearab">reset</button></span>
  <span><label>Size </label><input id="size" type="range" min="1" max="8" step="0.5" value="3"></span>
  <span class="sub" id="metrics"></span>
  <label id="explwrap" class="expl" hidden title="Images flagged as containing nudity are hidden unless this is ticked. The choice is shared by all explorer pages and remembered by the browser."><input type="checkbox" id="showexpl"> show explicit images <span id="explcount"></span></label>
</header>
<main>
  <div id="mapwrap"><canvas id="map"></canvas><div id="legend"></div><div id="tools"><button data-tool="pan" class="on" title="inspect / pan (V)">Pan</button>
    <button data-tool="rect" title="rectangle selection (R) - Shift adds, Alt removes">Box</button>
    <button data-tool="lasso" title="free-form lasso selection (L) - Shift adds, Alt removes">Lasso</button>
    <button id="selclear" hidden title="clear the selection (Esc)">Clear</button></div><div id="tip" hidden><img alt=""><span></span></div>
    <div id="hint">hover: inspect · click: pin · wheel: zoom · drag: pan · dbl-click: reset · legend: isolate</div></div>
  <div id="split" title="drag to resize the map and the panel · double-click to reset"></div>
  <aside>
    <div class="card" id="selcard" hidden><h2>Selection</h2>
      <h3 id="seltitle"></h3>
      <div class="row"><select id="selorder"><option value="random">random</option><option value="typical">most typical</option>
        <option value="outlier">outliers</option><option value="style">grouped by art style</option></select>
        <select id="selpagesize"><option>12</option><option selected>18</option><option>30</option><option>60</option></select>
        <span>per page</span></div>
      <div class="row"><button id="selprev">&lsaquo; previous</button><span id="selpos"></span>
        <button id="selnext">next &rsaquo;</button><button id="selclear2">clear</button></div>
      <div class="strip" id="selgallery"></div></div>
    <div class="card" id="groupcard" hidden><h2>Samples in the highlighted group</h2>
      <h3 id="grouptitle"></h3>
      <div class="row"><span id="groupn"></span><select id="gorder"><option value="random">random</option>
        <option value="typical">most typical</option><option value="outlier">outliers</option></select>
        <button id="gshuffle">shuffle</button></div>
      <div class="strip" id="gallery"></div></div>
    <div class="card" id="samplecard"><h2>Sample<span id="pin"></span></h2>
      <img id="thumb" alt="hover a point"><div id="palette"></div></div>
    <div class="card" id="modecard" hidden><h2 id="modetitle"></h2><div id="modehelp"></div><div class="strip" id="strip"></div></div>
    <div class="card"><h2>Details</h2><dl id="info"><div><dt>—</dt><dd>hover a point</dd></div></dl></div>
    <div class="card"><h2 id="comptitle">Art styles in this cluster</h2><div id="comp"></div></div>
  </aside>
</main>
__META_SCRIPT__<script>
// Per-image data that is the same for every model of a dataset (labels, thumbnails, palettes, attributes, ...) lives
// once in meta/<name>.js, loaded above and cached by the browser across pages.  The page keeps its sample indices
// (D.idx) and selects its rows from it here.
function hydrate(d) {
  if (!d.meta) return d;
  const M = window.EXPLORER_META;
  if (!M || M.name !== d.meta) {
    document.body.insertAdjacentHTML("afterbegin", `<p style="padding:16px;color:#f88">Could not load ` +
      `meta/${d.meta}.js: keep the meta/ folder next to this page.</p>`);
    throw new Error("missing meta/" + d.meta + ".js");
  }
  const ix = d.idx, pick = a => ix.map(i => a[i]);
  for (const k of d.meta_keys) {
    if (k === "theme") { d.theme = {}; for (const kk in M.theme) d.theme[kk] = pick(M.theme[kk]); }
    else d[k] = pick(M[k]);
  }
  for (const a of d.attrs || []) {                  // same encoding as the inline a.b64 (see the attribute decoder)
    if (!a.meta) continue;
    const m = M.attrs[a.key], bin = atob(m.b64), u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    if (m.kind === "num") { const q = new Uint16Array(u.buffer), span = (m.hi - m.lo) || 1;
      a.v = ix.map(i => q[i] === 65535 ? null : m.lo + q[i] / 65534 * span); }
    else { const q = m.kind === "color" ? new Int32Array(u.buffer) : new Int16Array(u.buffer); a.v = ix.map(i => q[i]); }
  }
  return d;
}
</script>
<script>
const D = hydrate(__DATA__);
const N = D.n, NC = D.classes.length;
const TAB20 = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f",
  "#bcbd22","#17becf","#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5","#c49c94","#f7b6d2",
  "#c7c7c7","#dbdb8d","#9edae5"];
function palette(k) {
  if (k <= 10) return TAB20.slice(0, k);
  if (k <= 20) return TAB20.slice(0, k);
  return Array.from({ length: k }, (_, i) => `hsl(${(i * 360 / k) | 0},70%,55%)`);
}
const STYLE_PAL = (D.style_colors && D.style_colors.length === NC) ? D.style_colors : palette(NC);  // fixed colour per style
const STYLE_YEAR = D.style_years || [];
const STYLE_ORDER = [...Array(NC).keys()].sort((a, b) => (STYLE_YEAR[a] ?? 9999) - (STYLE_YEAR[b] ?? 9999) || a - b);

// colour helpers for the shaded cluster colours
function hexToHsl(h) {
  const r = parseInt(h.slice(1, 3), 16) / 255, g = parseInt(h.slice(3, 5), 16) / 255, b = parseInt(h.slice(5, 7), 16) / 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), l = (mx + mn) / 2; let hh = 0, s = 0;
  if (mx !== mn) { const d = mx - mn; s = l > .5 ? d / (2 - mx - mn) : d / (mx + mn);
    hh = mx === r ? (g - b) / d + (g < b ? 6 : 0) : mx === g ? (b - r) / d + 2 : (r - g) / d + 4; hh *= 60; }
  return [hh, s, l];
}
function hslCss(h, s, l) { return `hsl(${((h % 360) + 360) % 360},${(s * 100).toFixed(0)}%,${(l * 100).toFixed(0)}%)`; }
// colours of k clusters: each takes the colour of its dominant art style; clusters sharing a dominant style become
// lighter / darker shades of that colour (largest cluster keeps the base colour)
function clusterStyleColors(lab, k) {
  const cnt = Array(k).fill(0), st = Array.from({ length: k }, () => Array(NC).fill(0));
  lab.forEach((l, i) => { cnt[l]++; st[l][D.lab[i]]++; });
  const dom = st.map(row => row.indexOf(Math.max(...row))), byStyle = {};
  dom.forEach((d, c) => { (byStyle[d] = byStyle[d] || []).push(c); });
  const colors = Array(k);
  for (const [d, cs] of Object.entries(byStyle)) {
    cs.sort((a, b) => cnt[b] - cnt[a]);
    const [h, s, l0] = hexToHsl(STYLE_PAL[+d]), m = cs.length;
    const dir = l0 > 0.55 ? -1 : 1, step = Math.min(0.17, 0.42 / Math.max(1, m - 1));   // shade towards the side with room
    cs.forEach((c, t) => {
      if (t === 0) { colors[c] = STYLE_PAL[+d]; return; }                     // the largest cluster keeps the base colour
      colors[c] = hslCss(h + 4 * t * dir, Math.min(1, s * (1 + 0.04 * t)), Math.max(0.2, Math.min(0.88, l0 + dir * step * t)));
    });
  }
  return { colors, dom, cnt, st };
}
const $ = id => document.getElementById(id);

// ── explicit images: flagged samples (D.nsfw = 1) are hidden unless "show explicit images" is ticked ──
// The tick is stored under one key, so the overview page and every explorer page share it; the default is hidden.
// Hidden samples are not drawn, cannot be hovered, selected or listed, and their image file is never requested.
const NSFW = D.nsfw || null;
const EXPL_KEY = "artExplorer.showExplicit";
let showExplicit = false;
try { showExplicit = localStorage.getItem(EXPL_KEY) === "1"; } catch (e) {}
const hidden = i => NSFW !== null && !showExplicit && NSFW[i] === 1;
const HIDDEN_IMG = "data:image/svg+xml;utf8," + encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" width="160" height="160"><rect width="160" height="160" fill="#222"/>' +
  '<text x="80" y="84" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#888">hidden</text></svg>');
const thumb = i => hidden(i) ? HIDDEN_IMG : D.thumbs[i];

// ── controls ──
const projsel = $("projsel"), spacesel = $("spacesel"), ksel = $("ksel"), colorsel = $("colorsel");
for (const [key, p] of Object.entries(D.proj)) projsel.add(new Option(p.name, key));
for (const [key, s] of Object.entries(D.spaces)) spacesel.add(new Option(s, key));
for (const k of D.ks) ksel.add(new Option(String(k), String(k)));
projsel.value = D.default_proj in D.proj ? D.default_proj : Object.keys(D.proj)[0];
ksel.value = String(D.ks.includes(D.default_k) ? D.default_k : D.ks[0]);
$("subtitle").textContent = D.subtitle;
if ((D.sets || []).length > 1) {                 // other sample sets / the full dataset of this representation
  $("setwrap").hidden = false;
  for (const st of D.sets) { const o = new Option(st.label, st.href); if (st.current) o.selected = true; $("setsel").add(o); }
  $("setsel").value = (D.sets.find(x => x.current) || D.sets[0]).href;
  $("setsel").addEventListener("change", () => { location.href = $("setsel").value; });
}
$("nav").innerHTML = `<a href="index.html">&#8962; overview</a> &middot; ` + D.siblings.map(s =>
  `<a href="${s.href}" class="${s.current ? "cur" : ""}">${s.name}</a>`).join("");
if ((D.variants || []).length > 1) {            // variants of this model (e.g. SimCLR augmentation sets)
  $("varwrap").hidden = false;
  for (const v of D.variants) { const o = new Option(v.label, v.href); if (v.current) o.selected = true; $("varsel").add(o); }
  $("varsel").addEventListener("change", () => { location.href = $("varsel").value; });
}
if (D.siblings.length > 8) {                     // many models: a drop-down instead of a link per model
  $("nav").innerHTML = `<a href="index.html">&#8962; overview</a> &middot; <select id="navsel" title="other representations of this dataset">` +
    D.siblings.map(s => `<option value="${s.href}"${s.current ? " selected" : ""}>${s.name}</option>`).join("") + `</select>`;
  $("navsel").addEventListener("change", () => { location.href = $("navsel").value; });
}

// ── state ──
const map = $("map"), g = map.getContext("2d");
let view = null, hover = -1, pinned = -1, isolate = null, bounds = [0, 1, 0, 1], raf = 0;
const curProj = () => D.proj[projsel.value].xy;
const curK = () => +ksel.value;
const curLab = () => D.clusters[spacesel.value][curK()];

// ── neighbour / interpolation data (decoded once) ──
function b64(s, T) {
  const bin = atob(s), u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return new T(u.buffer);
}
const NN = D.nn || null;
let KNN_IDX, KNN_SIM, PCS;
if (NN) {
  KNN_IDX = b64(NN.idx, NN.idx_t === "u4" ? Uint32Array : Uint16Array);
  KNN_SIM = b64(NN.sim, Uint8Array);
  PCS = b64(NN.pcs, Int16Array);                       // N x P, int16 -> scale
} else { document.getElementById("modesel").querySelectorAll("option:not([value=explore])").forEach(o => o.remove()); }
const SIMILARITY = (i, r) => KNN_SIM[i * NN.K + r] / 255;
let mode = "explore", qpt = -1, ptA = -1, ptB = -1, ov = null;   // ov = overlay computed for the active mode

function setBounds() {
  const xy = curProj(); let a = 1e30, b = -1e30, c = 1e30, d = -1e30;
  for (let i = 0; i < N; i++) { const [x, y] = xy[i];
    if (x < a) a = x; if (x > b) b = x; if (y < c) c = y; if (y > d) d = y; }
  bounds = [a, b, c, d];
}
function resetView() {
  const w = map.clientWidth, h = map.clientHeight, pad = 24;
  const s = Math.min((w - 2 * pad) / (bounds[1] - bounds[0] || 1), (h - 2 * pad) / (bounds[3] - bounds[2] || 1));
  view = { s, tx: w / 2 - s * (bounds[0] + bounds[1]) / 2, ty: h / 2 + s * (bounds[2] + bounds[3]) / 2 };
}
const toScreen = p => [view.tx + view.s * p[0], view.ty - view.s * p[1]];
function fitCanvas() {
  const r = window.devicePixelRatio || 1;
  map.width = map.clientWidth * r; map.height = map.clientHeight * r;
  g.setTransform(r, 0, 0, r, 0, 0);
}

// ── colouring ──
const hex = v => "#" + v.toString(16).padStart(6, "0");
const chips = cols => `<span class="chips">${cols.map(c => `<b style="background:${c}"></b>`).join("")}</span>`;
function viridis(t) {
  const s = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
  t = Math.min(1, Math.max(0, t)) * 4; const i = Math.min(3, Math.floor(t)), f = t - i;
  return `rgb(${s[i].map((v, j) => Math.round(v + f * (s[i + 1][j] - v)))})`;
}
function pct(a, q) { const s = Array.from(a).sort((x, y) => x - y); return s[Math.floor(q * (s.length - 1))]; }
const RANGE = {};   // robust 2%-98% range per continuous variable
for (const key of ["light", "chroma"]) if (D[key]) RANGE[key] = [pct(D[key], .02), pct(D[key], .98)];
// painting date: only some samples have one (parsed from the file name); the rest are shown grey
const YEARS = D.year || [];
const hasYear = YEARS.some(v => v !== null);
let YR = [0, 1];
if (hasYear) { const known = YEARS.filter(v => v !== null); YR = [pct(known, .02), pct(known, .98)]; }
else colorsel.querySelector('option[value="year"]').remove();

// extra attributes (dataset metadata, title keywords, CLIP zero-shot): one option per attribute, grouped
const ATTRS = D.attrs || [];
const ATTR = {};
ATTRS.forEach(a => {                              // compact base64 arrays -> plain arrays
  if (!a.b64) return;
  if (a.kind === "num") { const q = b64(a.b64, Uint16Array), span = (a.hi - a.lo) || 1; a.v = new Array(q.length);
    for (let i = 0; i < q.length; i++) a.v[i] = q[i] === 65535 ? null : a.lo + q[i] / 65534 * span; }
  else a.v = Array.from(b64(a.b64, a.kind === "color" ? Int32Array : Int16Array));
  delete a.b64;
});
{
  const groups = {};
  ATTRS.forEach(a => {
    ATTR["a:" + a.key] = a;
    if (a.kind === "num") { const v = a.v.filter(x => x !== null);
      a.mean = v.reduce((s, x) => s + x, 0) / (v.length || 1);
      a.std = Math.sqrt(v.reduce((s, x) => s + (x - a.mean) ** 2, 0) / (v.length || 1)) || 1; }
    (groups[a.group] = groups[a.group] || []).push(a);
  });
  for (const [name, list] of Object.entries(groups)) {
    const og = document.createElement("optgroup"); og.label = name;
    list.forEach(a => og.appendChild(new Option(a.label, "a:" + a.key)));
    colorsel.appendChild(og);
  }
}
const isTag = a => a.kind === "cat" && a.names.length === 2 && a.names[1] === "yes";

// Continuous variable -> viridis colour per point, plus NB clickable value bins in the legend
// (click = isolate the points in that range).  Values outside [lo, hi] fall into the end bins;
// null values form their own grey bin (-1).
function binned(v, lo, hi, nb, fmt, missingLabel) {
  const span = hi - lo || 1, bin = x => Math.min(nb - 1, Math.max(0, Math.floor((x - lo) / span * nb)));
  const cnt = Array(nb).fill(0); let miss = 0;
  v.forEach(x => { if (x === null) miss++; else cnt[bin(x)]++; });
  const legend = [];
  for (let b = nb - 1; b >= 0; b--) {                // high values on top
    const a = lo + span * b / nb, c = lo + span * (b + 1) / nb;
    const lab = b === 0 ? `\u2264 ${fmt(c)}` : b === nb - 1 ? `\u2265 ${fmt(a)}` : `${fmt(a)} \u2013 ${fmt(c)}`;
    legend.push([viridis((b + .5) / nb), `${lab}  (n=${cnt[b]})`, b]);
  }
  if (miss) legend.push(["#555", `${missingLabel}  (n=${miss})`, -1]);
  return { id: i => v[i] === null ? -1 : bin(v[i]),
           color: i => v[i] === null ? "#555" : viridis((v[i] - lo) / span), legend };
}

function attrScheme(a) {
  if (a.kind === "color") {               // value = packed RGB (-1 = n/a): drawn in that colour, legend = lightness bins
    const NB = 5, luma = x => ((x >> 16 & 255) * .299 + (x >> 8 & 255) * .587 + (x & 255) * .114) / 255;
    const ok = a.v.filter(x => x >= 0).map(luma);
    const lo = pct(ok, .02), hi = pct(ok, .98), span = hi - lo || 1;
    const bin = x => x < 0 ? -1 : Math.min(NB - 1, Math.max(0, Math.floor((luma(x) - lo) / span * NB)));
    const cnt = Array(NB).fill(0), sum = Array.from({ length: NB }, () => [0, 0, 0]); let miss = 0;
    a.v.forEach(x => { const b = bin(x); if (b < 0) miss++; else { cnt[b]++; sum[b][0] += x >> 16 & 255;
      sum[b][1] += x >> 8 & 255; sum[b][2] += x & 255; } });
    const legend = [];
    for (let b = NB - 1; b >= 0; b--) legend.push([`rgb(${sum[b].map(t => Math.round(t / (cnt[b] || 1)))})`,
      `${b === NB - 1 ? "lightest" : b === 0 ? "darkest" : "mid"} ${b + 1}/${NB}  (n=${cnt[b]})`, b]);
    if (miss) legend.push(["#333", `n/a  (n=${miss})`, -1]);
    return { id: i => bin(a.v[i]), color: i => a.v[i] < 0 ? "#333" : hex(a.v[i]), legend };
  }
  if (a.kind === "num") return binned(a.v, a.range[0], a.range[1], 8, x => x.toFixed(1), "n/a");
  const nm = a.names, cnt = Array(nm.length + 1).fill(0);
  a.v.forEach(x => cnt[x < 0 ? nm.length : x]++);
  const pal = isTag(a) ? ["#555", "#ff9f1c"] : palette(nm.length).map((c, j) => nm[j] === "(other)" ? "#777" : c);
  const colorOf = x => x < 0 ? "#333" : pal[x];
  const leg = nm.map((n, j) => [pal[j], `${n}  (n=${cnt[j]})`, j]);
  if (cnt[nm.length]) leg.push(["#333", `unknown  (n=${cnt[nm.length]})`, -1]);
  return { id: i => a.v[i], color: i => colorOf(a.v[i]), legend: isTag(a) ? leg.slice().reverse() : leg };
}

function scheme() {
  const cs = colorsel.value;
  if (cs in ATTR) return attrScheme(ATTR[cs]);
  if (cs === "style") {
    const cnt = Array(NC).fill(0); D.lab.forEach(l => cnt[l]++);
    return { id: i => D.lab[i], color: i => STYLE_PAL[D.lab[i]],
      legend: STYLE_ORDER.filter(j => cnt[j] > 0).map(j =>
        [STYLE_PAL[j], `${D.classes[j]}${STYLE_YEAR[j] ? " (~" + STYLE_YEAR[j] + ")" : ""}  n=${cnt[j]}`, j]) };
  }
  if (cs === "dominant")       // each point drawn in its painting's most dominant colour
    return { id: () => 0, fixed: true, color: i => hex(D.pal_c[i][0]), legend: [] };
  if (cs === "dominant_ref") {
    const m = D.ref_colors.length, cnt = Array(m).fill(0), st = Array.from({ length: m }, () => Array(NC).fill(0));
    D.dom_ref.forEach((r, i) => { cnt[r]++; st[r][D.lab[i]]++; });
    return { id: i => D.dom_ref[i], color: i => D.ref_colors[D.dom_ref[i]],
      legend: D.ref_colors.map((c, r) => { const j = st[r].indexOf(Math.max(...st[r]));
        return [c, `${c}  n=${cnt[r]}` + (cnt[r] ? ` · ${Math.round(100 * st[r][j] / cnt[r])}% ${D.classes[j]}` : ""), r]; }) };
  }
  if (cs === "light" || cs === "chroma") {
    const [a, b] = RANGE[cs];
    return binned(D[cs], a, b, 6, x => x.toFixed(0), "n/a");
  }
  if (cs === "year") {
    return binned(YEARS, YR[0], YR[1], 8, x => String(Math.round(x)), "no date");
  }
  if (cs === "theme") {
    const k = curK(), lab = D.theme[k], pal = D.theme_sw[k], cnt = Array(k).fill(0);
    lab.forEach(l => cnt[l]++);
    return { id: i => lab[i], color: i => pal[lab[i]][0],
      legend: cnt.map((n, c) => [pal[c][0], `${c} ${chips(pal[c])} n=${n}`, c]) };
  }
  if (cs === "cluster_style") {
    const k = curK(), lab = curLab(), { colors, dom, cnt, st } = clusterStyleColors(lab, k);
    const order = [...Array(k).keys()].sort((a, b) => (STYLE_YEAR[dom[a]] ?? 9999) - (STYLE_YEAR[dom[b]] ?? 9999) || cnt[b] - cnt[a]);
    return { id: i => lab[i], color: i => colors[lab[i]],
      legend: order.map(c => [colors[c], `${c}  n=${cnt[c]} · ${Math.round(100 * st[c][dom[c]] / cnt[c])}% ${D.classes[dom[c]]}`, c]) };
  }
  const k = curK(), lab = curLab(), pal = palette(k);
  const cnt = Array(k).fill(0), st = Array.from({ length: k }, () => Array(NC).fill(0));
  lab.forEach((l, i) => { cnt[l]++; st[l][D.lab[i]]++; });
  return { id: i => lab[i], color: i => pal[lab[i]],
    legend: cnt.map((n, c) => { const m = st[c].indexOf(Math.max(...st[c]));
      return [pal[c], `${c}  n=${n} · ${Math.round(100 * st[c][m] / n)}% ${D.classes[m]}`, c]; }) };
}

// ── modes ──
const MC = "#ffd54f", AC = "#4fc3f7", BC = "#ff8a65";
function tag(i, text, color) {
  return `<span class="tag" style="background:${color}">${text}</span>`;
}
function figure(i, caption, tagHtml, attr) {
  return `<figure data-i="${i}" ${attr || ""}><img src="${thumb(i)}" loading="lazy">${tagHtml || ""}
    <figcaption>${caption}</figcaption></figure>`;
}
function computeKnn() {
  const K = +$("kk").value; $("kkv").textContent = K;
  if (qpt < 0) { ov = null; return; }
  const ids = [], sims = [];
  for (let r = 0; r < NN.K && ids.length < K; r++) {            // flagged neighbours are skipped while explicit images are hidden
    const j = KNN_IDX[qpt * NN.K + r]; if (hidden(j)) continue;
    ids.push(j); sims.push(SIMILARITY(qpt, r)); }
  ov = { kind: "knn", q: qpt, ids, sims };
  const same = ids.filter(j => D.lab[j] === D.lab[qpt]).length;
  $("modetitle").textContent = "Nearest neighbours";
  $("modehelp").innerHTML = `Query: <b>${D.classes[D.lab[qpt]]}</b> · ${same} / ${K} neighbours share its art style ` +
    `(${Math.round(100 * same / K)}%). Cosine similarity in the full embedding space. Click an image to re-query.`;
  $("strip").innerHTML = figure(qpt, "query · " + D.classes[D.lab[qpt]], tag(qpt, "Q", MC), 'style="outline-color:' + MC + '"') +
    ids.map((j, r) => figure(j, `${sims[r].toFixed(2)} · ${D.classes[D.lab[j]]}`, tag(j, r + 1, "#fff"))).join("");
}
function computeInterp() {
  const S = +$("steps").value; $("stepsv").textContent = S;
  if (ptA < 0) { ov = null; return; }
  const P = NN.P, sc = NN.pcs_scale / 32767;
  const row = i => { const v = new Float32Array(P); for (let d = 0; d < P; d++) v[d] = PCS[i * P + d] * sc; return v; };
  if (ptB < 0) {
    ov = { kind: "interp", A: ptA, B: -1, path: [] };
    $("modetitle").textContent = "Interpolation";
    $("modehelp").textContent = "A is set (blue). Click a second point for B.";
    $("strip").innerHTML = figure(ptA, "A · " + D.classes[D.lab[ptA]], tag(ptA, "A", AC)); return;
  }
  const a = row(ptA), b = row(ptB), used = new Set([ptA, ptB]), path = [ptA];
  const v = new Float32Array(P);
  for (let s = 1; s < S - 1; s++) {                    // endpoints are A and B themselves
    const t = s / (S - 1);
    for (let d = 0; d < P; d++) v[d] = (1 - t) * a[d] + t * b[d];
    let best = -1, bd = 1e30;
    for (let i = 0; i < N; i++) {
      if (used.has(i) || hidden(i)) continue;
      let dist = 0, o = i * P;
      for (let d = 0; d < P; d++) { const x = PCS[o + d] * sc - v[d]; dist += x * x; if (dist >= bd) break; }
      if (dist < bd) { bd = dist; best = i; }
    }
    used.add(best); path.push(best);
  }
  path.push(ptB);
  ov = { kind: "interp", A: ptA, B: ptB, path };
  $("modetitle").textContent = "Interpolation A \u2192 B";
  $("modehelp").innerHTML = `At each of ${S} evenly spaced points on the straight line from <b style="color:${AC}">A</b> to ` +
    `<b style="color:${BC}">B</b> (in the leading ${NN.P} PCA dimensions, ${(100 * NN.pcs_var).toFixed(0)}% of the variance) ` +
    `the nearest not-yet-used sample is shown. Click two new points to start over.`;
  $("strip").innerHTML = path.map((j, s) => {
    const t = s / (S - 1), c = s === 0 ? AC : s === S - 1 ? BC : "#fff";
    return figure(j, `t=${t.toFixed(2)} · ${D.classes[D.lab[j]]}`, tag(j, s === 0 ? "A" : s === S - 1 ? "B" : s, c)); }).join("");
}
function applyMode() {
  mode = $("modesel").value; ov = null; pinned = -1; updateSide();
  $("knnctl").hidden = mode !== "knn"; $("interpctl").hidden = mode !== "interp";
  $("modecard").hidden = mode === "explore";
  $("hint").textContent = mode === "knn" ? "click a point: show its nearest neighbours · wheel: zoom · drag: pan · dbl-click: reset" :
    mode === "interp" ? "click point A, then point B · wheel: zoom · drag: pan · dbl-click: reset" :
    "hover: inspect · click: pin · wheel: zoom · drag: pan · dbl-click: reset · legend: isolate";
  if (mode === "knn") { $("modetitle").textContent = "Nearest neighbours"; $("modehelp").textContent = "Click a point on the map."; $("strip").innerHTML = ""; computeKnn(); }
  if (mode === "interp") { ptA = ptB = -1; $("modetitle").textContent = "Interpolation"; $("modehelp").textContent = "Click point A, then point B."; $("strip").innerHTML = ""; }
  schedule();
}
function modeClick(i) {
  if (i < 0) return;
  if (mode === "knn") { qpt = i; computeKnn(); show(i); }
  else if (mode === "interp") {
    if (ptA < 0 || ptB >= 0) { ptA = i; ptB = -1; } else if (i !== ptA) ptB = i; else return;
    computeInterp(); show(i);
  }
  schedule();
}
function drawOverlay(sc, xy) {
  const ids = ov.kind === "knn" ? [ov.q, ...ov.ids] : ov.path;
  g.globalAlpha = 0.18; g.fillStyle = "#111";            // veil over the (already drawn) cloud
  g.fillRect(0, 0, map.clientWidth, map.clientHeight);
  const r = +$("size").value, pt = i => toScreen(xy[i]);
  g.lineWidth = 1.5;
  if (ov.kind === "knn") {
    const [qx, qy] = pt(ov.q);
    ov.ids.forEach((j, k) => { const [x, y] = pt(j);
      g.globalAlpha = 0.25 + 0.6 * (ov.sims[k] ** 2); g.strokeStyle = MC;
      g.beginPath(); g.moveTo(qx, qy); g.lineTo(x, y); g.stroke(); });
  } else if (ov.B >= 0) {
    const [ax, ay] = pt(ov.A), [bx, by] = pt(ov.B);
    g.globalAlpha = .6; g.strokeStyle = "#fff"; g.setLineDash([5, 5]);
    g.beginPath(); g.moveTo(ax, ay); g.lineTo(bx, by); g.stroke(); g.setLineDash([]);
    g.globalAlpha = .95; g.strokeStyle = MC; g.lineWidth = 2; g.beginPath();
    ov.path.forEach((j, s) => { const [x, y] = pt(j); s ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
  }
  g.globalAlpha = 1; g.font = "11px system-ui"; g.textAlign = "center"; g.textBaseline = "middle";
  ids.forEach((j, k) => {
    const [x, y] = pt(j);
    const isQ = ov.kind === "knn" && k === 0, isA = ov.kind === "interp" && j === ov.A, isB = ov.kind === "interp" && j === ov.B;
    const ring = isQ ? MC : isA ? AC : isB ? BC : "#fff", big = isQ || isA || isB;
    g.fillStyle = sc.color(j); g.strokeStyle = ring; g.lineWidth = big ? 3 : 2;
    g.beginPath(); g.arc(x, y, r + (big ? 6 : 4), 0, 6.2832); g.fill(); g.stroke();
    const txt = isQ ? "Q" : isA ? "A" : isB ? "B" : ov.kind === "knn" ? k : ov.path.indexOf(j);
    g.fillStyle = "#000"; g.strokeStyle = "#fff"; g.lineWidth = 3;
    g.strokeText(String(txt), x, y - r - 14); g.fillText(String(txt), x, y - r - 14);
  });
}

function schedule() { if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); }); }
function draw() {
  const sc = scheme(), xy = curProj(), w = map.clientWidth, h = map.clientHeight;
  const r = +$("size").value, cur = pinned >= 0 ? pinned : hover;
  g.clearRect(0, 0, w, h);
  const selOn = selActive(), active = isolate !== null || selOn;
  for (let pass = 0; pass < 2; pass++) {       // dimmed (not isolated / not selected) first, then the rest on top
    g.globalAlpha = !active ? (cur >= 0 ? 0.55 : 0.8) : (pass === 0 ? 0.07 : 0.9);
    for (let i = 0; i < N; i++) {
      if (hidden(i)) continue;
      const on = (isolate === null || sc.id(i) === isolate) && (!selOn || selFlag[i] === 1);
      if (active && on !== (pass === 1)) continue;
      if (!active && pass === 1) continue;
      const [x, y] = toScreen(xy[i]);
      if (x < -5 || y < -5 || x > w + 5 || y > h + 5) continue;
      g.fillStyle = sc.color(i);
      if (N > 20000) g.fillRect(x - r, y - r, 2 * r, 2 * r);       // squares are much faster for 50k+ points
      else { g.beginPath(); g.arc(x, y, r, 0, 6.2832); g.fill(); }
    }
  }
  g.globalAlpha = 1;
  drawSelectionShapes();
  if (ov) drawOverlay(sc, xy);
  if (hilite >= 0 && !ov) {
    const [x, y] = toScreen(xy[hilite]);
    g.strokeStyle = "#ffd54f"; g.lineWidth = 3; g.beginPath(); g.arc(x, y, r + 7, 0, 6.2832); g.stroke();
  }
  if (cur >= 0 && !ov) {
    const [x, y] = toScreen(xy[cur]);
    g.fillStyle = sc.color(cur); g.strokeStyle = "#fff"; g.lineWidth = 2;
    g.beginPath(); g.arc(x, y, r + 4, 0, 6.2832); g.fill(); g.stroke();
  }
  $("legend").innerHTML = sc.legend.map(([c, t, id]) =>
    `<div data-id="${id}" class="${isolate !== null && isolate !== id ? "off" : ""}"><i style="background:${c}"></i>${t}</div>`).join("");
  const isTheme = colorsel.value === "theme";
  const m = isTheme ? (D.theme_metrics || {})[curK()] : (D.metrics[spacesel.value] || {})[curK()];
  $("metrics").textContent = !m ? "" : isTheme ?
    `colour themes, k=${curK()}: ARI vs style ${m.ari.toFixed(3)} · NMI ${m.nmi.toFixed(3)}` :
    `k=${curK()} (${D.spaces[spacesel.value]}): silhouette ${m.sil.toFixed(3)} · ARI vs style ${m.ari.toFixed(3)} · NMI ${m.nmi.toFixed(3)}`;
}

// ── side panel ──
function show(i) {
  $("pin").textContent = pinned >= 0 ? "· pinned (click to release)" : "";
  if (i < 0) return;
  if (hidden(i)) {          // flagged and explicit images are hidden: no image, no title, no metadata
    $("thumb").src = HIDDEN_IMG; $("palette").innerHTML = ""; $("comp").innerHTML = "";
    $("info").innerHTML = `<div><dt>sample</dt><dd>hidden (flagged as explicit)</dd></div>`; return;
  }
  $("thumb").src = thumb(i);
  const k = curK(), rows = [["sample", `${i}  (file ${D.idx[i]})`], ["file", D.files[i]],
    ["art style", D.classes[D.lab[i]]]];
  for (const [key, name] of Object.entries(D.spaces))
    rows.push([`cluster, ${key}`, `${D.clusters[key][k][i]} / ${k}`]);
  if (hasYear) rows.push(["year", YEARS[i] === null ? "unknown" : YEARS[i]]);
  const tags = [], nums = [];
  for (const a of ATTRS) {
    if (a.kind === "cat") {
      const x = a.v[i];
      if (isTag(a)) { if (x === 1) tags.push(a.label.replace("title has: ", "")); }
      else rows.push([a.label.replace(/ \(top \d+\)/, ""), x < 0 ? "unknown" : a.names[x]]);
    } else if (a.kind === "color") {
      if (a.v[i] >= 0) rows.push([a.label, `<span class="chips"><b style="background:${hex(a.v[i])};width:28px"></b></span> ${hex(a.v[i])}`]);
    } else if (a.v[i] !== null && a.group.startsWith("CLIP")) nums.push([(a.v[i] - a.mean) / a.std, a.label]);
    else if (a.v[i] !== null) rows.push([a.label, a.v[i]]);
  }
  if (ATTRS.some(isTag)) rows.push(["title keywords", tags.length ? tags.join(", ") : "—"]);
  if (nums.length) rows.push(["CLIP top concepts", nums.sort((p, q) => q[0] - p[0]).slice(0, 3)
    .map(([z, n]) => `${n} (${z >= 0 ? "+" : ""}${z.toFixed(1)}σ)`).join(", ")]);
  if (D.theme) rows.push(["colour theme", `${D.theme[k][i]} / ${k}`],
    ["lightness", D.light[i].toFixed(0)], ["colourfulness", D.chroma[i].toFixed(0)]);
  $("info").innerHTML = rows.map(([a, b]) => `<div><dt>${a}</dt><dd>${b}</dd></div>`).join("");
  if (D.pal_c) $("palette").innerHTML = D.pal_c[i].map((c, j) =>
    `<i title="${hex(c)} · ${(100 * D.pal_w[i][j]).toFixed(0)}%" style="flex:${D.pal_w[i][j]};background:${hex(c)}"></i>`).join("");
  const lab = colorsel.value === "theme" ? D.theme[k] : curLab(), c = lab[i], cnt = Array(NC).fill(0); let n = 0;
  for (let j = 0; j < N; j++) if (lab[j] === c) { cnt[D.lab[j]]++; n++; }
  const order = cnt.map((v, j) => [v, j]).filter(a => a[0] > 0).sort((a, b) => b[0] - a[0]).slice(0, 8);
  $("comp").innerHTML = `<div style="color:var(--muted);margin-bottom:4px">cluster ${c} · ${n} samples</div>` +
    order.map(([v, j]) => `<div class="bar"><span class="n">${D.classes[j]}</span>
      <span class="b"><i style="width:${100 * v / n}%;background:${STYLE_PAL[j]}"></i></span>
      <span class="p">${(100 * v / n).toFixed(0)}%</span></div>`).join("");
}

function nearest(mx, my) {
  const xy = curProj(), sc = scheme(); let best = -1, bd = 144;
  for (let i = 0; i < N; i++) {
    if (hidden(i)) continue;
    if (isolate !== null && sc.id(i) !== isolate) continue;
    if (selActive() && selFlag[i] !== 1) continue;
    const [x, y] = toScreen(xy[i]), d = (x - mx) ** 2 + (y - my) ** 2;
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}
// ── group view: with a legend group highlighted and no sample pinned, the side panel shows samples of that group ──
let hilite = -1, gSeed = 1;
const groupView = () => mode === "explore" && isolate !== null && pinned < 0;
function rng32(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a;
  t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
// order a set of samples: random (seeded) | typical (closest to the centre of the set) | outlier | style (grouped)
function orderMembers(m, order, seed) {
  const a = m.slice();
  const shuffle = () => { const r = rng32(seed); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
  if (order === "random" || a.length <= 1) return shuffle();
  if (order === "style") return shuffle().sort((p, q) => D.lab[p] - D.lab[q]);
  const dim = NN ? NN.P : 2, coord = NN ? (i, d) => PCS[i * dim + d] : (i, d) => curProj()[i][d];
  const c = new Float64Array(dim);                 // centre in the leading PCA dims (else the 2D map)
  for (const i of a) for (let d = 0; d < dim; d++) c[d] += coord(i, d) / a.length;
  const dist = new Map(a.map(i => { let s = 0; for (let d = 0; d < dim; d++) s += (coord(i, d) - c[d]) ** 2; return [i, s]; }));
  return a.sort((p, q) => order === "typical" ? dist.get(p) - dist.get(q) : dist.get(q) - dist.get(p));
}
function groupMembers() {
  const sc = scheme(), m = [];
  for (let i = 0; i < N; i++) if (sc.id(i) === isolate && !hidden(i)) m.push(i);
  const leg = sc.legend.find(l => l[2] === isolate);
  return { m, label: leg ? leg[1] : String(isolate) };
}
function isolateHiddenCount() {                   // members of the highlighted group that are hidden as explicit
  const sc = scheme(); let n = 0;
  for (let i = 0; i < N; i++) if (sc.id(i) === isolate && hidden(i)) n++;
  return n;
}
function renderGroup() {
  const { m, label } = groupMembers(), order = $("gorder").value, K = 18;
  $("grouptitle").innerHTML = label;
  $("groupn").textContent = `${m.length.toLocaleString()} samples`;
  const pick = orderMembers(m, order, gSeed * 7919 + isolate * 31).slice(0, K);
  const nHid = NSFW ? isolateHiddenCount() : 0;
  $("gallery").innerHTML = pick.length ? pick.map(i => `<figure data-i="${i}"><img src="${thumb(i)}" loading="lazy">
    <figcaption>${D.classes[D.lab[i]]}</figcaption></figure>`).join("") :
    `<p style="color:var(--muted);font-size:12px;grid-column:1/-1">All samples of this group are hidden because they are flagged as explicit. ` +
    `Tick &ldquo;show explicit images&rdquo; in the header to see them.</p>`;
  if (pick.length && nHid) $("groupn").textContent += ` (+ ${nHid.toLocaleString()} hidden as explicit)`;
  // composition of the group by art style
  const cnt = Array(NC).fill(0); m.forEach(i => cnt[D.lab[i]]++);
  $("comptitle").textContent = "Art styles in this group";
  $("comp").innerHTML = cnt.map((v, j) => [v, j]).filter(a => a[0] > 0).sort((a, b) => b[0] - a[0]).slice(0, 8).map(([v, j]) =>
    `<div class="bar"><span class="n">${D.classes[j]}</span><span class="b"><i style="width:${100 * v / m.length}%;background:${STYLE_PAL[j]}"></i></span>
     <span class="p">${(100 * v / m.length).toFixed(0)}%</span></div>`).join("");
  $("info").innerHTML = `<div><dt>group</dt><dd>${label}</dd></div><div><dt>samples</dt><dd>${m.length.toLocaleString()} of ${N.toLocaleString()}</dd></div>
    <div><dt>shown</dt><dd>${pick.length} (${order})</dd></div>`;
}
const selView = () => selActive() && pinned < 0;
const listView = () => selView() || groupView();
function updateSide() {
  const sel = selView(), grp = !sel && groupView();
  $("selcard").hidden = !sel; $("groupcard").hidden = !grp; $("samplecard").hidden = sel || grp;
  if (!sel && !grp) { hilite = -1; $("tip").hidden = true; $("comptitle").textContent = "Art styles in this cluster"; }
  if (sel) renderSelection(); else if (grp) renderGroup();
}
function tip(i, mx, my) {
  const t = $("tip");
  if (i < 0 || !listView()) { t.hidden = true; return; }
  t.hidden = false; t.querySelector("img").src = thumb(i); t.querySelector("span").textContent = D.classes[D.lab[i]];
  const w = map.clientWidth, h = map.clientHeight;
  t.style.left = Math.min(mx + 16, w - 160) + "px"; t.style.top = Math.min(my + 16, h - 180) + "px";
}
function setHover(i) {
  if (pinned >= 0 || i === hover) return;
  hover = i;
  if (i >= 0 && !listView()) show(i);
  schedule();
}
function togglePin(i) {
  pinned = (pinned >= 0 || i < 0) ? -1 : i; hover = i;
  if (pinned >= 0 || !listView()) show(pinned >= 0 ? pinned : i);
  updateSide(); schedule();
}

// ── selection tools: rectangle and free-form lasso (Shift = add to the selection, Alt = remove from it) ──
let tool = "pan", selFlag = null, selList = [], selShapes = [], selDraw = null, selSorted = [], selPage = 0;
const selActive = () => mode === "explore" && selList.length > 0;
function setTool(t) {
  tool = t;
  document.querySelectorAll("#tools button[data-tool]").forEach(b => b.classList.toggle("on", b.dataset.tool === t));
  map.style.cursor = t === "pan" ? "crosshair" : "cell";
  $("hint").textContent = t === "pan" ? "hover: inspect · click: pin · wheel: zoom · drag: pan · dbl-click: reset · legend: isolate" :
    "drag to select (Shift: add · Alt: remove) · wheel: zoom · V: pan tool · Esc: clear";
}
function pointInPoly(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1], xj = poly[j][0], yj = poly[j][1];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
function finishSelection() {
  const d = selDraw; selDraw = null;
  if (!d || !moved) { schedule(); return; }
  let poly;
  if (d.type === "rect") { const [a, b] = [d.pts[0], d.pts[d.pts.length - 1]];
    const x0 = Math.min(a[0], b[0]), x1 = Math.max(a[0], b[0]), y0 = Math.min(a[1], b[1]), y1 = Math.max(a[1], b[1]);
    if (x1 - x0 < 3 || y1 - y0 < 3) { schedule(); return; }
    poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]; }
  else { if (d.pts.length < 3) { schedule(); return; } poly = d.pts; }
  let bx0 = 1e9, bx1 = -1e9, by0 = 1e9, by1 = -1e9;
  for (const [x, y] of poly) { bx0 = Math.min(bx0, x); bx1 = Math.max(bx1, x); by0 = Math.min(by0, y); by1 = Math.max(by1, y); }
  const xy = curProj(), sc = scheme(), inside = [];
  for (let i = 0; i < N; i++) {
    if (hidden(i)) continue;
    if (isolate !== null && sc.id(i) !== isolate) continue;        // only what is visible can be selected
    const [x, y] = toScreen(xy[i]);
    if (x < bx0 || x > bx1 || y < by0 || y > by1) continue;
    if (d.type === "rect" || pointInPoly(x, y, poly)) inside.push(i);
  }
  const shape = { proj: projsel.value, poly: poly.map(([x, y]) => [(x - view.tx) / view.s, (view.ty - y) / view.s]) };
  let list;
  if (d.mod === "add") { const s = new Set(selList); inside.forEach(i => s.add(i)); list = [...s]; selShapes.push(shape); }
  else if (d.mod === "sub") { const rm = new Set(inside); list = selList.filter(i => !rm.has(i)); }
  else { list = inside; selShapes = [shape]; }
  setSelection(list);
}
function setSelection(list) {
  selList = list; selPage = 0;
  selFlag = new Uint8Array(N); list.forEach(i => { selFlag[i] = 1; });
  if (!list.length) selShapes = [];
  sortSelection(); hilite = -1;
  $("selclear").hidden = !list.length;
  pinned = -1; updateSide(); schedule();
}
function clearSelection() { selDraw = null; setSelection([]); }
function sortSelection() { selSorted = orderMembers(selList, $("selorder").value, 4242); selPage = 0; }
function renderSelection() {
  const n = selList.length, ps = +$("selpagesize").value, pages = Math.max(1, Math.ceil(n / ps));
  selPage = Math.min(selPage, pages - 1);
  const from = selPage * ps, pick = selSorted.slice(from, from + ps);
  $("seltitle").textContent = `${n.toLocaleString()} sample${n === 1 ? "" : "s"} selected` +
    (isolate !== null ? " (within the highlighted group)" : "");
  $("selpos").textContent = `${from + 1}\u2013${from + pick.length} of ${n.toLocaleString()} · page ${selPage + 1}/${pages}`;
  $("selprev").disabled = selPage === 0; $("selnext").disabled = selPage >= pages - 1;
  $("selgallery").innerHTML = pick.map(i => `<figure data-i="${i}"><img src="${thumb(i)}" loading="lazy">
    <figcaption>${D.classes[D.lab[i]]}</figcaption></figure>`).join("");
  const cnt = Array(NC).fill(0); selList.forEach(i => cnt[D.lab[i]]++);
  $("comptitle").textContent = "Art styles in the selection";
  $("comp").innerHTML = cnt.map((v, j) => [v, j]).filter(a => a[0] > 0).sort((a, b) => b[0] - a[0]).slice(0, 8).map(([v, j]) =>
    `<div class="bar"><span class="n">${D.classes[j]}</span><span class="b"><i style="width:${100 * v / n}%;background:${STYLE_PAL[j]}"></i></span>
     <span class="p">${(100 * v / n).toFixed(0)}%</span></div>`).join("");
  $("info").innerHTML = `<div><dt>selection</dt><dd>${n.toLocaleString()} of ${N.toLocaleString()} samples</dd></div>
    <div><dt>shown</dt><dd>${pick.length} (${$("selorder").value})</dd></div>`;
}
function drawSelectionShapes() {
  const toS = poly => poly.map(([x, y]) => [view.tx + view.s * x, view.ty - view.s * y]);
  const path = poly => { g.beginPath(); poly.forEach(([x, y], k) => k ? g.lineTo(x, y) : g.moveTo(x, y)); g.closePath(); };
  g.save(); g.lineWidth = 1.5; g.setLineDash([6, 4]);
  if (mode === "explore") for (const sh of selShapes) if (sh.proj === projsel.value) {
    path(toS(sh.poly)); g.fillStyle = "rgba(255,213,79,.07)"; g.fill(); g.strokeStyle = "rgba(255,213,79,.9)"; g.stroke(); }
  if (selDraw) {
    const pts = selDraw.type === "rect" ? (() => { const a = selDraw.pts[0], b = selDraw.pts[selDraw.pts.length - 1];
      return [[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]]]; })() : selDraw.pts;
    path(pts); g.fillStyle = "rgba(79,195,247,.10)"; g.fill(); g.strokeStyle = "#4fc3f7"; g.stroke();
  }
  g.restore();
}

let drag = null, moved = false;
map.addEventListener("mousedown", e => {
  moved = false;
  if (mode === "explore" && tool !== "pan") {
    selDraw = { type: tool, pts: [[e.offsetX, e.offsetY]], mod: e.shiftKey ? "add" : (e.altKey || e.ctrlKey) ? "sub" : "new" };
    return;
  }
  drag = [e.offsetX, e.offsetY];
});
window.addEventListener("mouseup", () => { if (selDraw) finishSelection(); drag = null; });
map.addEventListener("mousemove", e => {
  if (selDraw) { const p = selDraw.pts, last = p[p.length - 1], dx = e.offsetX - last[0], dy = e.offsetY - last[1];
    if (selDraw.type === "rect") { if (Math.abs(dx) + Math.abs(dy) > 3) moved = true; p[1] = [e.offsetX, e.offsetY]; }
    else if (dx * dx + dy * dy > 9) { moved = true; p.push([e.offsetX, e.offsetY]); }
    schedule(); return; }
  if (drag) { const dx = e.offsetX - drag[0], dy = e.offsetY - drag[1];
    if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
    view.tx += dx; view.ty += dy; drag = [e.offsetX, e.offsetY]; schedule(); return; }
  const i = nearest(e.offsetX, e.offsetY);
  setHover(i); tip(i, e.offsetX, e.offsetY);
});
map.addEventListener("mouseleave", () => { $("tip").hidden = true; });
map.addEventListener("click", e => { if (moved) return; const i = nearest(e.offsetX, e.offsetY);
  if (mode === "explore") togglePin(i); else modeClick(i); });
$("modesel").addEventListener("change", applyMode);
$("kk").addEventListener("input", () => { computeKnn(); schedule(); });
$("steps").addEventListener("input", () => { computeInterp(); schedule(); });
$("swapab").addEventListener("click", () => { [ptA, ptB] = [ptB, ptA]; if (ptA >= 0) computeInterp(); schedule(); });
$("clearab").addEventListener("click", () => { ptA = ptB = -1; ov = null; $("strip").innerHTML = ""; $("modehelp").textContent = "Click point A, then point B."; schedule(); });
$("strip").addEventListener("click", e => { const f = e.target.closest("figure[data-i]"); if (!f) return;
  const i = +f.dataset.i; if (mode === "knn") { qpt = i; computeKnn(); show(i); schedule(); } else show(i); });
map.addEventListener("dblclick", () => { resetView(); schedule(); });
document.querySelectorAll("#tools button[data-tool]").forEach(b => b.addEventListener("click", () => setTool(b.dataset.tool)));
$("selclear").addEventListener("click", clearSelection); $("selclear2").addEventListener("click", clearSelection);
$("selorder").addEventListener("change", () => { sortSelection(); renderSelection(); });
$("selpagesize").addEventListener("change", () => { selPage = 0; renderSelection(); });
$("selprev").addEventListener("click", () => { selPage = Math.max(0, selPage - 1); renderSelection(); });
$("selnext").addEventListener("click", () => { selPage++; renderSelection(); });
$("selgallery").addEventListener("mouseover", e => { const f = e.target.closest("figure[data-i]"); hilite = f ? +f.dataset.i : -1; schedule(); });
$("selgallery").addEventListener("mouseout", () => { hilite = -1; schedule(); });
$("selgallery").addEventListener("click", e => { const f = e.target.closest("figure[data-i]"); if (f) { hilite = -1; togglePin(+f.dataset.i); } });
window.addEventListener("keydown", e => {
  if (/^(INPUT|SELECT|TEXTAREA)$/.test((e.target.tagName || ""))) return;
  const k = e.key.toLowerCase();
  if (k === "r") setTool("rect"); else if (k === "l") setTool("lasso"); else if (k === "v" || k === "p") setTool("pan");
  else if (k === "escape") { if (selList.length) clearSelection(); else if (isolate !== null) { isolate = null; updateSide(); schedule(); } }
});
map.addEventListener("wheel", e => { e.preventDefault();
  const f = Math.exp(-e.deltaY * 0.0015);
  view.tx = e.offsetX - f * (e.offsetX - view.tx); view.ty = e.offsetY - f * (e.offsetY - view.ty);
  view.s *= f; schedule(); }, { passive: false });
$("legend").addEventListener("click", e => {
  const d = e.target.closest("div[data-id]"); if (!d || scheme().fixed) return;
  const id = +d.dataset.id; isolate = isolate === id ? null : id; hover = -1; updateSide(); schedule(); });
$("gorder").addEventListener("change", renderGroup);
$("gshuffle").addEventListener("click", () => { gSeed++; $("gorder").value = "random"; renderGroup(); });
$("gallery").addEventListener("mouseover", e => { const f = e.target.closest("figure[data-i]"); hilite = f ? +f.dataset.i : -1; schedule(); });
$("gallery").addEventListener("mouseout", () => { hilite = -1; schedule(); });
$("gallery").addEventListener("click", e => { const f = e.target.closest("figure[data-i]"); if (f) { hilite = -1; togglePin(+f.dataset.i); } });
projsel.addEventListener("change", () => { setBounds(); resetView(); schedule(); });
for (const el of [spacesel, ksel, colorsel]) el.addEventListener("change", () => {
  isolate = null; const cur = pinned >= 0 ? pinned : hover; if (cur >= 0) show(cur); updateSide(); schedule(); });
$("size").addEventListener("input", schedule);
window.addEventListener("resize", () => { fitCanvas(); resetView(); schedule(); });

// ── resizable split between the map and the side panel (width remembered in localStorage) ──
(function () {
  const main = document.querySelector("main"), split = $("split"), KEY = "explorer.sidePanelWidth", DEF = 360;
  const clamp = w => Math.max(240, Math.min(w, main.clientWidth * 0.8));
  const setSide = w => main.style.setProperty("--side", clamp(w) + "px");
  try { const w = parseFloat(localStorage.getItem(KEY)); if (w) setSide(w); } catch (e) {}
  let on = false;
  split.addEventListener("pointerdown", e => { on = true; split.setPointerCapture(e.pointerId);
    split.classList.add("drag"); document.body.classList.add("resizing"); });
  split.addEventListener("pointermove", e => { if (!on) return;
    setSide(main.getBoundingClientRect().right - 16 - e.clientX - 6); });
  const stop = () => { if (!on) return; on = false; split.classList.remove("drag"); document.body.classList.remove("resizing");
    try { localStorage.setItem(KEY, parseFloat(getComputedStyle(main).getPropertyValue("--side")) || DEF); } catch (e) {} };
  split.addEventListener("pointerup", stop); split.addEventListener("pointercancel", stop);
  split.addEventListener("dblclick", () => { setSide(DEF); try { localStorage.removeItem(KEY); } catch (e) {} });
  if (window.ResizeObserver) new ResizeObserver(() => { fitCanvas(); schedule(); }).observe($("mapwrap"));
})();

const firstVisible = () => { for (let i = 0; i < N; i++) if (!hidden(i)) return i; return -1; };
function refreshExplicit() {      // the "show explicit images" tick changed: bring every view in line with it
  if (pinned >= 0 && hidden(pinned)) pinned = -1;
  if (hover >= 0 && hidden(hover)) hover = -1;
  if (qpt >= 0 && hidden(qpt)) qpt = -1;
  if (ptA >= 0 && hidden(ptA)) ptA = -1;
  if (ptB >= 0 && hidden(ptB)) ptB = -1;
  if (selList.length) setSelection(selList.filter(i => !hidden(i)));
  if (mode === "knn") computeKnn(); else if (mode === "interp") computeInterp();
  const cur = pinned >= 0 ? pinned : hover, f0 = firstVisible();
  if (cur >= 0) show(cur); else if (f0 >= 0 && !listView()) show(f0);
  updateSide(); schedule();
}
if (NSFW) {
  const nf = NSFW.reduce((a, v) => a + v, 0);
  const lab = () => { $("explcount").textContent = `(${nf.toLocaleString()} of ${N.toLocaleString()} flagged${showExplicit ? "" : ", hidden"})`; };
  $("explwrap").hidden = false; $("showexpl").checked = showExplicit; lab();
  $("showexpl").addEventListener("change", () => {
    showExplicit = $("showexpl").checked;
    try { localStorage.setItem(EXPL_KEY, showExplicit ? "1" : "0"); } catch (e) {}
    lab(); refreshExplicit(); });
  window.addEventListener("storage", e => {        // another tab (e.g. the overview page) changed it
    if (e.key !== EXPL_KEY) return;
    showExplicit = e.newValue === "1"; $("showexpl").checked = showExplicit; lab(); refreshExplicit(); });
}
setBounds(); fitCanvas(); resetView(); draw(); { const f0 = firstVisible(); if (f0 >= 0) show(f0); }
</script></body></html>
"""


# ── per-image data shared by all pages of a dataset (meta/<name>.js) ──────────
# Every model's pages of one dataset show the same images, so their labels, thumbnails, palettes, colour themes,
# attributes, ... are identical.  They are written once per dataset to meta/<prefix>-<content hash>.js and each page
# keeps only its sample indices ``idx`` into it.  The hash in the name means a page always loads exactly the data it
# was checked against, even after the meta of a dataset changes (e.g. new attributes).

META_KEYS = ("lab", "files", "thumbs", "pal_c", "pal_w", "light", "chroma", "dom_ref", "year", "nsfw", "theme")
_META_PREFIX = "window.EXPLORER_META="
_META_CACHE: dict[str, dict] = {}


def write_meta(out_dir: str | Path, prefix: str, fields: dict) -> str:
    """Write the shared per-image ``fields`` of ALL images of a dataset (row i = image i of the representation file,
    i.e. the page indices ``idx``) to ``out_dir``/meta; returns the meta name to pass to :func:`write_explorer`."""
    m = {k: fields[k] for k in META_KEYS if fields.get(k) is not None}
    m["attrs"] = {a["key"]: {x: a[x] for x in ("kind", "lo", "hi", "b64") if x in a}
                  for a in fields.get("attrs") or [] if "b64" in a}
    body = json.dumps(m, separators=(",", ":"))
    name = f"{prefix}-{hashlib.sha1(body.encode()).hexdigest()[:10]}"
    m.update(name=name, n=len(m["files"]))
    path = Path(out_dir) / "meta" / f"{name}.js"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_META_PREFIX + json.dumps(m, separators=(",", ":")) + ";\n")
    _META_CACHE[name] = m
    return name


def load_meta(out_dir: str | Path, name: str) -> dict:
    if name not in _META_CACHE:
        text = (Path(out_dir) / "meta" / f"{name}.js").read_text()
        _META_CACHE[name] = json.loads(text[len(_META_PREFIX):].rstrip().rstrip(";"))
    return _META_CACHE[name]


def _decode_attr(a: dict) -> np.ndarray:
    raw = base64.b64decode(a["b64"])
    if a["kind"] == "num":
        q = np.frombuffer(raw, np.uint16).astype(float)
        return np.where(q == 65535, np.nan, a["lo"] + q / 65534 * ((a["hi"] - a["lo"]) or 1.0))
    return np.frombuffer(raw, np.int32 if a["kind"] == "color" else np.int16)


def _attr_matches(a: dict, m: dict, idx: np.ndarray) -> bool:
    if a.get("kind") != m["kind"] or "b64" not in a:
        return False
    v, w = _decode_attr(a), _decode_attr(m)[idx]
    if len(v) != len(w):
        return False
    if a["kind"] != "num":
        return bool(np.array_equal(v, w))
    nan = np.isnan(v)        # both are uint16-quantized over their own [lo, hi]: allow one step of each
    tol = (((a["hi"] - a["lo"]) or 1.0) + ((m["hi"] - m["lo"]) or 1.0)) / 65534
    return bool(np.array_equal(nan, np.isnan(w)) and np.all(np.abs(v[~nan] - w[~nan]) <= tol * 1.001))


def split_meta(payload: dict, out_dir: str | Path, meta: str) -> dict:
    """``payload`` without the fields that equal meta ``meta`` at the page's ``idx`` (any other field stays inline)."""
    M = load_meta(out_dir, meta)
    idx = np.asarray(payload.get("idx", []), dtype=np.int64)
    if not len(idx) or idx.min() < 0 or idx.max() >= M["n"]:
        return payload
    out, moved = dict(payload), []
    for k in META_KEYS:
        v, mv = payload.get(k), M.get(k)
        if v is None or mv is None:
            continue
        if k == "theme":
            ok = set(v) == set(mv) and all(v[kk] == [mv[kk][i] for i in idx] for kk in v)
        else:
            ok = v == [mv[i] for i in idx]
        if ok:
            del out[k]
            moved.append(k)
    attrs, n_attr = [], 0
    for a in payload.get("attrs") or []:
        m = M["attrs"].get(a["key"])
        if m is not None and _attr_matches(a, m, idx):
            a = {k: v for k, v in a.items() if k not in ("b64", "lo", "hi")}
            a["meta"] = 1
            n_attr += 1
        attrs.append(a)
    if not moved and not n_attr:
        return payload
    if "attrs" in payload:
        out["attrs"] = attrs
    out.update(meta=meta, meta_keys=moved)
    return out


def hydrate(payload: dict, out_dir: str | Path) -> dict:
    """Inverse of :func:`split_meta` (used when re-rendering pages): the page's rows of its meta put back inline."""
    if not payload.get("meta"):
        return payload
    M = load_meta(out_dir, payload["meta"])
    idx = np.asarray(payload["idx"], dtype=np.int64)
    d = {k: v for k, v in payload.items() if k not in ("meta", "meta_keys")}
    for k in payload["meta_keys"]:
        d[k] = {kk: [v[i] for i in idx] for kk, v in M[k].items()} if k == "theme" else [M[k][i] for i in idx]
    attrs = []
    for a in payload.get("attrs") or []:
        if a.get("meta"):
            m = M["attrs"][a["key"]]
            raw = np.frombuffer(base64.b64decode(m["b64"]),
                                {"num": np.uint16, "color": np.int32}.get(m["kind"], np.int16))[idx]
            a = {k: v for k, v in a.items() if k != "meta"}
            a["b64"] = base64.b64encode(np.ascontiguousarray(raw).tobytes()).decode()
            if m["kind"] == "num":
                a.update(lo=m["lo"], hi=m["hi"])
        attrs.append(a)
    if "attrs" in payload:
        d["attrs"] = attrs
    return d


def write_explorer(path: str | Path, payload: dict, title: str, meta: str | None = None) -> Path:
    """Write the explorer page for ``payload`` (see scripts/08_build_explorer.py).  With ``meta`` (from
    :func:`write_meta`, in the page's directory) the per-image fields found there are left out of the page."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if "style_colors" not in payload and payload.get("classes"):
        from src.visualization.style_colors import style_colors, style_years
        payload = {**payload, "style_colors": style_colors(payload["classes"]), "style_years": style_years(payload["classes"])}
    if meta:
        payload = split_meta(payload, path.parent, meta)
    script = f'<script src="meta/{payload["meta"]}.js"></script>\n' if payload.get("meta") else ""
    html = HTML_TEMPLATE.replace("__TITLE__", title).replace("__META_SCRIPT__", script)
    path.write_text(html.replace("__DATA__", json.dumps(payload, separators=(",", ":"))))
    return path
