"""
The feature-map viewer: a static page (no dependencies) that shows, for a curated set of paintings, the internal
feature maps of the supervised ResNet-50, the ImageNet ResNet-50 and the autoencoder, layer by layer.
The data (images, sprites, statistics) is written by scripts/11_build_featuremaps.py next to the page.
"""

from __future__ import annotations

from pathlib import Path

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Feature-map viewer</title>
<style>
:root { --bg:#0e0f12; --panel:#17191e; --panel2:#1d2027; --line:#2a2e37; --fg:#eceef2; --muted:#8c93a1; --accent:#ffb454; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
header { display:flex; flex-wrap:wrap; gap:10px 18px; align-items:center; padding:10px 18px; border-bottom:1px solid var(--line); }
header h1 { font-size:15px; margin:0; } header a { color:var(--muted); text-decoration:none; } header a:hover { color:var(--fg); }
header .sub { color:var(--muted); font-size:12px; }
label { color:var(--muted); font-size:12px; }
select, button, input[type=range] { background:var(--panel); color:var(--fg); border:1px solid var(--line); border-radius:6px; padding:4px 8px; font:inherit; }
button { cursor:pointer; } button:hover { border-color:#666; }
button.on { background:var(--accent); color:#000; border-color:var(--accent); }
.layout { display:grid; grid-template-columns:300px minmax(0,1fr); gap:14px; padding:14px 18px; height:calc(100vh - 56px); }
aside { display:flex; flex-direction:column; gap:8px; min-height:0; }
.chips { display:flex; flex-wrap:wrap; gap:6px; }
#thumbs { overflow:auto; display:grid; grid-template-columns:repeat(3,1fr); gap:6px; align-content:start; min-height:0; flex:1; padding-right:4px; }
#thumbs img { width:100%; aspect-ratio:1; object-fit:cover; border-radius:6px; cursor:pointer; outline:2px solid transparent; background:#000; }
#thumbs img:hover { outline-color:#fff6; } #thumbs img.sel { outline-color:var(--accent); }
main { overflow:auto; min-width:0; padding-right:4px; }
.tabs { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px; }
.tabs button { padding:5px 12px; }
.row { display:grid; grid-template-columns:minmax(260px,320px) minmax(0,1fr) minmax(200px,260px); gap:14px; align-items:start; }
@media (max-width:1250px) { .row { grid-template-columns:1fr 1fr; } .layout { grid-template-columns:1fr; height:auto; } }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px; }
.card h2 { margin:0 0 6px; font-size:11px; font-weight:600; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }
.stack { position:relative; width:100%; aspect-ratio:1; }
.stack canvas, .stack img { position:absolute; inset:0; width:100%; height:100%; image-rendering:auto; border-radius:6px; }
#sprite { width:100%; image-rendering:pixelated; border-radius:6px; background:#000; cursor:crosshair; display:block; }
.meta { color:var(--muted); font-size:12px; margin-top:6px; }
.meta b { color:var(--fg); font-weight:600; }
#tip { position:fixed; pointer-events:none; background:#000d; border:1px solid var(--line); border-radius:6px; padding:5px 8px; font-size:12px;
       z-index:10; display:none; }
#zoom { width:100%; image-rendering:pixelated; border-radius:6px; background:#000; margin-top:6px; display:none; }
.ctl { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:8px; }
.note { color:var(--muted); font-size:12px; }
</style></head>
<body>
<header>
  <h1>Feature-map viewer</h1>
  <a href="../index.html">&larr; overview</a>
  <span class="sub" id="sub"></span>
  <label id="explwrap" hidden style="color:#ffb74d;font-size:12px;cursor:pointer" title="Paintings flagged as containing nudity are hidden unless this is ticked (shared with the overview and all explorer pages)"><input type="checkbox" id="showexpl"> show explicit images <span id="explcount" class="sub"></span></label>
  <span style="margin-left:auto" class="sub">&larr; &rarr; image · &uarr; &darr; layer</span>
</header>
<div class="layout">
  <aside>
    <div class="chips" id="dsbtns"></div>
    <div><label>art style </label><select id="stylesel"></select></div>
    <div id="thumbs"></div>
  </aside>
  <main>
    <div class="tabs" id="modeltabs"></div>
    <div class="tabs" id="layertabs"></div>
    <div class="row">
      <div>
        <div class="card"><h2>Image <span id="imgtitle" style="text-transform:none;letter-spacing:0"></span></h2>
          <div class="stack"><canvas id="inputcv" width="256" height="256"></canvas><canvas id="overlay" width="256" height="256"></canvas></div>
          <div class="ctl" style="margin-top:8px"><label><input type="checkbox" id="ovon" checked> overlay</label>
            <select id="ovsrc"><option value="mean">mean activation</option></select>
            <input id="ovalpha" type="range" min="0" max="100" value="55" style="width:90px"></div>
          <div class="meta" id="ovnote">The overlay shows where the selected layer is most active (mean |activation| over all channels).
            For the transformers (DINOv2, MAE, CLIP) the CLS-token attention can be shown instead.</div>
        </div>
        <div class="card" id="reconcard" style="margin-top:12px" hidden><h2>Autoencoder reconstruction</h2>
          <img id="reconimg" style="width:100%;border-radius:6px" alt=""></div>
      </div>
      <div class="card">
        <h2>Feature maps <span id="chinfo" style="text-transform:none;letter-spacing:0"></span></h2>
        <div class="ctl"><label>show</label><select id="spmode"><option value="pcs">principal components (top 16)</option>
            <option value="channels">channels (64 most varied)</option></select>
          <label>colour map</label><select id="cmap"></select>
          <label><input type="checkbox" id="grid"> tile borders</label></div>
        <canvas id="sprite" width="512" height="512"></canvas>
        <canvas id="zoom" width="256" height="256"></canvas>
        <div class="meta" id="chmeta">Hover a tile for details; click to enlarge. Principal components are the directions along which the layer's
          activations vary most across the image: much easier to read than single channels.</div>
      </div>
      <div>
        <div class="card"><h2>Principal components as RGB</h2>
          <img id="pcaimg" style="width:100%;border-radius:6px;image-rendering:pixelated;background:#000" alt="">
          <div class="meta">First three principal components of the activation vector at every pixel, one per colour channel:
            similar colours = similar features.</div></div>
        <div class="card" style="margin-top:12px"><h2>Layer</h2><div class="meta" id="stats"></div></div>
      </div>
    </div>
    <div class="card" id="headscard" style="margin-top:14px" hidden>
      <h2>Attention heads <span id="headsinfo" style="text-transform:none;letter-spacing:0"></span></h2>
      <canvas id="heads" width="768" height="256" style="width:100%;max-width:900px;border-radius:6px;background:#000;cursor:crosshair"></canvas>
      <div class="meta">Where the CLS token looks in each attention head of this block (bright = high attention), drawn over the image.
        Heads often specialise: some follow object outlines, others textures or the background.</div>
    </div>
    <p class="note" id="note"></p>
  </main>
</div>
<div id="tip"></div>
<script>
const $ = id => document.getElementById(id);
const DATA = "data/";
const CMAPS = {
  gray: [[0,0,0],[255,255,255]],
  viridis: [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]],
  inferno: [[0,0,4],[87,16,110],[188,55,84],[249,142,9],[252,255,164]],
  magma: [[0,0,4],[81,18,124],[183,55,121],[252,137,97],[252,253,191]],
  turbo: [[48,18,59],[70,130,250],[40,220,170],[200,240,50],[250,130,20],[122,4,3]],
};
function lut(stops) {
  const out = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) { const t = i / 255 * (stops.length - 1), k = Math.min(stops.length - 2, Math.floor(t)), f = t - k;
    for (let c = 0; c < 3; c++) out[i * 3 + c] = stops[k][c] + f * (stops[k + 1][c] - stops[k][c]); }
  return out;
}
const LUTS = Object.fromEntries(Object.entries(CMAPS).map(([k, v]) => [k, lut(v)]));
for (const k of Object.keys(CMAPS)) $("cmap").add(new Option(k, k));
$("cmap").value = "viridis";

let META = null, cur = { ds: "artbench", style: "all", id: null, model: "cnn_supervised", layer: "layer4" }, stats = null, spriteImg = null, meanImg = null, attnImg = null, pcsImg = null, rollImg = null, headsImg = null, inputImg = null;

// explicit images: flagged paintings (nsfw = 1 in index.json) are hidden unless the shared tick is set (same key as the overview page)
const EXPL_KEY = "artExplorer.showExplicit";
let showExplicit = false;
try { showExplicit = localStorage.getItem(EXPL_KEY) === "1"; } catch (e) {}
const isHidden = im => im.nsfw === 1 && !showExplicit;
function visible() { return META.images.filter(im => im.dataset === cur.ds && !isHidden(im) && (cur.style === "all" || im.style === cur.style)); }
function explicitChanged() {
  const nf = META.images.filter(i => i.nsfw === 1).length;
  $("explcount").textContent = `(${nf} of ${META.images.length} flagged${showExplicit ? "" : ", hidden"})`;
  if (cur.id && !visible().find(i => i.id === cur.id)) { cur.id = null; stats = null; }
  setDataset(cur.ds);
}
function save() { try { location.hash = `${cur.id}|${cur.model}|${cur.layer}`; } catch (e) {} }

function renderThumbs() {
  $("thumbs").innerHTML = visible().map(im =>
    `<img src="${DATA}${im.id}/input.jpg" data-id="${im.id}" title="${im.style}" loading="lazy" class="${im.id === cur.id ? "sel" : ""}">`).join("");
}
function renderTabs() {
  $("modeltabs").innerHTML = Object.entries(META.models).map(([k, m]) => `<button data-m="${k}" class="${k === cur.model ? "on" : ""}">${m.title}</button>`).join("");
  const layers = META.models[cur.model].layers;
  if (!layers.find(l => l.key === cur.layer)) cur.layer = layers[layers.length - 1].key;
  $("layertabs").innerHTML = layers.map(l => `<button data-l="${l.key}" class="${l.key === cur.layer ? "on" : ""}">${l.title}</button>`).join("");
}
function loadImg(src) { return new Promise(res => { const i = new Image(); i.onload = () => res(i); i.onerror = () => res(null); i.src = src; }); }

async function show() {
  if (!cur.id) return;
  const im = META.images.find(x => x.id === cur.id), base = `${DATA}${cur.id}/`, tag = `${cur.model}__${cur.layer}`;
  $("imgtitle").textContent = `· ${im.style} · ${im.file}`;
  $("reconcard").hidden = META.models[cur.model].kind !== "ae";
  if (META.models[cur.model].kind === "ae") $("reconimg").src = base + "recon.jpg";
  $("pcaimg").src = `${base}${tag}__pca.jpg`;
  if (!stats || stats.id !== cur.id) { try { stats = { id: cur.id, d: await (await fetch(base + "stats.json", { cache: "no-store" })).json() }; } catch (e) { stats = null; } }
  const st0 = stats && stats.d[`${cur.model}/${cur.layer}`], hasAttn = !!(st0 && st0.has_attn);
  const opt = (flag, name) => flag ? loadImg(`${base}${tag}__${name}`) : Promise.resolve(null);
  [spriteImg, meanImg, pcsImg, attnImg, rollImg, headsImg, inputImg] = await Promise.all([
    loadImg(`${base}${tag}__sprite.jpg`), loadImg(`${base}${tag}__mean.png`), loadImg(`${base}${tag}__pcs.jpg`),
    opt(hasAttn, "attn.png"), opt(st0 && st0.has_roll, "roll.png"), opt(st0 && st0.heads, "heads.png"), loadImg(base + "input.jpg")]);
  const prev = $("ovsrc").value;
  $("ovsrc").innerHTML = `<option value="mean">mean activation</option>` + (hasAttn ? `<option value="attn">CLS attention (mean of heads)</option>` : "") +
    (rollImg ? `<option value="roll">attention rollout (flow from the input)</option>` : "");
  const vit = META.models[cur.model].kind === "vit";
  $("ovsrc").value = [...$("ovsrc").options].some(o => o.value === prev) && prev !== "mean" ? prev : (vit && rollImg ? "roll" : vit && hasAttn ? "attn" : "mean");
  drawInput(); drawSprite(); drawOverlay(); drawHeads(); renderStats();
  renderTabs(); document.querySelectorAll("#thumbs img").forEach(i => i.classList.toggle("sel", i.dataset.id === cur.id));
  save();
}
function recolor(img, cmap, w, h, alphaFn) {
  const c = document.createElement("canvas"); c.width = w; c.height = h; const g = c.getContext("2d");
  g.imageSmoothingEnabled = true; g.drawImage(img, 0, 0, w, h);
  const d = g.getImageData(0, 0, w, h), L = LUTS[cmap];
  for (let i = 0; i < d.data.length; i += 4) { const v = d.data[i];
    d.data[i] = L[v * 3]; d.data[i + 1] = L[v * 3 + 1]; d.data[i + 2] = L[v * 3 + 2]; d.data[i + 3] = alphaFn ? alphaFn(v) : 255; }
  g.putImageData(d, 0, 0); return c;
}
function spriteState() {                                  // which sprite is shown, and its tile grid
  const pcs = $("spmode").value === "pcs" && pcsImg;
  return pcs ? { img: pcsImg, n: 4, pcs: true } : { img: spriteImg, n: META.grid, pcs: false };
}
function drawSprite() {
  const cv = $("sprite"), g = cv.getContext("2d"), sp = spriteState();
  if (!sp.img) { g.clearRect(0, 0, 512, 512); return; }
  g.imageSmoothingEnabled = false; g.drawImage(recolor(sp.img, $("cmap").value, 512, 512), 0, 0);
  if ($("grid").checked) { g.strokeStyle = "#000a"; g.lineWidth = 1; const t = 512 / sp.n;
    for (let i = 1; i < sp.n; i++) { g.beginPath(); g.moveTo(i * t, 0); g.lineTo(i * t, 512); g.moveTo(0, i * t); g.lineTo(512, i * t); g.stroke(); } }
  $("zoom").style.display = "none";
}
// each model is shown exactly the part of the painting it sees (frame = that region, as fractions of the 256 px image)
function frameOf() { return META.models[cur.model].frame || [0, 0, 1, 1]; }
function drawInput() {
  const g = $("inputcv").getContext("2d"), f = frameOf();
  g.clearRect(0, 0, 256, 256);
  if (inputImg) g.drawImage(inputImg, f[0] * 256, f[1] * 256, (f[2] - f[0]) * 256, (f[3] - f[1]) * 256, 0, 0, 256, 256);
}
function drawOverlay() {
  const cv = $("overlay"), g = cv.getContext("2d"); cv.width = 256; cv.height = 256; g.clearRect(0, 0, 256, 256);
  if (!$("ovon").checked || !meanImg) return;
  const v_ = $("ovsrc").value, a = $("ovalpha").value / 100,
        src = v_ === "attn" && attnImg ? attnImg : v_ === "roll" && rollImg ? rollImg : meanImg;
  g.drawImage(recolor(src, $("cmap").value === "gray" ? "inferno" : $("cmap").value, 256, 256,
    v => Math.round(255 * a * Math.min(1, v / 80 + .15))), 0, 0);       // the maps cover exactly the visible region
}
function drawHeads() {
  const card = $("headscard"), st = stats && stats.d[`${cur.model}/${cur.layer}`];
  if (!st || !st.heads || !headsImg || !inputImg) { card.hidden = true; return; }
  card.hidden = false;
  const nh = st.heads, cols = st.head_cols, rows = Math.ceil(nh / cols), T = 128, cv = $("heads"), g = cv.getContext("2d");
  cv.width = cols * T; cv.height = rows * T; $("headsinfo").textContent = `· ${nh} heads of ${META.models[cur.model].layers.find(l => l.key === cur.layer).title}`;
  const f = frameOf(), cm = $("cmap").value === "gray" ? "inferno" : $("cmap").value;
  const heat = recolor(headsImg, cm, headsImg.width, headsImg.height, v => Math.round(255 * Math.min(1, v / 90 + .08)));
  g.clearRect(0, 0, cv.width, cv.height);
  for (let h = 0; h < nh; h++) {
    const c = h % cols, r = Math.floor(h / cols), x0 = c * T, y0 = r * T;
    g.drawImage(inputImg, f[0] * 256, f[1] * 256, (f[2] - f[0]) * 256, (f[3] - f[1]) * 256, x0, y0, T, T);
    g.drawImage(heat, c * 64, r * 64, 64, 64, x0, y0, T, T);
    g.fillStyle = "#000b"; g.fillRect(x0, y0, 34, 15); g.fillStyle = "#fff"; g.font = "11px system-ui"; g.fillText("head " + h, x0 + 3, y0 + 11);
  }
}
function renderStats() {
  const s = stats && stats.d[`${cur.model}/${cur.layer}`];
  if (!s) { $("stats").textContent = ""; $("chinfo").textContent = ""; return; }
  const [C, H, W] = s.shape;
  const pc = $("spmode").value === "pcs" && pcsImg;
  $("chinfo").textContent = pc ? `· first 16 principal components of ${C} channels` : `· top ${Math.min(META.grid * META.grid, C)} of ${C} channels by variance`;
  $("stats").innerHTML = `<b>${C}</b> channels of <b>${H}&times;${W}</b><br>zero activations: <b>${(100 * s.sparsity).toFixed(0)}%</b><br>` +
    `mean <b>${s.mean.toFixed(3)}</b> · std <b>${s.std.toFixed(3)}</b>`;
}
function tileAt(e) {
  const sp = spriteState(), r = $("sprite").getBoundingClientRect(), t = r.width / sp.n;
  const cx = Math.floor((e.clientX - r.left) / t), cy = Math.floor((e.clientY - r.top) / t), k = cy * sp.n + cx;
  const s = stats && stats.d[`${cur.model}/${cur.layer}`];
  return { sp, cx, cy, k, s, ok: s && cx >= 0 && cy >= 0 && cx < sp.n && cy < sp.n &&
    k < (sp.pcs ? (s.pc_var || []).length : s.channels.length) };
}
function tileText(t) {
  return t.sp.pcs ? `component <b>${t.k + 1}</b> · explains ${(100 * t.s.pc_var[t.k]).toFixed(1)}% of the layer's variance`
                  : `channel <b>${t.s.channels[t.k]}</b> · mean ${t.s.chmean[t.k].toFixed(3)} · rank ${t.k + 1} by variance`;
}
$("sprite").addEventListener("mousemove", e => {
  const t = tileAt(e), tip = $("tip");
  if (!t.ok) { tip.style.display = "none"; return; }
  tip.style.display = "block"; tip.style.left = e.clientX + 14 + "px"; tip.style.top = e.clientY + 14 + "px"; tip.innerHTML = tileText(t);
});
$("sprite").addEventListener("mouseleave", () => { $("tip").style.display = "none"; });
$("sprite").addEventListener("click", e => {
  const t = tileAt(e); if (!t.ok) return;
  const z = $("zoom"), g = z.getContext("2d"), T = 512 / t.sp.n;
  z.style.display = "block"; g.imageSmoothingEnabled = false; g.drawImage($("sprite"), t.cx * T, t.cy * T, T, T, 0, 0, 256, 256);
  $("chmeta").innerHTML = tileText(t);
});
$("heads").addEventListener("mousemove", e => {
  const st = stats && stats.d[`${cur.model}/${cur.layer}`]; if (!st || !st.heads) return;
  const r = $("heads").getBoundingClientRect(), tw = r.width / st.head_cols, th = r.height / Math.ceil(st.heads / st.head_cols);
  const k = Math.floor((e.clientY - r.top) / th) * st.head_cols + Math.floor((e.clientX - r.left) / tw), tip = $("tip");
  if (k < 0 || k >= st.heads) { tip.style.display = "none"; return; }
  tip.style.display = "block"; tip.style.left = e.clientX + 14 + "px"; tip.style.top = e.clientY + 14 + "px"; tip.innerHTML = `attention head <b>${k}</b>`;
});
$("heads").addEventListener("mouseleave", () => { $("tip").style.display = "none"; });

function setDataset(ds) {
  cur.ds = ds; cur.style = "all";
  const styles = [...new Set(META.images.filter(i => i.dataset === ds).map(i => i.style))];
  $("stylesel").innerHTML = `<option value="all">all styles (${META.images.filter(i => i.dataset === ds).length})</option>` + styles.map(s => `<option>${s}</option>`).join("");
  $("dsbtns").innerHTML = ["artbench", "wikiart"].filter(d => META.images.some(i => i.dataset === d)).map(d =>
    `<button data-ds="${d}" class="${d === ds ? "on" : ""}">${d === "artbench" ? "ArtBench-10" : "WikiArt"}</button>`).join("");
  renderThumbs();
  const v = visible(); if (!v.find(i => i.id === cur.id)) { cur.id = v[0] && v[0].id; stats = null; }
  show();
}
$("dsbtns").addEventListener("click", e => { const b = e.target.closest("button[data-ds]"); if (b) setDataset(b.dataset.ds); });
$("stylesel").addEventListener("change", () => { cur.style = $("stylesel").value; renderThumbs(); const v = visible(); if (v.length && !v.find(i => i.id === cur.id)) { cur.id = v[0].id; stats = null; show(); } });
$("thumbs").addEventListener("click", e => { const i = e.target.closest("img[data-id]"); if (i) { cur.id = i.dataset.id; stats = null; show(); } });
$("modeltabs").addEventListener("click", e => { const b = e.target.closest("button[data-m]"); if (b) { cur.model = b.dataset.m; renderTabs(); show(); } });
$("layertabs").addEventListener("click", e => { const b = e.target.closest("button[data-l]"); if (b) { cur.layer = b.dataset.l; show(); } });
for (const id of ["cmap", "grid"]) $(id).addEventListener("change", () => { drawSprite(); drawOverlay(); drawHeads(); });
$("spmode").addEventListener("change", () => { drawSprite(); renderStats(); });
$("ovon").addEventListener("change", drawOverlay); $("ovsrc").addEventListener("change", drawOverlay); $("ovalpha").addEventListener("input", drawOverlay);
window.addEventListener("keydown", e => {
  if (/^(INPUT|SELECT)$/.test(e.target.tagName)) return;
  const v = visible(), i = v.findIndex(x => x.id === cur.id), layers = META.models[cur.model].layers, li = layers.findIndex(l => l.key === cur.layer);
  if (e.key === "ArrowRight" && i < v.length - 1) { cur.id = v[i + 1].id; stats = null; show(); }
  else if (e.key === "ArrowLeft" && i > 0) { cur.id = v[i - 1].id; stats = null; show(); }
  else if (e.key === "ArrowDown" && li < layers.length - 1) { cur.layer = layers[li + 1].key; show(); }
  else if (e.key === "ArrowUp" && li > 0) { cur.layer = layers[li - 1].key; show(); }
  else return;
  e.preventDefault();
});

fetch(DATA + "index.json", { cache: "no-store" }).then(r => r.json()).then(m => {
  META = m;
  $("sub").textContent = `${m.images.length} held-out paintings · ${Object.keys(m.models).length} models`;
  $("note").textContent = (m.note || "") + ". Sprites show the most varied channels of each layer; their grey values are normalised per channel. For the transformers the channels are the 768 embedding dimensions of the patch tokens laid out on their patch grid (16x16 for DINOv2, 14x14 for the MAEs, 7x7 for CLIP). Each model is shown exactly the part of the painting it sees: DINOv2 and the MAEs the central 224 of 256 px, the others the whole square.";
  const h = (location.hash || "").slice(1).split("|"), im = h[0] && m.images.find(i => i.id === h[0]);
  if (im && !isHidden(im)) { cur.id = im.id; cur.ds = im.dataset; if (m.models[h[1]]) cur.model = h[1]; if (h[2]) cur.layer = h[2]; }
  if (m.images.some(i => i.nsfw === 1)) {
    $("explwrap").hidden = false; $("showexpl").checked = showExplicit;
    $("showexpl").addEventListener("change", () => {
      showExplicit = $("showexpl").checked;
      try { localStorage.setItem(EXPL_KEY, showExplicit ? "1" : "0"); } catch (e) {}
      explicitChanged(); });
    window.addEventListener("storage", e => { if (e.key === EXPL_KEY) { showExplicit = e.newValue === "1"; $("showexpl").checked = showExplicit; explicitChanged(); } });
    $("explcount").textContent = `(${m.images.filter(i => i.nsfw === 1).length} of ${m.images.length} flagged${showExplicit ? "" : ", hidden"})`;
  }
  setDataset(cur.ds);
}).catch(() => { $("sub").textContent = "no data yet: run scripts/11_build_featuremaps.py"; });
</script></body></html>
"""


def write_viewer(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HTML)
    return path
