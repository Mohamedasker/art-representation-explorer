// Shared state of the lecture page: the fish, the number of features in use (2 or 3), derived quantities per feature
// count, colour schemes, cached t-SNE / UMAP runs and the tab switching.

let SET = "continuum", FISH = DATA.sets[SET].fish;
const N = FISH.length;                      // both populations have the same number of fish
const GROUP = () => DATA.sets[SET].group;   // 0 / 1 per fish in the cluster population, else null
const defaultColor = () => GROUP() ? "group" : "length";
const FEAT = ["length", "thickness", "tail length"], FEAT_SHORT = ["L", "T", "F"];
let DIM = 2;
const SAMPLE9 = { ids: null };                  // the nine fish of the PCA tab, shown again in the result step of t-SNE / UMAP

const SPACES = {};
const tercile = v => {                         // 0 / 1 / 2 by rank
  const o = [...v.keys()].sort((a, b) => v[a] - v[b]), c = new Int8Array(v.length);
  o.forEach((i, r) => { c[i] = Math.min(2, Math.floor(3 * r / v.length)); });
  return c;
};
function space(d = DIM) {
  const key = `${SET}${d}`; if (SPACES[key]) return SPACES[key];
  const X = Float64Array.from(FISH.flatMap(r => r.slice(0, d))), pca = pcaSteps(X, N, d), D2 = sqDistMatrix(X, N, d);
  const sd = Array.from({ length: d }, (_, k) => Math.sqrt(pca.C[k * d + k]));
  let sizeCls, shapeCls;
  if (GROUP()) { sizeCls = tercile(FISH.map(r => r[0])); shapeCls = tercile(FISH.map(r => r[1] / r[0])); }      // PC1 is no longer simply "size": use the measurements
  else {
    const flipShape = pca.W[1 * d + 1] < 0; sizeCls = tercile(Array.from({ length: N }, (_, i) => pca.Z[i * d]));
    shapeCls = tercile(Array.from({ length: N }, (_, i) => (flipShape ? -1 : 1) * pca.Z[i * d + 1]));
  }
  const tailCls = tercile(FISH.map(r => r[2]));
  const Xs = Float64Array.from(X, (v, i) => (v - pca.mu[i % d]) / sd[i % d]);          // standardised features (unit variance)
  return SPACES[key] = { d, X, pca, D2, sd, Xs, D2s: sqDistMatrix(Xs, N, d), sizeCls, shapeCls, tailCls, order: [...Array(N).keys()].sort((a, b) => X[a * d] - X[b * d]) };
}
const sp = () => space(DIM);
const tailOf = (i, d = DIM) => d === 3 || GROUP() ? FISH[i][2] : 0.3 * FISH[i][0];       // in the 2-number version of the first population the tail is just part of the drawing

// ── colour schemes ──────────────────────────────────────────────────────────────────────────────────────────────

const CLS_COL = ["#b39ddb", "#4db6ac", "#f06292"];
function colorSchemes(d = DIM) {
  const S = space(d), out = [];
  const cont = (key, label, k) => {
    const v = FISH.map(r => r[k]), lo = Math.min(...v), hi = Math.max(...v), nb = 6, span = hi - lo || 1, bin = x => Math.min(nb - 1, Math.floor((x - lo) / span * nb));
    const cnt = Array(nb).fill(0); v.forEach(x => cnt[bin(x)]++);
    out.push({ key, label, id: i => bin(v[i]), color: i => viridis((v[i] - lo) / span), value: i => v[i], lo, hi,
      legend: [...Array(nb).keys()].reverse().map(b => [viridis((b + .5) / nb), `${fmt(lo + span * b / nb, 0)} – ${fmt(lo + span * (b + 1) / nb, 0)} cm  (n=${cnt[b]})`, b]) });
  };
  const cat = (key, label, cls, names) => {
    const cnt = [0, 0, 0]; cls.forEach(c => cnt[c]++);
    out.push({ key, label, id: i => cls[i], color: i => CLS_COL[cls[i]], names, legend: [2, 1, 0].map(c => [CLS_COL[c], `${names[c]}  (n=${cnt[c]})`, c]) });
  };
  cont("length", "length", 0); cont("thickness", "thickness", 1); if (d === 3) cont("tail", "tail length", 2);
  if (GROUP()) {
    const g = GROUP(), cols = DATA.sets[SET].group_colors, names = DATA.sets[SET].group_names, cnt = names.map(() => 0); g.forEach(x => cnt[x]++);
    out.unshift({ key: "group", label: "population (ground truth)", id: i => g[i], color: i => cols[g[i]], names, legend: names.map((_, c) => [cols[c], `${names[c]}  (n=${cnt[c]})`, c]).reverse() });
  }
  cat("size", GROUP() ? "size class (terciles of length)" : "size class (terciles of PC1)", S.sizeCls, ["small", "medium", "large"]);
  cat("shape", GROUP() ? "shape class (terciles of thickness / length)" : "shape class (terciles of PC2)", S.shapeCls, ["slender", "average", "stocky"]);
  if (d === 3) cat("tailcls", "tail class", S.tailCls, ["short tail", "medium tail", "long tail"]);
  return out;
}

// ── cached runs of t-SNE and UMAP ────────────────────────────────────────────────────────────────────────────────

const TSNE_DEFAULT = { perplexity: 30, lr: 10, exag: 12, init: "pca", seed: 1, iters: 1000 };
const UMAP_DEFAULT = { nNeighbors: 15, minDist: 0.1, init: "spectral", seed: 1, epochs: 500 };
const RUNS = new Map(), KEEP = new Set();
const tsneKey = (d, p) => `${SET}|tsne|${d}|${JSON.stringify(p)}`, umapKey = (d, p) => `${SET}|umap|${d}|${JSON.stringify(p)}`;
function cached(key, fn, keep = false) {            // `keep`: the default runs (shown in the explore tab) are never evicted
  if (!RUNS.has(key)) {
    const v = fn(); RUNS.set(key, v); if (keep) KEEP.add(key);
    if (RUNS.size > 14) for (const k of RUNS.keys()) if (!KEEP.has(k)) { RUNS.delete(k); break; }
  }
  return RUNS.get(key);
}
const NN_K = 10;                                // neighbourhood size of the "k-NN preserved" curves
function preservationCurve(frames, stride, d) {
  const S = space(d), inNN = knnSets(S.D2, N, NN_K), t = [], v = [];
  for (let f = 0; f < frames.length; f += stride) { t.push(f); v.push(knnPreservation(inNN, frames[f], N, NN_K)); }
  if (t[t.length - 1] !== frames.length - 1) { t.push(frames.length - 1); v.push(knnPreservation(inNN, frames[frames.length - 1], N, NN_K)); }
  return { t, v };
}
function getTsne(d, p = TSNE_DEFAULT) {
  return cached(tsneKey(d, p), () => {
    const S = space(d), aff = cached(`${SET}|aff|${d}|${p.perplexity}`, () => tsneAffinities(S.D2, N, p.perplexity));
    const Y0 = tsneInit(p.init, N, p.seed, S.X, d), run = tsneRun(aff.P, Y0, N, { lr: p.lr, exag: p.exag, iters: p.iters });
    return { p, aff, Y0, frames: run.frames, kl: run.kl, pres: preservationCurve(run.frames, 10, d), T: p.iters, exagIters: 250 };
  }, p === TSNE_DEFAULT);
}
function getUmap(d, p = UMAP_DEFAULT) {
  return cached(umapKey(d, p), () => {
    const S = space(d), graph = cached(`${SET}|graph|${d}|${p.nNeighbors}`, () => umapGraph(S.D2, N, p.nNeighbors)), { a, b } = umapFitAB(p.minDist);
    let Y0;
    if (p.init === "random") Y0 = umapRandomInit(N, p.seed);
    else if (p.init === "pca") { Y0 = new Float64Array(2 * N); let m = 0; for (let i = 0; i < N; i++) { Y0[2 * i] = S.pca.Z[i * d]; Y0[2 * i + 1] = S.pca.Z[i * d + 1]; m = Math.max(m, Math.abs(Y0[2 * i]), Math.abs(Y0[2 * i + 1])); }
      for (let k = 0; k < 2 * N; k++) Y0[k] = Y0[k] / m * 10; }
    else Y0 = umapSpectralInit(graph.W, N, p.seed);
    const run = umapRun(graph.W, Y0, N, a, b, { epochs: p.epochs, seed: p.seed + 100 });
    return { p, graph, a, b, Y0, frames: run.frames, pres: preservationCurve(run.frames, 5, d), T: p.epochs };
  }, p === UMAP_DEFAULT);
}
// run `fn` after the browser had a chance to paint a "computing" message
function later(fn) { setTimeout(fn, 30); }

// ── tooltip with a fish (step-through tabs) ───────────────────────────────────────────────────────────────────────

function fishTip(i, x, y, extra = "") {
  const t = $("gtip");
  if (i < 0) { t.hidden = true; return; }
  const c = t.querySelector("canvas"), r = devicePixelRatio || 1, w = 120, h = 64; c.width = w * r; c.height = h * r; c.style.width = w + "px"; c.style.height = h + "px";
  const g = c.getContext("2d"); g.scale(r, r); const k = FISH_K(w, h), F = tailOf(i);
  drawFish(g, w / 2 + F * k / 2, h / 2, FISH[i][0], FISH[i][1], F, k, "#9ad");
  t.querySelector("span").innerHTML = `<b>fish ${i + 1}</b> &middot; L ${fmt(FISH[i][0])} &middot; T ${fmt(FISH[i][1])}${DIM === 3 ? ` &middot; F ${fmt(FISH[i][2])}` : ""}${extra}`;
  t.hidden = false; t.style.left = Math.min(x + 14, innerWidth - 150) + "px"; t.style.top = Math.min(y + 14, innerHeight - 110) + "px";
}

// ── tabs ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const TABS = {};
let CUR = "explore";
function showTab(name) {
  CUR = name;
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === name));
  for (const k of Object.keys(TABS)) $("tab-" + k).hidden = k !== name;
  fishTip(-1); TABS[name].show();
  try { history.replaceState(null, "", "#" + name); } catch (e) {}
}
function setDataset(k) {
  SET = k; FISH = DATA.sets[k].fish; SAMPLE9.ids = null; $("setsel").value = k;
  for (const t of Object.keys(TABS)) TABS[t].onDim("set");
  TABS[CUR].show();
}
function setDim(d) {
  DIM = d;
  document.querySelectorAll("[data-dim]").forEach(b => b.classList.toggle("on", +b.dataset.dim === d));
  $("dimdesc").textContent = d === 2 ? "each fish = (length, thickness) in cm: an ellipse" : "each fish = (length, thickness, tail length) in cm";
  for (const k of Object.keys(TABS)) TABS[k].onDim();
  TABS[CUR].show();
}
function initApp() {
  TABS.explore = makeExplore(); TABS.pca = makePca($("tab-pca")); TABS.tsne = makeEmbed($("tab-tsne"), "tsne"); TABS.umap = makeEmbed($("tab-umap"), "umap");
  document.querySelectorAll("#tabs button").forEach(b => b.addEventListener("click", () => showTab(b.dataset.tab)));
  document.querySelectorAll("[data-dim]").forEach(b => b.addEventListener("click", () => setDim(+b.dataset.dim)));
  $("setsel").innerHTML = Object.entries(DATA.sets).map(([k, s]) => `<option value="${k}">${s.label}</option>`).join("");
  $("setsel").addEventListener("change", () => setDataset($("setsel").value));
  window.addEventListener("resize", () => TABS[CUR].show());
  const want = (location.hash || "").slice(1);
  setDim(2); showTab(want in TABS ? want : "explore");
}
window.addEventListener("DOMContentLoaded", initApp);
