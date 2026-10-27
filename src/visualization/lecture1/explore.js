// "Explore" tab: the fish as a scatter plot, with the same interactions as the embedding explorer (hover, pin, colour by,
// legend isolation, nearest neighbours, interpolation A -> B) and the same projections (feature plane, PCA, t-SNE, UMAP).

function makeExplore() {
  const map = $("map"), plot = new Plot(map, 2), g = map.getContext("2d");
  let projKey = "f01", schemeKey = "length", mode = "explore", hover = -1, pinned = -1, isolate = null, qpt = -1, ptA = -1, ptB = -1, ov = null;
  let dirty = true, raf = 0, PR = [], SC = [], scr = new Float64Array(3 * N);

  // ── projections of the current fish set ──
  function projections(d) {
    const S = space(d), W = S.pca.W, ev = S.pca.ratio, out = [];
    const pair = (a, b) => ({ key: `f${a}${b}`, name: `Features: ${FEAT[a]} vs ${FEAT[b]}`, dim: 2, labels: [`${FEAT[a]} (cm)`, `${FEAT[b]} (cm)`], kind: "feat",
      get: () => Float64Array.from({ length: 2 * N }, (_, k) => FISH[k >> 1][k & 1 ? b : a]), lin: v => [v[a], v[b]], axes: [a, b] });
    const pcs = (a, b) => ({ key: `pc${a}${b}`, name: `PCA: PC${a + 1} vs PC${b + 1} (${fmt(ev[a] * 100, 0)}% / ${fmt(ev[b] * 100, 0)}%)`, dim: 2, kind: "pc", pc: [a, b],
      labels: [`PC${a + 1}`, `PC${b + 1}`], get: () => Float64Array.from({ length: 2 * N }, (_, k) => S.pca.Z[(k >> 1) * d + (k & 1 ? b : a)]),
      lin: v => [a, b].map(j => v.reduce((s, x, k) => s + (x - S.pca.mu[k]) * W[k * d + j], 0)) });
    if (d === 3) out.push({ key: "f3d", name: "Features: all three (3-D, drag to rotate)", dim: 3, kind: "feat3", labels: FEAT.map(f => f + " (cm)"), get: () => S.X, lin: v => v });
    out.push(pair(0, 1)); if (d === 3) out.push(pair(0, 2), pair(1, 2));
    out.push(pcs(0, 1)); if (d === 3) out.push(pcs(0, 2), pcs(1, 2));
    out.push({ key: "tsne", name: "t-SNE", dim: 2, kind: "map", algo: "tsne", labels: ["t-SNE 1", "t-SNE 2"] });
    out.push({ key: "umap", name: "UMAP", dim: 2, kind: "map", algo: "umap", labels: ["UMAP 1", "UMAP 2"] });
    return out;
  }
  const proj = () => PR.find(p => p.key === projKey) || PR[0];
  const scheme = () => SC.find(s => s.key === schemeKey) || SC[0];
  function coords(p) {                       // flat (N x dim) array of the projection, or null while a map is still being computed
    if (p.algo) {
      const key = p.algo === "tsne" ? tsneKey(DIM, TSNE_DEFAULT) : umapKey(DIM, UMAP_DEFAULT);
      if (!RUNS.has(key)) { $("busy").hidden = false; later(() => { (p.algo === "tsne" ? getTsne : getUmap)(DIM); $("busy").hidden = true; setBounds(); about(); schedule(); }); return null; }
      const r = RUNS.get(key); return r.frames[r.frames.length - 1];
    }
    return p.xyc || (p.xyc = p.get());
  }
  function setBounds() {
    const p = proj(), xy = coords(p); if (!xy) return;
    const r = rangeOf(xy, N, p.dim, { margin: 0.07 }); plot.dim = p.dim; plot.fit(r.lo, r.hi); plot.reset();
  }
  const vec = (xy, i, dim) => dim === 2 ? [xy[2 * i], xy[2 * i + 1]] : [xy[3 * i], xy[3 * i + 1], xy[3 * i + 2]];

  // ── controls ──
  function rebuild() {
    PR = projections(DIM); SC = colorSchemes(DIM);
    const ps = $("projsel"), cs = $("colorsel");
    ps.innerHTML = PR.map(p => `<option value="${p.key}">${p.name}</option>`).join(""); cs.innerHTML = SC.map(s => `<option value="${s.key}">${s.label}</option>`).join("");
    if (!PR.some(p => p.key === projKey)) projKey = PR[0].key; if (!SC.some(s => s.key === schemeKey)) schemeKey = "length";
    ps.value = projKey; cs.value = schemeKey; isolate = null; ov = null; hover = pinned = -1; dirty = false;
    setBounds(); applyMode(true);
  }

  // ── nearest neighbours / interpolation ──
  const metric = () => $("metricsel").value === "std" ? { D2: sp().D2s, unit: "σ", X: sp().Xs } : { D2: sp().D2, unit: " cm", X: sp().X };
  function computeKnn() {
    const K = +$("kk").value; $("kkv").textContent = K;
    if (qpt < 0) { ov = null; return; }
    const m = metric(), ids = neighbourOrder(m.D2, N, qpt).slice(0, K), d = ids.map(j => Math.sqrt(m.D2[qpt * N + j]));
    const p = proj(), xy = coords(p);
    let kept = null;
    if (xy) {                                  // how many of the K neighbours are also among the K nearest points ON THIS MAP
      const D = sqDistMatrix(xy, N, p.dim), onmap = new Set(neighbourOrder(D, N, qpt).slice(0, K)); kept = ids.filter(j => onmap.has(j)).length;
    }
    ov = { kind: "knn", q: qpt, ids, dist: d, unit: m.unit };
    $("modetitle").textContent = "Nearest neighbours";
    $("modehelp").innerHTML = `Query fish ${qpt + 1}: its ${K} nearest fish by Euclidean distance in the original ${DIM}-number description (${$("metricsel").value === "std" ? "each feature divided by its standard deviation" : "centimetres"}). ` +
      (kept === null ? "" : `<b>${kept} of ${K}</b> of them are also among its ${K} nearest points on the map shown. `) + `Click a fish to re-query.`;
    $("strip").innerHTML = figureHTML(qpt, "query", MC, MC) + ids.map((j, r) => figureHTML(j, `${fmt(d[r], 1)}${m.unit}`, "#fff", null, r + 1)).join("");
    $("strip2").innerHTML = ""; fillFigures();
  }
  function computeInterp() {
    const S = +$("steps").value; $("stepsv").textContent = S;
    if (ptA < 0) { ov = null; return; }
    $("modetitle").textContent = "Interpolation";
    if (ptB < 0) { ov = { kind: "interp", A: ptA, B: -1, path: [], ideal: [] }; $("modehelp").textContent = "A is set (blue). Click a second fish for B."; $("strip").innerHTML = figureHTML(ptA, "A", AC, AC); $("strip2").innerHTML = ""; fillFigures(); return; }
    const d = DIM, a = FISH[ptA].slice(0, d), b = FISH[ptB].slice(0, d), used = new Set([ptA, ptB]), path = [ptA], ideal = [a];
    for (let s = 1; s < S - 1; s++) {
      const t = s / (S - 1), v = a.map((x, k) => (1 - t) * x + t * b[k]); let best = -1, bd = Infinity;
      for (let i = 0; i < N; i++) { if (used.has(i)) continue; let dist = 0; for (let k = 0; k < d; k++) dist += (FISH[i][k] - v[k]) ** 2; if (dist < bd) { bd = dist; best = i; } }
      used.add(best); path.push(best); ideal.push(v);
    }
    path.push(ptB); ideal.push(b);
    ov = { kind: "interp", A: ptA, B: ptB, path, ideal };
    $("modehelp").innerHTML = `Straight line from <b style="color:${AC}">A</b> to <b style="color:${BC}">B</b> in the ${d}-number description, ${S} evenly spaced points. ` +
      `<b>Top row:</b> the fish with exactly the interpolated numbers (it need not exist). <b>Bottom row:</b> the nearest fish of the data set. Click two new fish to start over.`;
    $("strip").innerHTML = `<div class="rowlabel">interpolated values (new fish)</div>` + ideal.map((v, s) => figureHTML(null, `t=${fmt(s / (S - 1), 2)}`, s === 0 ? AC : s === S - 1 ? BC : "#fff", null, s === 0 ? "A" : s === S - 1 ? "B" : s, v)).join("");
    $("strip2").innerHTML = `<div class="rowlabel">nearest real fish</div>` + path.map((j, s) => figureHTML(j, s === 0 || s === S - 1 ? `fish ${j + 1}` : `fish ${j + 1}`, s === 0 ? AC : s === S - 1 ? BC : "#fff", null, s === 0 ? "A" : s === S - 1 ? "B" : s)).join("");
    fillFigures();
  }
  // figures are placeholders filled with canvases afterwards (the strip is built from HTML strings)
  function figureHTML(i, caption, tagColor, outline, tagText, vals) {
    const tag = tagText !== undefined ? `<span class="tag" style="background:${tagColor}">${tagText}</span>` : (outline ? `<span class="tag" style="background:${tagColor}">Q</span>` : "");
    return `<figure ${i !== null ? `data-i="${i}"` : ""} data-v="${vals ? vals.join(",") : ""}" style="${outline ? "outline-color:" + outline : ""}"><span class="fc"></span>${tag}<figcaption>${caption}</figcaption></figure>`;
  }
  function fillFigures() {
    const sc = scheme();
    document.querySelectorAll("#modecard figure").forEach(f => {
      const i = f.dataset.i !== undefined ? +f.dataset.i : -1, vs = f.dataset.v ? f.dataset.v.split(",").map(Number) : null;
      const L = vs ? vs[0] : FISH[i][0], T = vs ? vs[1] : FISH[i][1], F = vs ? (DIM === 3 ? vs[2] : 0.3 * vs[0]) : tailOf(i);
      f.querySelector(".fc").replaceWith(fishCanvas(L, T, F, vs ? "#cfd8dc" : sc.color(i), 120, 70));
    });
  }
  function applyMode(initial) {
    mode = $("modesel").value; ov = null; pinned = -1;
    $("knnctl").hidden = mode !== "knn"; $("interpctl").hidden = mode !== "interp"; $("modecard").hidden = mode === "explore"; $("samplecard").hidden = false;
    const base = proj().dim === 3 ? " · drag: rotate · wheel: zoom · dbl-click: reset" : " · wheel: zoom · drag: pan · dbl-click: reset";
    $("hint").textContent = (mode === "knn" ? "click a fish: show its nearest neighbours" : mode === "interp" ? "click fish A, then fish B" : "hover: inspect · click: pin · legend: isolate") + base;
    if (mode === "knn") { $("modetitle").textContent = "Nearest neighbours"; $("modehelp").textContent = "Click a fish on the map."; $("strip").innerHTML = ""; $("strip2").innerHTML = ""; computeKnn(); }
    if (mode === "interp") { ptA = ptB = -1; $("modetitle").textContent = "Interpolation"; $("modehelp").textContent = "Click fish A, then fish B."; $("strip").innerHTML = ""; $("strip2").innerHTML = ""; }
    about(); schedule();
  }
  function modeClick(i) {
    if (i < 0) return;
    if (mode === "knn") { qpt = i; computeKnn(); showFish(i); }
    else if (mode === "interp") { if (ptA < 0 || ptB >= 0) { ptA = i; ptB = -1; } else if (i !== ptA) ptB = i; else return; computeInterp(); showFish(i); }
    schedule();
  }

  // ── side panel ──
  function showFish(i) {
    $("pin").textContent = pinned >= 0 ? "· pinned (click to release)" : "";
    if (i < 0) return;
    const S = sp(), sc = scheme(), c = document.createElement("canvas"), r = devicePixelRatio || 1, w = 340, h = 150;
    c.width = w * r; c.height = h * r; c.style.height = h + "px"; const cg = c.getContext("2d"); cg.scale(r, r);
    const F = tailOf(i), k = FISH_K(w, h) * 0.95; drawFish(cg, w / 2 + F * k / 2, h / 2, FISH[i][0], FISH[i][1], F, k, sc.color(i), 1, "#ffffff55");
    cg.fillStyle = "#666"; cg.font = "10px system-ui"; cg.fillText("drawn to scale: 1 cm = " + fmt(k, 1) + " px", 6, h - 6);
    $("bigfish").replaceChildren(c);
    const p = proj(), xy = coords(p), z = S.pca.Z, rows = [["fish", `${i + 1} of ${N}`], ["length", `${fmt(FISH[i][0])} cm`], ["thickness", `${fmt(FISH[i][1])} cm`]];
    rows.push(DIM === 3 ? ["tail length", `${fmt(FISH[i][2])} cm`] : ["tail", "drawn, not measured"]);
    rows.push(["thickness / length", fmt(FISH[i][1] / FISH[i][0], 2)], ["size class", ["small", "medium", "large"][S.sizeCls[i]]], ["shape class", ["slender", "average", "stocky"][S.shapeCls[i]]],
      ["PC1 · PC2 score", `${fmt(z[i * DIM], 1)} · ${fmt(z[i * DIM + 1], 1)}`]);
    if (xy && p.kind === "map") rows.push(["map position", Array.from({ length: p.dim }, (_, k) => fmt(xy[p.dim * i + k], 1)).join(", ")]);
    $("info").innerHTML = rows.map(([a, b]) => `<div><dt>${a}</dt><dd>${b}</dd></div>`).join("");
  }
  function about() {
    const p = proj(), S = sp(), W = S.pca.W, d = DIM, ev = S.pca.ratio, load = j => Array.from({ length: d }, (_, k) => `${fmt(W[k * d + j], 2)}·${FEAT_SHORT[k]}`).join(" + ").replace(/\+ −/g, "− ");
    let h;
    if ((p.kind === "feat" || p.kind === "feat3") && GROUP() && DATA.sets[SET].kind === "clusters") h = `<p>Every fish is a point whose coordinates are its measurements. Fish of the same kind have similar measurements, so the points form <b>groups</b>; how well they are separated depends on the data set chosen in the header. Distances on this plot are real distances (cm).</p>`;
    else if (p.kind === "feat" || p.kind === "feat3") h = `<p>Every fish is a point whose coordinates are its measurements. The cloud is <b>long and narrow</b>: the fish vary a lot along one direction (big fish are long <i>and</i> thick) and little along the perpendicular one (slender vs stocky at the same size). These are the two <b>axes of variation</b> that PCA will find.</p>` +
      `<p>Distances on this plot are real distances (cm), so nearest neighbours and interpolation can be read off directly. Try <b>Mode &rarr; nearest neighbours</b>.</p>`; 
    else if (p.kind === "pc") h = `<p>PCA keeps the same points but rotates the axes onto the directions of largest variance. <b>PC1</b> = ${load(0)} (${fmt(ev[0] * 100, 0)}% of the variance)${GROUP() ? "" : " is the <b>size</b> of the fish"}; <b>PC2</b> = ${load(1)} (${fmt(ev[1] * 100, 1)}%)${GROUP() ? "." : " separates <b>slender from stocky</b> fish."}</p>` +
      `<p>Nothing is distorted: distances on the PC1/PC2 plane are the real distances with the remaining components dropped (see the tab <i>PCA step by step</i> for the recipe).</p>`;
    else if (p.algo === "tsne") { const r = RUNS.get(tsneKey(DIM, TSNE_DEFAULT)); h = `<p><b>t-SNE</b> places the fish on the plane so that fish that were neighbours stay neighbours (perplexity ${TSNE_DEFAULT.perplexity}, ${TSNE_DEFAULT.iters} gradient steps${r ? `, final KL divergence ${fmt(r.kl[r.kl.length - 1], 3)}` : ""}).</p>` +
      `<p>The axes have no meaning and only local structure counts. Use the <b>nearest-neighbours mode</b> to compare: neighbours in the measurement space are mostly close on the map too. t-SNE gives coordinates for these fish only; a new fish needs a new optimisation.</p>`; }
    else h = `<p><b>UMAP</b> builds a weighted neighbour graph of the fish (${UMAP_DEFAULT.nNeighbors} neighbours each, min_dist ${UMAP_DEFAULT.minDist}) and lays it out on the plane by pulling connected fish together and pushing others apart.</p>` +
      `<p>As for t-SNE the axes have no meaning; the global shape of the cloud may be stretched or bent. Compare with PCA: with only ${d} numbers per fish the data really is (nearly) flat, so all methods give a similar picture, and the differences show how each treats distances.</p>`;
    $("about").innerHTML = h + clusterNote(p);
  }
  // extra paragraph for the populations with groups of fish (texts come with the data, see src/visualization/lecture1/__init__.py)
  function clusterNote(p) {
    if (!GROUP()) return "";
    const s = DATA.sets[SET], what = p.kind === "feat" || p.kind === "feat3" ? (DIM === 3 ? s.note3 : s.note2) :
      p.kind === "pc" ? "PCA only rotates the axes: a group is visible along the components that carry its direction, but it is not a separate 'island' unless a component happens to separate it." :
      "A neighbourhood-based map should show groups of similar fish as islands: the neighbours of a fish are fish of its own group (try the nearest-neighbours mode).";
    return `<p><b style="color:${s.group_colors[s.group_colors.length - 1]}">${s.title}:</b> ${s.note} ${what}</p>`;
  }

  // ── drawing ──
  function schedule() { if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); }); }
  function draw() {
    plot.resize(); plot.clear();
    const p = proj(), xy = coords(p), sc = scheme(), r = +$("size").value, cur = pinned >= 0 ? pinned : hover, fishMode = $("markersel").value === "fish";
    if (!xy) { $("legend").innerHTML = ""; return; }
    plot.dim = p.dim;
    if (p.dim === 3) plot.axes3D(p.labels); else if (p.kind === "feat") plot.axes2D(p.labels[0], p.labels[1]); else if (p.kind === "pc") plot.axes2D(p.labels[0], p.labels[1], { origin: true });
    const order = [...Array(N).keys()];
    for (let i = 0; i < N; i++) { const q = plot.P(vec(xy, i, p.dim)); scr[3 * i] = q[0]; scr[3 * i + 1] = q[1]; scr[3 * i + 2] = q[2]; }
    if (p.dim === 3) order.sort((a, b) => scr[3 * a + 2] - scr[3 * b + 2]);
    const active = isolate !== null, K = FISH_PX(r), ghost = ov || cur >= 0, R3 = plot.s * 0.5 * Math.hypot(...plot.hi.map((h, k) => h - plot.lo[k])) || 1;
    g.save();
    for (const i of order) {
      const on = !active || sc.id(i) === isolate, depth = p.dim === 3 ? 0.6 + 0.4 * (scr[3 * i + 2] / R3 + 1) / 2 : 1;
      g.globalAlpha = (on ? (ghost ? 0.55 : 0.85) : 0.08) * depth;
      if (fishMode) drawFish(g, scr[3 * i], scr[3 * i + 1], FISH[i][0], FISH[i][1], tailOf(i), K, sc.color(i));
      else { g.fillStyle = sc.color(i); g.beginPath(); g.arc(scr[3 * i], scr[3 * i + 1], r * (p.dim === 3 ? 0.7 + 0.5 * depth : 1), 0, TAU); g.fill(); }
    }
    g.restore();
    if (ov) drawOverlay(p, xy, sc, r);
    if (cur >= 0 && !ov) { g.save(); g.strokeStyle = "#fff"; g.lineWidth = 2; g.beginPath(); g.arc(scr[3 * cur], scr[3 * cur + 1], r + 5, 0, TAU); g.stroke(); g.restore(); }
    $("legend").innerHTML = sc.legend.map(([c, t, id]) => `<div data-id="${id}" class="${isolate !== null && isolate !== id ? "off" : ""}"><i style="background:${c}"></i>${t}</div>`).join("");
  }
  const FISH_PX = r => r * 0.2;
  function drawOverlay(p, xy, sc, r) {
    g.save(); g.globalAlpha = 0.25; g.fillStyle = "#111"; g.fillRect(0, 0, plot.W, plot.H); g.restore();
    const pt = i => [scr[3 * i], scr[3 * i + 1]], K = FISH_PX(r) * 1.7, fishMode = $("markersel").value === "fish";
    const mark = (i, text, ring) => {
      const [x, y] = pt(i);
      if (fishMode) drawFish(g, x, y, FISH[i][0], FISH[i][1], tailOf(i), K, sc.color(i), 1, ring); else { g.fillStyle = sc.color(i); g.strokeStyle = ring; g.lineWidth = 2.5; g.beginPath(); g.arc(x, y, r + 4, 0, TAU); g.fill(); g.stroke(); }
      g.save(); g.font = "11px system-ui"; g.textAlign = "center"; g.textBaseline = "middle"; g.fillStyle = "#000"; g.strokeStyle = "#fff"; g.lineWidth = 3;
      g.strokeText(String(text), x, y - r - 12); g.fillText(String(text), x, y - r - 12); g.restore();
    };
    if (ov.kind === "knn") {
      const [qx, qy] = pt(ov.q);
      if (p.kind === "feat" && $("metricsel").value === "raw") {           // the neighbourhood is a disc of radius = distance to the K-th neighbour
        const c = plot.P([FISH[ov.q][p.axes[0]], FISH[ov.q][p.axes[1]]]), R = ov.dist[ov.dist.length - 1] * plot.s;
        g.save(); g.setLineDash([5, 5]); g.strokeStyle = MC; g.globalAlpha = 0.8; g.beginPath(); g.arc(c[0], c[1], R, 0, TAU); g.stroke(); g.restore();
      }
      ov.ids.forEach(j => { const [x, y] = pt(j); g.save(); g.globalAlpha = 0.7; g.strokeStyle = MC; g.lineWidth = 1.5; g.beginPath(); g.moveTo(qx, qy); g.lineTo(x, y); g.stroke(); g.restore(); });
      ov.ids.forEach((j, k) => mark(j, k + 1, "#fff")); mark(ov.q, "Q", MC);
    } else {
      const path = ov.path;
      if (ov.B >= 0) {
        if (p.dim === 2 && proj().lin) {                                   // the true straight line and the ideal points on it (linear maps only)
          const ideal = ov.ideal.map(v => plot.P(proj().lin(v)));
          g.save(); g.setLineDash([5, 5]); g.strokeStyle = "#fff"; g.globalAlpha = 0.6; g.beginPath(); ideal.forEach(([x, y], s) => s ? g.lineTo(x, y) : g.moveTo(x, y)); g.stroke(); g.restore();
          ideal.forEach(([x, y]) => { g.save(); g.strokeStyle = "#fff"; g.lineWidth = 1.5; g.beginPath(); g.arc(x, y, 3.5, 0, TAU); g.stroke(); g.restore(); });
        }
        g.save(); g.strokeStyle = MC; g.lineWidth = 2; g.globalAlpha = 0.95; g.beginPath(); path.forEach((j, s) => { const [x, y] = pt(j); s ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke(); g.restore();
      }
      path.forEach((j, s) => mark(j, j === ov.A ? "A" : j === ov.B ? "B" : s, j === ov.A ? AC : j === ov.B ? BC : "#fff"));
      if (ov.B < 0) mark(ov.A, "A", AC);
    }
  }

  // ── events ──
  function nearest(mx, my) {
    const r = +$("size").value, lim = Math.max(10, r + 5) ** 2; let best = -1, bd = lim, bz = -1e9;
    const sc = scheme();
    for (let i = 0; i < N; i++) {
      if (isolate !== null && sc.id(i) !== isolate) continue;
      const d = (scr[3 * i] - mx) ** 2 + (scr[3 * i + 1] - my) ** 2;
      if (d < bd || (d === bd && scr[3 * i + 2] > bz)) { bd = d; best = i; bz = scr[3 * i + 2]; }
    }
    return best;
  }
  function tip(i, mx, my) {
    const t = $("tip"); if (i < 0 || mode !== "explore") { t.hidden = true; return; }
    const c = t.querySelector("canvas"), r = devicePixelRatio || 1, w = 120, h = 64; c.width = w * r; c.height = h * r; c.style.width = w + "px"; c.style.height = h + "px";
    const cg = c.getContext("2d"); cg.scale(r, r); const k = FISH_K(w, h), F = tailOf(i); drawFish(cg, w / 2 + F * k / 2, h / 2, FISH[i][0], FISH[i][1], F, k, scheme().color(i));
    t.querySelector("span").textContent = `fish ${i + 1} · L ${fmt(FISH[i][0])} · T ${fmt(FISH[i][1])}`;
    t.hidden = false; t.style.left = Math.min(mx + 14, plot.W - 150) + "px"; t.style.top = Math.min(my + 14, plot.H - 100) + "px";
  }
  plot.bind({
    onMove: (x, y) => { const i = nearest(x, y); if (pinned < 0 && i !== hover) { hover = i; if (i >= 0) showFish(i); schedule(); } tip(i, x, y); },
    onClick: (x, y) => { const i = nearest(x, y); if (mode === "explore") { pinned = (pinned >= 0 || i < 0) ? -1 : i; hover = i; if (pinned >= 0 || i >= 0) showFish(pinned >= 0 ? pinned : i); schedule(); } else modeClick(i); },
    onLeave: () => { $("tip").hidden = true; }, onChange: schedule,
  });
  $("projsel").addEventListener("change", () => { projKey = $("projsel").value; ov = null; setBounds(); if (mode === "knn") computeKnn(); about(); applyHint(); schedule(); });
  function applyHint() { const b = proj().dim === 3 ? " · drag: rotate · wheel: zoom · dbl-click: reset" : " · wheel: zoom · drag: pan · dbl-click: reset";
    $("hint").textContent = (mode === "knn" ? "click a fish: show its nearest neighbours" : mode === "interp" ? "click fish A, then fish B" : "hover: inspect · click: pin · legend: isolate") + b; }
  $("colorsel").addEventListener("change", () => { schemeKey = $("colorsel").value; isolate = null; if (ov) { if (mode === "knn") computeKnn(); else computeInterp(); } if (hover >= 0 || pinned >= 0) showFish(pinned >= 0 ? pinned : hover); schedule(); });
  $("markersel").addEventListener("change", schedule); $("size").addEventListener("input", schedule);
  $("modesel").addEventListener("change", () => applyMode());
  $("kk").addEventListener("input", () => { computeKnn(); schedule(); }); $("metricsel").addEventListener("change", () => { computeKnn(); schedule(); });
  $("steps").addEventListener("input", () => { computeInterp(); schedule(); });
  $("swapab").addEventListener("click", () => { [ptA, ptB] = [ptB, ptA]; if (ptA >= 0) computeInterp(); schedule(); });
  $("clearab").addEventListener("click", () => { ptA = ptB = -1; ov = null; $("strip").innerHTML = ""; $("strip2").innerHTML = ""; $("modehelp").textContent = "Click fish A, then fish B."; schedule(); });
  $("strip").addEventListener("click", e => { const f = e.target.closest("figure[data-i]"); if (!f) return; const i = +f.dataset.i; if (mode === "knn") { qpt = i; computeKnn(); showFish(i); schedule(); } else showFish(i); });
  $("strip2").addEventListener("click", e => { const f = e.target.closest("figure[data-i]"); if (f) showFish(+f.dataset.i); });
  $("legend").addEventListener("click", e => { const d = e.target.closest("div[data-id]"); if (!d) return; const id = +d.dataset.id; isolate = isolate === id ? null : id; schedule(); });
  window.addEventListener("keydown", e => { if (CUR !== "explore" || /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName || "")) return; if (e.key === "Escape") { isolate = null; pinned = -1; schedule(); } });

  return {
    onDim(what) { dirty = true; for (const p of PR) delete p.xyc; if (what === "set") { schemeKey = defaultColor(); qpt = ptA = ptB = -1; } },
    show() { if (dirty) rebuild(); plot.resize(); schedule(); showFish(pinned >= 0 ? pinned : hover >= 0 ? hover : 0); },
  };
}
