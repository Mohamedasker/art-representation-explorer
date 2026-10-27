// "t-SNE step by step" and "UMAP step by step" tabs.  Both show the fish in their measurement space (left) and the 2-D map
// being built (right); the steps walk through the algorithm and everything is computed live in the browser (algos.js), so
// perplexity / n_neighbors / min_dist / initialisation / learning rate can be changed and the whole run is redone.

function makeEmbed(root, kind) {
  const T = kind === "tsne", id = s => kind + s;
  root.innerHTML = `
    <div class="stepbar" id="${id("Steps")}"></div>
    <div class="toolbar" id="${id("Ctl")}"></div>
    <main class="steps">
      <div class="plotcol">
        <div class="twin">
          <div class="plotwrap"><div class="title" id="${id("InTitle")}"></div><canvas id="${id("In")}"></canvas><div class="cap" id="${id("InCap")}"></div></div>
          <div class="plotwrap"><div class="title" id="${id("MapTitle")}"></div><canvas id="${id("Map")}"></canvas><div class="cap" id="${id("MapCap")}"></div>
            <div class="busy" id="${id("Busy")}" hidden style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:#111a;z-index:3">computing…</div></div>
        </div>
        <div class="below" id="${id("Below")}"><canvas id="${id("C1")}"></canvas><canvas id="${id("C2")}"></canvas></div>
      </div>
      <aside><div class="card" id="${id("Text")}"></div><div class="card" id="${id("Math")}"></div></aside>
    </main>`;
  const inCv = $(id("In")), mapCv = $(id("Map")), inPlot = new Plot(inCv, 2), mapPlot = new Plot(mapCv, 2), gI = inCv.getContext("2d"), gM = mapCv.getContext("2d");
  const STEPS = T ? ["Data", "Similarities in the data", "Initialise the map", "Similarities on the map", "Gradient and update", "Iterate", "Result"]
                  : ["Data", "Nearest neighbours", "Fuzzy weights", "Fuzzy union", "Initialise the layout", "Optimise the layout", "Result"];
  const UNIT = T ? "iteration" : "epoch", NB = 8;                       // NB: input-space neighbours of the focus fish that are highlighted
  let step = 0, focus = -1, hov = -1, it = 0, run = null, dirty = true, colorKey = "length", playing = 0, speed = 1, raf = 0;
  let tp = { ...TSNE_DEFAULT }, up = { ...UMAP_DEFAULT }, box = null, nf = null, nfRes = null;
  const maxIt = () => run ? (T ? run.T : run.T) : 0, itMax = () => T && step === 4 ? Math.min(60, maxIt()) : maxIt();
  const scheme = () => { const l = colorSchemes(DIM); return l.find(s => s.key === colorKey) || l[0]; };
  const frameY = () => run.frames[Math.min(it, run.frames.length - 1)];
  const exagAt = t => T && t < run.exagIters ? run.p.exag : 1;
  const momAt = t => t < run.exagIters ? 0.5 : 0.8;
  const posIn = i => Array.from(sp().X.slice(i * DIM, i * DIM + DIM));
  const nbrs = () => neighbourOrder(sp().D2, N, focus).slice(0, NB);

  // ── setting up / recomputing ──
  function setup() {
    const S = sp(); box = rangeOf(S.X, N, DIM, { margin: 0.08 }); inPlot.dim = DIM;
    const c = S.pca.mu; focus = 0; let bd = Infinity;
    for (let i = 0; i < N; i++) { let dd = 0; for (let k = 0; k < DIM; k++) dd += (S.X[i * DIM + k] - c[k]) ** 2; if (dd < bd) { bd = dd; focus = i; } }
    nf = Array.from(c).map(v => v + 0.0); nf[0] += 1.5; nf[1] -= 1.0; nfRes = null; run = null; dirty = false;
  }
  function ensure(cb) {
    const key = T ? tsneKey(DIM, tp) : umapKey(DIM, up);
    if (RUNS.has(key)) { run = RUNS.get(key); it = Math.min(it, itMax()); cb && cb(); return; }
    $(id("Busy")).hidden = false;
    later(() => { run = T ? getTsne(DIM, tp) : getUmap(DIM, up); $(id("Busy")).hidden = true; it = Math.min(it, itMax()); nfRes = null; cb && cb(); });
  }

  // ── left panel: the measurement space ──
  function pointsIn() {                                              // screen positions of all fish (+ depth)
    const out = new Float64Array(3 * N), S = sp();
    for (let i = 0; i < N; i++) { const q = inPlot.P(Array.from(S.X.slice(i * DIM, i * DIM + DIM))); out[3 * i] = q[0]; out[3 * i + 1] = q[1]; out[3 * i + 2] = q[2]; }
    return out;
  }
  let scrI = new Float64Array(3 * N), scrM = new Float64Array(3 * N);
  const depthA = (scr, i, R) => DIM === 3 ? 0.6 + 0.4 * (scr[3 * i + 2] / R + 1) / 2 : 1;
  function drawIn() {
    inPlot.dim = DIM; inPlot.fit(box.lo, box.hi); inPlot.clear();
    if (DIM === 2) inPlot.axes2D("length (cm)", "thickness (cm)"); else inPlot.axes3D(["length (cm)", "thickness (cm)", "tail (cm)"]);
    scrI = pointsIn(); const sc = scheme(), S = sp(), g = gI, R3 = inPlot.s * 0.5 * Math.hypot(...inPlot.hi.map((h, k) => h - inPlot.lo[k])) || 1;
    const order = [...Array(N).keys()]; if (DIM === 3) order.sort((a, b) => scrI[3 * a + 2] - scrI[3 * b + 2]);
    const fx = posIn(focus), near = step >= 1 ? new Set(nbrs()) : new Set();
    let w = null;                                                    // per-fish weights shown by size / opacity (step dependent)
    if (T && step === 1) { const c = run.aff.cond; w = Float64Array.from({ length: N }, (_, j) => c[focus * N + j]); }
    if (!T && step === 2) { w = Float64Array.from({ length: N }, (_, j) => run.graph.A[focus * N + j]); }
    if (!T && step === 3) { w = Float64Array.from({ length: N }, (_, j) => run.graph.W[focus * N + j]); }
    const wmax = w ? Math.max(1e-12, ...w) : 1;
    // underlays: kernels / graph edges
    if (T && step === 1) {
      const sg = run.aff.sigma[focus];
      const planes = DIM === 2 ? [[[1, 0], [0, 1]]] : [[[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 0, 1]], [[0, 1, 0], [0, 0, 1]]];
      for (const [r, a] of [[sg, 0.9], [2 * sg, 0.55], [3 * sg, 0.3]]) for (const [u, v] of planes) inPlot.ellipse(fx, u, v, r, r, MC, 1.3, a, r === sg && DIM === 2 ? "#ffd54f22" : null);
    }
    if (!T && step >= 3) edgesOn(inPlot, scrI, run.graph.W, step === 3 ? 0.9 : 0.5);
    if (!T && step === 1) {
      const gr = run.graph, k = gr.k, planes = DIM === 2 ? [[[1, 0], [0, 1]]] : [[[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 0, 1]], [[0, 1, 0], [0, 0, 1]]];
      for (const [r, a] of [[gr.rho[focus], 0.9], [gr.dist[focus * k + k - 1], 0.6]]) for (const [u, v] of planes) inPlot.ellipse(fx, u, v, r, r, MC, 1.2, a);
    }
    if (!T && (step === 1 || step === 2)) {
      const gr = run.graph, k = gr.k;
      for (let r = 0; r < k; r++) { const j = gr.idx[focus * k + r], wj = gr.A[focus * N + j]; inPlot.line(fx, posIn(j), MC, step === 2 ? 1 + 2.5 * wj : 1.2, null, step === 2 ? 0.25 + 0.7 * wj : 0.6); }
    }
    g.save();
    for (const i of order) {
      let r = 3.6 * (DIM === 3 ? 0.8 + 0.4 * depthA(scrI, i, R3) : 1), a = 0.85 * depthA(scrI, i, R3);
      if (w && i !== focus) { const u = Math.sqrt(w[i] / wmax); r = 2 + 8 * u; a = (0.15 + 0.85 * u) * depthA(scrI, i, R3); }
      g.globalAlpha = a; g.fillStyle = sc.color(i); g.beginPath(); g.arc(scrI[3 * i], scrI[3 * i + 1], r, 0, TAU); g.fill();
      if (near.has(i) && !w) { g.globalAlpha = 0.9; g.strokeStyle = MC; g.lineWidth = 1.5; g.beginPath(); g.arc(scrI[3 * i], scrI[3 * i + 1], r + 2.5, 0, TAU); g.stroke(); }
    }
    g.restore();
    ring(gI, scrI, focus, 8, MC, 2.5); if (hov >= 0) ring(gI, scrI, hov, 7, "#fff", 2);
    if (step === 6 && nf) { const p = inPlot.P(nf); g.save(); star(g, p[0], p[1], 9, "#fff", "#e91e63"); g.restore(); inPlot.text(nf, "new fish", "#fff", 12, -10, 11, "left", true); }
    $(id("InTitle")).textContent = `Measurement space (${DIM} numbers per fish)`;
    $(id("InCap")).textContent = DIM === 3 ? "drag to rotate" : "";
  }
  function ring(g, scr, i, r, col, lw) { g.save(); g.strokeStyle = col; g.lineWidth = lw; g.beginPath(); g.arc(scr[3 * i], scr[3 * i + 1], r, 0, TAU); g.stroke(); g.restore(); }
  function star(g, x, y, r, fill, stroke) { g.beginPath(); for (let k = 0; k < 10; k++) { const rr = k % 2 ? r * 0.45 : r, a = -Math.PI / 2 + k * Math.PI / 5; k ? g.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a)) : g.moveTo(x + rr * Math.cos(a), y + rr * Math.sin(a)); } g.closePath(); g.fillStyle = fill; g.fill(); g.strokeStyle = stroke; g.lineWidth = 2; g.stroke(); }
  // all edges of a symmetric weight matrix as faint lines between the plotted points (the focus fish's edges highlighted)
  function edgesOn(plot, scr, W, hot) {
    const g = plot.g; g.save(); g.lineCap = "round";
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      const w = W[i * N + j]; if (w < 0.08) continue; const f = i === focus || j === focus;
      g.globalAlpha = f ? 0.25 + hot * 0.7 * w : 0.03 + 0.3 * w; g.strokeStyle = f ? MC : "#cfd8dc"; g.lineWidth = f ? 1 + 2 * w : 0.8;
      g.beginPath(); g.moveTo(scr[3 * i], scr[3 * i + 1]); g.lineTo(scr[3 * j], scr[3 * j + 1]); g.stroke();
    }
    g.restore();
  }

  // ── right panel: heat map of the similarities, or the map ──
  function drawMap() {
    mapPlot.dim = 2; const sc = scheme(), g = gM; $(id("MapCap")).textContent = "";
    const heat = T ? (step === 1 ? run.aff.P : null) : (step === 2 ? run.graph.A : step === 3 ? run.graph.W : null);
    if (step === 0 || heat || (!T && step === 1)) {
      mapPlot.resize(); mapPlot.clear();
      if (heat) { drawHeat(mapCv, heat, N, sp().order, focus); $(id("MapCap")).textContent = "rows and columns: fish sorted by length · yellow: the focus fish"; $(id("MapTitle")).textContent = T ? "Similarity matrix P (symmetrised)" : step === 2 ? "Directed weights w(i→j) (not symmetric)" : "Fuzzy union: symmetric weights"; }
      else { const q = mapCv.getContext("2d"), r = devicePixelRatio || 1; q.setTransform(r, 0, 0, r, 0, 0); q.fillStyle = "#777"; q.font = "13px system-ui"; q.textAlign = "center";
        const lines = step === 0 ? [`The goal: ${N} points in the plane`, "(the map), one per fish,", "that keeps neighbours together."] : ["The map appears in a later step."];
        lines.forEach((t, k) => q.fillText(t, mapCv.clientWidth / 2, mapCv.clientHeight / 2 - 10 + k * 20)); $(id("MapTitle")).textContent = "Map (2-D)"; }
      return;
    }
    const Y = frameY(), r = rangeOf(Y, N, 2, { margin: 0.12 }); mapPlot.fit(r.lo, r.hi); mapPlot.clear();
    const spread = Math.max(r.hi[0] - r.lo[0], r.hi[1] - r.lo[1]); mapPlot.axes2D("", "", { grid: false, sci: spread < 0.01 });
    for (let i = 0; i < N; i++) { const q = mapPlot.P([Y[2 * i], Y[2 * i + 1]]); scrM[3 * i] = q[0]; scrM[3 * i + 1] = q[1]; scrM[3 * i + 2] = 0; }
    if (!T && step >= 4) edgesOn(mapPlot, scrM, run.graph.W, 0.9);
    let w = null; const near = step >= 1 ? new Set(nbrs()) : new Set();
    if (T && step === 3) { const { num, Z } = tsneQ(Y, N); let s = 0; const row = Float64Array.from({ length: N }, (_, j) => j === focus ? 0 : num[focus * N + j] / Z); row.forEach(v => { s += v; }); w = row.map(v => v / s); }
    const wmax = w ? Math.max(1e-12, ...w) : 1;
    g.save();
    for (let i = 0; i < N; i++) {
      let rr = 3.6, a = 0.85;
      if (w && i !== focus) { const u = Math.sqrt(w[i] / wmax); rr = 2 + 8 * u; a = 0.15 + 0.85 * u; }
      g.globalAlpha = a; g.fillStyle = sc.color(i); g.beginPath(); g.arc(scrM[3 * i], scrM[3 * i + 1], rr, 0, TAU); g.fill();
      if (near.has(i) && !w) { g.globalAlpha = 0.9; g.strokeStyle = MC; g.lineWidth = 1.5; g.beginPath(); g.arc(scrM[3 * i], scrM[3 * i + 1], rr + 2.5, 0, TAU); g.stroke(); }
    }
    g.restore();
    if (T && step === 4) forcesT(Y); if (!T && step === 5) forcesU(Y);
    if (step === 6 && nfRes) { g.save(); nfRes.path.forEach((p, k) => { const q = mapPlot.P(p); g.globalAlpha = 0.7; g.fillStyle = "#fff"; g.beginPath(); g.arc(q[0], q[1], k === 0 ? 3.5 : 1.8, 0, TAU); g.fill(); });
      const q = mapPlot.P(nfRes.y); star(g, q[0], q[1], 10, "#fff", "#e91e63"); g.restore(); mapPlot.text(nfRes.y, "new fish", "#fff", 13, -11, 11, "left", true); }
    ring(gM, scrM, focus, 8, MC, 2.5); if (hov >= 0) ring(gM, scrM, hov, 7, "#fff", 2);
    $(id("MapTitle")).textContent = `Map (2-D) · ${UNIT} ${it}`;
  }
  // arrows on the focus fish: how every other fish pulls (green) or pushes (red) it - normalised so the largest is 70 px
  function arrowsFrom(Y, items, net) {
    const g = gM, fx = [Y[2 * focus], Y[2 * focus + 1]], p0 = mapPlot.P(fx), top = items.slice().sort((a, b) => Math.hypot(b.v[0], b.v[1]) - Math.hypot(a.v[0], a.v[1])).slice(0, 10);
    const mx = Math.max(1e-300, ...top.map(t => Math.hypot(t.v[0], t.v[1])));
    for (const t of top) { const L = 70 * Math.hypot(t.v[0], t.v[1]) / mx, ang = Math.atan2(-t.v[1], t.v[0]), a = t.col;
      g.save(); g.globalAlpha = 0.9; g.strokeStyle = g.fillStyle = a; g.lineWidth = 2; const x1 = p0[0] + L * Math.cos(ang), y1 = p0[1] + L * Math.sin(ang);
      g.beginPath(); g.moveTo(p0[0], p0[1]); g.lineTo(x1, y1); g.stroke(); g.beginPath(); g.moveTo(x1, y1); g.lineTo(x1 - 7 * Math.cos(ang - 0.4), y1 - 7 * Math.sin(ang - 0.4)); g.lineTo(x1 - 7 * Math.cos(ang + 0.4), y1 - 7 * Math.sin(ang + 0.4)); g.fill(); g.restore();
      if (t.j !== undefined) { const q = mapPlot.P([Y[2 * t.j], Y[2 * t.j + 1]]); g.save(); g.globalAlpha = 0.5; g.strokeStyle = a; g.lineWidth = 1; g.setLineDash([2, 3]); g.beginPath(); g.moveTo(p0[0], p0[1]); g.lineTo(q[0], q[1]); g.stroke(); g.restore();
        g.save(); g.strokeStyle = a; g.lineWidth = 2; g.beginPath(); g.arc(q[0], q[1], 6, 0, TAU); g.stroke(); g.restore(); } }
    if (net) { const q = mapPlot.P([fx[0] + net[0], fx[1] + net[1]]); gM.save(); gM.strokeStyle = gM.fillStyle = "#fff"; gM.lineWidth = 2.5; const ang = Math.atan2(q[1] - p0[1], q[0] - p0[0]), L = Math.hypot(q[0] - p0[0], q[1] - p0[1]);
      if (L > 3) { gM.beginPath(); gM.moveTo(p0[0], p0[1]); gM.lineTo(q[0], q[1]); gM.stroke(); gM.beginPath(); gM.moveTo(q[0], q[1]); gM.lineTo(q[0] - 9 * Math.cos(ang - 0.4), q[1] - 9 * Math.sin(ang - 0.4)); gM.lineTo(q[0] - 9 * Math.cos(ang + 0.4), q[1] - 9 * Math.sin(ang + 0.4)); gM.fill(); } gM.restore(); }
  }
  function forcesT(Y) {
    const e = exagAt(it), { grad, num, Z } = tsneGradient(run.aff.P, Y, N, e), items = [];
    for (let j = 0; j < N; j++) if (j !== focus) {
      const p = run.aff.P[focus * N + j], q = num[focus * N + j] / Z, c = 4 * (e * p - q) * num[focus * N + j];       // dC/dy_i term of fish j; the force is its negative
      items.push({ j, v: [-c * (Y[2 * focus] - Y[2 * j]), -c * (Y[2 * focus + 1] - Y[2 * j + 1])], col: e * p > q ? GOOD : BAD });
    }
    const next = run.frames[Math.min(it + 1, run.frames.length - 1)], d = [next[2 * focus] - Y[2 * focus], next[2 * focus + 1] - Y[2 * focus + 1]];
    arrowsFrom(Y, items, d);
    const q = mapPlot.P([next[2 * focus], next[2 * focus + 1]]); gM.save(); gM.strokeStyle = "#fff"; gM.setLineDash([3, 3]); gM.beginPath(); gM.arc(q[0], q[1], 7, 0, TAU); gM.stroke(); gM.restore();
    $(id("MapCap")).textContent = "green: pulled towards (p > q) · red: pushed away (p < q) · arrows normalised · white: the actual update";
  }
  function forcesU(Y) {
    const { a, b, graph } = run, items = [], rng = mulberry32(it * 977 + focus);
    for (let j = 0; j < N; j++) { const w = graph.W[focus * N + j]; if (w < 0.1 || j === focus) continue;
      const dx = Y[2 * focus] - Y[2 * j], dy = Y[2 * focus + 1] - Y[2 * j + 1], d2 = dx * dx + dy * dy, c = -2 * a * b * Math.pow(d2, b - 1) / (a * Math.pow(d2, b) + 1) * w;
      items.push({ j, v: [c * dx, c * dy], col: GOOD }); }
    for (let t = 0; t < 5; t++) { const j = Math.floor(rng() * N); if (j === focus) continue;
      const dx = Y[2 * focus] - Y[2 * j], dy = Y[2 * focus + 1] - Y[2 * j + 1], d2 = dx * dx + dy * dy, c = 2 * b / ((0.001 + d2) * (a * Math.pow(d2, b) + 1));
      items.push({ j, v: [c * dx, c * dy], col: BAD }); }
    arrowsFrom(Y, items, null);
    $(id("MapCap")).textContent = "green: attraction along the edges of the fish (weight w) · red: repulsion from 5 random fish · arrows normalised";
  }

  // ── charts below the two plots ──
  function drawCharts() {
    const c1 = $(id("C1")), c2 = $(id("C2")), below = $(id("Below")), S = sp(), hasCharts = !(step === 0 || (!T && step === 4) || (T && step === 2));
    below.hidden = !hasCharts; if (!hasCharts) return;
    c1.hidden = c2.hidden = false;
    const top = (arr, k) => arr.map((v, j) => [v, j]).filter(x => x[1] !== focus).sort((a, b) => b[0] - a[0]).slice(0, k);
    if (T) {
      const cond = run.aff.cond, sg = run.aff.sigma[focus];
      if (step === 1) {
        const row = Array.from({ length: N }, (_, j) => cond[focus * N + j]), tp12 = top(row, 12);
        chartBars(c1, tp12.map(([v, j]) => ({ label: `${j + 1}`, label2: `${fmt(Math.sqrt(S.D2[focus * N + j]), 1)}`, v, color: MC })), { title: "p(j|i) of the 12 most similar fish (label: fish, distance in cm)", fmtv: v => v.toFixed(3) });
        const dmax = Math.sqrt(Math.max(...Array.from({ length: N }, (_, j) => S.D2[focus * N + j]))), xs = Array.from({ length: 101 }, (_, k) => dmax * k / 100);
        chartLines(c2, { series: [{ x: xs, y: xs.map(x => Math.exp(-x * x / (2 * sg * sg))), color: MC }], xmin: 0, xmax: dmax, ymin: 0, ymax: 1, title: `Gaussian kernel, σ = ${fmt(sg, 2)} cm  (dots: the other fish)`, xlabel: "distance to the focus fish (cm)",
          points: Array.from({ length: N }, (_, j) => j).filter(j => j !== focus).map(j => ({ x: Math.sqrt(S.D2[focus * N + j]), y: Math.exp(-S.D2[focus * N + j] / (2 * sg * sg)), color: "#cfd8dc", r: 2 })) });
      } else if (step === 3 || step === 4) {
        const Y = frameY(), { num, Z } = tsneQ(Y, N), P = run.aff.P, e = step === 4 ? exagAt(it) : 1;
        const pr = Array.from({ length: N }, (_, j) => j === focus ? 0 : P[focus * N + j]), qr = Array.from({ length: N }, (_, j) => j === focus ? 0 : num[focus * N + j] / Z);
        const sp_ = pr.reduce((s, x) => s + x, 0), sq_ = qr.reduce((s, x) => s + x, 0), tops = top(pr, 12);
        if (step === 3) {
          chartBars(c1, tops.map(([v, j]) => ({ label: `${j + 1}`, label2: "", v: v / sp_, color: MC, v2: qr[j] / sq_, color2: AC })), { title: "p_ij and q_ij of the 12 fish most similar in the data (each row normalised)", legend: [[MC, "p"], [AC, "q"]], fmtv: v => v.toFixed(3) });
          const dm = 6, xs = Array.from({ length: 101 }, (_, k) => dm * k / 100);
          chartLines(c2, { series: [{ x: xs, y: xs.map(x => Math.exp(-x * x / 2)), color: MC, dash: [4, 3] }, { x: xs, y: xs.map(x => 1 / (1 + x * x)), color: AC }], xmin: 0, xmax: dm, ymin: 0, ymax: 1, title: "kernels: Gaussian (σ = 1, dashed) vs Student-t (1 d.o.f.): heavy tail", xlabel: "distance" });
        } else {
          chartBars(c1, tops.map(([v, j]) => { const f = 4 * (e * v - qr[j]) * num[focus * N + j]; return { label: `${j + 1}`, v: Math.abs(f), color: f > 0 ? GOOD : BAD }; }),
            { title: "size of the pull (green) / push (red) of each fish on the focus fish", fmtv: v => v.toExponential(0) });
          klChart(c2, true);
        }
      } else { klChart(c1, false); presChart(c2); }
    } else {
      const gr = run.graph, k = gr.k, nb = Array.from({ length: k }, (_, r) => ({ j: gr.idx[focus * k + r], d: gr.dist[focus * k + r] })), rho = gr.rho[focus], sg = gr.sigma[focus];
      if (step === 1) {
        chartBars(c1, nb.map((n, r) => ({ label: `${n.j + 1}`, label2: "", v: n.d, color: r === 0 ? AC : MC })), { title: "distances (cm) to the nearest fish; the nearest one defines ρ (blue)", fmtv: v => fmt(v, 1) });
        chartBars(c2, [{ label: "ρ (nearest)", v: rho, color: AC }, { label: "farthest of k", v: nb[k - 1].d, color: MC }, { label: "mean of all fish", v: Math.sqrt(S.D2.reduce((s, x) => s + x, 0) / (N * N)), color: "#78909c" }], { title: "scale of the neighbourhood (cm)", fmtv: v => fmt(v, 1) });
      } else if (step === 2) {
        const dmax = nb[k - 1].d * 1.25, xs = Array.from({ length: 101 }, (_, q) => dmax * q / 100);
        chartLines(c1, { series: [{ x: xs, y: xs.map(x => Math.exp(-Math.max(0, x - rho) / sg)), color: MC }], xmin: 0, xmax: dmax, ymin: 0, ymax: 1, title: `w = exp(−(d − ρ)/σ), ρ = ${fmt(rho, 2)}, σ = ${fmt(sg, 2)}`, xlabel: "distance (cm)",
          vlines: [{ x: rho, color: AC, label: "ρ" }], points: nb.map(n => ({ x: n.d, y: Math.exp(-Math.max(0, n.d - rho) / sg), color: "#fff", r: 3 })) });
        chartBars(c2, nb.map(n => ({ label: `${n.j + 1}`, v: gr.A[focus * N + n.j], color: MC })), { title: `weights w(i→j): they sum to ${fmt(nb.reduce((s, n) => s + gr.A[focus * N + n.j], 0), 2)} = log₂(${gr.nNeighbors})`, fmtv: v => fmt(v, 2) });
      } else if (step === 3) {
        const ids = Array.from({ length: N }, (_, j) => j).filter(j => j !== focus && gr.W[focus * N + j] > 0).sort((a, b) => gr.W[focus * N + b] - gr.W[focus * N + a]).slice(0, 10);
        chartBars(c1, ids.map(j => ({ label: `${j + 1}`, v: gr.A[focus * N + j], color: MC, v2: gr.A[j * N + focus], color2: AC })), { title: "w(i→j) and w(j→i) of the focus fish i", legend: [[MC, "i→j"], [AC, "j→i"]], fmtv: v => fmt(v, 2) });
        chartBars(c2, ids.map(j => ({ label: `${j + 1}`, v: gr.W[focus * N + j], color: GOOD })), { title: "fuzzy union  a + b − a·b", fmtv: v => fmt(v, 2) });
      } else {
        const dm = 4, xs = Array.from({ length: 121 }, (_, q) => dm * q / 120), md = up.minDist;
        chartLines(c1, { series: [{ x: xs, y: xs.map(x => x < md ? 1 : Math.exp(-(x - md))), color: "#78909c", dash: [4, 3] }, { x: xs, y: xs.map(x => 1 / (1 + run.a * Math.pow(x, 2 * run.b))), color: AC }], xmin: 0, xmax: dm, ymin: 0, ymax: 1,
          title: `map similarity v(d) = 1/(1 + a d^2b): a = ${fmt(run.a, 2)}, b = ${fmt(run.b, 2)}  (dashed: target for min_dist ${md})`, xlabel: "distance on the map" });
        presChart(c2);
      }
    }
  }
  function klChart(cv, early) {
    const n = early ? Math.min(60, run.T) : run.T, xs = Array.from({ length: n + 1 }, (_, t) => t), ys = xs.map(t => run.kl[t]), lo = Math.min(...ys), hi = Math.max(...ys);
    chartLines(cv, { series: [{ x: xs, y: ys, color: GOOD }], xmin: 0, xmax: n, ymin: lo, ymax: hi, ydec: 2, title: "KL(P‖Q) of the map (not exaggerated) vs iteration", xlabel: "iteration", logy: false,
      shades: early || run.exagIters >= n ? [] : [{ x0: 0, x1: run.exagIters, color: "#ffffff0c", label: "early exaggeration" }], vlines: [{ x: Math.min(it, n), color: "#fff" }], points: [{ x: Math.min(it, n), y: run.kl[Math.min(it, n)], color: "#fff" }] });
  }
  function presChart(cv) {
    const pr = run.pres, cur = pr.v[Math.min(pr.v.length - 1, Math.round(it / (pr.t[1] - pr.t[0])))];
    chartLines(cv, { series: [{ x: pr.t, y: pr.v, color: AC }], xmin: 0, xmax: run.T, ymin: 0, ymax: 1, ydec: 1, title: `share of each fish's ${NN_K} nearest neighbours that are still its neighbours on the map`, xlabel: UNIT,
      vlines: [{ x: it, color: "#fff" }], points: [{ x: it, y: cur, color: "#fff" }] });
  }

  // ── texts and numbers ──
  const E = x => x === 0 ? "0" : Math.abs(x) < 0.01 || Math.abs(x) >= 1000 ? x.toExponential(2).replace("e-", "e−").replace("e+", "e") : fmt(x, 3);
  const eq = h => `<div class="eq">${h}</div>`;
  function renderText() {
    if (!run && step > 0) return;
    const S = sp(), fi = focus + 1; let title = STEPS[step], text = "", math = "";
    const pos = fx => `(${Array.from(fx).map(v => fmt(v, 1)).join(", ")})`;
    if (T) {
      const aff = run && run.aff;
      if (step === 0) { text = `<p><b>t-SNE</b> (t-distributed stochastic neighbour embedding) only needs the <b>distances</b> between the fish. Goal: find a point <i>y</i><sub>i</sub> in the plane for every fish <i>x</i><sub>i</sub> (the <b>map</b>) so that fish that are close in the measurement space (left) are close on the map (right).</p>` +
        `<p><b>Click a fish</b> (in either plot) to follow it through all steps; it is ringed in yellow, its ${NB} nearest fish in the measurement space have yellow rings.</p>`;
        math = `<div class="eq"><i>x</i><sub>i</sub> ∈ ℝ<sup>${DIM}</sup> &nbsp;→&nbsp; <i>y</i><sub>i</sub> ∈ ℝ<sup>2</sup>,&nbsp; <i>i</i> = 1 … ${N}</div><div class="note">fish ${fi} = ${pos(posIn(focus))} cm</div>`; }
      else if (step === 1) {
        const sg = aff.sigma[focus], H = aff.entropy[focus];
        text = `<p>For fish <i>i</i> every other fish <i>j</i> gets the probability of being its neighbour from a <b>Gaussian kernel</b> centred on <i>i</i>: nearby fish get a high value, far ones almost 0 (size / opacity of the dots; yellow circles: 1σ, 2σ, 3σ).</p>` +
          `<p>The width σ<sub>i</sub> is found for every fish separately so that the <b>perplexity</b> (the effective number of neighbours) equals the chosen value: dense regions get a small σ, sparse ones a large σ. Finally the two directions are averaged: <i>p</i><sub>ij</sub> = (<i>p</i><sub>j|i</sub> + <i>p</i><sub>i|j</sub>)/2<i>N</i> – the matrix on the right.</p>`;
        math = eq(`<i>p</i><sub>j|i</sub> = exp(−‖<i>x</i><sub>i</sub>−<i>x</i><sub>j</sub>‖² / 2σ<sub>i</sub>²) / Σ<sub>k≠i</sub> exp(−‖<i>x</i><sub>i</sub>−<i>x</i><sub>k</sub>‖² / 2σ<sub>i</sub>²)`) +
          eq(`<i>p</i><sub>ij</sub> = (<i>p</i><sub>j|i</sub> + <i>p</i><sub>i|j</sub>) / 2<i>N</i>`) +
          `<div class="note">fish ${fi}: σ = <b>${fmt(sg, 2)} cm</b>, entropy H = ${fmt(H, 3)} nats, perplexity e<sup>H</sup> = <b>${fmt(Math.exp(H), 1)}</b> (target ${tp.perplexity}). Larger perplexity → wider neighbourhoods.</div>`;
      } else if (step === 2) {
        text = `<p>Give every fish a starting point on the map. Either <b>tiny random numbers</b> (std 10<sup>−4</sup>) or the first two <b>principal components</b> scaled down to the same size (PCA initialisation, more stable and reproducible). The axis labels show how tiny the map still is.</p>` +
          `<p>Nothing has been optimised yet: the map only reflects the initial positions. Change the initialisation in the toolbar and see how the later steps depend on it.</p>`;
        const Y0 = run.Y0; math = eq(`<i>y</i><sub>i</sub><sup>(0)</sup> ~ 𝒩(0, 10<sup>−8</sup> <i>I</i>)  &nbsp;or&nbsp;  scaled (PC1, PC2)`) + `<div class="note">fish ${fi}: <i>y</i><sup>(0)</sup> = (${E(Y0[2 * focus])}, ${E(Y0[2 * focus + 1])})</div>`;
      } else if (step === 3) {
        const Y = frameY(), { num, Z } = tsneQ(Y, N);
        text = `<p>On the map similarity is measured with a <b>Student-t kernel</b> (one degree of freedom). Its heavy tail lets moderately dissimilar fish sit far apart on the map, which avoids the crowding of everything in the middle.</p>` +
          `<p>The map is good when <i>q</i><sub>ij</sub> ≈ <i>p</i><sub>ij</sub>. The mismatch is the <b>Kullback–Leibler divergence</b>. Dot size now shows <i>q</i><sub>j|i</sub> on the map; scrub the iteration to watch it approach the <i>p</i> pattern (bars).</p>`;
        math = eq(`<i>q</i><sub>ij</sub> = (1 + ‖<i>y</i><sub>i</sub>−<i>y</i><sub>j</sub>‖²)<sup>−1</sup> / Σ<sub>k≠l</sub> (1 + ‖<i>y</i><sub>k</sub>−<i>y</i><sub>l</sub>‖²)<sup>−1</sup>`) +
          eq(`<i>C</i> = KL(<i>P</i>‖<i>Q</i>) = Σ<sub>i≠j</sub> <i>p</i><sub>ij</sub> log(<i>p</i><sub>ij</sub>/<i>q</i><sub>ij</sub>)`) + `<div class="note">at ${UNIT} ${it}: <i>C</i> = <b>${fmt(run.kl[it], 3)}</b></div>`;
      } else if (step === 4) {
        const Y = frameY(), e = exagAt(it), g = tsneGradient(run.aff.P, Y, N, e), gx = g.grad[2 * focus], gy = g.grad[2 * focus + 1], mom = momAt(it), prev = it > 0 ? run.frames[it - 1] : Y;
        const dpx = Y[2 * focus] - prev[2 * focus], dpy = Y[2 * focus + 1] - prev[2 * focus + 1], dx = mom * dpx - run.p.lr * gx, dy = mom * dpy - run.p.lr * gy;
        text = `<p>To reduce the cost, move every point downhill. The <b>gradient</b> is a sum over all other fish: fish with <i>p</i> > <i>q</i> are too far apart on the map and <b>pull</b> (green), fish with <i>p</i> < <i>q</i> are too close and <b>push</b> (red).</p>` +
          `<p>Update: a step against the gradient plus <b>momentum</b> (a fraction of the previous step). For the first 250 iterations all <i>p</i><sub>ij</sub> are multiplied by ${run.p.exag} (<b>early exaggeration</b>) so that clusters can move through each other. Use ▶ / the slider to take single steps.</p>`;
        math = eq(`∂<i>C</i>/∂<i>y</i><sub>i</sub> = 4 Σ<sub>j</sub> (<i>p</i><sub>ij</sub> − <i>q</i><sub>ij</sub>) (<i>y</i><sub>i</sub> − <i>y</i><sub>j</sub>) (1 + ‖<i>y</i><sub>i</sub>−<i>y</i><sub>j</sub>‖²)<sup>−1</sup>`) +
          eq(`<i>y</i><sub>i</sub><sup>(t+1)</sup> = <i>y</i><sub>i</sub><sup>(t)</sup> − η ∂<i>C</i>/∂<i>y</i><sub>i</sub> + α (<i>y</i><sub>i</sub><sup>(t)</sup> − <i>y</i><sub>i</sub><sup>(t−1)</sup>)`) +
          `<table class="mat" style="width:100%"><tr><td>fish ${fi}, ${UNIT} ${it}</td><td>x</td><td>y</td></tr><tr><td>position <i>y</i><sup>(t)</sup></td><td>${E(Y[2 * focus])}</td><td>${E(Y[2 * focus + 1])}</td></tr>` +
          `<tr><td>gradient (p × ${e})</td><td>${E(gx)}</td><td>${E(gy)}</td></tr><tr><td>−η · gradient, η = ${run.p.lr}</td><td>${E(-run.p.lr * gx)}</td><td>${E(-run.p.lr * gy)}</td></tr>` +
          `<tr><td>α · previous step, α = ${mom}</td><td>${E(mom * dpx)}</td><td>${E(mom * dpy)}</td></tr><tr><td><b>step</b></td><td>${E(dx)}</td><td>${E(dy)}</td></tr></table>`;
      } else if (step === 5) {
        text = `<p>Repeat the update for all fish at once: ${run.T} iterations in total, the first ${run.exagIters} with early exaggeration and momentum 0.5, the rest with momentum 0.8. The cost falls quickly at first and then slowly; the map stops changing (<b>convergence</b>).</p>` +
          `<p>The lower right curve measures how many of each fish's ${NN_K} nearest neighbours in the measurement space are still its neighbours on the map.</p>`;
        math = `<div class="note">${UNIT} <b>${it}</b> of ${run.T} · KL(P‖Q) = <b>${fmt(run.kl[it], 3)}</b> · neighbours kept: <b>${fmt(100 * run.pres.v[Math.min(run.pres.v.length - 1, Math.round(it / 10))], 0)}%</b></div>` +
          `<div class="note">${it < run.exagIters ? "early exaggeration phase: all attractions are ×" + run.p.exag + ", so the cloud is squeezed together. " + (GROUP() ? "The groups of fish form their own clumps that are pulled apart from each other in the second phase." : "These fish form one continuous cloud rather than separate clusters, so it first collapses onto a line (the size axis) and unfolds in the second phase.") + " Set the exaggeration to 1 in the toolbar to skip this phase." : "normal phase: the squeezed cloud unfolds and the local neighbourhoods are refined"}</div>`;
      } else {
        const Y = run.frames[run.T], ids = SAMPLE9.ids || S.order.filter((_, k) => k % 17 === 0).slice(0, 9);
        text = `<p>The final map: <b>${N} points, no formula</b>. Nothing like “multiply by <i>W</i>” exists in t-SNE: the coordinates are simply what the optimisation produced. Axes have no meaning, only neighbourhoods do, and distances between far apart clusters should not be trusted.</p>` +
          `<p>A <b>new fish</b> therefore needs its own little optimisation: compute its probabilities <i>p</i><sub>j</sub> to the existing fish and move one point until its <i>q</i> matches (star; the dots show the path). Move the sliders to try.</p>`;
        math = matrixHTML(ids.map(i => [fmt(Y[2 * i], 1), fmt(Y[2 * i + 1], 1)]), { head: ["y₁", "y₂"], rowHead: ids.map(i => `fish ${i + 1}`), cls: "br" }) +
          `<div class="note">map coordinates <i>Y</i> (${N}×2; nine fish shown). Final KL = <b>${fmt(run.kl[run.T], 3)}</b>.</div>` + newFishHTML();
      }
    } else {
      const gr = run && run.graph;
      if (step === 0) { text = `<p><b>UMAP</b> also uses only distances between the fish. It first builds a weighted <b>graph</b> of who is near whom (a fuzzy description of the data's shape) and then draws that graph in the plane.</p><p><b>Click a fish</b> (in either plot) to follow it through the steps.</p>`;
        math = `<div class="eq"><i>x</i><sub>i</sub> ∈ ℝ<sup>${DIM}</sup> &nbsp;→&nbsp; graph &nbsp;→&nbsp; <i>y</i><sub>i</sub> ∈ ℝ<sup>2</sup></div><div class="note">fish ${fi} = ${pos(posIn(focus))} cm</div>`; }
      else if (step === 1) {
        text = `<p>For every fish find its <b>${gr.nNeighbors} nearest neighbours</b> (counting itself, so ${gr.k} others; yellow lines, circles at the distance of the nearest and of the farthest). The distance ρ<sub>i</sub> to the <b>nearest</b> neighbour is recorded: every fish is considered fully connected to at least one other fish.</p>`;
        math = eq(`ρ<sub>i</sub> = min<sub>j≠i</sub> ‖<i>x</i><sub>i</sub> − <i>x</i><sub>j</sub>‖`) + `<div class="note">fish ${fi}: ρ = <b>${fmt(gr.rho[focus], 2)} cm</b>, farthest of the ${gr.k} neighbours: ${fmt(gr.dist[focus * gr.k + gr.k - 1], 2)} cm. Change <b>n_neighbors</b> in the toolbar: it sets how local the graph is.</div>`;
      } else if (step === 2) {
        text = `<p>Each of the ${gr.k} edges gets a <b>fuzzy weight</b> between 0 and 1: the nearest neighbour gets 1, farther ones decay exponentially (dot size and line width). The scale σ<sub>i</sub> is chosen per fish so that the weights add up to log₂(n_neighbors): in a sparse region the weights decay more slowly.</p>` +
          `<p>The matrix on the right holds all these directed weights w(i→j): it is <b>not symmetric</b>, because j can be among the neighbours of i without i being among those of j.</p>`;
        math = eq(`<i>w</i>(<i>i</i>→<i>j</i>) = exp(− (<i>d</i><sub>ij</sub> − ρ<sub>i</sub>) / σ<sub>i</sub>)`) + eq(`Σ<sub>j</sub> <i>w</i>(<i>i</i>→<i>j</i>) = log₂(n_neighbors) = ${fmt(Math.log2(gr.nNeighbors), 2)}`) +
          `<div class="note">fish ${fi}: ρ = ${fmt(gr.rho[focus], 2)}, σ = <b>${fmt(gr.sigma[focus], 2)}</b>; sum of weights = ${fmt(Array.from({ length: gr.k }, (_, r) => gr.A[focus * N + gr.idx[focus * gr.k + r]]).reduce((s, x) => s + x, 0), 3)}</div>`;
      } else if (step === 3) {
        const jb = Array.from({ length: N }, (_, j) => j).filter(j => j !== focus && gr.A[focus * N + j] > 0 && gr.A[j * N + focus] > 0 && gr.A[focus * N + j] < 1)[0];
        text = `<p>Combine the two directions with a <b>fuzzy union</b>: the weight of the edge {i, j} is the probability that at least one of the two directed edges exists. The result is a symmetric weighted graph (matrix on the right, all edges on the left, fish ${fi}'s edges in yellow): the <b>fuzzy simplicial set</b> that represents the shape of the data.</p>`;
        math = eq(`<i>w</i><sub>ij</sub> = <i>a</i> + <i>b</i> − <i>a</i><i>b</i>,&nbsp; <i>a</i> = <i>w</i>(<i>i</i>→<i>j</i>),&nbsp; <i>b</i> = <i>w</i>(<i>j</i>→<i>i</i>)`) +
          (jb !== undefined ? `<div class="note">example, fish ${fi} and ${jb + 1}: a = ${fmt(gr.A[focus * N + jb], 2)}, b = ${fmt(gr.A[jb * N + focus], 2)} → w = ${fmt(gr.W[focus * N + jb], 2)}</div>` : "") +
          `<div class="note">the graph has ${(() => { let c = 0; for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) if (gr.W[i * N + j] > 0) c++; return c; })()} edges</div>`;
      } else if (step === 4) {
        text = `<p>Start the map from a <b>spectral layout</b>: the eigenvectors of the graph's Laplacian give every fish two coordinates such that connected fish get similar values (alternatives in the toolbar: random, or the first two principal components). The graph is drawn on the map; the next step makes the layout better.</p>`;
        math = eq(`<i>L</i> = <i>I</i> − <i>D</i><sup>−1/2</sup> <i>W</i> <i>D</i><sup>−1/2</sup>,&nbsp; <i>y</i> = eigenvectors 2, 3 of <i>L</i>`) + `<div class="note">initialisation: <b>${up.init}</b></div>`;
      } else if (step === 5) {
        const e = it, al = 1 - e / run.T;
        text = `<p>On the map, two fish are “connected” with probability <i>v</i> = 1/(1 + <i>a d</i><sup>2<i>b</i></sup>); <i>a</i> and <i>b</i> come from <b>min_dist</b> (how tightly points may pack). Stochastic gradient descent makes <i>v</i> match the graph weights <i>w</i>: for every edge the two fish are <b>pulled</b> together (stronger for heavier edges), and for a few random fish they are <b>pushed</b> apart.</p>` +
          `<p>The learning rate falls from 1 to 0 over the ${run.T} epochs. Green / red arrows show these forces for the focus fish at the current epoch.</p>`;
        math = eq(`<i>v</i><sub>ij</sub> = 1 / (1 + <i>a</i> ‖<i>y</i><sub>i</sub>−<i>y</i><sub>j</sub>‖<sup>2<i>b</i></sup>)`) +
          eq(`<i>C</i> = Σ<sub>ij</sub> [ <i>w</i> log(<i>w</i>/<i>v</i>) + (1−<i>w</i>) log((1−<i>w</i>)/(1−<i>v</i>)) ]`) +
          `<div class="note">epoch <b>${e}</b> of ${run.T} · learning rate ${fmt(al, 2)} · min_dist ${up.minDist} → a = ${fmt(run.a, 3)}, b = ${fmt(run.b, 3)} · neighbours kept: <b>${fmt(100 * run.pres.v[Math.min(run.pres.v.length - 1, Math.round(e / 5))], 0)}%</b></div>` +
          `<div class="note">(UMAP's cross-entropy is minimised only approximately: the sampled updates need not lower it at every step, so we show how many neighbours are kept instead.)</div>`;
      } else {
        const Y = run.frames[run.T], ids = SAMPLE9.ids || S.order.filter((_, k) => k % 17 === 0).slice(0, 9);
        text = `<p>The final layout. As for t-SNE the coordinates are a result of the optimisation; the axes carry no meaning and global distances can be distorted. Unlike t-SNE, UMAP keeps the graph, so a <b>new fish</b> can be placed cheaply (as <code>UMAP.transform</code> does): find its nearest existing fish, compute fuzzy weights, start at their weighted average position on the map and refine a few epochs with only the new point moving.</p>`;
        math = matrixHTML(ids.map(i => [fmt(Y[2 * i], 1), fmt(Y[2 * i + 1], 1)]), { head: ["y₁", "y₂"], rowHead: ids.map(i => `fish ${i + 1}`), cls: "br" }) +
          `<div class="note">map coordinates <i>Y</i> (${N}×2; nine fish shown).</div>` + newFishHTML();
      }
    }
    $(id("Text")).innerHTML = `<h3>${step + 1} · ${title}</h3>${text}`; $(id("Math")).innerHTML = `<h2>Maths and numbers</h2>${math}`;
    bindNewFish();
  }
  // ── placing a new fish (last step) ──
  function newFishHTML() {
    const ranges = [[8, 46, 0.5], [2, 22, 0.5], [2, 14, 0.5]];
    const sl = k => `<div class="ctrl"><label style="min-width:84px">${FEAT[k]}</label><input type="range" class="nf" data-k="${k}" min="${ranges[k][0]}" max="${ranges[k][1]}" step="${ranges[k][2]}" value="${nf[k]}"><span class="val" id="nfv${k}">${fmt(nf[k], 1)} cm</span></div>`;
    return `<h2 style="margin-top:12px">Place a new fish</h2>${Array.from({ length: DIM }, (_, k) => sl(k)).join("")}<div id="nfFish" style="margin:6px 0"></div><div class="note" id="nfNote"></div>`;
  }
  function bindNewFish() {
    if (step !== 6) return;
    root.querySelectorAll("input.nf").forEach(el => el.oninput = () => { nf[+el.dataset.k] = +el.value; $(`nfv${el.dataset.k}`).textContent = fmt(+el.value, 1) + " cm"; placeNew(); drawAll(); });
    placeNew();
  }
  function placeNew() {
    if (step !== 6 || !run) return;
    const S = sp(), d2 = new Float64Array(N); for (let j = 0; j < N; j++) { let s = 0; for (let k = 0; k < DIM; k++) s += (nf[k] - S.X[j * DIM + k]) ** 2; d2[j] = s; }
    const Y = run.frames[run.T];
    nfRes = T ? tsneNewPoint(d2, Y, N, tp.perplexity) : umapNewPoint(d2, Y, N, up.nNeighbors, run.a, run.b);
    const c = $("nfFish"); if (c) { c.replaceChildren(fishCanvas(nf[0], nf[1], DIM === 3 ? nf[2] : 0.3 * nf[0], "#cfd8dc", 140, 60)); }
    const nn = [...Array(N).keys()].sort((a, b) => d2[a] - d2[b]).slice(0, 3); const note = $("nfNote");
    if (note) note.innerHTML = T ? `Optimised in ${nfRes.path.length - 1} steps from the p-weighted average of its neighbours' positions. Nearest existing fish: ${nn.map(j => j + 1).join(", ")}.` :
      `Neighbours: ${nfRes.neighbours.slice(0, 3).map(j => j + 1).join(", ")} … (weights ${nfRes.weights.slice(0, 3).map(w => fmt(w, 2)).join(", ")} …); σ = ${fmt(nfRes.sigma, 2)}.`;
  }

  // ── toolbar ──
  function controls() {
    const c = $(id("Ctl")), S = colorSchemes(DIM);
    let h = `<span><label>Colour fish by </label><select id="${id("Col")}">${S.map(s => `<option value="${s.key}" ${s.key === colorKey ? "selected" : ""}>${s.label}</option>`).join("")}</select></span>`;
    if (T && step >= 1) h += `<span class="ctrl"><label>perplexity</label><input id="${id("Perp")}" type="range" min="2" max="60" step="1" value="${tp.perplexity}"><span class="val" id="${id("PerpV")}">${tp.perplexity}</span></span>`;
    if (!T && step >= 1) h += `<span class="ctrl"><label>n_neighbors</label><input id="${id("NN")}" type="range" min="3" max="40" step="1" value="${up.nNeighbors}"><span class="val" id="${id("NNV")}">${up.nNeighbors}</span></span>`;
    if (!T && step >= 4) h += `<span class="ctrl"><label>min_dist</label><select id="${id("MD")}">${[0, 0.1, 0.25, 0.5, 1].map(v => `<option ${v === up.minDist ? "selected" : ""}>${v}</option>`).join("")}</select></span>`;
    if (step >= (T ? 2 : 4)) h += `<span class="ctrl"><label>initialisation</label><select id="${id("Init")}">${(T ? [["pca", "PCA"], ["random", "random"]] : [["spectral", "spectral"], ["pca", "PCA"], ["random", "random"]]).map(([v, l]) => `<option value="${v}" ${(T ? tp.init : up.init) === v ? "selected" : ""}>${l}</option>`).join("")}</select><button id="${id("Seed")}" title="new random numbers">↻</button></span>`;
    if (T && step >= 4 && step <= 5) h += `<span class="ctrl"><label>learning rate η</label><input id="${id("LR")}" type="range" min="1" max="100" step="1" value="${tp.lr}"><span class="val" id="${id("LRV")}">${tp.lr}</span></span>` +
      `<span class="ctrl"><label>early exaggeration</label><input id="${id("Ex")}" type="range" min="1" max="24" step="1" value="${tp.exag}"><span class="val" id="${id("ExV")}">${tp.exag}</span></span>`;
    if (step >= (T ? 3 : 5) && step <= 5) h += `<span class="ctrl"><button id="${id("Reset")}" title="back to the start">reset</button><button id="${id("Prev")}" title="one step back">◀ 1</button><button id="${id("Play")}">${playing ? "pause" : "▶ play"}</button><button id="${id("Next")}" title="one step forward">1 ▶</button>` +
      `<input id="${id("It")}" type="range" min="0" max="${itMax()}" step="1" value="${it}"><span class="val" id="${id("ItV")}" style="min-width:8em">${UNIT} ${it}</span>` +
      `<select id="${id("Spd")}" title="playback speed">${[["1", "slow"], ["3", "normal"], ["9", "fast"]].map(([v, l]) => `<option value="${v}" ${+v === speed ? "selected" : ""}>${l}</option>`).join("")}</select></span>`;
    c.innerHTML = h;
    const on = (s, ev, fn) => { const e = $(id(s)); if (e) e.addEventListener(ev, fn); };
    on("Col", "change", e => { colorKey = e.target.value; drawAll(); });
    on("Perp", "input", e => { $(id("PerpV")).textContent = e.target.value; }); on("Perp", "change", e => { tp.perplexity = +e.target.value; recompute(); });
    on("NN", "input", e => { $(id("NNV")).textContent = e.target.value; }); on("NN", "change", e => { up.nNeighbors = +e.target.value; recompute(); });
    on("MD", "change", e => { up.minDist = +e.target.value; recompute(); });
    on("Init", "change", e => { if (T) tp.init = e.target.value; else up.init = e.target.value; it = 0; recompute(); });
    on("Seed", "click", () => { if (T) tp.seed++; else up.seed++; it = 0; recompute(); });
    on("LR", "input", e => { $(id("LRV")).textContent = e.target.value; }); on("LR", "change", e => { tp.lr = +e.target.value; recompute(); });
    on("Ex", "input", e => { $(id("ExV")).textContent = e.target.value; }); on("Ex", "change", e => { tp.exag = +e.target.value; recompute(); });
    on("It", "input", e => { stop(); it = +e.target.value; drawAll(); }); on("Spd", "change", e => { speed = +e.target.value; });
    on("Reset", "click", () => { stop(); it = 0; syncIt(); drawAll(); }); on("Prev", "click", () => { stop(); it = Math.max(0, it - 1); syncIt(); drawAll(); });
    on("Next", "click", () => { stop(); it = Math.min(itMax(), it + 1); syncIt(); drawAll(); }); on("Play", "click", () => playing ? stop() : play());
  }
  function syncIt() { const s = $(id("It")); if (s) { s.max = itMax(); s.value = it; } const v = $(id("ItV")); if (v) v.textContent = `${UNIT} ${it}`; }
  function play() {
    if (it >= itMax()) it = 0; playing = 1; const b = $(id("Play")); if (b) b.textContent = "pause";
    const f = () => { if (!playing) return; it = Math.min(itMax(), it + speed * (T ? 1 : 1)); syncIt(); drawAll(); if (it >= itMax()) stop(); else raf = requestAnimationFrame(f); };
    raf = requestAnimationFrame(f);
  }
  function stop() { playing = 0; cancelAnimationFrame(raf); const b = $(id("Play")); if (b) b.textContent = "▶ play"; }
  function recompute() { stop(); run = null; ensure(() => { nfRes = null; refresh(); }); }

  // ── glue ──
  function drawAll() {
    if (!run && step > 0) return;
    renderText(); drawIn(); if (run) drawMap(); else drawMapEmpty(); drawCharts(); syncIt();
  }
  function drawMapEmpty() {
    mapPlot.resize(); mapPlot.clear(); const q = mapCv.getContext("2d"); q.fillStyle = "#777"; q.font = "13px system-ui"; q.textAlign = "center";
    [`The goal: ${N} points in the plane`, "(the map), one per fish,", "that keeps neighbours together."].forEach((t, k) => q.fillText(t, mapCv.clientWidth / 2, mapCv.clientHeight / 2 - 10 + k * 20)); $(id("MapTitle")).textContent = "Map (2-D)";
  }
  function refresh() { controls(); drawAll(); }
  function buildStepBar() {
    $(id("Steps")).innerHTML = STEPS.map((s, i) => `<button data-s="${i}" class="${i === step ? "on" : ""}"><b>${i + 1}</b>${s}</button>`).join("") + `<span class="arrows">← → keys</span>`;
    $(id("Steps")).querySelectorAll("button").forEach(b => b.onclick = () => go(+b.dataset.s));
  }
  function go(i) {
    stop(); step = Math.max(0, Math.min(STEPS.length - 1, i)); buildStepBar(); fishTip(-1); hov = -1;
    it = T ? [0, 0, 0, 25, 0, 0, maxIt()][step] : [0, 0, 0, 0, 0, 0, maxIt()][step];     // t-SNE step 4: iteration 25 has some structure to compare p and q
    ensure(() => { it = Math.min(it, itMax()); refresh(); if (step === 5) play(); });
  }
  function nearestIn(scr, x, y, lim = 12) { let best = -1, bd = lim * lim; for (let i = 0; i < N; i++) { const d = (scr[3 * i] - x) ** 2 + (scr[3 * i + 1] - y) ** 2; if (d < bd) { bd = d; best = i; } } return best; }
  const clientOf = (cv, x, y) => { const r = cv.getBoundingClientRect(); return [r.left + x, r.top + y]; };
  function mapShowsPoints() { return run && !(step === 0 || (T && step === 1) || (!T && step <= 3)); }
  inPlot.bind({ onMove: (x, y) => { const i = nearestIn(scrI, x, y); if (i !== hov) { hov = i; drawIn(); if (mapShowsPoints()) drawMap(); } fishTip(i, ...clientOf(inCv, x, y)); },
    onClick: (x, y) => { const i = nearestIn(scrI, x, y); if (i >= 0) { focus = i; drawAll(); } }, onLeave: () => { hov = -1; fishTip(-1); drawIn(); if (mapShowsPoints()) drawMap(); }, onChange: drawIn, rotate: null, pan: false, zoomable: true });
  mapPlot.bind({ onMove: (x, y) => { if (!mapShowsPoints()) return; const i = nearestIn(scrM, x, y); if (i !== hov) { hov = i; drawIn(); drawMap(); } fishTip(i, ...clientOf(mapCv, x, y)); },
    onClick: (x, y) => { if (!mapShowsPoints()) return; const i = nearestIn(scrM, x, y); if (i >= 0) { focus = i; drawAll(); } }, onLeave: () => { hov = -1; fishTip(-1); drawIn(); if (mapShowsPoints()) drawMap(); }, rotate: false, pan: false, zoomable: false });
  window.addEventListener("keydown", e => { if (CUR !== kind || /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName || "")) return; if (e.key === "ArrowRight") go(step + 1); else if (e.key === "ArrowLeft") go(step - 1); });

  return {
    onDim(what) { dirty = true; stop(); if (what === "set") colorKey = defaultColor(); },
    show() { if (dirty) { setup(); it = 0; } buildStepBar(); inPlot.dim = DIM; refresh(); ensure(() => { it = Math.min(it, itMax()); refresh(); }); },
  };
}
