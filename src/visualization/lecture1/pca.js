// "PCA step by step" tab: nine fish are taken through the PCA recipe - data, mean, centring, covariance matrix,
// eigenvectors, transformation - with the numbers (matrices) next to the geometry, in 2-D or 3-D.

function makePca(root) {
  root.innerHTML = `
    <div class="stepbar" id="pcaSteps"></div>
    <div class="toolbar" id="pcaCtl"></div>
    <main class="steps">
      <div class="plotcol"><div class="plotwrap"><div class="title" id="pcaTitle"></div><canvas id="pcaCanvas"></canvas><div class="cap" id="pcaCap"></div></div>
        <canvas id="pcaSub" style="width:100%;height:120px;background:var(--panel);border:1px solid var(--line);border-radius:6px" hidden></canvas></div>
      <aside><div class="card" id="pcaText"></div><div class="card" id="pcaMath"></div></aside>
    </main>`;
  const cv = $("pcaCanvas"), plot = new Plot(cv, 2), g = cv.getContext("2d");
  const STEPS = ["Data", "Mean", "Centre", "Covariance matrix", "Eigenvectors", "Transformation"];
  let step = 0, dirty = true, sampleSeed = 1, sample = [], st = null, X9 = null, tC = 0, tR = 0, flat = 0, keep = 2, entry = [0, 1], theta = 0.4, showAll = true, hl = -1, anim = 0;
  let ranges = null, d = 2, rotf = null;

  // ── the nine fish: spread over the cloud (along PC1) and alternating above / below its axis (PC2) ──
  function pickSample(seed) {
    const S = space(DIM), rng = mulberry32(seed * 7919 + 13), grp = GROUP();
    if (grp && DATA.sets[SET].kind === "clusters") {                          // groups only: spread the nine fish over the groups, and within a group along PC1
      const ng = DATA.sets[SET].group_names.length, out = [];
      for (let g = 0; g < ng; g++) {
        const m = [...Array(N).keys()].filter(i => grp[i] === g).sort((a, b) => S.pca.Z[a * DIM] - S.pca.Z[b * DIM]), k = Math.floor(9 / ng) + (g < 9 % ng ? 1 : 0);
        for (let q = 0; q < k; q++) out.push(m[Math.min(m.length - 1, Math.floor((q + 0.15 + 0.7 * rng()) * m.length / k))]);
      }
      return out;
    }
    const nCl = grp ? 2 : 0, nb = 9 - nCl;
    const ord = [...Array(N).keys()].filter(i => !grp || grp[i] === 0).sort((a, b) => S.pca.Z[a * DIM] - S.pca.Z[b * DIM]);
    const sd2 = Math.sqrt(S.pca.values[1]), base = [1.1, -1.1, 0.8, -0.8, 0.4, -0.4, 0.2, -0.2, 0].slice(0, nb);
    let t = base;                                                 // an order of these offsets that is not correlated with the position along PC1
    for (let tries = 0; tries < 200; tries++) {
      const p = base.slice(); for (let i = p.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; }
      t = p; if (Math.abs(p.reduce((s, x, b) => s + (b - (nb - 1) / 2) * x, 0)) < 0.7) break;
    }
    const out = t.map((off, b) => {
      const lo = Math.floor(b * ord.length / nb), hi = Math.floor((b + 1) * ord.length / nb); let best = -1, bd = Infinity;
      for (let k = lo; k < hi; k++) { const i = ord[k], dd = Math.abs(S.pca.Z[i * DIM + 1] - off * sd2 * 1.1); if (dd < bd) { bd = dd; best = i; } }
      return best;
    });
    if (nCl) { const cl = [...Array(N).keys()].filter(i => grp[i] === 1); for (let k = 0; k < nCl; k++) out.push(cl[Math.floor((k + rng() * 0.8) * cl.length / nCl)]); }
    return out;
  }
  function setup() {
    d = DIM; plot.dim = d; const S = space(d);
    sample = pickSample(sampleSeed); SAMPLE9.ids = sample;
    X9 = Float64Array.from(sample.flatMap(i => FISH[i].slice(0, d)));
    st = pcaSteps(X9, 9, d);
    rotf = rotationOf(st.W, d);
    // fixed view boxes: raw cloud, centred cloud and cloud in PC coordinates (all fish + the sample)
    const centredAll = Float64Array.from(S.X, (v, k) => v - st.mu[k % d]), pcAll = matmul(centredAll, st.W, N, d, d);
    ranges = { raw: rangeOf(S.X, N, d, { margin: 0.08 }), cen: rangeOf(centredAll, N, d, { margin: 0.08, zero: true }), pc: rangeOf(pcAll, N, d, { margin: 0.08, zero: true }) };
    dirty = false;
  }
  // rotation by Wᵀ (raw centred coordinates -> PC coordinates), interpolated by t in [0, 1]: x -> Rot(axis, -t*angle) x
  function rotationOf(W, d) {
    if (d === 2) { const phi = Math.atan2(W[2], W[0]); return (x, t) => { const c = Math.cos(t * phi), s = Math.sin(t * phi); return [c * x[0] + s * x[1], -s * x[0] + c * x[1]]; }; }
    const tr = W[0] + W[4] + W[8], th = Math.acos(Math.max(-1, Math.min(1, (tr - 1) / 2))), sn = Math.sin(th);
    if (sn < 1e-9) return x => x.slice();
    const ax = [(W[7] - W[5]) / (2 * sn), (W[2] - W[6]) / (2 * sn), (W[3] - W[1]) / (2 * sn)];
    return (x, t) => { const a = -t * th, c = Math.cos(a), s = Math.sin(a), dot = ax[0] * x[0] + ax[1] * x[1] + ax[2] * x[2];
      const cr = [ax[1] * x[2] - ax[2] * x[1], ax[2] * x[0] - ax[0] * x[2], ax[0] * x[1] - ax[1] * x[0]];
      return [0, 1, 2].map(k => x[k] * c + cr[k] * s + ax[k] * dot * (1 - c)); };
  }
  const curC = () => step < 2 ? 0 : step === 2 ? tC : 1, curR = () => step < 5 ? 0 : tR, curF = () => step < 5 ? 0 : flat;
  // raw vector -> vector in the current frame (centred by curC, rotated by curR, flattened onto the first `keep` PCs by curF)
  function xf(v) {
    let x = v.map((a, k) => a - curC() * st.mu[k]);
    if (curR() > 0) x = rotf(x, curR());
    if (curF() > 0) x = x.map((a, k) => k >= keep ? a * (1 - curF()) : a);
    return x;
  }
  const mix = (a, b, t) => ({ lo: a.lo.map((v, k) => lerp(v, b.lo[k], t)), hi: a.hi.map((v, k) => lerp(v, b.hi[k], t)) });
  function viewBox() {
    if (step <= 1) return ranges.raw;
    if (step === 2) return mix(ranges.raw, ranges.cen, tC);
    if (step < 5) return ranges.cen;
    return mix(ranges.cen, ranges.pc, tR);
  }
  const rowOf = i => Array.from(X9.slice(i * d, i * d + d));
  const labels = () => curR() > 0.5 ? ["PC1", "PC2", "PC3"].slice(0, d) : d === 2 ? ["length (cm)", "thickness (cm)"] : ["length (cm)", "thickness (cm)", "tail (cm)"];

  // ── drawing ──
  const FK = () => d === 2 ? 1.25 : 0.9;                       // pixels per cm of the fish pictures
  function drawFishAt(i, v, alpha = 0.92, ring = null) {
    const p = plot.P(v), k = FK(), j = sample[i], F = tailOf(j, d);
    drawFish(g, p[0] + F * k / 2, p[1], FISH[j][0], FISH[j][1], F, k, "#7fb3ff", alpha, ring);
    g.save(); g.font = "600 11px system-ui"; g.fillStyle = "#fff"; g.strokeStyle = "#000"; g.lineWidth = 3; g.textAlign = "center";
    g.strokeText(String(i + 1), p[0], p[1] - FISH[j][1] * k / 2 - 6); g.fillText(String(i + 1), p[0], p[1] - FISH[j][1] * k / 2 - 6); g.restore();
  }
  function draw() {
    if (dirty) setup();
    plot.dim = d; const box = viewBox(); plot.fit(box.lo, box.hi); plot.clear();
    const lab = labels();
    if (d === 2) plot.axes2D(lab[0], lab[1], { origin: step >= 2 }); else plot.axes3D(lab, { origin: step >= 2 });
    const S = space(d), P9 = [...Array(9).keys()].map(i => xf(rowOf(i)));
    g.save();
    if (showAll) for (let i = 0; i < N; i++) { const q = plot.P(xf(Array.from(S.X.slice(i * d, i * d + d)))); g.globalAlpha = 0.2; g.fillStyle = "#aaa"; g.beginPath(); g.arc(q[0], q[1], 2.2, 0, TAU); g.fill(); }
    g.restore();
    if (d === 3) P9.forEach(v => plot.line(v, [v[0], plot.lo[1], v[2]], "#fff", 1, [2, 3], 0.18));
    const mu = xf(Array.from(st.mu)), origin = Array(d).fill(0);
    if (step === 1) {                                                // lines to the mean + the mean itself
      P9.forEach(v => plot.line(v, mu, "#fff", 1, [3, 4], 0.35));
    }
    if (step === 3) drawCovariance(P9);
    if (step === 4) drawEigen(P9);
    if (step === 5) drawTransform(P9);
    P9.forEach((v, i) => drawFishAt(i, v, hl >= 0 && hl !== i ? 0.45 : 0.95, hl === i ? "#fff" : null));
    if (step >= 1 && step <= 2) { plot.dot(mu, 6, MC, "#000", 1, 1.5); plot.text(mu, "mean μ", MC, 10, -12, 12, "left", true); }
    if (step === 0) $("pcaCap").textContent = `${N} fish drawn faintly; the ${9} labelled fish are the sample`;
    $("pcaTitle").textContent = `${STEPS[step]}`;
  }
  function drawCovariance(P9) {
    const [a, b] = entry; let sum = 0;
    P9.forEach((v, i) => {
      const x = v[a], y = v[b], pos = x * y >= 0, hot = hl === i, area = x * y;
      const q = [0, 0, 0], corners = [q.slice(), q.slice(), q.slice(), q.slice()]; corners[1][a] = x; corners[2][a] = x; corners[2][b] = y; corners[3][b] = y;
      plot.poly(corners.map(c => c.slice(0, d)), pos ? GOOD : BAD, pos ? GOOD : BAD, hl >= 0 ? (hot ? 0.55 : 0.08) : 0.2, hot ? 2 : 1);
      plot.line(v, corners[1].slice(0, d), "#fff", 1, [2, 3], hl >= 0 && !hot ? 0.1 : 0.5); plot.line(v, corners[3].slice(0, d), "#fff", 1, [2, 3], hl >= 0 && !hot ? 0.1 : 0.5);
      sum += area;
    });
    $("pcaCap").innerHTML = `rectangle of fish <i>i</i>: ${FEAT_SHORT[a]}<sub>i</sub> · ${FEAT_SHORT[b]}<sub>i</sub> (green: both deviations on the same side of the mean, red: opposite sides)`;
  }
  function drawEigen(P9) {
    const lam = st.values, W = st.W, col = j => Array.from({ length: d }, (_, k) => W[k * d + j]), cols = [AC, BC, "#b39ddb"];
    const o = Array(d).fill(0);
    if (d === 2) { for (const k of [1, 2]) plot.ellipse(o, col(0), col(1), k * Math.sqrt(lam[0]), k * Math.sqrt(lam[1]), "#fff", 1, 0.25 + 0.15 * (2 - k)); }
    else { for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) plot.ellipse(o, col(p), col(q), 2 * Math.sqrt(lam[p]), 2 * Math.sqrt(lam[q]), "#fff", 1, 0.22); }
    for (let j = 0; j < d; j++) {
      const tip = col(j).map(x => x * 2 * Math.sqrt(lam[j]));
      plot.arrow(o, tip, cols[j], 3, 12); plot.text(tip, `PC${j + 1}`, cols[j], 6, -8, 13, "left", true);
    }
    if (d === 2) {                                                  // a trial direction: the variance of the fish projected onto it
      const u = [Math.cos(theta), Math.sin(theta)], R = Math.sqrt(lam[0]) * 3.2;
      plot.line([-R * u[0], -R * u[1]], [R * u[0], R * u[1]], "#fff", 1.5, [6, 5], 0.9);
      P9.forEach(v => { const t = v[0] * u[0] + v[1] * u[1], pr = [t * u[0], t * u[1]]; plot.line(v, pr, "#fff", 1, null, 0.45); plot.dot(pr, 3.5, "#fff", null, 0.95); });
      plot.arrow([0, 0], [u[0] * Math.sqrt(lam[0]) * 1.2, u[1] * Math.sqrt(lam[0]) * 1.2], "#fff", 2, 9);
    }
    $("pcaCap").textContent = d === 2 ? "ellipses: 1σ and 2σ contours of the covariance · arrows: 2√λ along each eigenvector" : "ellipses: 2σ contours of the covariance in the three planes · arrows: 2√λ along each eigenvector · drag to rotate";
  }
  function drawTransform(P9) {
    // the principal axes are carried along with the rotation: they end up along the coordinate axes
    const cols = [AC, BC, "#b39ddb"], o = Array(d).fill(0), col = j => Array.from({ length: d }, (_, k) => st.W[k * d + j] * 2 * Math.sqrt(st.values[j]));
    for (let j = 0; j < d; j++) plot.arrow(o, curR() > 0 ? rotf(col(j), curR()) : col(j), cols[j], 2.5, 10, j < keep || curF() === 0 ? 1 : 0.35);
    if (curR() > 0.999 && keep < d) P9.forEach((v, i) => {                      // drop the last coordinates: project onto the first `keep` axes
      const full = rotf(rowOf(i).map((a, k) => a - st.mu[k]), 1), target = full.map((a, k) => k >= keep ? 0 : a);
      plot.line(full, target, "#fff", 1, [3, 3], 0.4); plot.dot(target, 3.5, "#fff", null, 0.9);
    });
    $("pcaCap").textContent = curR() > 0.999 ? (keep < d ? `keeping ${keep} of ${d} components: ${fmt(100 * st.ratio.slice(0, keep).reduce((s, x) => s + x, 0), 1)}% of the variance` : "all components kept: just a rotation, nothing is lost") : "the cloud is rotated into the PC axes";
  }

  // ── numbers next to the picture ──
  const num = (x, k = 1) => fmt(x, k);
  const mat = (rows, o = {}) => matrixHTML(rows.map(r => r.map(x => typeof x === "number" ? num(x) : x)), { cls: "br", ...o });
  const heads = () => d === 2 ? ["L", "T"] : ["L", "T", "F"];
  const idx = () => [...Array(9).keys()].map(i => `${i + 1}`);
  function vsub(name, j) { return `<i>${name}</i><sub>${j}</sub>`; }
  function render() {
    if (dirty) setup();
    const lam = st.values, ratio = st.ratio, W = st.W, C = st.C, H = heads(), n = 9;
    const S = space(d), allAng = Math.atan2(S.pca.W[1 * d + 0], S.pca.W[0]) * 180 / Math.PI, smpAng = Math.atan2(W[1 * d + 0], W[0]) * 180 / Math.PI;
    let title = "", text = "", math = "";
    if (step === 0) {
      title = "1 · The data";
      text = `<p>Nine fish, each described by ${d} numbers (${d === 2 ? "length and thickness" : "length, thickness and tail length"}, in cm). Every fish is a <b>point</b> in ${d}-dimensional space; its picture is drawn at that point. The faint dots are all ${N} fish: the nine labelled ones were picked to cover the spread of the cloud${GROUP() ? " (" + (DATA.sets[SET].kind === "clusters" ? "two or three from every group" : "two of them are long-tail fish") + ")" : ""}.</p>` +
        `<p>Written as a <b>data matrix</b> <i>X</i> there is one row per fish and one column per measurement.</p>`;
      math = `<div class="eq"><i>X</i> ∈ ℝ<sup>${n}×${d}</sup></div>` + mat(sample.map((_, i) => rowOf(i)), { head: H, rowHead: idx() });
    } else if (step === 1) {
      title = "2 · The mean";
      text = `<p>The <b>mean fish</b> μ is the average of every column. It is the centre of mass of the cloud (yellow cross); the dotted lines show the fish relative to it.</p>`;
      math = `<div class="eq"><i>μ</i> = <span> (1/<i>n</i>) Σ<sub><i>i</i></sub> <i>x</i><sub><i>i</i></sub></span> = ${vecT(st.mu)}</div>` +
        mat(sample.map((_, i) => rowOf(i)), { head: H, rowHead: idx(), foot: [Array.from(st.mu, x => num(x, 2))] }) +
        `<div class="note">bottom row (yellow): the column means, e.g. μ<sub>L</sub> = (${sample.map((_, i) => num(X9[i * d])).slice(0, 3).join(" + ")} + …) / ${n} = ${num(st.mu[0], 2)}</div>`;
    } else if (step === 2) {
      title = "3 · Centre the data";
      text = `<p>Subtract μ from every row. The cloud is shifted so that its mean lies at the origin; shape and distances between fish do not change. Afterwards every column of <i>X</i><sub>c</sub> sums to zero.</p><p class="note">Drag the slider (or press play) to move the cloud.</p>`;
      const Xc = sample.map((_, i) => Array.from(st.Xc.slice(i * d, i * d + d)).map(x => num(x)));
      math = `<div class="mrow">${mat(sample.map((_, i) => rowOf(i)), { head: H, rowHead: idx() })}<span class="op">− <i>μ</i> =</span>` +
        mat(Xc, { head: H.map(h => h + "<sub>c</sub>"), rowHead: idx(), foot: [Array.from({ length: d }, (_, k) => num(Array.from({ length: n }, (_, i) => st.Xc[i * d + k]).reduce((s, x) => s + x, 0), 1))] }) + `</div>` +
        `<div class="eq"><i>X</i><sub>c</sub> = <i>X</i> − <b>1</b><i>μ</i><sup>T</sup>, μ = ${vecT(st.mu)}</div><div class="note">bottom row: column sums = 0</div>`;
    } else if (step === 3) {
      title = "4 · The covariance matrix";
      const [a, b] = entry, name = a === b ? `var(${H[a]})` : `cov(${H[a]}, ${H[b]})`;
      text = `<p>The <b>covariance matrix</b> <i>C</i> says how the measurements vary together. Diagonal entries are <b>variances</b> (average squared deviation from the mean); off-diagonal entries are <b>covariances</b>: for each fish multiply its two deviations – the area of the rectangle in the picture – and average.</p>` +
        `<p>Mostly green rectangles mean the two measurements grow together (big fish are thick), red ones that one grows while the other shrinks. <b>Click an entry of <i>C</i></b> to see how it is built.</p>`;
      const rows = sample.map((_, i) => [st.Xc[i * d + a], st.Xc[i * d + b], st.Xc[i * d + a] * st.Xc[i * d + b]]);
      const tot = rows.reduce((s, r) => s + r[2], 0);
      math = `<div class="eq"><i>C</i> = <span>1/(<i>n</i>−1)</span> <i>X</i><sub>c</sub><sup>T</sup><i>X</i><sub>c</sub></div>` +
        matrixHTML(Array.from({ length: d }, (_, r) => Array.from({ length: d }, (_, c) => num(C[r * d + c], 2))), { cls: "br", head: H, rowHead: H, cellCls: (r, c) => "pick" + (r === a && c === b ? " sel" : "") }) +
        `<div class="note">${name} = (1/${n - 1}) Σ ${H[a]}<sub>c</sub>·${H[b]}<sub>c</sub> = ${num(tot, 1)} / ${n - 1} = <b style="color:var(--accent)">${num(tot / (n - 1), 2)}</b></div>` +
        matrixHTML(rows.map(r => r.map(x => num(x, 1))), { cls: "br", head: [`${H[a]}<sub>c</sub>`, `${H[b]}<sub>c</sub>`, "product"], rowHead: idx(), hl, foot: [["", "Σ", num(tot, 1)]] });
    } else if (step === 4) {
      title = "5 · Eigenvectors of C";
      const v = j => Array.from({ length: d }, (_, k) => W[k * d + j]);
      text = `<p>PCA looks for the direction in which the cloud is most spread out. The <b>eigenvectors</b> <i>v</i> of <i>C</i> (<i>C v</i> = λ <i>v</i>) are exactly the principal axes; the <b>eigenvalue</b> λ is the variance of the fish along that axis.</p>` +
        (d === 2 ? `<p>Move the slider: the variance of the projections onto a direction is largest for the direction of PC1 and smallest for the one orthogonal to it. The axes are perpendicular because <i>C</i> is symmetric.</p>` : `<p>The three eigenvectors are perpendicular; PC1 is the direction of largest variance, PC2 the largest among those perpendicular to it, and so on. Drag the plot to rotate it.</p>`) +
        `<p class="note">${GROUP() ? `With all ${N} fish PC1 is ${vecT(Array.from({ length: d }, (_, k) => S.pca.W[k * d]))}; these ${n} fish give ${vecT(Array.from({ length: d }, (_, k) => W[k * d]))}.` : `With all ${N} fish PC1 would point at ${num(allAng, 0)}° in the (${H[0]}, ${H[1]}) plane; these ${n} fish give ${num(smpAng, 0)}°.`}</p>`;
      const rowsE = Array.from({ length: d }, (_, j) => `<tr><td style="color:${[AC, BC, "#b39ddb"][j]}">PC${j + 1}</td><td>\u03bb = ${num(lam[j], 2)}</td><td>${num(100 * ratio[j], 1)}%</td><td>v = (${v(j).map(x => num(x, 2)).join(", ")})</td></tr>`).join("");
      const Cv = j => Array.from({ length: d }, (_, r) => v(j).reduce((s, x, c) => s + C[r * d + c] * x, 0));
      math = `<div class="eq"><i>C</i> <i>v</i><sub><i>j</i></sub> = λ<sub><i>j</i></sub> <i>v</i><sub><i>j</i></sub></div>` +
        `<table class="mat" style="width:100%">${rowsE}</table>` +
        `<div class="note">check for PC1: <i>C v</i><sub>1</sub> = (${Cv(0).map(x => num(x, 2)).join(", ")}) and λ<sub>1</sub> <i>v</i><sub>1</sub> = (${v(0).map(x => num(lam[0] * x, 2)).join(", ")})</div>` +
        (d === 2 ? `<div class="eq" id="pcaVar"></div>` : "") +
        `<div class="bars">${Array.from({ length: d }, (_, j) => `<div class="bar"><span class="n">PC${j + 1}</span><span class="b"><i style="width:${100 * ratio[j]}%;background:${[AC, BC, "#b39ddb"][j]}"></i></span><span class="p">${num(100 * ratio[j], 1)}% of variance</span></div>`).join("")}</div>`;
    } else {
      title = "6 · The transformation";
      const kept = 100 * ratio.slice(0, keep).reduce((s, x) => s + x, 0);
      text = `<p>Put the eigenvectors as columns into a matrix <i>W</i>. The new coordinates of every fish are its projections onto the principal axes: <b><i>Z</i> = <i>X</i><sub>c</sub> <i>W</i></b>. Geometrically the cloud is <b>rotated</b> so that PC1 lies along the first axis (slider).</p>` +
        `<p>To reduce dimensions keep only the first <i>k</i> columns of <i>Z</i>: the fish are dropped onto the first ${keep < d ? keep : d} principal axes. ${keep < d ? `That keeps <b>${num(kept, 1)}%</b> of the variance.` : "With all columns nothing is lost."}</p>`;
      const Zr = sample.map((_, i) => Array.from(st.Z.slice(i * d, i * d + d)).map((x, j) => num(x)));
      math = `<div class="mrow">${mat(sample.map((_, i) => Array.from(st.Xc.slice(i * d, i * d + d))), { head: H.map(h => h + "<sub>c</sub>"), rowHead: idx() })}<span class="op">·</span>` +
        mat(Array.from({ length: d }, (_, r) => Array.from({ length: d }, (_, c) => num(W[r * d + c], 2))), { head: Array.from({ length: d }, (_, j) => `v<sub>${j + 1}</sub>`) }) + `<span class="op">=</span>` +
        mat(Zr, { head: Array.from({ length: d }, (_, j) => `PC${j + 1}`), rowHead: idx(), foot: [Array.from({ length: d }, (_, j) => num(lam[j], 2))] }) + `</div>` +
        `<div class="note">bottom row: variance of each new coordinate = the eigenvalues λ<sub>j</sub> (and the new coordinates are uncorrelated).</div>`;
    }
    $("pcaText").innerHTML = `<h3>${title}</h3>${text}`; $("pcaMath").innerHTML = `<h2>Numbers</h2>${math}`;
    controls(); varianceCurve();
  }

  // ── controls ──
  function controls() {
    const c = $("pcaCtl");
    let h = `<button id="pcaNew" title="pick another nine fish">↻ new sample of 9</button><label class="note"><input type="checkbox" id="pcaAll" ${showAll ? "checked" : ""}> show all ${N} fish</label>`;
    if (step === 2) h += `<span class="ctrl"><label>subtract the mean</label><input id="pcaT" type="range" min="0" max="1" step="0.01" value="${tC}"><button id="pcaPlay">▶ play</button></span>`;
    if (step === 4 && d === 2) h += `<span class="ctrl"><label>direction</label><input id="pcaTheta" type="range" min="0" max="180" step="0.5" value="${(theta * 180 / Math.PI + 180) % 180}"><span class="val" id="pcaThetaV"></span><button id="pcaSnap">snap to PC1</button></span>`;
    if (step === 5) h += `<span class="ctrl"><label>rotate to the PC axes</label><input id="pcaR" type="range" min="0" max="1" step="0.01" value="${tR}"><button id="pcaPlay">▶ play</button></span>` +
      `<span class="ctrl"><label>keep</label>${Array.from({ length: d }, (_, k) => `<button data-keep="${d - k}" class="${keep === d - k ? "on" : ""}">${d - k} PC${d - k > 1 ? "s" : ""}</button>`).join("")}</span>` +
      `<span class="ctrl"><label>drop the rest</label><input id="pcaF" type="range" min="0" max="1" step="0.01" value="${flat}" ${tR < 0.999 ? "disabled" : ""}></span>`;
    c.innerHTML = h;
    $("pcaNew").onclick = () => { sampleSeed++; dirty = true; hl = -1; refresh(); };
    $("pcaAll").onchange = e => { showAll = e.target.checked; draw(); };
    if ($("pcaT")) $("pcaT").oninput = e => { tC = +e.target.value; draw(); };
    if ($("pcaR")) $("pcaR").oninput = e => { tR = +e.target.value; $("pcaF").disabled = tR < 0.999; draw(); };
    if ($("pcaF")) $("pcaF").oninput = e => { flat = +e.target.value; draw(); };
    if ($("pcaPlay")) $("pcaPlay").onclick = () => tween(step === 2 ? "tC" : "tR");
    if ($("pcaTheta")) { $("pcaTheta").oninput = e => { theta = +e.target.value * Math.PI / 180; draw(); varianceCurve(); }; $("pcaSnap").onclick = () => { theta = Math.atan2(st.W[2], st.W[0]); $("pcaTheta").value = (theta * 180 / Math.PI + 180) % 180; draw(); varianceCurve(); }; }
    c.querySelectorAll("[data-keep]").forEach(b => b.onclick = () => { keep = +b.dataset.keep; controls(); render(); draw(); });
  }
  function tween(which) {
    cancelAnimationFrame(anim); const t0 = performance.now(), dur = 1600;
    const set = v => { if (which === "tC") { tC = v; if ($("pcaT")) $("pcaT").value = v; } else { tR = v; if ($("pcaR")) { $("pcaR").value = v; $("pcaF").disabled = v < 0.999; } } };
    set(0); const f = now => { const u = Math.min(1, (now - t0) / dur); set(u * u * (3 - 2 * u)); draw(); if (u < 1) anim = requestAnimationFrame(f); }; anim = requestAnimationFrame(f);
  }
  // variance of the projections onto the direction theta, as a curve over all directions (step 4, 2-D only)
  function varianceCurve() {
    const sub = $("pcaSub"); sub.hidden = !(step === 4 && d === 2);
    if (sub.hidden) return;
    const r = devicePixelRatio || 1, w = sub.clientWidth, h = sub.clientHeight; sub.width = w * r; sub.height = h * r;
    const q = sub.getContext("2d"); q.scale(r, r); q.clearRect(0, 0, w, h);
    const C = st.C, v = t => { const c = Math.cos(t), s = Math.sin(t); return c * c * C[0] + 2 * c * s * C[1] + s * s * C[3]; }, top = st.values[0] * 1.1;
    const X = t => 40 + (w - 56) * t / Math.PI, Y = val => h - 24 - (h - 40) * val / top;
    q.strokeStyle = "#444"; q.lineWidth = 1; q.beginPath(); q.moveTo(40, Y(0)); q.lineTo(w - 16, Y(0)); q.stroke();
    q.fillStyle = "#888"; q.font = "10px system-ui"; q.textAlign = "center";
    for (const a of [0, 45, 90, 135, 180]) { q.fillText(a + "°", X(a * Math.PI / 180), h - 10); }
    q.textAlign = "left"; q.fillText("variance of the projections vs. direction", 44, 12);
    q.strokeStyle = "#e6e6e6"; q.lineWidth = 2; q.beginPath(); for (let i = 0; i <= 180; i++) { const t = i * Math.PI / 180; i ? q.lineTo(X(t), Y(v(t))) : q.moveTo(X(t), Y(v(t))); } q.stroke();
    const a1 = ((Math.atan2(st.W[2], st.W[0]) + Math.PI) % Math.PI), a2 = (a1 + Math.PI / 2) % Math.PI;
    for (const [a, c, nm, val] of [[a1, AC, "PC1", st.values[0]], [a2, BC, "PC2", st.values[1]]]) { q.strokeStyle = c; q.setLineDash([3, 3]); q.beginPath(); q.moveTo(X(a), 18); q.lineTo(X(a), Y(0)); q.stroke(); q.setLineDash([]); q.fillStyle = c; q.beginPath(); q.arc(X(a), Y(val), 4, 0, TAU); q.fill(); q.textAlign = "center"; q.fillText(nm, X(a), 28); }
    const th = ((theta % Math.PI) + Math.PI) % Math.PI; q.fillStyle = "#fff"; q.beginPath(); q.arc(X(th), Y(v(th)), 5, 0, TAU); q.fill();
    $("pcaThetaV").textContent = `${fmt(th * 180 / Math.PI, 0)}°`;
    if ($("pcaVar")) $("pcaVar").innerHTML = `variance along ${fmt(th * 180 / Math.PI, 0)}° : <b>u</b><sup>T</sup><i>C</i><b>u</b> = ${fmt(v(th), 2)} &nbsp; (max λ<sub>1</sub> = ${fmt(st.values[0], 2)})`;
  }

  function refresh() { render(); draw(); }
  // step bar and plot interaction
  function buildStepBar() {
    $("pcaSteps").innerHTML = STEPS.map((s, i) => `<button data-s="${i}" class="${i === step ? "on" : ""}"><b>${i + 1}</b>${s}</button>`).join("") + `<span class="arrows">← → keys</span>`;
    $("pcaSteps").querySelectorAll("button").forEach(b => b.onclick = () => go(+b.dataset.s));
  }
  function go(i) {
    cancelAnimationFrame(anim); step = Math.max(0, Math.min(STEPS.length - 1, i)); hl = -1; fishTip(-1);
    buildStepBar(); if (step === 2) tC = 0; if (step === 5) { tR = 0; flat = 0; }
    refresh(); if (step === 2 || step === 5) tween(step === 2 ? "tC" : "tR");
  }
  function nearestSample(x, y) {
    let best = -1, bd = 28 * 28;
    for (let i = 0; i < 9; i++) { const p = plot.P(xf(rowOf(i))), dd = (p[0] - x) ** 2 + (p[1] - y) ** 2; if (dd < bd) { bd = dd; best = i; } }
    return best;
  }
  plot.bind({
    onMove: (x, y) => { const i = nearestSample(x, y); if (i !== hl) { hl = i; draw(); const rows = $("pcaMath").querySelectorAll("tr[data-r]"); rows.forEach(r => r.classList.toggle("hl", +r.dataset.r === i && r.closest("table").rows.length > 3)); } fishTip(i >= 0 ? sample[i] : -1, ...clientOf(x, y)); },
    onLeave: () => { hl = -1; fishTip(-1); draw(); }, onChange: draw, rotate: null, pan: false, zoomable: true,
  });
  const clientOf = (x, y) => { const r = cv.getBoundingClientRect(); return [r.left + x, r.top + y]; };
  $("pcaMath").addEventListener("click", e => {
    const td = e.target.closest("td.pick"); if (!td) return;
    entry = [+td.dataset.r, +td.dataset.c]; render(); draw();
  });
  $("pcaMath").addEventListener("mouseover", e => { const tr = e.target.closest("tr[data-r]"); if (!tr || tr.closest("table").rows.length < 4) return; if (step === 3 || step <= 2 || step === 5) { const i = +tr.dataset.r; if (i !== hl && i < 9) { hl = i; draw(); } } });
  $("pcaMath").addEventListener("mouseleave", () => { if (hl >= 0) { hl = -1; draw(); } });
  window.addEventListener("keydown", e => { if (CUR !== "pca" || /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName || "")) return; if (e.key === "ArrowRight") go(step + 1); else if (e.key === "ArrowLeft") go(step - 1); });

  return {
    onDim(what) { dirty = true; keep = DIM; if (what === "set") sampleSeed = 1; },
    show() { buildStepBar(); keep = Math.min(keep, DIM); if (dirty) { setup(); } refresh(); },
  };
}
